"""/data.py

Tools for loading data.
"""

import errno
import itertools
import json
import os
import pandas as pd


def load_json_from_path(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise error

    return None


def json_to_pandas_dataframe(dictionary):
    columns = list(dictionary.keys())
    rows = sorted(list(set(itertools.chain.from_iterable([
        list(dictionary[k].keys())
        for k in columns
    ]))), key=lambda x: int(x))

    # map(list, zip(*data)) is a quick trick to transpose
    # a list of lists
    data = list(map(list, zip(*([rows] + [
        [
            dictionary[column][r] if r in dictionary[column] else None
            for r in rows
        ]
        for column in columns
    ]))))
    df = pd.DataFrame(data, columns=['id'] + columns)
    df.set_index('id')

    return df


def load_training_test_data(training_data_path, test_data_path):
    return (
        json_to_pandas_dataframe(load_json_from_path(training_data_path)),
        json_to_pandas_dataframe(load_json_from_path(test_data_path))
    )

"""/utils/dataframe.py

Utilities to clean out the data
in the dataframe.
"""

import datetime
import functools
import itertools
import json
import numpy as np
import operator
import pandas as pd
import pprint
import re
import spacy

from collections import Counter, deque


def string_to_category_name(string):
    return string.lower().replace(" ", "_")


def categories_from_column(data_frame, column):
    return list(set(list(itertools.chain.from_iterable(
        data_frame[column].tolist()
    ))))


def normalize_whitespace(string):
    return re.sub(r"\s+", " ", string)


def normalize_category(category):
    return normalize_whitespace(re.sub(r"[\*\-\!\&]", " ", category.lower())).strip()


def normalize_categories(categories):
    return [
        normalize_category(c) for c in categories
    ]


def sliding_window(sequence, n):
    """Returns a sliding window of width n over data from sequence."""
    it = iter(sequence)
    window = deque((next(it, None) for _ in range(n)), maxlen=n)

    yield list(window)

    for element in it:
        window.append(element)
        yield list(window)


def create_ngrams(content, n):
    for ngram in sliding_window(content.split(), n):
        yield " ".join(ngram)


def create_ngrams_up_to_n(content, n):
    for i in range(n):
        yield from create_ngrams(content, i)


def count_ngrams_up_to_n(content, n):
    return Counter(list(create_ngrams_up_to_n(content, n)))


def remove_small_or_stopwords_from_ranking(ranking, nlp, min_len):
    for word, rank in ranking:
        if nlp.vocab[word].is_stop or len(word) < min_len:
            continue

        yield word, rank


def column_list_to_category_flags(data_frame, column, grams):
    categories = [
        "{}_{}".format(column, string_to_category_name(n))
        for n in grams
    ]
    row_cleaned_categories = [
        normalize_category(" ".join(r))
        for r in data_frame[column].tolist()
    ]
    category_flags = pd.DataFrame.from_records([
        [1 if gram in r else 0 for gram in grams]
        for r in row_cleaned_categories
    ], columns=categories)

    return pd.concat((data_frame, category_flags), axis=1)


def remap_column(data_frame, column, new_column, mapping):
    data_frame[new_column] = data_frame[column].transform(mapping)
    return data_frame


def remap_date_column_to_days_before(data_frame,
                                     column,
                                     new_column,
                                     reference_date):
    data_frame[new_column] = data_frame[column].transform(
        lambda x: (reference_date - datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).days
    )
    return data_frame


def map_categorical_column_to_category_ids(train_data_frame,
                                           test_data_frame,
                                           column,
                                           new_column,
                                           min_freq=1):
    category_counts = Counter(train_data_frame[column]) + Counter(test_data_frame[column])
    category_to_unknown_mapping = {
        category: category if count >= min_freq else "Unknown"
        for category, count in category_counts.items()
    }
    category_to_id_map = {
        category: i
        for i, category in enumerate(sorted([
            category_to_unknown_mapping[c] for c in
            (set(train_data_frame[column]) | set(test_data_frame[column]))
        ]))
    }
    id_to_category_map = {
        i: category
        for category, i in category_to_id_map.items()
    }

    return (
        category_to_unknown_mapping,
        category_to_id_map,
        id_to_category_map,
        remap_column(train_data_frame,
                     column,
                     new_column,
                     lambda x: category_to_id_map[category_to_unknown_mapping[x]]),
        remap_column(test_data_frame,
                     column,
                     new_column,
                     lambda x: category_to_id_map[category_to_unknown_mapping[x]])
    )


def remap_columns_with_transform(train_data_frame,
                                 test_data_frame,
                                 column,
                                 new_column,
                                 transform):
    """Remove some columns with a transform."""
    return (
        remap_column(train_data_frame,
                     column,
                     new_column,
                     transform),
        remap_column(test_data_frame,
                     column,
                     new_column,
                     transform)
    )


def normalize_description(description):
    """Normalize the description field."""
    description = description.lower()
    description = re.sub(r"<[^<]+?(>|$)", " ", description)
    description = re.sub(r"[0-9\-]+", " ", description)
    description = re.sub(r"[a-z0-9]@[a-z0-9]\.[a-z]", " ", description)
    description = re.sub(r"[\!]+", "! ", description)
    description = re.sub(r"[\-\:]", " ", description)
    description = re.sub("\*", " ", description)
    return re.sub(r"\s+", " ", description)


def add_epsilon(array):
    return np.array([a + 10e-10 if a == 0 else a for a in array])


def numerical_feature_engineering_on_dataframe(dataframe,
                                               numerical_columns):
    """Do per-dataframe feature engineering."""
    for lhs_column, rhs_column in itertools.combinations(numerical_columns, 2):
        dataframe['{}_add_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] + dataframe[rhs_column]
        dataframe['{}_sub_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] - dataframe[rhs_column]
        dataframe['{}_mul_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] * dataframe[rhs_column]
        dataframe['{}_div_{}'.format(lhs_column, rhs_column)] = dataframe[lhs_column] / add_epsilon(dataframe[rhs_column])

    return dataframe


def numerical_feature_engineering(train_data_frame,
                                  test_data_frame,
                                  numerical_columns):
    """Add, subtract, divide, multiply, exponentiate and take log."""
    return (
        numerical_feature_engineering_on_dataframe(train_data_frame,
                                                   numerical_columns),
        numerical_feature_engineering_on_dataframe(test_data_frame,
                                                   numerical_columns),
    )


def normalize_eastwest(eastwest):
    eastwest = eastwest.lower().strip()

    if not eastwest:
        return ""

    if eastwest[0] == "e":
        return "e"
    elif eastwest[0] == "w":
        return "w"
    else:
        return ""


STREET_MAPPING = {
    "st": "street",
    "ave": "avenue",
    "pl": "place",
    "rd": "road"
}


def normalize_name(name):
    m = re.match(r"(?P<address>[\w\s]+)(?P<st>st|street|ave|avenue|place|pl|road|rd).*",
                 name.lower().strip())

    if not m:
        return name.lower().strip()

    return "{address} {street}".format(
        address=m.groupdict()["address"].strip(),
        street=STREET_MAPPING.get(m.groupdict()["st"], m.groupdict()["st"])
    )


def normalize_address(address_dict):
    return "{eastwest} {name}".format(
        eastwest=normalize_eastwest(address_dict["eastwest"] or ""),
        name=normalize_name(address_dict["name"] or "")
    )


def parse_address_components_from_address(address):
    m = re.match(r"(?P<number>[0-9]*\s+)?\s*(?P<eastwest>East|West|E\s|W\s)?\s*(?P<name>[A-Za-z0-9\.\-\s]*)",
                 normalize_whitespace(address),
                 flags=re.IGNORECASE)
    return {
        "normalized": normalize_address(m.groupdict()) if m is not None else address
    }


def parse_address_components_for_column(dataframe, column):
    return pd.concat((dataframe, pd.DataFrame.from_records([
        {
            "{}_{}".format(column, key): value for key, value in
            parse_address_components_from_address(cell).items()
        }
        for cell in dataframe[column]
    ])), axis=1)


def parse_address_components(train_data_frame,
                             test_data_frame,
                             columns):
    return (
        functools.reduce(lambda df, c: parse_address_components_for_column(df,
                                                                           c),
                         columns,
                         train_data_frame),
        functools.reduce(lambda df, c: parse_address_components_for_column(df,
                                                                           c),
                         columns,
                         test_data_frame)
    )


def count_json_column(dataframe, column):
    return pd.DataFrame([
        len(c) for c in dataframe[column]
    ])


def count_json(train_data_frame,
               test_data_frame,
               column):
    train_data_frame["{}_count".format(column)] = count_json_column(train_data_frame,
                                                                    column)
    test_data_frame["{}_count".format(column)] = count_json_column(test_data_frame,
                                                                   column)

    return train_data_frame, test_data_frame

"""/feedback2vec.py

Given some raw string of feedback and a label (good/bad), build
a model capable of predicting whether the feedback was good or bad.

To do this we have a character encoder which encodes the
characters in the dataset as one-hot encoded letters. We then pass each
character in the stream through an embedding layer, then through a forward
and backward LSTM. The output is then passed to a fully connected
layer which predicts if the feedback was good or bad.

The theory is that we learn representations in the embedding layer which
put the feedback into an appropriate vector space.
"""

import argparse
import torch
import math
import numpy as np
import pandas as pd
import json

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn.utils import shuffle


def maybe_cuda(tensor):
    """CUDAifies a tensor if possible."""
    if torch.cuda.is_available():
        return tensor.cuda()

    return tensor.cpu()


class Doc2Vec(nn.Module):
    """Doc2Vec model, based on Tweet2Vec."""

    def __init__(self,
                 embedding_size,
                 hidden_layer_size,
                 vocab_size,
                 output_size,
                 batch_size):
        super().__init__()

        self.hidden_dim = hidden_layer_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # One hidden layers for each direction
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden = (maybe_cuda(torch.randn(2, batch_size, self.hidden_dim)),
                       maybe_cuda(torch.randn(2, batch_size, self.hidden_dim)))
        self.lstm = nn.LSTM(embedding_size,
                            self.hidden_dim,
                            num_layers=1,
                            bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim * 2, output_size)

    def sentence_embedding(self, sentence):
        self.hidden = (maybe_cuda(torch.randn(2, self.batch_size, self.hidden_dim)),
                       maybe_cuda(torch.randn(2, self.batch_size, self.hidden_dim)))
        embeddings = self.embedding(sentence)
        out, self.hidden = self.lstm(embeddings.view(-1, self.batch_size, self.embedding_size),
                                     self.hidden)
        added = self.hidden[0] + self.hidden[1]
        return added / torch.norm(added)

    def forward(self, sentence):
        embedding = self.sentence_embedding(sentence)
        lin = self.linear(embedding.view(-1, self.hidden_dim * 2))
        return F.softmax(lin, dim=1)


def train_model(model, optimizer, epochs, sentence_tensors, label_tensors):
    for epoch in range(epochs):
        total_loss = 0
        loss_criterion = nn.CrossEntropyLoss()

        shuffled_sentence_tensors, shuffled_label_tensors = shuffle(
            sentence_tensors, label_tensors
        )

        progressable_tensors = tqdm(
            zip(shuffled_sentence_tensors, shuffled_label_tensors),
            total=len(shuffled_label_tensors),
            desc="Processing sentence vectors"
        )

        for sentence_tensor, label_tensor in progressable_tensors:
            optimizer.zero_grad()

            preds = model(sentence_tensor)
            loss = loss_criterion(preds,
                                  label_tensor)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progressable_tensors.set_postfix(loss=loss.item())

        print('Epoch', epoch, 'total loss', total_loss)

        with torch.no_grad():
            print("Model Accuracy:",
                  compute_model_accuracy(model,
                                         shuffled_sentence_tensors[0],
                                         shuffled_label_tensors[0]))

    return model


def characters_to_one_hot_lookups(all_characters):
    set_characters = set(all_characters)
    character_to_one_hot = {
        c: i for i, c in enumerate(sorted(set_characters))
    }
    one_hot_to_character = {
        c: i for i, c in enumerate(sorted(set_characters))
    }

    return character_to_one_hot, one_hot_to_character


def character_sequence_to_matrix(sentence, character_to_one_hot):
    return np.array([character_to_one_hot[c] for c in sentence])


def compute_model_accuracy(model, sentence_tensors, label_tensors):
    """A floating point value of how accurate the model was at predicting each label."""
    predictions = np.argmax(model(maybe_cuda(sentence_tensors)).detach().cpu().numpy(), axis=1).flatten()
    labels = label_tensors.detach().cpu().numpy().flatten()
    return len([p for p in (predictions == labels) if p == True]) / len(predictions)


def pad_sentence(sentence, padding):
    truncated = sentence[:padding]
    return truncated + (" " * (padding - len(truncated)))


def to_batches(sentences, batch_size, pad_value):
    for i in range(math.ceil(len(sentences) / batch_size)):
        batch = [
            sentences[i * batch_size + j]
            for j in range(min((batch_size, len(sentences[i * batch_size:]))))
        ]
        padding = [
            pad_value
            for k in range(max(0, batch_size - len(sentences[i * batch_size:])))
        ]
        yield torch.stack(batch + padding, dim=0)


def documents_to_vectors_model(train_documents,
                               test_documents,
                               labels,
                               epochs,
                               parameters,
                               learning_rate,
                               load=None,
                               save=None,
                               sentence_length=1000,
                               batch_size=200):
    """Convert some documents to vectors based on labels."""
    character_to_one_hot, one_hot_to_character = characters_to_one_hot_lookups(
        "".join(train_documents) + "".join(test_documents)
    )
    train_sentence_tensors = list(to_batches([
        maybe_cuda(torch.tensor(character_sequence_to_matrix(pad_sentence(sentence, sentence_length),
                                                             character_to_one_hot), dtype=torch.long))
        for sentence in train_documents
    ], batch_size, maybe_cuda(torch.tensor([character_to_one_hot[" "] for i in range(sentence_length)]))))

    label_tensors = list(to_batches([maybe_cuda(torch.tensor(i)) for i in labels],
                                    batch_size,
                                    maybe_cuda(torch.tensor(0))))

    model = maybe_cuda(Doc2Vec(parameters,
                               parameters * 2,
                               len(character_to_one_hot.keys()), max(labels) + 1,
                               batch_size))

    if not load:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        try:
            train_model(model,
                        optimizer,
                        epochs,
                        train_sentence_tensors,
                        label_tensors)
        except KeyboardInterrupt:
            print("Interrupted, saving current state now")
    else:
        model.load_state_dict(torch.load(load))

    if save:
        torch.save(model.state_dict(), save)

    return character_to_one_hot, one_hot_to_character, model


def generate_document_vector_embeddings_from_model(model,
                                                   character_to_one_hot,
                                                   sentences,
                                                   sentence_length,
                                                   batch_size):
    # Generate the embeddings for all of our documents now
    sentence_tensors = list(to_batches([
        maybe_cuda(torch.tensor(character_sequence_to_matrix(pad_sentence(sentence, sentence_length),
                                                             character_to_one_hot), dtype=torch.long))
        for sentence in sentences
    ], batch_size, maybe_cuda(torch.tensor([character_to_one_hot[" "] for i in range(sentence_length)]))))
    with torch.no_grad():
        return np.row_stack([
            model.sentence_embedding(sentence_tensor).detach().cpu().numpy()
            for sentence_tensor in sentence_tensors
        ]).reshape(-1, model.hidden_dim * 2)[:len(sentences)]


def column_to_doc_vectors(train_data_frame,
                          test_data_frame,
                          description_column,
                          target_column,
                          document_vector_column,
                          epochs=100,
                          parameters=40,
                          learning_rate=0.01,
                          load=None,
                          save=None,
                          sentence_length=1000,
                          batch_size=200):
    """Convert some description columns to document vector columns."""
    train_descriptions = list(train_data_frame[description_column])
    test_descriptions = list(test_data_frame[description_column])
    labels = list(train_data_frame[target_column])

    character_to_one_hot, one_hot_to_character, model = documents_to_vectors_model(
        train_descriptions,
        test_descriptions,
        labels,
        epochs,
        parameters,
        learning_rate,
        load=load,
        save=save,
        sentence_length=sentence_length,
        batch_size=batch_size
    )

    train_description_vectors = pd.DataFrame(
        generate_document_vector_embeddings_from_model(
            model,
            character_to_one_hot,
            train_descriptions,
            sentence_length,
            batch_size
        )
    )

    test_description_vectors = pd.DataFrame(
        generate_document_vector_embeddings_from_model(
            model,
            character_to_one_hot,
            test_descriptions,
            sentence_length,
            batch_size
        )
    )

    return (
        pd.concat((train_data_frame, train_description_vectors), axis=1),
        pd.concat((test_data_frame, test_description_vectors), axis=1)
    )


"""/utils/model.py

Models to use with the data.

This module creates pipelines, which depending on the underlying
model, will one-hot encode categorical data or just leave it as is,
converting it to a number. All the returned models satisfy the
sklearn estimator API, so we can use them with grid search/evolutionary
algorithms for hyperparameter search if we want to.
"""

import numpy as np
import pandas as pd

import xgboost as xgb

from category_encoders import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def sklearn_pipeline_steps(categorical_columns, verbose=False):
    return (
        ('one_hot',
         OneHotEncoder(cols=categorical_columns, verbose=verbose)),
        ('scaling', StandardScaler())
    )


def basic_logistic_regression_pipeline(categorical_columns,
                                       verbose=False):
    return Pipeline((
        *sklearn_pipeline_steps(categorical_columns, verbose=verbose),
        ('xgb', xgb.XGBClassifier(
            n_estimators=1000,
            seed=42,
            objective='multi:softprob',
            subsample=0.8,
            colsample_bytree=0.8,
        ))
    ))


def calculate_statistics(statistics, test_labels, predictions):
    return {
        k: s(test_labels, predictions)
        for k, s in statistics.items()
    }


def format_statistics(calculated_statistics):
    return ", ".join([
        "{0}: {1:.2f}".format(k, s)
        for k, s in calculated_statistics.items()
    ])


def prediction_accuracy(labels, predictions):
    return (
        len([a for a, b in zip(labels, predictions) if a == b]) / len(predictions)
    )


def fit_one_split(model, features, labels, statistics, train_index, test_index):
    train_data, train_labels = features.iloc[train_index], labels[train_index]
    test_data, test_labels = features.iloc[test_index], labels[test_index]

    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    return (
        test_labels,
        predictions,
        calculate_statistics(statistics, test_labels, predictions)
    )


def test_model_with_k_fold_cross_validation(model,
                                            features,
                                            labels,
                                            statistics,
                                            n_splits=5,
                                            random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_labels, predictions = [], []

    for i, (train_index, test_index) in enumerate(kf.split(features, labels)):
        fold_test_labels, fold_predictions, calculated_statistics = fit_one_split(
            model,
            features,
            labels,
            statistics,
            train_index,
            test_index
        )
        print('Fold', i, format_statistics(calculated_statistics))
        test_labels.extend(fold_test_labels)
        predictions.extend(fold_predictions)

    return (
        calculate_statistics(statistics, test_labels, predictions),
        test_labels,
        predictions,
        model
    )


def get_prediction_probabilities_with_columns(model,
                                              test_dataframe,
                                              keep_columns):
    return pd.concat((test_dataframe[keep_columns],
                      pd.DataFrame(model.predict_proba(test_dataframe.drop(keep_columns, axis=1)))),
                     axis=1)

import datetime
import itertools
import json
import operator
import os
import pandas as pd
import pprint
import numpy as np
import re
import spacy
import torch

from collections import Counter, deque
from sklearn.metrics import mean_squared_error

nlp = spacy.load("en")
torch.cuda.is_available()
(TRAIN_DATAFRAME, TEST_DATAFRAME) = \
  load_training_test_data(os.path.join('..', 'input', 'train.json'),
                          os.path.join('..', 'input', 'test.json'))
TRAIN_DATAFRAME.head()
normalized_categories = sorted(normalize_categories(categories_from_column(TRAIN_DATAFRAME, 'features')))
normalized_categories[:50]
most_common_ngrams = sorted(count_ngrams_up_to_n(" ".join(normalized_categories), 3).most_common(),
                            key=lambda x: (-x[1], x[0]))
most_common_ngrams[:50]
most_common_ngrams = sorted(list(remove_small_or_stopwords_from_ranking(most_common_ngrams, nlp, 3)),
                            key=lambda x: (-x[1], x[0]))
most_common_ngrams[:50]
TRAIN_DATAFRAME = column_list_to_category_flags(TRAIN_DATAFRAME, 'features', list(map(operator.itemgetter(0), most_common_ngrams[:100])))
TEST_DATAFRAME = column_list_to_category_flags(TEST_DATAFRAME, 'features', list(map(operator.itemgetter(0), most_common_ngrams[:100])))
TRAIN_DATAFRAME.head(5)
TRAIN_DATAFRAME = remap_date_column_to_days_before(TRAIN_DATAFRAME, "created", "created_days_ago", datetime.datetime(2017, 1, 1))
TEST_DATAFRAME = remap_date_column_to_days_before(TEST_DATAFRAME, "created", "created_days_ago", datetime.datetime(2017, 1, 1))
TRAIN_DATAFRAME["created_days_ago"].head(5)
TRAIN_DATAFRAME = remap_column(TRAIN_DATAFRAME, "interest_level", "label_interest_level", lambda x: {
    "high": 0,
    "medium": 1,
    "low": 2
}[x])
# The TEST_DATAFRAME does not have an interest_level column, so we
# instead add it and replace it with all zeros
TEST_DATAFRAME["label_interest_level"] = 0
TRAIN_DATAFRAME["label_interest_level"].head(5)
(BUILDING_ID_UNKNOWN_REMAPPING,
 BUILDING_ID_TO_BUILDING_CATEGORY,
 BUILDING_CATEGORY_TO_BUILDING_ID,
 TRAIN_DATAFRAME,
 TEST_DATAFRAME) = map_categorical_column_to_category_ids(
    TRAIN_DATAFRAME,
    TEST_DATAFRAME,
    'building_id',
    'building_id_category',
    min_freq=40
)
(MANAGER_ID_UNKNOWN_REMAPPING,
 MANAGER_ID_TO_MANAGER_CATEGORY,
 MANAGER_CATEGORY_TO_MANAGER_ID,
 TRAIN_DATAFRAME,
 TEST_DATAFRAME) = map_categorical_column_to_category_ids(
    TRAIN_DATAFRAME,
    TEST_DATAFRAME,
    'manager_id',
    'manager_id_category',
    min_freq=40
)
import imp

TRAIN_DATAFRAME, TEST_DATAFRAME = parse_address_components(
    TRAIN_DATAFRAME,
    TEST_DATAFRAME,
    [
        "display_address",
        "street_address"
    ]
)
(DISP_ADDR_ID_UNKNOWN_REMAPPING,
 DISP_ADDR_TO_DISP_ADDR_CATEGORY,
 DISP_ADDR_CATEGORY_TO_DISP_ADDR_ID,
 TRAIN_DATAFRAME,
 TEST_DATAFRAME) = map_categorical_column_to_category_ids(
    TRAIN_DATAFRAME,
    TEST_DATAFRAME,
    'display_address_normalized',
    'display_address_category',
    min_freq=10
)
TRAIN_DATAFRAME, TEST_DATAFRAME = count_json(
    TRAIN_DATAFRAME,
    TEST_DATAFRAME,
    "photos"
)
NUMERICAL_COLUMNS = [
    'bathrooms',
    'bedrooms',
    'price',
    'latitude',
    'longitude',
    'photos_count'
]

TRAIN_DATAFRAME, TEST_DATAFRAME = numerical_feature_engineering(
    TRAIN_DATAFRAME,
    TEST_DATAFRAME,
    NUMERICAL_COLUMNS
)
TRAIN_DATAFRAME, TEST_DATAFRAME = remap_columns_with_transform(
    TRAIN_DATAFRAME,
    TEST_DATAFRAME,
    'description',
    'clean_description',
    normalize_description
)
#TRAIN_DATAFRAME, TEST_DATAFRAME = column_to_doc_vectors(
#    TRAIN_DATAFRAME,
#    TEST_DATAFRAME,
#    'clean_description',
#    'label_interest_level',
#    'description_vector',
#    epochs=1000,
#    parameters=200,
#    learning_rate=0.01,
#    save='word_embedding.npy',
#    batch_size=100
#)
DROP_COLUMNS = [
    'id',
    'created',
    'building_id',
    'clean_description',
    'description',
    'features',
    'display_address',
    'display_address_normalized',
    # We keep listing_id in the dataframe
    # since we'll need it later
    # 'listing_id',
    'manager_id',
    'photos',
    'street_address',
    'street_address_normalized',
    'interest_level',
]
TRAIN_DATAFRAME = TRAIN_DATAFRAME.drop(DROP_COLUMNS, axis=1)
# TEST_DATAFRAME doesn't have interest_level, so we remove it
# from the DROP_COLUMNS
TEST_DATAFRAME = TEST_DATAFRAME.drop(DROP_COLUMNS[:-1], axis=1)
TRAIN_DATAFRAME.head(5)
[(i, x) for i, x in enumerate(np.all(np.isfinite(TRAIN_DATAFRAME.drop(['listing_id', 'label_interest_level'], axis=1)), axis=0)) if not x]
TRAIN_DATAFRAME.drop(['listing_id', 'label_interest_level'], axis=1).columns[119]
TEST_DATAFRAME.columns[109]
np.argwhere(~np.isfinite(TEST_DATAFRAME.drop(['listing_id', 'label_interest_level'], axis=1).as_matrix()))
CATEGORICAL_COLUMNS = ('building_id_category', 'manager_id_category')
(LINEAR_MODEL_STATISTICS,
 LINEAR_MODEL_LABELS,
 LINEAR_MODEL_PREDICTIONS,
 LINEAR_MODEL) = test_model_with_k_fold_cross_validation(
    basic_logistic_regression_pipeline(CATEGORICAL_COLUMNS),
    TRAIN_DATAFRAME.drop(['listing_id', 'label_interest_level'], axis=1).astype(float),
    TRAIN_DATAFRAME['label_interest_level'],
    {
        "mse_loss": mean_squared_error,
        "accuracy": prediction_accuracy
    },
    n_splits=2
)

print('Linear Model', format_statistics(LINEAR_MODEL_STATISTICS))
pd.set_option("display.max_columns",200)
pd.set_option("display.max_rows",500)
TRAIN_DATAFRAME[TRAIN_DATAFRAME.isnull().any(axis=1)]
table = get_prediction_probabilities_with_columns(LINEAR_MODEL,
                                                  TEST_DATAFRAME.drop('label_interest_level', axis=1),
                                                  ['listing_id'])
table.columns = ['listing_id', 'high', 'medium', 'low']
table.to_csv('submission.csv', columns=['listing_id', 'high', 'medium', 'low'], index=False)