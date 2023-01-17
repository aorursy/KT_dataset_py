import pandas as pd



train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv', na_filter=False)

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv', na_filter=False)
train_df.sample(10)
train_df.shape
train_df['sentiment'].value_counts()
test_df.sample(10)
test_df.shape
def jaccard(str1, str2):

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    if len(a) + len(b) == len(c):

        return 0.0

    return float(len(c)) / (len(a) + len(b) - len(c))
def eval_jaccard(y_pred, y_true):

    return sum(jaccard(sp, st) for sp, st in zip(y_pred, y_true)) / len(y_true)
eval_jaccard(train_df["text"], train_df["selected_text"])
SEED = 2020
def to_spacy_format(X, Y, S):

    data = []

    for x, y, s in zip(X, Y, S):

        if not x:

            print(f"'{x}': no tweet given")

            continue

        if y and (s == "positive" or s == "negative"):            

            start = x.find(y)

            if start < 0:

                print("Can't find a phrase: skipping...")

                continue

            ex = (x, {"entities": [(start, start + len(y), s)]})

        else:

            ex = (x, {"entities": []})

        data.append(ex)

    return data
def from_spacy_format(data):

    X, Y = [], []

    for x, ann in data:

        X.append(x)

        if ann["entities"]:

            s = []

            for e in ann["entities"]:

                s.append(x[e[0]: e[1]])                

            Y.append(" ".join(s))

        else:

            Y.append(x)

    return X, Y
def predict_spacy(nlp, X):

    Y = []

    for x in X:

        doc = nlp(x)

        if doc.ents:

            y = " ".join([e.text for e in doc.ents])

        else:

            y = x

        Y.append(y)

    return Y
def baseline(data):

    X, Y = from_spacy_format(data)

    return eval_jaccard(X, Y)
from pathlib import Path



model_dir = Path("/kaggle/working/model/")

model_dir.mkdir(exist_ok=True)
import random

import numpy as np

from spacy.util import minibatch, compounding, decaying





def train_model(nlp, train_data, model_dir,

                valid_data=None, blank=False,

                epochs=100, 

                dropouts=decaying(0.6, 0.2, 1e-6), 

                batch_sizes=compounding(1.0, 16.0, 1.0 + 1e-4)

               ):

    

    x_train, y_train = from_spacy_format(train_data)    

    

    if valid_data:

        best_score = 0

        x_valid, y_valid = from_spacy_format(valid_data)

    

    random.seed(SEED)

    

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    else:

        ner = nlp.get_pipe("ner")

        

    ner.add_label("positive")

    ner.add_label("negative")

    

    # get names of other pipes to disable them during training

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    

    # only train NER

    with nlp.disable_pipes(*other_pipes):

        

        # reset and initialize the weights randomly

        if blank:

            optimizer = nlp.begin_training()

        else:

            optimizer = nlp.resume_training()

        

        for i in range(epochs):

            random.shuffle(train_data)

            losses = {}

            

            # batch up the examples using spaCy's minibatch

            batches = minibatch(train_data, size=batch_sizes)

            for batch in batches:

                texts, annotations = zip(*batch)

                dropout = next(dropouts)

                

                nlp.update(

                    texts,  # batch of texts

                    annotations,  # batch of annotations

                    sgd=optimizer,

                    drop=dropout,  # dropout - make it harder to memorise data

                    losses=losses,

                )

            

            loss = losses['ner']

            batch_size = next(batch_sizes)

            dropout = next(dropouts)

            

            y_pred = predict_spacy(nlp, x_train)

            tr_score = eval_jaccard(y_pred, y_train)

            

            message = f"epoch {i + 1}: batch_size={batch_size:.1f}, dropout={dropout:.3f}, loss={loss:.3f}, tr_score={tr_score:.3f}"

            

            if valid_data:

                y_pred = predict_spacy(nlp, x_valid)

                val_score = eval_jaccard(y_pred, y_valid)          

                

                if val_score > best_score:

                    best_score = val_score

                    if not model_dir.exists():

                        model_dir.mkdir(parents=True)

                    nlp.to_disk(model_dir)

                

                message = f"{message}, val_score={val_score:.3f}"

            

            print(message)
import spacy

from sklearn.model_selection import train_test_split, StratifiedKFold





def cross_validate(df, model_dir, folds=10, epochs=10, blank=True):

    print(f"Cross-validation: folds={folds}")

        

    data = to_spacy_format(df.text, df.selected_text, df.sentiment)

    

    models = []    

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    

    for fold, (train_idx, valid_idx) in enumerate(skf.split(data, df.sentiment), start=1): 

        print(f'Fold: {fold}')

        

        train_set = set(train_idx)

        valid_set = set(valid_idx)

        

        train_data = [x for i, x in enumerate(data) if i in train_set]

        valid_data = [x for i, x in enumerate(data) if i in valid_set]

        

        baseline_score = baseline(valid_data)

        print(f"Baseline={baseline_score:.3f}")

        

        if blank:

            model = spacy.blank("en")

        else:

            model = spacy.load("en_core_web_sm")

        

        dropouts=decaying(0.5, 0.3, 1e-6)

        batch_sizes=compounding(4.0, 16.0, 1.0 + 1e-4)

        

        model_path = Path(model_dir.joinpath(f"spacy_fold_{fold}"))

        model_path.mkdir(parents=True)

        

        train_model(model, train_data, model_path, valid_data,

                    dropouts=dropouts, batch_sizes=batch_sizes, 

                    blank=blank, epochs=epochs)

        

        models.append(model)

    

    return models
train_pn_df = train_df[(train_df["sentiment"] == "positive") | (train_df["sentiment"] == "negative")]
%%time



models = cross_validate(train_pn_df, model_dir, folds=10, epochs=7, blank=True)
import numpy as np

import math



def mode(array, prefer_min=True):

    (values, counts) = np.unique(array, return_counts=True)

    max_count = np.max(counts)

    best = math.inf if prefer_min else - math.inf

    for i in range(len(values)):

        if counts[i] == max_count:

            if prefer_min and best > values[i]:

                best = values[i]

            if not prefer_min and best < values[i]:

                best = values[i]

    return best
def majority_vote(ys, x):

    starts, ends = [], []

    for y in ys:

        start = x.find(y)

        end = start + len(y)

        starts.append(start)

        ends.append(end)

    y = x[mode(starts, prefer_min=True): mode(ends, prefer_min=False)]

    return y
def predict_all(models):

    Y = []

    for x, s in zip(test_df.text, test_df.sentiment):

        if s == "neutral":

            Y.append(x)

        else:

            ys = []

            for model in models:

                ys.append(predict_spacy(model, [x])[0])

            Y.append(majority_vote(ys, x))

    return Y
%%time



Y_pred = predict_all(models)
test_df["selected_text"] = Y_pred
test_df.sample(10)
sub_df = pd.DataFrame({

    'textID': test_df['textID'],

    'selected_text': Y_pred

})
sub_df.to_csv('/kaggle/working/submission.csv', index=False)
sub_df.sample(10)
import shutil

shutil.rmtree(model_dir)