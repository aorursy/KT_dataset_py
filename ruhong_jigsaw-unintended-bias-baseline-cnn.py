# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from collections import Counter

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding

from keras.layers import Input

from keras.layers import Conv1D

from keras.layers import MaxPooling1D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Dense

from keras.optimizers import RMSprop

from keras.models import Model

from keras.models import load_model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/glove-global-vectors-for-word-representation"))

print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))



# Any results you write to the current directory are saved as output.
# Config

SEED = 11

TARGET = 'target'

TEXT = 'comment_text'

PREDICTION = 'prediction'

IDENTITY = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim',

            'black', 'white', 'psychiatric_or_mental_illness']

MAX_NUM_WORDS = 10000

MAX_SEQUENCE_LENGTH = 250

TEST_SIZE = 0.01

EMBEDDINGS_PATH = '../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'

EMBEDDINGS_DIMENSION = 100

DROPOUT_RATE = 0.3

LEARNING_RATE = 1e-5

NUM_EPOCHS = 10

BATCH_SIZE = 128

CONTRACTIONS = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not", 

                "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",

                "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will",

                "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",

                "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",

                "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",

                "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",

                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",

                "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have",

                "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",

                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",

                "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have",

                "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",

                "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",

                "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",

                "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",

                "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", "it's": "it is" }

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')



# Make sure all comment_text values are strings

train[TEXT] = train[TEXT].astype(str) 



# Convert taget and identity columns to booleans

def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    

def convert_dataframe_to_bool(df):

    bool_df = df.copy()

    for col in [TARGET] + IDENTITY:

        convert_to_bool(bool_df, col)

    return bool_df



train = convert_dataframe_to_bool(train)

train.shape
# All comments must be truncated or padded to be the same length.

def pad_text(texts, tokenizer):

    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)





def _embeddings_index(path):

    """Load embeddings"""

    embeddings_index = {}

    with open(path) as f:

        for line in f:

            values = line.split()

            word = values[0]

            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs

    return embeddings_index





def _unknown_words(vocab, embeddings_index):

    known_words_count_unique = 0

    known_words_count = 0

    unknown_words_count = 0

    unknown_words = Counter()

    for word in vocab.keys():

        em = embeddings_index.get(word, None)

        c = vocab.get(word)

        if em is None:

            unknown_words_count += c

            unknown_words[word] += c

        else:

            known_words_count_unique += 1

            known_words_count += c

    print('Found embeddings for {:.3%} of vocab'.format(known_words_count_unique / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(known_words_count / (known_words_count + unknown_words_count)))

    return unknown_words





def add_lowercase_to_embeddings(embeddings_index, vocab):

    c = 0

    for word in vocab:

        l = word.lower()

        if word in embeddings_index and l not in embeddings_index:  

            embeddings_index[l] = embeddings_index[word]

            c += 1

    print('add_lowercase_to_embeddings: added {} words'.format(c))





def clean_contractions(str, mapping=CONTRACTIONS):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        str = str.replace(s, "'")

    res = ' '.join([mapping[t] if t in mapping else t for t in str.split(' ')])

    return res

    



train[TEXT] = train[TEXT].apply(lambda x: clean_contractions(x))

# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(train[TEXT])

vocab = Counter(tokenizer.word_counts)

print('most common vocab: {}'.format(vocab.most_common(100)))







    
embeddings_index = _embeddings_index(EMBEDDINGS_PATH)

# Glove does not have contractions such as "it's", "that's"

#print("it is={}".format("it is" in embeddings_index))

#print("that is={}".format("that is" in embeddings_index))

add_lowercase_to_embeddings(embeddings_index, vocab)

unknown_words = _unknown_words(vocab, embeddings_index)

print('{} unknown words'.format(len(unknown_words)))

print('most common unknown words: {}'.format(unknown_words.most_common(1000)))
train_df, validate_df = train_test_split(train, test_size=TEST_SIZE, random_state=SEED)

train_df.shape, validate_df.shape
def _embedding_matrix(embeddings_index, tokenizer, dimension_size):

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, dimension_size))

    for word, i in tokenizer.word_index.items():

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    return embedding_matrix





def train_model(train_df, validate_df, tokenizer, embeddings_index):

    """Define and train a Convolutional Neural Net for classifying toxic comments"""

    # Prepare data

    train_text = pad_text(train_df[TEXT], tokenizer)

    train_labels = to_categorical(train_df[TARGET])

    validate_text = pad_text(validate_df[TEXT], tokenizer)

    validate_labels = to_categorical(validate_df[TARGET])



    embedding_matrix = _embedding_matrix(embeddings_index, tokenizer, dimension_size=EMBEDDINGS_DIMENSION)

    



    # Create model layers.

    def get_convolutional_neural_net_layers():

        """Returns (input_layer, output_layer)"""

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        embedding_layer = Embedding(len(tokenizer.word_index) + 1,

                                    EMBEDDINGS_DIMENSION,

                                    weights=[embedding_matrix],

                                    input_length=MAX_SEQUENCE_LENGTH,

                                    trainable=False)

        x = embedding_layer(sequence_input)

        x = Conv1D(128, 2, activation='relu', padding='same')(x)

        x = MaxPooling1D(5, padding='same')(x)

        x = Conv1D(128, 3, activation='relu', padding='same')(x)

        x = MaxPooling1D(5, padding='same')(x)

        x = Conv1D(128, 4, activation='relu', padding='same')(x)

        x = MaxPooling1D(40, padding='same')(x)

        x = Flatten()(x)

        x = Dropout(rate=DROPOUT_RATE, seed=SEED)(x)

        x = Dense(128, activation='relu')(x)

        preds = Dense(2, activation='softmax')(x)

        return sequence_input, preds



    # Compile model.

    print('compiling model...')

    input_layer, output_layer = get_convolutional_neural_net_layers()

    model = Model(input_layer, output_layer)

    model.compile(loss='categorical_crossentropy',

                  optimizer=RMSprop(lr=LEARNING_RATE),

                  metrics=['acc'])



    # Train model.

    print('training model...')

    history = model.fit(train_text,

                  train_labels,

                  batch_size=BATCH_SIZE,

                  epochs=NUM_EPOCHS,

                  validation_data=(validate_text, validate_labels),

                  verbose=2)



    return model, history



model, history = train_model(train_df, validate_df, tokenizer, embeddings_index)

model.summary()
plt.figure(figsize=(12,8))

plt.plot(history.history['acc'], label='Train Accuracy')

plt.plot(history.history['val_acc'], label='Validation Accuracy')

plt.show()
validate_df[PREDICTION] = model.predict(pad_text(validate_df[TEXT], tokenizer))[:, 1]

validate_df.head()
class JigsawEvaluator:



    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):

        self.y = (y_true >= 0.5).astype(int)

        self.y_i = (y_identity >= 0.5).astype(int)

        self.n_subgroups = self.y_i.shape[1]

        self.power = power

        self.overall_model_weight = overall_model_weight



    @staticmethod

    def _compute_auc(y_true, y_pred):

        try:

            return roc_auc_score(y_true, y_pred)

        except ValueError:

            return np.nan



    def _compute_subgroup_auc(self, i, y_pred):

        """Restrict the data set to only the examples that mention the specific identity subgroup.

        A low value in this metric means the model does a poor job of distinguishing

        between toxic and non-toxic comments that mention the identity."""

        mask = self.y_i[:, i] == 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def _compute_bpsn_auc(self, i, y_pred):

        """BPSN (Background Positive, Subgroup Negative) AUC:

        Restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not.

        A low value in this metric means that the model confuses non-toxic examples

        that mention the identity with toxic examples that do not,

        likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity."""

        mask = self.y_i[:, i] + self.y == 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def _compute_bnsp_auc(self, i, y_pred):

        """BNSP (Background Negative, Subgroup Positive) AUC:

        Restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not.

        A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not,

        likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity."""

        mask = self.y_i[:, i] + self.y != 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def _compute_bias_metrics_for_model(self, y_pred):

        records = np.zeros((3, self.n_subgroups))

        for i in range(self.n_subgroups):

            records[0, i] = self._compute_subgroup_auc(i, y_pred)

            records[1, i] = self._compute_bpsn_auc(i, y_pred)

            records[2, i] = self._compute_bnsp_auc(i, y_pred)

        return records



    def _calculate_overall_auc(self, y_pred):

        return roc_auc_score(self.y, y_pred)



    def _power_mean(self, array):

        total = sum(np.power(array, self.power))

        return np.power(total / len(array), 1 / self.power)



    def get_final_metric(self, y_pred):

        bias_metrics = self._compute_bias_metrics_for_model(y_pred)

        bias_score = np.average([

            self._power_mean(bias_metrics[0]),

            self._power_mean(bias_metrics[1]),

            self._power_mean(bias_metrics[2])

        ])

        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)

        bias_score = (1 - self.overall_model_weight) * bias_score

        return overall_score + bias_score
y_true = validate_df[TARGET].values

y_identity = validate_df[IDENTITY].values



# predict

y_pred = validate_df[PREDICTION].values



# evaluate

evaluator = JigsawEvaluator(y_true, y_identity)

auc_score = evaluator.get_final_metric(y_pred)

print('auc_score={}'.format(auc_score))
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

submission[PREDICTION] = model.predict(pad_text(test[TEXT], tokenizer))[:, 1]

submission.head()
submission.shape
submission.to_csv('submission.csv')

print(os.listdir("."))