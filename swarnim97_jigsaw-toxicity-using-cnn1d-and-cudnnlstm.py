# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tqdm import tqdm

tqdm.pandas()

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
print(train.shape)

print(test.shape)
EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



def load_embeddings(embed_dir=EMBEDDING_PATH):

    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))

    return embedding_index
embeddings_index = load_embeddings()
print('Found %s word vectors.' % len(embeddings_index))
def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
train_sentences = train["comment_text"].progress_apply(lambda x: x.split()).values

test_sentences = test["comment_text"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    print("Total words common in both vocabulary and in embeddings_index",len(a))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
del train_vocab

del test_vocab

del train_sentences

del test_sentences

del train_oov

del test_oov
import gc

gc.collect()
contraction_mapping = {"1950's": "1950s", "1983's": "1983", "ain't": "is not", "aren't": "are not", "Bretzing's": "", "Bundycon's": "Bundycon", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "C'mon": "Come on", "Denzel's": "Denzel", "didn't": "did not",  "doesn't": "does not", "Don't": "Do not", "don't": "do not", "Farmer's": "Farmers", "FBI's": "FBI", "Ferguson's": "Ferguson", "Hammond's": "Hammond", "hadn't": "had not", "hasn't": "has not", "Haven't": "Have not", "haven't": "have not", "he'd": "he would", "Here's": "Here is", "here's": "here is","he'll": "he will", "he's": "he is", "He's": "He is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "I'd": "I had", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "I'm": "I am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "It's": "it is", "Kay's": "Kay", "let's": "let us", "Let's": "let us", "ma'am": "madam", "mayn't": "may not", "Medford's": "Medford", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "Murphy's": "Murphys", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "Paula's": "Paula", "Portland's": "Portlands", "Portlander's": "Portlanders", "publication's": "publications", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "She's": "She is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "Tastebud's": "Tastebuds", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "That's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "There's": "There is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "They're": "They are", "they've": "they have", "to've": "to have", "Trump's": "trump is", "U.S.": "United state", "U.S": "United state", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "We'll": "We will", "we'll've": "we will have", "Wendy's": "Wendy", "we're": "we are", "We're": "We are", "we've": "we have", "We've": "We have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "What's": "What is",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "Who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "Wouldn't": "Would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "You'd": "You had","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "You're": "you are", "you've": "you have", "Zoo's": "zoos", "zoo's": "zoos" }
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
train['comment_text'] = train['comment_text'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

test['comment_text'] = test['comment_text'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))
train_sentences = train["comment_text"].progress_apply(lambda x: x.split()).values

test_sentences = test["comment_text"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
del train_vocab

del test_vocab

del train_sentences

del test_sentences

del train_oov

del test_oov
gc.collect()
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
train['comment_text'] = train['comment_text'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

test['comment_text'] = test['comment_text'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))
train_sentences = train["comment_text"].progress_apply(lambda x: x.split()).values

test_sentences = test["comment_text"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
del train_vocab

del test_vocab

del train_sentences

del test_sentences

del train_oov

del test_oov
gc.collect()
from sklearn.model_selection import train_test_split, StratifiedKFold

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
print('Found %s word vectors.' % len(embeddings_index))
train['target'] = np.where(train['target'] >= 0.5, 1, 0)
train_df, validate_df = train_test_split(train, test_size=0.1, stratify=train['target'])
max_features = 120000
tokenizer_obj = Tokenizer(num_words=max_features)

tokenizer_obj.fit_on_texts(list(train['comment_text']) +list(test['comment_text']))

print(train_df.shape)

print(validate_df.shape)
word_index = tokenizer_obj.word_index

print('Found %s unique tokens.' % len(word_index))
max_length = max([len(s.split()) for s in list(train['comment_text'])])

print(max_length)
max_length = 256

X_train_pad = tokenizer_obj.texts_to_sequences(train_df['comment_text'])

y_train = train_df['target'].values

X_test_pad = tokenizer_obj.texts_to_sequences(validate_df['comment_text'])

y_test = validate_df['target'].values
x_test = test['comment_text'].fillna('').values

test_sequences = tokenizer_obj.texts_to_sequences(x_test)

Test_pad = pad_sequences(test_sequences, maxlen=max_length)

print(Test_pad.shape)
X_train_pad = pad_sequences(X_train_pad, maxlen=max_length)

X_test_pad = pad_sequences(X_test_pad, maxlen=max_length)
nb_words = min(max_features, len(word_index))
print('shape of X_train_pad tensor:', X_train_pad.shape)

print('shape of y_train tensor:', y_train.shape)

print('shape pf X_test_pad tensor:', X_test_pad.shape)

print('shape of y_test tensor:', y_test.shape)
del train_df

del test

del train

del tokenizer_obj
gc.collect()
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU, CuDNNLSTM, SpatialDropout1D

from keras.layers import Bidirectional, BatchNormalization

from keras.layers.embeddings import Embedding

from keras.initializers import Constant

from keras.layers import Dense, Dropout, Activation

from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):

    embedding_matrix = np.zeros((max_features, 300))

    for word, i in tqdm(word_index.items(),disable = not verbose):

        if lower:

            word = word.lower()

        if i >= max_features: continue

        try:

            embedding_vector = embeddings_index[word]

        except:

            embedding_vector = embeddings_index["unknown"]

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    return embedding_matrix

embedding_matrix = build_embedding_matrix(word_index, embeddings_index, max_features)
len(embedding_matrix)
nb_words
EMBEDDING_DIM=300
model = Sequential()

embedding_layer = Embedding(nb_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),input_length=max_length, trainable=False)

model.add(embedding_layer)

model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(CuDNNLSTM(EMBEDDING_DIM, return_sequences=True)))

model.add(Conv1D(64, 5, activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(32, kernel_initializer='normal', activation='relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
import matplotlib.pyplot as plt

from sklearn import metrics
skfolds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for j, (train_index, test_index) in enumerate(skfolds.split(X_train_pad, y_train)):

    print('\nFold ',j)

    X_train_folds = X_train_pad[train_index]

    y_train_folds = y_train[train_index]

    X_test_fold = X_train_pad[test_index]

    y_test_fold = y_train[test_index]

    history = model.fit(X_train_folds, y_train_folds, batch_size=2048, epochs=2, validation_data=(X_test_fold, y_test_fold))

    val_y = model.predict([X_test_fold], batch_size=1024, verbose=1)

    for thresh in np.arange(0.1, 0.501, 0.01):

        thresh = np.round(thresh, 2)

        print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_test_fold, (val_y>thresh).astype(int))))

    print('roc_auc_score',metrics.roc_auc_score(y_test_fold, val_y))

    plt.plot(history.history['loss'])

    plt.show()
val_y = model.predict([X_test_pad], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_test, (val_y>thresh).astype(int))))
metrics.roc_auc_score(y_test, val_y)
y_pred = model.predict(Test_pad)
final = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
final['prediction'] = y_pred
final.to_csv('submission.csv', index=False)