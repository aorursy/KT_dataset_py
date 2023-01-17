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
train = pd.read_csv('../input/twitter-sentimentanalytics-vidhya/train_tweet.csv', encoding='utf-8')

test = pd.read_csv('../input/twitter-sentimentanalytics-vidhya/test_tweets.csv', encoding='utf-8')
train.head()
train.shape
test.head()
test.shape
embeddings_index = {}

f = open('../input/glove840b300dtxt/glove.840B.300d.txt', encoding='utf-8')

for line in f:

    values = line.split(" ")

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()
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
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
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
train_oov[:10]
test_oov[:10]
mispell_dict = {'bihday': 'birthday'}
def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
train['tweet'] = train['tweet'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

test['tweet'] = test['tweet'].progress_apply(lambda x: correct_spelling(x, mispell_dict))
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
import re
def clean_text(text):

    text = text.replace('\n', ' ')

    result = re.sub(r"http\S+", "website", text)

    result = re.sub(r"pic\S+", "website", result)

    result = re.sub(r"@\S+", "", result)

    result = re.sub(r"#\S+", "", result)

    return result
train['tweet'] = train['tweet'].apply(lambda x: clean_text(x))

test['tweet'] = test['tweet'].apply(lambda x: clean_text(x))
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
contraction_mapping = {"ain't": "is not", "aren't": "are not", "Can't": "cannot", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "C'mon": "Come on", "didn't": "did not",  "doesn't": "does not", "Don't": "Do not", "don't": "do not", "Farmer's": "Farmers", "father's": 'fathers', "FBI's": "FBI", "hadn't": "had not", "hasn't": "has not", "Haven't": "Have not", "haven't": "have not", "he'd": "he would", "Here's": "Here is", "here's": "here is","he'll": "he will", "he's": "he is", "He's": "He is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "I'd": "I had", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "It's": "it is", "let's": "let us", "Let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "She's": "She is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "That's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "There's": "There is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "They're": "They are", "they've": "they have", "to've": "to have", "U.S.": "United state", "U.S": "United state", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "We'll": "We will", "we'll've": "we will have", "we're": "we are", "We're": "We are", "we've": "we have", "We've": "We have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "What's": "What is",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "Who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "Wouldn't": "Would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "You'd": "You had","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "You're": "you are", "you've": "you have"}
"fathers" in embeddings_index
def known_contractions(embed):

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known
print(known_contractions(embeddings_index))
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
train['tweet'] = train['tweet'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

test['tweet'] = test['tweet'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown
print(unknown_punct(embeddings_index, punct))
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
train['tweet'] = train['tweet'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

test['tweet'] = test['tweet'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
train.head()
train_df, validate_df = train_test_split(train, test_size=0.05, stratify=train['label'])
tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(list(train['tweet']) +list(test['tweet']))
print(train_df.shape)

print(validate_df.shape)
word_index = tokenizer_obj.word_index

print('Found %s unique tokens.' % len(word_index))
max_length = max([len(s.split()) for s in list(train['tweet'])])
print(max_length)
max_length = 55
X_train_pad = tokenizer_obj.texts_to_sequences(train_df['tweet'])

y_train = train_df['label'].values

X_test_pad = tokenizer_obj.texts_to_sequences(validate_df['tweet'])

y_test = validate_df['label'].values
x_test = test['tweet'].fillna('').values

test_sequences = tokenizer_obj.texts_to_sequences(x_test)

Test_pad = pad_sequences(test_sequences, maxlen=max_length)

print(Test_pad.shape)
len(X_train_pad)
X_train_pad = pad_sequences(X_train_pad, maxlen=max_length)

X_test_pad = pad_sequences(X_test_pad, maxlen=max_length)
EMBEDDING_DIM = 300
num_words = len(word_index)+1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word,i in word_index.items():

  if i>num_words:

    continuelabel

  embedding_vector = embeddings_index.get(word)

  if embedding_vector is not None:

    # words not found in embedding index will be all-zeros

    embedding_matrix[i] = embedding_vector
print(num_words)
count=0

for word,i in word_index.items():

    if word in embeddings_index:

        count = count + 1

print(count)
print('shape of X_train_pad tensor:', X_train_pad.shape)

print('shape of y_train tensor:', y_train.shape)

print('shape pf X_test_pad tensor:', X_test_pad.shape)

print('shape of y_test tensor:', y_test.shape)
print(train_df.shape)

print(validate_df.shape)
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU, CuDNNLSTM, SpatialDropout1D

from keras.layers import Bidirectional, BatchNormalization

from keras.layers.embeddings import Embedding

from keras.initializers import Constant

from keras.layers import Dense, Dropout, Activation

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
model = Sequential()

embedding_layer = Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),input_length=max_length, trainable=False)

model.add(embedding_layer)

model.add(SpatialDropout1D(0.2))

model.add((CuDNNLSTM(EMBEDDING_DIM)))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train_pad, y_train, batch_size=256, epochs=8, validation_data=(X_test_pad, y_test))
from sklearn import metrics
val_y = model.predict([X_test_pad], verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_test, (val_y>thresh).astype(int))))
metrics.roc_auc_score(y_test, val_y)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])

plt.show()
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



plt.clf()

acc = history_dict['acc']

val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



val_y = (val_y > 0.1).astype(int)
metrics.accuracy_score(y_test, val_y)
metrics.confusion_matrix(y_test, val_y)
from sklearn.metrics import classification_report



cr = classification_report(y_test, val_y)

print(cr)