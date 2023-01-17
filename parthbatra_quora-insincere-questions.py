# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import io

import warnings

import gc

import numpy as np

import pandas as pd

import re

from tqdm import tqdm

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from gensim.models import KeyedVectors

import nltk

from sklearn.model_selection import train_test_split

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras import backend as K

from keras import layers

from keras.utils.np_utils import to_categorical

from keras import backend as K

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from keras import models

from keras import regularizers

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import gc
train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')

test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
pd.options.display.max_colwidth = 150
def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),#!?\'\`]", " ", string)     

    string = re.sub(r"\'s", " \'s", string) 

    string = re.sub(r"\'ve", " \'ve", string) 

    string = re.sub(r"n\'t", " n\'t", string) 

    string = re.sub(r"\'re", " \'re", string) 

    string = re.sub(r"\'d", " \'d", string) 

    string = re.sub(r"\'ll", " \'ll", string) 

    string = re.sub(r",", " , ", string)

    string = re.sub(r"!", " ! ", string) 

    string = re.sub(r"\(", " ( ", string) 

    string = re.sub(r"\)", " ) ", string) 

    string = re.sub(r"\?", " ? ", string) 

    string = re.sub(r"\s{2,}", " ", string)

    return string.strip()



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



puncts_rem = [ '•', '@', '£', '·', '`', '→', '°', '€', '♥', '←',  '§', 'Â', '█',  'à', '…', 

 '★',   '●', 'â', '►',  '¢',  '¬', '░', '¶', '↑', '±', '¿', '▾',  '¦',  '¥', '▓',  

 '▒',  '▼', '▪', '†', '■',  '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅' 

 '↓', '、', '│',  '»',  '♪', '╩', '╚', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',   '‡',  ]



puncts_keep = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '″', '′','<','›',

          '：','∙', '）', '，','”','“', '（', '—', '‹', '─','_', '{', '}','^','═','×','≤','−','-','’','²','√','½', '³','¼','⊕','~','¹', '‘', '∞','║', '―', '®','©','™',]



def remove_punct(x):

    x = str(x)

    for punct in puncts_rem:

        x = x.replace(punct,'')

    return x



def clean_text2(x):

    x = str(x)

    for punct in puncts_keep:

        x = x.replace(punct, f' {punct} ')



def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x



mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",

                "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",

                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",

                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",

                "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",

                "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",

                "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 

                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us",

                "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",

                "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",

                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",

                "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",

                "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",

                "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have",

                "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", 

                "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will",

                "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 

                "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",

                "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",

                "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", 

                "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",

                "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",

                "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", 

                "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",

                "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", 

                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",

                "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',

                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',

                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',

                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',

                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 

                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',

                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',

                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',

                'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
def count_char(x):

    x = str(x)

    return len(x)

  

def count_char_minus_space(x):

    x = str(x)

    return len(x.replace(' ',''))
#Removing Misspellings

train['clean_text'] = train['question_text'].apply(lambda x: replace_typical_misspell(x))

test['clean_text'] = test['question_text'].apply(lambda x: replace_typical_misspell(x))



#Removing Unwanted Characters

train['clean_text'] = train['clean_text'].apply(lambda x: remove_punct(x))

test['clean_text'] = test['clean_text'].apply(lambda x: remove_punct(x))



#Cleaning Numbers

train['clean_text'] = train['clean_text'].apply(lambda x: clean_numbers(x))

test['clean_text'] = test['clean_text'].apply(lambda x: clean_numbers(x))



#Removing Capitals

train['clean_text'] = train['clean_text'].apply(lambda x: x.lower())

test['clean_text'] = test['clean_text'].apply(lambda x: x.lower())
  #Seperating all words

train['clean_text'] = train['clean_text'].apply(lambda x: clean_str(x))

test['clean_text'] = test['clean_text'].apply(lambda x: clean_str(x))



#Seperating words from remaining punctuations

train['spaced_text'] = train['clean_text'].apply(lambda x: clean_text(x))

test['spaced_text'] = test['clean_text'].apply(lambda x: clean_text(x))
train['spaced_text'].loc[:5]
plt.figure(figsize = (7,5))

sns.set(style = 'darkgrid')

ax = sns.countplot(x = 'target', data=train, linewidth = 0.1)

for patch in ax.patches :

        current_width = patch.get_width()

        new_width = current_width/4

        diff = current_width - new_width



        # we change the bar width

        patch.set_width(new_width)

        patch.set_x(patch.get_x() + diff/2)

plt.plot()

#Number of words per sentence

train['sent_length'] = train['clean_text'].apply(lambda x: len(x.split()))

test['sent_length'] = test['clean_text'].apply(lambda x: len(x.split()))
fig = plt.figure(figsize=(14,5))

#fig.tight_layout(pad = 3.0)

plt.subplot(1,2,1)

sns.set(style = 'darkgrid')

ax1 = sns.distplot(train['sent_length'], bins = 8, kde = False, color = 'darkcyan')

plt.yscale('log')

plt.title('Training Set: Sentence Length')

plt.xlabel('Word Count')

plt.ylabel('Number of Sentences')



plt.subplot(1,2,2)

sns.set(style = 'darkgrid')

ax2 = sns.distplot(test['sent_length'], bins = 8, kde = False, color = 'c')

plt.yscale('log')

plt.title('Test Set: Sentence Length')

plt.xlabel('Word Count')

plt.ylabel('Number of Sentences')



plt.subplots_adjust(wspace = 0.5)

plt.plot()  
#Number of characters per sentence

train['num_char'] = train['clean_text'].apply(lambda x: count_char_minus_space(x))

test['num_char'] = test['clean_text'].apply(lambda x: count_char_minus_space(x))
fig = plt.figure(figsize=(14,5))

#fig.tight_layout(pad = 3.0)

plt.subplot(1,2,1)

sns.set(style = 'darkgrid')

ax1 = sns.distplot(train['num_char'], bins = 8, kde = False, color='sandybrown')

plt.yscale('log')

plt.title('Training Set: Number of Characters')

plt.xlabel('Character Count')

plt.ylabel('Number of Sentences')



plt.subplot(1,2,2)

sns.set(style = 'darkgrid')

ax2 = sns.distplot(test['num_char'], bins = 8, kde = False,color = 'darkgoldenrod')

plt.yscale('log')

plt.title('Test Set: Number of Characters')

plt.xlabel('Character Count')

plt.ylabel('Number of Sentences')



plt.subplots_adjust(wspace = 0.5)

plt.plot()  
plt.figure(figsize = (7,5))

ax = sns.violinplot(x = 'target', y = 'sent_length', data = train)

plt.plot()
sincere = train.loc[train['target']==1]

insincere = train.loc[train['target']==0]
fig = plt.figure(figsize=(14,5))

#fig.tight_layout(pad = 3.0)

plt.subplot(1,2,1)

sns.set(style = 'darkgrid')

ax1 = sns.distplot(sincere['num_char'], bins = 8, kde = False, color='sandybrown')

plt.yscale('log')

plt.title('Sincere: Number of Characters')

plt.xlabel('Character Count')

plt.ylabel('Number of Sentences')



plt.subplot(1,2,2)

sns.set(style = 'darkgrid')

ax2 = sns.distplot(insincere['num_char'], bins = 8, kde = False,color = 'darkgoldenrod')

plt.yscale('log')

plt.title('Insincere: Number of Characters')

plt.xlabel('Character Count')

plt.ylabel('Number of Sentences')



plt.subplots_adjust(wspace = 0.5)

plt.plot()  
print('Percentage of Sincere Questions: '+ str(len(sincere)) + '/' + str(len(train)))

print('Percentage of Sincere Questions: {:.2f}'.format((len(sincere)/len(train))*100))

print('Percentage of Insincere Questions: '+ str(len(insincere)) + '/' + str(len(train)))

print('Percentage of Insincere Questions: {:.2f}'.format((len(insincere)/len(train))*100))
fig = plt.figure(figsize=(14,5))

#fig.tight_layout(pad = 3.0)

plt.subplot(1,2,1)

sns.set(style = 'darkgrid')

ax1 = sns.distplot(sincere['sent_length'], bins = 8, kde = False, color='sandybrown')

plt.yscale('log')

plt.title('Sincere: Number of Characters')

plt.xlabel('Character Count')

plt.ylabel('Number of Sentences')



plt.subplot(1,2,2)

sns.set(style = 'darkgrid')

ax2 = sns.distplot(insincere['sent_length'], bins = 8, kde = False,color = 'darkgoldenrod')

plt.yscale('log')

plt.title('Insincere: Number of Characters')

plt.xlabel('Character Count')

plt.ylabel('Number of Sentences')



plt.subplots_adjust(wspace = 0.5)

plt.plot()  
!wget -P /root/input/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
EMBEDDING_FILE = '/root/input/GoogleNews-vectors-negative300.bin.gz' # from above

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
def build_vocab(sentences):

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                   vocab[word] = 1

    

    return vocab
sentences = train["spaced_text"].apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print( {k + ':' + str(vocab[k]) for k in list(vocab)[:5]})
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

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
oov = check_coverage(vocab,word2vec)
#removing Numbers/Hashes not having embeddings

train['spaced_text'] = train['spaced_text'].apply(lambda x: re.sub("\S*#+\S*",'',str(x)))

test['spaced_text'] = train['spaced_text'].apply(lambda x: re.sub("\S*#+\S*",'',str(x)))
#Cleaning for word2vec

def prep_cleaning(text):

    preposition_removal = ['a','and','of','to']

    text = str(text)

    for prep in preposition_removal:

        text = re.sub('\s' + prep + '\s',' ',text)

  

    return text



def remove_unnecessary_punct(text):

    text = str(text)

    punctuation = ['?','!','\.',',','/','"','$','%','\'','(',')','*','+','-','/',':',';','<','=','>','@','[','\\',']','^','_','`','{','|','}','~','“','”','’']

    for punct in punctuation:

        text = text.replace(punct,'')

    return text
train['spaced_text'] = train['spaced_text'].apply(lambda x: remove_unnecessary_punct(x))

test['spaced_text'] = test['spaced_text'].apply(lambda x: remove_unnecessary_punct(x))
train['spaced_text'] = train['spaced_text'].apply(lambda x: prep_cleaning(x))

test['spaced_text'] = test['spaced_text'].apply(lambda x: prep_cleaning(x))
sentences = train["spaced_text"].apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

oov = check_coverage(vocab,word2vec)
gc.collect()
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
train['spaced_text'] = train['spaced_text'].apply(lambda x: list(filter(None,x.split(' '))))

test['spaced_text'] = test['spaced_text'].apply(lambda x: list(filter(None,x.split(' '))))
train['spaced_text'] = train['spaced_text'].apply(lambda x: [word for word in x if word not in stop_words])

test['spaced_text'] = test['spaced_text'].apply(lambda x: [word for word in x if word not in stop_words])
train['clean_text'] = train['spaced_text'].apply(lambda x: ' '.join(x))

test['clean_text'] = test['spaced_text'].apply(lambda x: ' '.join(x))
## Removing Sentences with length 0

rem_indices = train.loc[(train['spaced_text'].apply(lambda x: len(x))==0)].index.values

train.drop(rem_indices,axis = 0,inplace = True)

train.reset_index(inplace = True)
max(train['clean_text'].apply(lambda x: len(x.split(' '))))
plt.figure()

sns.distplot(train['spaced_text'].apply(lambda x: len(x)))

plt.plot()
print('Gauging the upper limit as {}. No of rows: {}'.format(str(55),str(len(train.loc[train['spaced_text'].apply(lambda x: len(x)) >55]))))
max_features = 20000                                  #Number of most common words we are going to use.

NB_WORDS = min(max_features, len(vocab))

VAL_SIZE = 1000

NB_EPOCHS = 20

MAX_LENGTH = 55                                         #max(train['clean_text'].apply(lambda x: len(x.split(' ')))) ---> Too much paddding will lead to noise.

WORD2VEC_DIM = 300
tokenizer = Tokenizer(num_words = NB_WORDS,lower = True, split = ' ',oov_token = '<UNK>')
tokenizer.fit_on_texts(list(train['spaced_text']))
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((max_features, 300))

for word, i in tokenizer.word_index.items():

	if i < max_features:

		try:

			embedding_vector = word2vec[word]

			if embedding_vector is not None:

				# words not found in embedding index will be all-zeros.

				embedding_matrix[i] = embedding_vector

		except KeyError:

			continue



## Embedding_martix[0] contains all zeroes as tokenizer index starts from 1
train['tokenized'] = tokenizer.texts_to_sequences(train['spaced_text'])

X_test = tokenizer.texts_to_sequences(test['spaced_text'])
X_train,X_val,Y_train, Y_val = train_test_split(train['tokenized'],train['target'], test_size = 0.1, random_state = 42)
X_test = tokenizer.texts_to_sequences(test['spaced_text'])
X_train = pad_sequences(X_train, maxlen = MAX_LENGTH, padding = 'post')

X_val = pad_sequences(X_val, maxlen = MAX_LENGTH, padding = 'post')

X_test = pad_sequences(X_test, maxlen = MAX_LENGTH, padding = 'post')
from keras import backend as K



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
inp = layers.Input(name = 'inputs', shape =(MAX_LENGTH,))

x = layers.Embedding(NB_WORDS, WORD2VEC_DIM, weights=[embedding_matrix],trainable = False)(inp)

x = layers.Flatten()(x)

x = layers.Dense(10,name = 'FC1', activation = 'relu')(x)

pred = layers.Dense(1, name = 'output_layer',activation = 'sigmoid')(x) ##For K=2 Softmax==Sigmoid.    ##When using Softmax, output nodes == number of classes

model = models.Model(input = inp, output = pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',recall_m,precision_m,f1_m])

model.summary()
model.fit(X_train, Y_train, batch_size=512, epochs=10, validation_data=(X_val, Y_val))
gc.collect()
Y = pd.get_dummies(train['target'])

X_train,X_val,Y_train, Y_val = train_test_split(train['tokenized'],Y, test_size = 0.1, random_state = 42)

X_test = tokenizer.texts_to_sequences(test['spaced_text'])
X_train = pad_sequences(X_train, maxlen = MAX_LENGTH, padding = 'post')

X_val = pad_sequences(X_val, maxlen = MAX_LENGTH, padding = 'post')

X_test = pad_sequences(X_test, maxlen = MAX_LENGTH, padding = 'post')
num_filters = 100

embedding_dim = embedding_matrix.shape[1]



inp_01 = layers.Input(shape = (MAX_LENGTH,))

embedding_layer = layers.Embedding(NB_WORDS,embedding_dim, weights = [embedding_matrix],input_length = MAX_LENGTH,trainable = False)(inp_01)



conv_01 = layers.Conv1D(filters = num_filters ,kernel_size = 3, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)

conv_02 = layers.Conv1D(filters = num_filters ,kernel_size = 4, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)

conv_03 = layers.Conv1D(filters = num_filters ,kernel_size = 5, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)



max_p01 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 3 + 1)(conv_01)

max_p01 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 4 + 1)(conv_02)

max_p02 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 5 + 1)(conv_03)



concatenated = layers.Concatenate(axis = -1)([max_p01, max_p01, max_p02])



flatten = layers.Flatten()(concatenated)

dropout = layers.Dropout(0.5)(flatten)



CNN_pred_01 = layers.Dense(1, activation = 'sigmoid')(dropout)
CNN_model_01 = models.Model(inp_01, CNN_pred_01)

CNN_model_01.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision_m, recall_m, f1_m])

CNN_model_01.summary()
history_CNN_01 = CNN_model_01.fit(X_train, Y_train, batch_size = 512, epochs = 100, validation_data = (X_val,Y_val))
#Defining the Model

inp0 = layers.Input(shape= [MAX_LENGTH])

embedding_layer = layers.Embedding(NB_WORDS,embedding_dim, weights = [embedding_matrix],input_length = MAX_LENGTH,trainable = False)(inp0)

lstm = layers.LSTM(units = 55)(embedding_layer)

x = layers.BatchNormalization()(lstm)

dense = layers.Dense(10, activation = 'relu')(x)

pred = layers.Dense(1, activation = 'sigmoid')(dense)
lstm_model = models.Model(input = inp0, output = pred)

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',recall_m,precision_m,f1_m])

lstm_model.summary()
history_lstm = lstm_model.fit(X_train, Y_train, batch_size = 512, epochs = 8, validation_data = (X_val,Y_val))
plt.plot()

plt.plot(history_lstm.history['acc'], label = 'Train Accuracy')

plt.plot(history_lstm.history['val_acc'], label = 'Validation Accuracy')

plt.show()
num_filters = 100

embedding_dim = embedding_matrix.shape[1]



#Defining the Model

inp02 = layers.Input(shape = (MAX_LENGTH,))

embedding_layer = layers.Embedding(NB_WORDS, WORD2VEC_DIM, weights=[embedding_matrix],trainable = True)(inp02)

conv0 = layers.Conv1D(filters = num_filters, kernel_size = 3, activation='relu', kernel_regularizer = keras.regularizers.l2(3))(embedding_layer)

conv1 = layers.Conv1D(filters = num_filters, kernel_size = 4, activation='relu', kernel_regularizer = keras.regularizers.l2(3))(embedding_layer)

conv2 = layers.Conv1D(filters = num_filters, kernel_size = 5, activation='relu', kernel_regularizer = keras.regularizers.l2(3))(embedding_layer)



max_p0 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 3 + 1)(conv0)

max_p1 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 4 + 1)(conv1)

max_p2 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 5 + 1)(conv2)





concatenated = layers.Concatenate(axis = -1)([max_p0,max_p1,max_p2])



flatten = layers.Flatten()(concatenated)

dropout = layers.Dropout(0.1)(flatten)



CNNpred = layers.Dense(1, activation = 'sigmoid')(dropout)
CNN_model_trainable = models.Model(input = inp02, output = CNNpred)

CNN_model_trainable.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision_m, recall_m, f1_m])

CNN_model_trainable.summary()
hist_CNN_train = CNN_model_trainable.fit(X_train, Y_train, batch_size = 512, epochs = 100, validation_data = (X_val,Y_val))