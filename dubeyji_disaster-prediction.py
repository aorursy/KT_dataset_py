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
import tensorflow as tf

import operator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *

import pandas as pd

import numpy as np

from gensim.models import Word2Vec, KeyedVectors

import gensim

import time

import random

import matplotlib.pyplot as plt

import string

import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences

import multiprocessing

import nltk

from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import stopwords

from nltk import word_tokenize, pos_tag

import functools

from gensim.models.fasttext import FastText

from nltk.corpus import wordnet

import re

from nltk.tokenize.treebank import TreebankWordDetokenizer

from wordsegment import load, segment
disaster_text = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv", encoding='utf-8')
print(disaster_text.columns)
disaster_text.shape
disaster_text.head()
disaster_text['keyword'] = disaster_text['keyword'].astype(str)

disaster_text.loc[disaster_text['keyword'] == 'nan','keyword'] = ''
disaster_text['location'] = disaster_text['location'].astype(str)

disaster_text.loc[disaster_text['location'] == 'nan','location'] = ''
df_mislabeled = disaster_text.groupby(['text']).nunique().sort_values(by='target', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']

df_mislabeled.index.tolist()
disaster_text['target_relabeled'] = disaster_text['target'].copy() 



disaster_text.loc[disaster_text['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == 'To fight bioterrorism sir.', 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_relabeled'] = 1

disaster_text.loc[disaster_text['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_relabeled'] = 1

disaster_text.loc[disaster_text['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_relabeled'] = 1

disaster_text.loc[disaster_text['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_relabeled'] = 1

disaster_text.loc[disaster_text['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == "Caution: breathing may be hazardous to your health.", 'target_relabeled'] = 1

disaster_text.loc[disaster_text['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_relabeled'] = 0

disaster_text.loc[disaster_text['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_relabeled'] = 0
disaster_text['target'] = disaster_text['target_relabeled']
def to_lower(x):

    x = x.lower()

    return(x)



def tokenization(x):

    x = word_tokenize(x)

    return(x)



def get_pos(word):

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)



def stemming_and_lemmatization(x):

    lemmatizer = WordNetLemmatizer()

    y = [lemmatizer.lemmatize(w, get_pos(w)) for w in x]

    return(y)



def remove_stopwords(x):

    return([word for word in x if word not in stopwords.words('english')])



def remove_punctuations(x):

    x = re.sub(re.compile('((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',flags=re.MULTILINE), '', x)

    x = re.sub('[^\w\s]','',x)

    x = re.sub("[^a-zA-Z\s]+", '', x)

    return(x)



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)

    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase



def hashtags(phrase):

    phrase = " ".join(segment(phrase))

    return phrase



def spellCheck(phrase):

    spell = Speller(lang='en')

    phrase = "".join([spell(w)+" " for w in phrase.split()])

    return phrase



def text_preprocessing(x):

    x = to_lower(x)

    x = decontracted(x)

    x = remove_punctuations(x)

    #x = hashtags(x)

    #x = spellCheck(x)

    #print('spellcheck')

    #x = tokenization(x)

    #x = remove_stopwords(x)

    #x = stemming_and_lemmatization(x)

    return(x)
p_time1 = time.time()

count = 0

load()

disaster_text['preprocessed_text'] = disaster_text['text'].apply(text_preprocessing)

p_time2 = time.time()

disaster_text.head(10)
p_time2 - p_time1
from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec(glove_input_file="/kaggle/input/glove-pretrained/glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
cores = multiprocessing.cpu_count()

w_time1 = time.time()

word2VecModel = gensim.models.KeyedVectors.load_word2vec_format('gensim_glove_vectors.txt', binary=False)  

w_time2 = time.time()
w_time2 - w_time1
#disaster_text['preprocessed_text'] = disaster_text['preprocessed_text'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
def word2token(word,model):

    try:

        return model.vocab[word].index

    # If word is not in index return 0. I realize this means that this

    # is the same as the word of index 0 (i.e. most frequent word), but 0s

    # will be padded later anyway by the embedding layer (which also

    # seems dirty but I couldn't find a better solution right now)

    except KeyError:

        return 0
#Create an iterator that formats data from the dataset proper for

# LSTM training



# Sequences will be padded or truncated to this length

MAX_SEQUENCE_LENGTH = 700



# Samples of categories with less than this number of samples will be ignored

DROP_THRESHOLD = 10000



class SequenceIterator:

    def __init__(self, dataset, drop_threshold, seq_length):

        self.dataset = dataset



        self.translator = str.maketrans('', '', string.punctuation + '–')

        self.categories, self.ccount = np.unique(dataset.target, return_counts=True)

        

        self.seq_length = seq_length

        

        

    def __iter__(self):

        for sent, lang in zip(self.dataset.preprocessed_text, self.dataset.target):

            # Make all characters lower-case

            sent = sent.lower()

            

            # Clean string of all punctuation

            sent = sent.translate(self.translator)



            words = np.array([word2token(w, word2VecModel) for w in sent.split(' ')[:self.seq_length] if w != ''])

                                

            yield (words, lang)



sequences = SequenceIterator(disaster_text, DROP_THRESHOLD, MAX_SEQUENCE_LENGTH)



# Used for generating the labels in the set

cat_dict = {k: v for k, v in zip(sequences.categories, range(len(sequences.categories)))}



set_x = []

set_y = []

for w, c in sequences:

    set_x.append(w)

    set_y.append(cat_dict[c])

    

# Padding sequences with 0.

set_x = pad_sequences(set_x, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', value=0)

set_y = np.array(set_y)



print(set_x.shape)

print(set_y.shape)
VALID_PER = 0.00 # Percentage of the whole set that will be separated for validation



total_samples = set_x.shape[0]

n_val = int(VALID_PER * total_samples)

n_train = total_samples - n_val



random_i = random.sample(range(total_samples), total_samples)

train_x = set_x[random_i[:n_train]]

train_y = set_y[random_i[:n_train]]

val_x = set_x[random_i[n_train:n_train+n_val]]

val_y = set_y[random_i[n_train:n_train+n_val]]



print("Train Shapes - X: {} - Y: {}".format(train_x.shape, train_y.shape))

print("Val Shapes - X: {} - Y: {}".format(val_x.shape, val_y.shape))



categories, ccount = np.unique(train_y, return_counts=True)

n_categories = len(categories)
w2v_weights = word2VecModel.vectors
vocab_size, embedding_size = w2v_weights.shape



lstm_model = Sequential()



# Keras Embedding layer with Word2Vec weights initialization

lstm_model.add(Embedding(input_dim=vocab_size,

                    output_dim=300,

                    weights=[w2v_weights],

                    input_length=700,

                    mask_zero=True,

                    trainable=True))

lstm_model.add(Bidirectional(LSTM(128, return_sequences = True)))

lstm_model.add(Bidirectional(LSTM(128)))

lstm_model.add(Dense(n_categories, activation='softmax'))

lstm_model

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = lstm_model.fit(train_x, train_y, epochs=5, batch_size=256,

                     verbose=1)
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv", encoding='utf-8')
test_data['keyword'] = test_data['keyword'].astype(str)

test_data.loc[test_data['keyword'] == 'nan','keyword'] = ''
test_data['location'] = test_data['location'].astype(str)

test_data.loc[test_data['location'] == 'nan','location'] = ''
pt_time1 = time.time()

test_data['preprocessed_text'] = test_data['text'].apply(text_preprocessing)

pt_time2 = time.time()

test_data.head(10)
test_data.columns
pt_time2 - pt_time1
#test_data['preprocessed_text'] = test_data['preprocessed_text'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))

test_data['target'] = 2
def make_lstm_format_output(X):

    string1 = X

    string1 = pd.DataFrame(string1, columns =['preprocessed_text'])

    string1['target'] = "X"



    sequence1 = SequenceIterator(string1, DROP_THRESHOLD, MAX_SEQUENCE_LENGTH)



    # Used for generating the labels in the set

    cat_dict = {k: v for k, v in zip(sequence1.categories, range(len(sequence1.categories)))}



    set_x = []

    set_y = []

    for w, c in sequence1:

        set_x.append(w)

        set_y.append(cat_dict[c])



    # Padding sequences with 0.

    set_x = pad_sequences(set_x, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', value=0)

    return set_x
test = make_lstm_format_output(test_data)
predicted_values = history.model.predict(test)
predicted_values = list(np.argmax(predicted_values, axis = 1))
print(len(predicted_values))

print(test_data.shape)
predictions = pd.merge(test_data[['id']], pd.DataFrame(predicted_values, columns = ['target']),left_index= True, right_index = True)
predictions.to_csv('Submission.csv',index = False)