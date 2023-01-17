# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd

biorxiv = pd.read_csv('/kaggle/input/cordcsvs/biorxiv_medrxiv.csv')

noncomm_sub = pd.read_csv('/kaggle/input/cordcsvs/noncomm_use_subset.csv')

comm_use_sub =  pd.read_csv('/kaggle/input/cordcsvs/comm_use_subset.csv')

metadata =  pd.read_csv('/kaggle/input/cordcsvs/metadata.csv')

custom_license =  pd.read_csv('/kaggle/input/cordcsvs/custom_license.csv')

pieces = (biorxiv, noncomm_sub, comm_use_sub, custom_license)

df_minus_metadata = pd.concat(pieces, ignore_index = True)

df_minus_metadata.head(10)
# Thanks, Andy White! https://www.kaggle.com/ajrwhite/covid-19-thematic-tagging-with-regular-expressions
def get_terms():
    term_list = ['covid',
                    'coronavirus disease 19',
                    'sars cov 2', # Note that search function replaces '-' with ' '
                    '2019 ncov',
                    '2019ncov',
                    r'2019 n cov\b',
                    r'2019n cov\b',
                    'ncov 2019',
                    r'\bn cov 2019',
                    'coronavirus 2019',
                    'wuhan pneumonia',
                    'wuhan virus',
                    'wuhan coronavirus',
                    r'coronavirus 2\b']
    return term_list
!pip install fuzzywuzzy
!pip install python-levenshtein
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 

text_df = df_minus_metadata

text_df['paragraphs'] = text_df['text'].str.split('\n\n')

combo_dict = {'paper_id':[], 'title':[], 'authors':[], 'term':[],'partial_ratio':[], 'paragraphs':[]}


term_list = get_terms()

for ind in text_df.index:
    title = text_df['title'][ind]
    paras = text_df['paragraphs'][ind]
    paper_id = text_df['paper_id'][ind]
    authors = text_df['authors'][ind]
    text = text_df['text'][ind]
    affils = text_df['affiliations'][ind]
    abstract = text_df['abstract'][ind]
    bib = text_df['bibliography'][ind]
    raw_auth = text_df['raw_authors'][ind]
    raw_bib = text_df['raw_bibliography'][ind]
    
    for term in term_list:
        for p in paras:
            if len(term) < len(p):
                partial_ratio = fuzz.partial_ratio(term, p)

                if partial_ratio >= 90:
                    combo_dict["title"].append(title)
                    combo_dict["paper_id"].append(paper_id)
                    combo_dict["authors"].append(authors)
                    combo_dict["paragraphs"].append(p)
                    combo_dict["term"].append(term)
                    combo_dict["partial_ratio"].append(partial_ratio)
results_df = pd.DataFrame.from_dict(combo_dict)

results_df = results_df.drop_duplicates()
results_df.head(5)
# due to computational limitations of notebook, uploaded as input data
results_df.to_csv('relevant_covidpapers.csv')
dataset = pd.read_csv('/kaggle/input/labeled-relevant-papers/relevant_covidpapers_labeled.csv')
dataset = dataset[['paragraphs', 'label (1-11)']]

dataset.rename(columns = {'paragraphs':'paragraph','label (1-11)':'label'}, inplace = True)

dataset = dataset.dropna()
dataset.head(10)
import matplotlib.pyplot as plt
dataset.groupby('label').count().plot.bar(ylim=0)
plt.show()
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
import nltk
nltk.download('wordnet')
X = dataset['paragraph']
y = dataset['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result
processed_docs = []

for doc in X_train:
    processed_docs.append(preprocess(doc))

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
'''
OPTIONAL STEP
Remove very rare and very common words:

- words appearing less than 15 times
- words appearing in more than 10% of all documents
'''

dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
'''
Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
words and how many times those words appear. Save this to 'bow_corpus'
'''
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
'''
Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
'''
# TODO
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 10, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
num = 0
unseen_document = X_test.iloc[num]
print(unseen_document)
# Data preprocessing step for the unseen document
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sqlite3 import Error
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import pickle
X = dataset['paragraph']
y = dataset['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
stemmer = PorterStemmer()
words = stopwords.words("english")
dataset['cleaned'] = dataset['paragraph'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
final_features = vectorizer.fit_transform(dataset['cleaned']).toarray()
final_features.shape # (373, 7475) so out of 373 paragraphs we got 7475 features
#first we split our dataset into testing and training set:
# this block is to split the dataset into training and testing set 
X = dataset['cleaned']
Y = dataset['label']
indices = dataset.index.values

X_train, X_test,indices_train,indices_test = train_test_split(X, indices, test_size=0.25, random_state=42)

y_train, y_test = Y[indices_train],  Y[indices_test]

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 42)



# instead of doing these steps one at a time, we can use a pipeline to complete them all at once
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', RandomForestClassifier())])
# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)
with open('RandomForest.pickle', 'wb') as f:
    pickle.dump(model, f)
ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))
dataset.loc[indices_test,'pred_test'] = model.predict(X_test)
dataset
import pandas as pd
df = pd.read_csv('/kaggle/input/labeled-relevant-papers/relevant_covidpapers_labeled.csv')
df = df[['paragraphs', 'label (1-11)']]

df.rename(columns = {'paragraphs':'paragraph','label (1-11)':'label'}, inplace = True)

df = df.dropna()
df.head(10)

import itertools
import os

import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils, optimizers, models



from keras import backend as K
tf.random.set_seed(1234)

# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

train_size = int(len(df) * .7)
train_posts = df['paragraph'][:train_size]
train_tags = df['label'][:train_size]

test_posts = df['paragraph'][train_size:]
test_tags = df['label'][train_size:]

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts)  # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 300

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

rmsprop = optimizers.rmsprop(lr=0.001)
sgd = optimizers.sgd(lr=0.001)
adam = optimizers.adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

train_acc = model.evaluate(x_train, y_train, verbose=0)
test_acc = model.evaluate(x_test, y_test, verbose=0)

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
# # serialize model to JSON/save
# model_json = model.to_json()
# with open("BoW_ann_model.json", "w+") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])
