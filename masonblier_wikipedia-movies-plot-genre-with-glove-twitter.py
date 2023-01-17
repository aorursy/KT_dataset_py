# code based on https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle



import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.metrics import mean_squared_error, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier

from nltk import word_tokenize

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load data 

train = pd.read_csv('../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv')

# encode genre strings to unique numbers

lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(train['Genre'].values)

sns.distplot(y)
# collect single-word genre values

genre_counts = {}

for rowgenre in train['Genre']:

    # skip compound genres

    if re.search(r'[^\w]', rowgenre) != None:

        continue

    if not rowgenre in genre_counts:

        genre_counts[rowgenre] = 0

    genre_counts[rowgenre] += 1

# keep any single-word genre with >= 2 entries

simpler_genres = []

sorted_keys = sorted(genre_counts.keys(), key=lambda x: -genre_counts[x])

for genre in sorted_keys:

    if genre_counts[genre] >= 500:

        simpler_genres.append(genre)

(len(simpler_genres), simpler_genres[0:7])
# create new encoded y using simplified genres list

def getSimplerGenre(rowgenre):

    if rowgenre in simpler_genres:

        return rowgenre

    for s_genre in simpler_genres:

        if s_genre in rowgenre:

            return s_genre

    return 'unknown'

simpler_genre_set = [getSimplerGenre(g) for g in train['Genre']]

# count / describe simplified genre distribution

simpler_genre_set_count = {}

for sg in simpler_genre_set:

    if not sg in simpler_genre_set_count:

        simpler_genre_set_count[sg] = 0

    simpler_genre_set_count[sg] += 1

sns.barplot(x=list(simpler_genre_set_count.values()),y=list(simpler_genre_set_count.keys()),log=True)
# create unified set of modified genre/plots

simpler_df = pd.DataFrame({'genre':simpler_genre_set, 'plot':train['Plot'].values})

# save unknowns for later use

unknown_df = simpler_df[simpler_df.genre == 'unknown']

# exclude unknowns from simpler_df set

simpler_df.drop(unknown_df.index, inplace=True)

simpler_df.head()
# encode new y values for plotting purposes

lbl_enc2 = preprocessing.LabelEncoder()

y = lbl_enc2.fit_transform(simpler_df['genre'].values)

# split train/validation data

train_x, validation_x, train_y, validation_y = train_test_split(simpler_df['plot'].values, y, stratify=y, test_size=0.1, shuffle=True)

# show word count distributions of both sets

wordcounter = np.vectorize(lambda s: len(s.split(' ')))

f, axes = plt.subplots(1, 2, sharey=True)

sns.distplot( wordcounter(train_x) , color="skyblue", label="Training", ax=axes[0])

sns.distplot( wordcounter(validation_x) , color="red", label="Validation", ax=axes[1])
# load glove-twitter embeddings

f = open('../input/glove-twitter/glove.twitter.27B.25d.txt')

embeddings_index = {}

for line in f.readlines():

    values = line.split()

    word = values[0] # first column is word, rest are coefs

    embeddings_index[word] = np.asarray(values[1:], dtype='float32')

f.close()

print(f"Found {len(embeddings_index.keys())} embeddings")
# build vector calculated from embeddings of each word in sentence

def buildSentenceVector(s):

    lower_s = str(s).lower() # normalize case

    tokens = word_tokenize(lower_s) # tokenize

    # only use words of alphabet characters, no stop words

    words = [w for w in tokens if (not w in stop_words) and w.isalpha()] 

    # build matrix

    M = []

    for w in words:

        if w in embeddings_index:

            M.append(embeddings_index[w])

    # sum all coef terms across sentence

    v = np.array(M).sum(axis=0)

    # exit if v failed to calculate

    if type(v) != np.ndarray:

        return np.zeros(300)

    # normalized output vector

    return v / np.sqrt((v ** 2).sum())

# build train/validation vectors for all rows

train_vectors = [buildSentenceVector(x) for x in train_x]

validation_vectors = [buildSentenceVector(x) for x in validation_x]

(len(train_vectors), len(validation_vectors))
train_x_glove = pd.DataFrame(train_vectors)

validation_x_glove = pd.DataFrame(validation_vectors)

validation_x_glove.head()
# use BaggingClassifier to chose hyperparamters automatically

clf = BaggingClassifier()

clf.fit(train_x_glove, train_y)

# I'm not sure what these warnings mean

None
# get predictions

predictions = clf.predict(validation_x_glove)

# check mean squared error

percor = np.sum(predictions == validation_y) / len(validation_y)

sqmse = np.sqrt(mean_squared_error(validation_y, predictions))

print ("correct: %0.1f ; sqrt(mse(...)): %0.3f" % (percor,sqmse))
# from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig



# display

validation_y_texts = lbl_enc2.inverse_transform(validation_y)

prediction_texts = lbl_enc2.inverse_transform(predictions)

cm = confusion_matrix(validation_y_texts, prediction_texts)

class_names = list(lbl_enc2.classes_)

print_confusion_matrix(cm, class_names)

None
# grab random sample

unknown_sample = unknown_df.sample(n=50)['plot'].values

unknown_vectors = pd.DataFrame([buildSentenceVector(x) for x in unknown_sample])

# predict and get texts

unknown_predictions = clf.predict(unknown_vectors)

unknown_guesses = lbl_enc2.inverse_transform(unknown_predictions)

# display

pd.options.display.max_colwidth = 200

samples_df = pd.DataFrame({'plot':unknown_sample, 'guess':unknown_guesses})

samples_df.groupby('guess').apply(lambda df: df.sample(1))