# Standard data science and classification tools

import pandas as pd 

import numpy as np

import nltk

import re



# NLTK and NLP libraries

from nltk import word_tokenize, pos_tag

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet

import gensim

from gensim import corpora



# Libraries to create features from text data

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



# Sklearn essentials and models

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm 



# Metrics

from sklearn import metrics 

from sklearn.metrics import log_loss

from sklearn.metrics import classification_report



# Cross validation libraries

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate



# For quick graphing

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Library to over-sample and under-sample

from imblearn.combine import SMOTETomek
train_df = pd.read_csv('../input/train.csv', header = 0)

test_df = pd.read_csv('../input/test.csv', header = 0, index_col = 0)
train_df.isnull().any()
train_df['posts'].apply(lambda x: x.split('|||')).apply(lambda x : len(x)).mode()
train_df['posts'].apply(lambda x: x.split('|||')).apply(lambda x : len(x)).min()
train_df['posts'].apply(lambda x: x.split('|||')).apply(lambda x : len(x)).max()
train_df['mind'] = train_df['type'].apply(lambda x: 0 if x[0] == 'I' else 1)

train_df['energy'] = train_df['type'].apply(lambda x: 0 if x[1] == 'S' else 1)

train_df['nature'] = train_df['type'].apply(lambda x: 0 if x[2] == 'F' else 1)

train_df['tactics'] = train_df['type'].apply(lambda x: 0 if x[3] == 'P' else 1)
plt.figure(figsize=(10, 10))

plt.title('Heatmap showing correlation between each label category', fontsize = 14)

sns.heatmap(train_df[['mind', 'energy', 'nature', 'tactics']].corr(), cmap = 'coolwarm', square = True)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
# Before the the column is dropped a list of the unique personality types is created for later use

profile_types = list(train_df['type'].unique())

profile_types = [x.lower() for x in profile_types]



# Dropping the type column

train_df.drop('type', axis = 1, inplace = True)
train_df.head()
plt.figure(figsize=(10, 10))



plt.subplot(2,2,1)

plt.title('Mind', fontsize = 14)

train_df['mind'].value_counts().plot(kind='bar')

plt.xticks(ticks = range(2), labels = ['I', 'E'])



plt.subplot(2,2,2)

plt.title('Energy', fontsize = 14)

train_df['energy'].value_counts().plot(kind='bar')

plt.xticks(ticks = range(2), labels = ['N', 'S'])



plt.subplot(2,2,3)

plt.title('Nature', fontsize = 14)

train_df['nature'].value_counts().plot(kind='bar')

plt.xticks(ticks = range(2), labels = ['F', 'T'])



plt.subplot(2,2,4)

plt.title('Tactics', fontsize = 14)

train_df['tactics'].value_counts().plot(kind='bar')

plt.xticks(ticks = range(2), labels = ['P', 'J'])



plt.show()
train_df['posts'].iloc[0]
re.sub(r'http\S+', 'url', train_df['posts'].iloc[0], flags=re.MULTILINE)
no_urls = re.sub(r'http\S+', 'url', train_df['posts'].iloc[0], flags=re.MULTILINE)

re.sub('[^a-zA-Z]', ' ', no_urls)
# Using profiles types that was defined earlier on



only_letters = re.sub('[^a-zA-Z]', ' ', no_urls)



for x in profile_types:

        only_letters = re.sub(x, '', only_letters)

        

print(only_letters)
lower = only_letters.lower()

words_nltk = word_tokenize(lower)

print(words_nltk)
# Creating a set of stopwords

stops = set(stopwords.words('english')) # A set is quicker to search through

stops.remove('me') # Keeping the word 'me'



words_nltk = [w for w in words_nltk if not w in stops]



print(words_nltk)
def get_wordnet_pos(word):

        """Map POS tag to first character lemmatize() accepts"""

        tag = nltk.pos_tag([word])[0][1][0].upper()

        tag_dict = {"J": wordnet.ADJ,

                    "N": wordnet.NOUN,

                    "V": wordnet.VERB,

                    "R": wordnet.ADV}



        return tag_dict.get(tag, wordnet.NOUN)



lem = WordNetLemmatizer() # Instantiate the lemmatizer

words_nltk = [lem.lemmatize(w, get_wordnet_pos(w)) for w in words_nltk]



print(words_nltk)
def pre_process(post):

    ''' Performs preprocessing on a block of text

    Input is a block of text that is a single string

    Output is a single string of of preprocessed posts

    '''

    

    # Remove URLs

    no_urls = re.sub(r'http\S+', 'url', post, flags=re.MULTILINE)

    

    # Remove non-letters

    only_letters = re.sub('[^a-zA-Z]', ' ', no_urls)

    

    # Remove mentions on personality types

    for x in profile_types:

        only_letters = re.sub(x, '', only_letters)

    

    # Tokenize

    lower = only_letters.lower()

    words_nltk = word_tokenize(lower)

       

    # Remove stopwords

    stops = set(stopwords.words('english')) # set is quicker to search through

    stops.remove('me')

    words_nltk = [w for w in words_nltk if not w in stops]

    

    # Define a function to obtain the parts of speech

    def get_pos(word):

        ''' Map POS tag to first character lemmatize() accepts

        '''

        

        # Identify the first letter in the POS tag

        tag = nltk.pos_tag([word])[0][1][0].upper()

        # Create a dictionary to use with the lemmatizer

        tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}



        return tag_dict.get(tag, wordnet.NOUN)



    # Lemmatize using parts of speech

    lem = WordNetLemmatizer()

    words_nltk = [lem.lemmatize(w, get_pos(w)) for w in words_nltk]

    

    # Combine into one string

    combined = ' '.join(words_nltk)

    

    return combined



# Adapted from...
train_df['posts'] = train_df['posts'].apply(pre_process)
test_df['posts'] = test_df['posts'].apply(pre_process)
X = train_df[['posts']]

y = train_df[['mind','energy','nature','tactics']]

X_t = test_df['posts'] # The test file to do the predictions on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False, random_state=42)
# Instantiate the count vectorizer

vect = CountVectorizer(ngram_range=(1, 1), min_df=0.2, max_df=0.5, max_features = 10000)
# Using a sparse dataframe to convert the sparse matrix output into a much easier to manipulate form 

X_train_count = pd.SparseDataFrame(vect.fit_transform(X_train['posts']),

                         columns=vect.get_feature_names(), 

                         default_fill_value=0)
X_test_count = pd.SparseDataFrame(vect.transform(X_test['posts']),

                         columns=vect.get_feature_names(), 

                         default_fill_value=0)
X_t_count = pd.SparseDataFrame(vect.transform(X_t),

                         columns=vect.get_feature_names(), 

                         default_fill_value=0)
# Instantiate the tf-idf vectorizer

vectorizer = TfidfVectorizer(min_df=0.2, max_df=0.5, max_features = 10000)
# Using a sparse dataframe to convert the sparse matrix output into a much easier to manipulate form

X_train_tfidf = pd.SparseDataFrame(vectorizer.fit_transform(X_train['posts']),

                         columns=vectorizer.get_feature_names(), 

                         default_fill_value=0)
X_test_tfidf= pd.SparseDataFrame(vectorizer.transform(X_test['posts']),

                         columns=vectorizer.get_feature_names(), 

                         default_fill_value=0)
X_t_tfidf =  pd.SparseDataFrame(vectorizer.transform(X_t),

                         columns=vectorizer.get_feature_names(), 

                         default_fill_value=0)
features_train = pd.concat([X_train_tfidf, X_train_count], axis = 1)
features_test = pd.concat([X_train_tfidf, X_train_count], axis = 1)
features = pd.concat([X_t_tfidf, X_t_count], axis = 1)
features_train.shape
svdT = TruncatedSVD(n_components=100)

svdTFit = svdT.fit_transform(features_train)

svdTFit_test = svdT.fit_transform(features_test)
svdTFit_t = svdT.transform(features)
svdTFit.shape
scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(svdTFit)

scaled_df_test = scaler.fit_transform(svdTFit_test)
scaled_df_t = scaler.transform(svdTFit_t)
smt = SMOTETomek(ratio='auto')

X_smt_mind, y_smt_mind = smt.fit_sample(scaled_df, y_train['mind'])
np.bincount(y_smt_mind)
model = LogisticRegression() # Instantiate the model

model.fit(X_smt_mind, y_smt_mind)
mind_train = model.predict(scaled_df)

mind_test = model.predict(scaled_df_test)
print('Train accuracy', metrics.accuracy_score(y_train['mind'], mind_test))

print('Test accuracy', metrics.accuracy_score(y_test['mind'], mind_test))
print('Classification Report', classification_report(y_test['mind'], mind_test))
print('Log Loss', log_loss(y_test['mind'], mind_test))
mind_prediction = model.predict(scaled_df_t)
X_smt_energy, y_smt_energy = smt.fit_sample(scaled_df, y_train['energy'])
np.bincount(y_smt_energy)
model = LogisticRegression()

model.fit(X_smt_energy, y_smt_energy)
energy_train = model.predict(scaled_df)

energy_test = model.predict(scaled_df_test)
print('Train accuracy', metrics.accuracy_score(y_train['energy'], energy_test))

print('Test accuracy', metrics.accuracy_score(y_test['energy'], energy_test))
print('Classification Report', classification_report(y_test['energy'], energy_test))
print('Log Loss', log_loss(y_test['energy'], energy_test))
energy_prediction = model.predict(scaled_df_t)
X_smt_nature, y_smt_nature = smt.fit_sample(scaled_df, y_train['nature'])
np.bincount(y_smt_nature)
model = LogisticRegression()

model.fit(X_smt_nature, y_smt_nature)
nature_train = model.predict(scaled_df)

nature_test = model.predict(scaled_df_test)
print('Train accuracy', metrics.accuracy_score(y_train['nature'], nature_test))

print('Test accuracy', metrics.accuracy_score(y_test['nature'], nature_test))
print('Classification Report', classification_report(y_test['nature'], nature_test))
print('Log Loss', log_loss(y_test['nature'], nature_test))
nature_prediction = model.predict(scaled_df_t)
X_smt_tactics, y_smt_tactics = smt.fit_sample(scaled_df, y_train['tactics'])
np.bincount(y_smt_tactics)
model = LogisticRegression()

model.fit(X_smt_tactics, y_smt_tactics)
tactics_train = model.predict(scaled_df)

tactics_test = model.predict(scaled_df_test)
print('Train accuracy', metrics.accuracy_score(y_train['tactics'], tactics_test))

print('Test accuracy', metrics.accuracy_score(y_test['tactics'], tactics_test))
print('Classification Report', classification_report(y_test['tactics'], nature_test))
print('Log Loss', log_loss(y_test['tactics'], tactics_test))
tactics_prediction = model.predict(scaled_df_t)
mind = pd.DataFrame(mind_prediction, columns = ['Mind'])

energy = pd.DataFrame(energy_prediction, columns = ['Energy'])

nature = pd.DataFrame(nature_prediction, columns = ['Nature'])

tactics = pd.DataFrame(tactics_prediction, columns = ['Tactics'])
submission = pd.concat([mind, energy, nature, tactics], axis = 1)
submission = submission.reset_index()

submission = submission.rename(columns={'index':'id'})

submission['id'] = submission['id'] + 1

submission.to_csv("new.csv", index = False)