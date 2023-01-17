# Load the necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

import re



import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load the data



df = pd.read_csv('/kaggle/input/imdb_movie_data.csv')
print('*'*10 + 'First 10 rows' + '*'*10)

print(df.head(10))

print("")

print('*'*10 + 'Information' + '*'*10)

print(df.info())

print("")

print('*'*10 + 'Null values' + '*'*10)

print(df.isnull().any())

print("")

print('*'*10 + 'Duplicate values' + '*'*10)

print(df.duplicated(subset='review').value_counts())



sns.heatmap(df.isnull(),cmap='viridis',cbar=False,yticklabels=False)
df.drop_duplicates(subset='review', inplace=True)
sns.distplot(df.sentiment,kde=False)
# Let us check one data row



df.loc[0,'review']
def cleaner(text):

    # Remove html objects

    text = re.sub('<[^<]*>','',text)

    

    # Temporarily store emoticons

    emoticons = ''.join(re.findall('[:;=]-+[\)\(pPD]+',text))

    

    # Remove non-word characters and combine back the emoticons

    text = re.sub('\W+',' ',text.lower()) + emoticons.replace('-','')

    

    return text
# let us check the function if it works



cleaner(df.loc[0,'review'])
# Apply the function to whole dataset



df['review'] = df['review'].apply(cleaner)

df.head(10)
porter = PorterStemmer()



def token_porter(text):

    return [porter.stem(word) for word in text.split()]



# We will also tokenize without porter

def token(text):

    return text.split()



# We will pass the 2 functions in our GridSearchCV
tfidf = TfidfVectorizer(lowercase=False)



# Also load the stopwords from nltk library

stop = stopwords.words('english')
X = df.iloc[:,0].to_numpy()

y = df.iloc[:,1].to_numpy()



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)
# Initialize parameters

param_grid = [{'vect__stop_words':[stop, None],

               'vect__tokenizer':[token, token_porter],

               'clf__penalty':['l2'],

               'clf__C':[1, 10, 100]},

              {'vect__use_idf':[False],

               'vect__stop_words':[stop, None], 

               'vect__tokenizer':[token, token_porter],

               'clf__penalty':['l2'],

               'clf__C':[1, 10, 100]}

             ]



# Use pipeline to build composite estimator

lr_tfidf = Pipeline([('vect', tfidf),

                     ('clf', LogisticRegression(tol=0.01, random_state=0))])



gs = GridSearchCV(lr_tfidf, 

                  param_grid, 

                  scoring='accuracy',

                  cv=5,

                  n_jobs=1,

                  verbose=0)
# Fit our model to the train dataset

gs.fit(X_train, y_train)
print('Best parameter settings: %s' % gs.best_params_)

print('CV Accuracy:%.3f' % gs.best_score_)
# Get our best classifier settings

clf = gs.best_estimator_



print('Test Accuracy: %.3f' % clf.score(X_test, y_test))