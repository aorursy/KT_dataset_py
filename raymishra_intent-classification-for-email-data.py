# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelBinarizer
lb= LabelBinarizer()
df= pd.read_csv("../input/traindata.csv")
df.dropna(axis= 0)
df

#Regular expressions help us deal with characters and numerics and modify them according to our requirement 
import re
#import the nltk library
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))
from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()
#creating a function to encapsulate our entire preprocessing and feature engineering steps
def processing(df):
    #get the no. of punctuations
    df['specialchars'] = df['text'].apply(lambda x : len(x) - len(re.findall('[\w]', x)))
    #lowering and removing punctuation
    df['processed'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
    #removing the numerical values and working only with text values
    df['processed'] = df['processed'].apply(lambda x : re.sub('[^a-zA-Z]', " ", x ))
    #removing the stopwords
    df['processed'] = df['processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in sw ]))
    #stemming the words
    df['processed'] = df['processed'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
    #count the no. of words in the text
    df['words'] = df['processed'].apply(lambda x: len(x.split(' ')))
    df["Intent"]= lb.fit_transform(df["Intent"])
    return df
    
df= processing(df)   
df.head()


from sklearn.model_selection import train_test_split

def Xysplit(df):
    features= [x for x in df.columns.values if x not in ['text', 'Intent']]
    target = 'Intent'
    X= df[features]
    y= df[target]
    return X, y
X, y =Xysplit(df)

from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


text = Pipeline([
                ('selector', TextSelector(key='processed')),
                ('vec', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                 ])


from sklearn.preprocessing import StandardScaler


words =  Pipeline([
                ('selector', NumberSelector(key='words')),
                ('standard', StandardScaler())
            ])
specialchars =  Pipeline([
                ('selector', NumberSelector(key='specialchars')),
                ('standard', StandardScaler()),
            ])
from sklearn.pipeline import FeatureUnion
feats = FeatureUnion([('text', text),              
                      ('words', words),
                      ('specialchars', specialchars),
                      ])

from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ('features', feats),
    ('classifier', RandomForestClassifier())
])
from sklearn.model_selection import RandomizedSearchCV
parameters = { 
    'features__text__vec__ngram_range': [(1,1), (1,2)],
    'classifier__min_samples_leaf' : [1, 2, 3],
    'classifier__min_samples_split' : [2, 3, 4],
    'classifier__max_depth' : [30, 40, 50, 55],
    'classifier__n_estimators' : [1200, 1300, 1350, 1375]

              }

clf= RandomizedSearchCV(pipeline, parameters, cv= 5, n_jobs= -1)
clf.fit(X, y)
#predicting on test data
submission = pd.read_csv('../input/testdata.csv')

#preprocessing
submission = processing(submission)

#splitting
X_test, y_test= Xysplit(submission)
predictions = clf.predict(X_test)
#printing the accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))