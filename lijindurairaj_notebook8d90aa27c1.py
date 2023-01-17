import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_colwidth', -1)
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import Counter
np.set_printoptions(threshold=np.inf)
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV,train_test_split
_data=pd.read_csv('../input/nlp-getting-started/train.csv')
_data


class data_processing:
    
    def remove_special_characters(self,data):
        _d=[]
        pattern='[a-zA-Z\s]'
        for words in data:
             _d.append(''.join([_d for _d in words if re.search(pattern, _d)]))
        return _d
        
    def vectorize_data(self,data):
        vectorizer_obj=CountVectorizer(max_features=500,stop_words='english',max_df=.80,min_df=10)
        return vectorizer_obj.fit_transform(data)
    
    def train_test_split(self,x,y):
        return train_test_split(x,y,train_size=0.85,random_state=42)
    
    def main(self,data):
        special_char_removed=self.remove_special_characters(data['text'])
        x=self.vectorize_data(special_char_removed)
        return self.train_test_split(x,data['target'])
    
    

_processing=data_processing()
x_train,x_test,y_train,y_test=_processing.main(_data)



svm_pipeline=Pipeline([
    ('vectorizer',CountVectorizer(stop_words='english')),
    ('svm',SVC())
])
nb_pipeline=Pipeline([
    ('nb_vectorizer',CountVectorizer(stop_words='english')),
    ('nb',MultinomialNB())
])

pipelines=[svm_pipeline,nb_pipeline]
parameters=[
    {
        'vectorizer__max_features':[500,750,1000],
        'vectorizer__max_df':[0.60,0.75,0.85],
        'vectorizer__min_df':[50,100,150],
        'svm__C':[10,100,150],
        'svm__kernel':['linear','rbf']
    },
    {
        'nb_vectorizer__max_features':[500,750,1000],
        'nb_vectorizer__max_df':[0.60,0.75,0.85],
        'nb_vectorizer__min_df':[50,100,150],
        'nb__alpha':[1,10,40]
    }
]


for pipe_index in list(range(0,len(pipelines))):
    grid_search_cv=GridSearchCV(pipelines[pipe_index],parameters[pipe_index],n_jobs=-1,cv=5)
    grid_search_cv.fit(x_train,y_train)