import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/winemag-data-130k-v2.csv')
data.info()
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
data=data.dropna(subset=['price'])
print("Total number of examples: ", data.shape[0])
print("Number of examples with the same title and description: ", data[data.duplicated(['description','title'])].shape[0])
data=data.drop_duplicates(['description','title'])
data=data.reset_index(drop=True)
data=data.fillna(-1)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
from wordcloud import WordCloud, STOPWORDS
import re

from nltk.tokenize import RegexpTokenizer
data['description']= data['description'].str.lower()
data['description']= data['description'].apply(lambda elem: re.sub('[^a-zA-Z]',' ', elem))  
data['description']
tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = data['description'].apply(tokenizer.tokenize)
words_descriptions.head()
all_words = [word for tokens in words_descriptions for word in tokens]
data['description_lengths']= [len(tokens) for tokens in words_descriptions]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
from collections import Counter
count_all_words = Counter(all_words)
count_all_words.most_common(100)
stopword_list = stopwords.words('english')
ps = PorterStemmer()
words_descriptions = words_descriptions.apply(lambda elem: [word for word in elem if not word in stopword_list])
words_descriptions = words_descriptions.apply(lambda elem: [ps.stem(word) for word in elem])
data['description_cleaned'] = words_descriptions.apply(lambda elem: ' '.join(elem))
all_words = [word for tokens in words_descriptions for word in tokens]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
count_all_words = Counter(all_words)
count_all_words.most_common(100)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv

def prepare_dataframe(vect, data, features=True):
    vectorized=vect.fit_transform(data['description_cleaned']).toarray()
    vectorized=pd.DataFrame(vectorized)
    if features == True:
        X=data.drop(columns=['points','Unnamed: 0','description','description_cleaned'])
        X=X.fillna(-1)
        print(X.columns)
        X=pd.concat([X.reset_index(drop=True),vectorized.reset_index(drop=True)],axis=1)
        categorical_features_indices =[0,1,3,4,5,6,7,8,9,10]
    else:
        X=vectorized
        categorical_features_indices =[]
    y=data['points']
    return X,y,categorical_features_indices
#model definintion and training.
def perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test,categorical_features_indices,name):
    model = CatBoostRegressor(
        random_seed = 100,
        loss_function = 'RMSE',
        iterations=800,
    )
    
    model.fit(
        X_train, y_train,
        cat_features = categorical_features_indices,
        verbose=False,
        eval_set=(X_valid, y_valid)
    )
    
    print(name+" technique RMSE on training data: "+ model.score(X_train, y_train).astype(str))
    print(name+" technique RMSE on test data: "+ model.score(X_test, y_test).astype(str))
    
def prepare_variable(vect, data, features_append=True):
    X, y , categorical_features_indices = prepare_dataframe(vect, data,features_append)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, 
                                                        random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                        random_state=52)
    return X_train, y_train,X_valid, y_valid,X_test, y_test, categorical_features_indices
vect= CountVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)
training_variable=prepare_variable(vect, data)
perform_model(*training_variable, 'Bag of Words Counts')
vect= TfidfVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)
training_variable=prepare_variable(vect, data)
perform_model(*training_variable, 'TF-IDF')

vect= CountVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)
training_variable=prepare_variable(vect, data, False)
perform_model(*training_variable, 'Bag of Words Counts')
vect= TfidfVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)
training_variable=prepare_variable(vect, data, False)
perform_model(*training_variable, 'TF-IDF')