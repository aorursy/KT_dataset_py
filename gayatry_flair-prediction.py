#importing necessary libraries

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import stopwords

import string

import seaborn

import warnings

warnings.filterwarnings("ignore",category= DeprecationWarning)

warnings.filterwarnings('ignore')



%matplotlib inline
#Reading the data

data = pd.read_csv('/kaggle/input/reddit-india-flair-detection/datafinal.csv')



dataf = data.copy()

data
data.columns
data = data.drop(['score','url','comms_num','author','timestamp'],axis=1)
data.head()
data['title'][0]
data['body'][0]
data['combined_features'][0]
data['comments'][0]
data = data.drop(['combined_features'],axis=1)

data.head()
data.info()
data.describe()
data['flair'].unique()
data.groupby('flair')['title'].describe()
data.groupby('flair')['id'].describe()
data.groupby('flair')['body'].describe()
data.groupby('flair')['comments'].describe()
#Dropping the rows corresponding to date-time flairs

f = data['flair'].dropna()

regx = re.compile(r"[\d]{1,2}-[\d]{1,2}-[\d]{4} [\d]{1,2}:[\d]{1,2}")

for __ in f:

    #print(flair)

    x = regx.search(__)

    if x is not None:

        #print(x.group())

        d = data[data.flair == x.group()]

        #print(d)

        data = data.drop(d.index)
data['flair'].unique()
data[data['flair'] == np.nan].describe()
data = data.dropna(subset=['flair'])
data.info()
data['text'] = data['title'].astype(str) + data['body'].astype(str) + data['comments'].astype(str)
data_final = data[['flair','id','text']]

data_final.head()
data_final.describe()
data_final.groupby('flair')['text'].describe()
data_final['text'] = data_final['text'].str.replace("[^a-zA-Z0-9 \n.]"," ")
"""Now we have clean data!!!"""

data_final.head(10) 
"""

    1. Removing all punctuation

    2. Removing stop-words

    3. Returns a clean text

"""

def clean_txt(mess):

    

    nonpunc = [char for char in mess if char not in string.punctuation] #list of strings which are non-punc

    

    nonpunc = "".join(nonpunc) #join back to form the whole string

    

    return [word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]    
data_final['text'] = data_final['text'].apply(clean_txt)
from nltk import WordNetLemmatizer



le = WordNetLemmatizer()



data_final['text'] = data_final['text'].apply(lambda x : [le.lemmatize(word) for word in x])

data_final['text'] = data_final['text'].apply(lambda x : " ".join(x))

data_final['text']
qwe = data_final['text'].copy()

qwe
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer = clean_txt).fit(data_final['text'])
text_bow = bow_transformer.transform(data_final['text'])
print(f'Shape of sparse matrix is {text_bow.shape}')

print(f'Length of dictionary is {len(bow_transformer.vocabulary_)}')

print(f'Number of non=zero occusrances is {text_bow.nnz}')

sparsity = (text_bow.nnz/(text_bow.shape[0]*text_bow.shape[1]))*100

print('Sparsity :', sparsity)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(text_bow)

text_tfidf = tfidf_transformer.transform(text_bow)
print(f'Shape of tfidf of text is {text_tfidf.shape}')
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(text_tfidf,data_final['flair'])
predictions = model.predict(text_tfidf)
from sklearn.metrics import classification_report
print(classification_report(data_final['flair'],predictions))
from sklearn.pipeline import Pipeline
pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=clean_txt)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB()),

])
from sklearn.model_selection import train_test_split
text_train, text_test, flair_train, flair_test = train_test_split(data_final['text'],data_final['flair'])
pipeline.fit(text_train,flair_train)
predictions = pipeline.predict(text_test)
print(classification_report(flair_test,predictions))
print(text_test)
ids = [data_final.iloc[int(i)]['id'] for i in text_test.index]

Predicted_df = pd.DataFrame({'ID':ids,'Text':text_test,'PredictedFlair':predictions}).reset_index(drop=True)
Predicted_df