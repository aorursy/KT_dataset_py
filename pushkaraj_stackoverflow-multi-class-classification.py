# Import necessary libraries

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import gc
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read data

df=pd.read_csv('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')
df.Y.value_counts()
# Create feature CreationYear and remove feature CreationDate

df['CreationYear']=df.CreationDate.apply(lambda val:int(val.split()[0].split('-')[0]))

del df['CreationDate']

gc.collect()



# Label encode target feature

df.Y.replace({'LQ_CLOSE':0,'HQ':1,'LQ_EDIT':2},inplace=True)



# Create train and test dataframes

train_df=df[['Title','Body','Tags','Y']][df.CreationYear<2019].copy()

test_df=df[['Title','Body','Tags','Y']][df.CreationYear>=2019].copy()



# Delete main dataframe to clear some memory

del df

gc.collect()



train_df.shape, test_df.shape
def clean_tags(string):

    return ((string.replace('><',' ')).replace('<','')).replace('>','')



for df in [train_df,test_df]:

    df['Tags']=list(map(lambda val:clean_tags(val), df.Tags.values))
from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from tqdm.notebook import tqdm

import re
def clean_body_text(df):

    # Create list of english stopwords from nltk library

    stop_words = set(stopwords.words('english'))



    # Create a list to save body text of all questions

    body_text=[]

    # Create a list to indicate if code snippet is present in the body

    code_indicator=[]

    reference_link_indicator=[]

    image_indicator=[]



    for ind in tqdm(range(df.shape[0])):



        # Create a BeautifulSoup object

        q_body=df['Body'].values[ind].lower()

        soup=BeautifulSoup(q_body)

        

        # To check if body contains code snippet

        if len(soup.findAll('code'))>0:

            code_indicator.append(1)

            # Find all code tags and replace them with empty string ''

            for code_text in soup.findAll('code'):

                code_text.replace_with('')

        else:

            code_indicator.append(0)

        

        # To check if body contains reference link tag

        if len(soup.findAll('a'))>0:

            reference_link_indicator.append(1)

        else:

            reference_link_indicator.append(0)



        # To check if body contains image

        if len(soup.findAll('img'))>0:

            image_indicator.append(1)

        else:

            image_indicator.append(0)            



        # Create a list to save all <p> tag text of a question into a list

        text=[]

        for line in soup.findAll('p'):

            line=line.get_text()

            line=line.replace('\n','')

            line=re.sub(r'[^A-Za-z0-9]', ' ', line)

            line=' '.join([word for word in line.split() if not word in stop_words])

            text.append(line)



        body_text.append(' '.join(text))



    return body_text, code_indicator, reference_link_indicator, image_indicator
train_df['body_text'],train_df['code_indicator'],train_df['reference_link_indicator'],train_df['image_indicator']=clean_body_text(train_df)

test_df['body_text'],test_df['code_indicator'],test_df['reference_link_indicator'],test_df['image_indicator']=clean_body_text(test_df)
def clean_title_text(df):

    # Create list of english stopwords from nltk library

    stop_words = set(stopwords.words('english'))

    title_text=[]

    for ind in range(df.shape[0]):

        text=df.Title.values[ind].lower()

        text=text.replace('\n','')

        text=re.sub(r'[^A-Za-z0-9]', ' ', text)

        text=' '.join([word for word in text.split() if not word in stop_words])



        title_text.append(text)

        

    return title_text
train_df['title_text']=clean_title_text(train_df)

test_df['title_text']=clean_title_text(test_df)
del train_df['Title'], train_df['Body'], test_df['Title'], test_df['Body']

gc.collect()
train_y=train_df['Y']

test_y=test_df['Y']



del train_df['Y'], test_df['Y']

gc.collect()
train_df.shape, test_df.shape, train_y.shape, test_y.shape
from sklearn.feature_extraction.text import TfidfVectorizer

import scipy
train_tfidf=[]

test_tfidf=[]

for feat in tqdm(train_df.select_dtypes(include='object').columns):

    vectorizer=TfidfVectorizer(ngram_range=(1,4),max_features=10000)

    train_tfidf.append(vectorizer.fit_transform(train_df[feat]))

    test_tfidf.append(vectorizer.transform(test_df[feat]))
train_tfidf=scipy.sparse.hstack(train_tfidf).tocsr()

test_tfidf=scipy.sparse.hstack(test_tfidf).tocsr()



train_tfidf.shape, test_tfidf.shape
train_x=scipy.sparse.hstack([train_tfidf, train_df[['code_indicator','reference_link_indicator','image_indicator']].values]).tocsr()

test_x=scipy.sparse.hstack([test_tfidf, test_df[['code_indicator','reference_link_indicator','image_indicator']].values]).tocsr()
train_x.shape, test_x.shape
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
lr=LogisticRegression(max_iter=1000,n_jobs=-1)

lr.fit(train_x,train_y)
train_y_pred=lr.predict(train_x)

test_y_pred=lr.predict(test_x)
print('Mean accuracy score:',lr.score(train_x,train_y))



fig,ax=plt.subplots(figsize=(8,8))

sns.heatmap(metrics.confusion_matrix(train_y,train_y_pred),annot=True,cbar=False,fmt='d',cmap='Reds')

ax.set_ylabel('True label',fontsize=14)

ax.set_xlabel('Predicted label',fontsize=14)

ax.set_title('Confusion matrix: Train set prediction',fontsize=16);
print('Mean accuracy score:',lr.score(test_x,test_y))



fig,ax=plt.subplots(figsize=(8,8))

sns.heatmap(metrics.confusion_matrix(test_y,test_y_pred),annot=True,cbar=False,fmt='d',cmap='Reds')

ax.set_ylabel('True label',fontsize=14)

ax.set_xlabel('Predicted label',fontsize=14)

ax.set_title('Confusion matrix: Test set prediction',fontsize=16);