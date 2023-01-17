import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import math

import re   

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from matplotlib import rcParams

from wordcloud import WordCloud



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', None) 

pd.set_option('display.max_rows', None)  

pd.set_option('display.max_colwidth', -1) 
#importing data and storing it in pandas DataFrame

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
#checking head and tail of train data

train.head()
train.shape
train.info()
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
print(train.target.value_counts())

plt.figure(figsize=(8,5))

sns.countplot(train.target)

plt.show()
#creating a new column which will consist of the length of text in each row. 

train['len_text'] = np.NaN

for i in range(0,len(train['text'])):

    train['len_text'][i]=(len(train['text'][i]))

train.len_text = train.len_text.astype(int)
#creating subplots to see distribution of length of tweet

sns.set_style("darkgrid")

f, (ax1, ax2) = plt.subplots(figsize=(12,6),nrows=1, ncols=2,tight_layout=True)

sns.distplot(train[train['target']==1]["len_text"],bins=30,ax=ax1)

sns.distplot(train[train['target']==0]["len_text"],bins=30,ax=ax2)

ax1.set_title('\n Distribution of length of tweet labelled Disaster\n')

ax2.set_title('\nDistribution of length of tweet labelled No Disaster\n ')

ax1.set_ylabel('Frequency')
# word cloud for words related to Disaster 

text=" ".join(post for post in train[train['target']==1].text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(11,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
# word cloud for words related to No Disaster 

text=" ".join(post for post in train[train['target']==0].text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to No Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()

# Import Tokenizer

from nltk.tokenize import RegexpTokenizer

#Instantiate Tokenizer

tokenizer = RegexpTokenizer(r'\w+') 
#changing the contents of selftext to lowercase

train.loc[:,'text'] = train.text.apply(lambda x : str.lower(x))



#removing hyper link, latin characters and digits

train['text']=train['text'].str.replace('http.*.*', '',regex = True)

train['text']=train['text'].str.replace('û.*.*', '',regex = True)

train['text']=train['text'].str.replace(r'\d+','',regex= True)
# "Run" Tokenizer

train['tokens'] = train['text'].map(tokenizer.tokenize)
train.head()
#assigning stopwords to a variable

stop = stopwords.words("english")
# adding this stop word to list of stopwords as it appears on frequently occuring word

item=['amp']



stop.extend(item)
#removing stopwords from tokens

train['tokens']=train['tokens'].apply(lambda x: [item for item in x if item not in stop])
lemmatizer = WordNetLemmatizer()
lemmatize_words=[]

for i in range (len(train['tokens'])):

    word=''

    for j in range(len(train['tokens'][i])):

        lemm_word=lemmatizer.lemmatize(train['tokens'][i][j])#lemmatize

        

        word=word + ' '+lemm_word # joining tokens into sentence    

    lemmatize_words.append(word) # storing in list
#creating a new column to store the result

train['lemmatized']=lemmatize_words



#displaying first 5 rows of dataframe

train.head()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
#defining X and y for the model

X = train['lemmatized']

y = train['target']
# Spliting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# pipeline will consist of two stages:

# 1.Instantiating countVectorizer

# 2.Instantiating logistic regression model



pipe = Pipeline([

    ('cvec', TfidfVectorizer()),  

    ('lr', LogisticRegression()) 

])
tuned_params = {

    'cvec__max_features': [2500, 3000, 3500],

    'cvec__min_df': [2,3],

    'cvec__max_df': [.9, .95],

    'cvec__ngram_range': [(1,1), (1,2)]

}

gs = GridSearchCV(pipe, param_grid=tuned_params, cv=3) # Evaluating model on unseen data



model_lr=gs.fit(X_train, y_train) # Fitting model



# This is the average of all cv folds for a single 

#combination of the parameters specified in the tuned_params

print(gs.best_score_) 



#displaying the best values of parameters

gs.best_params_
gs.score(X_train, y_train)
gs.score(X_test, y_test)
# Generating predictions!

predictions_lr = model_lr.predict(X_test)
# Importing the confusion matrix function

from sklearn.metrics import confusion_matrix



# Generating confusion matrix

confusion_matrix(y_test, predictions_lr)
#interpreting confusion matrix

tn, fp, fn, tp = confusion_matrix(y_test, predictions_lr).ravel()



#values with coreesponding labels

print("True Negatives: %s" % tn)

print("False Positives: %s" % fp)

print("False Negatives: %s" % fn)

print("True Positives: %s" % tp)
# word cloud for Frequntly occuring words related to Disaster

text=" ".join(post for post in train[train['target']==1].lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()

# word cloud for Frequntly occuring words related to No Disaster

text=" ".join(post for post in train[train['target']==0].lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to No Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
#reading the test data

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test.head()
#creating a new column- length 

# this gives the length of the post

test['length'] = np.NaN

for i in range(0,len(test['text'])):

    test['length'][i]=(len(test['text'][i]))

test.length = test.length.astype(int)
# word cloud for Frequntly occuring words in test dataframe

text=" ".join(post for post in test.text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words in test dataframe \n\n',fontsize=18)

plt.axis("off")

plt.show()
# Instantiate Tokenizer

tokenizer = RegexpTokenizer(r'\w+')



#changing the contents of selftext to lowercase

test.loc[:,'text'] = test.text.apply(lambda x : str.lower(x))



#removing hyper link and latin characters

test['text']=test['text'].str.replace('http.*.*', '',regex = True)

test['text']=test['text'].str.replace('û.*.*', '',regex = True)

test['text']=test['text'].str.replace(r'\d+','',regex= True)



# "Run" Tokenizer

test['tokens'] = test['text'].map(tokenizer.tokenize)
test.head()
#removing stopwords from tokens

test['tokens']=test['tokens'].apply(lambda x: [item for item in x if item not in stop])
lemmatize_words=[]

for i in range (len(test['tokens'])):

    word=''

    for j in range(len(test['tokens'][i])):

        lemm_word=lemmatizer.lemmatize(test['tokens'][i][j])#lemmatize

        

        word=word + ' '+lemm_word # joining tokens into sentence    

    lemmatize_words.append(word) # store in list
#creating a new column to store the result

test['lemmatized']=lemmatize_words



#displaying first 5 rows of dataframe

test.head()
# word cloud for Frequntly occuring words in test dataframe after lemmatizing

text=" ".join(post for post in test.lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words in test dataframe \n\n',fontsize=18)

plt.axis("off")

plt.show()
predictions_kaggle = model_lr.predict(test['lemmatized'])
# Creating an empty data frame

submission_kaggle = pd.DataFrame()
# Assigning values to the data frame-submission_kaggle

submission_kaggle['Id'] = test.id

submission_kaggle['target'] = predictions_kaggle
# Head of submission_kaggle

submission_kaggle.head()
# saving data as  final_kaggle.csv

submission_kaggle.loc[ :].to_csv('final_kaggle.csv',index=False)