# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

import string

import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train  = pd.read_csv('../input/train.csv',sep='~')

test = pd.read_csv('../input/test.csv',sep='~')
train.head()
test.head()
#Keeping only the necessary features

train_Desc_Only = train[['Description','Is_Response']]
train_Desc_Only.head()
#Checking for possible Nuetral Values

train['Is_Response'].unique()
#Splitting into positive and Negative comments to check for possible Imbalance

train_Desc_Only_Pos = train_Desc_Only[train_Desc_Only['Is_Response']=='Good']

train_Desc_Only_Pos = train_Desc_Only_Pos['Description']

train_Desc_Only_Neg = train_Desc_Only[train_Desc_Only['Is_Response']=='Bad']

train_Desc_Only_Neg = train_Desc_Only_Neg['Description']
train_Desc_Only_Pos.head()
train_Desc_Only_Neg.head()
#We have twice as many positive Responses than negative responses

train_Desc_Only_Pos.shape[0],train_Desc_Only_Neg.shape[0]

#We will deal with this class Imbalance after deploying all the cleaning functions
#Adding a Sentiment Column to the dataset

x=[]

for i in train['Is_Response']:

    if i=='Good':

        x.append(1)

    elif i == 'Bad':

        x.append(0)
train['sentiment'] = x
train.head()
#Total number of words in Description.

base_list=[]

for i in train['Description']:

    i=str(i)

    for j in i.split():

        base_list.append(j)

#Set of Unique words

base_uniq = set(base_list)

print("Total number of words: {} and Total number of unique words {}".format(len(base_list),len(base_uniq)))     
#checking for alphabets

x_words=[]

for i in base_uniq:

    if i.isalpha()==True:

        x_words.append(i)

len(set(x_words))

#Hence this makes us believe there are many special characters present
#Replacing special characters 

train['Description'] = train['Description'].str.replace("[^a-zA-Z#]", " ")
#Total number of words.

base_list=[]

for i in train['Description']:

    i=str(i)

    for j in i.split():

        base_list.append(j)

#Set of Unique words

base_uniq = set(base_list)

print("Total number of words: {} and Total number of unique words {}".format(len(base_list),len(base_uniq))) 
#checking for alphabets

x_words=[]

for i in base_uniq:

    if i.isalpha()==True:

        x_words.append(i)

len(set(x_words))

#Still few special characters exist we will now set a probe as to which all special characters still exist
#checking further for Special Charcters

x_words_Special=[]

for i in base_uniq:

    if i.isalpha()==False:

        x_words_Special.append(i)

print(set(x_words_Special),len(set(x_words_Special)))

#So we now need to check and replace for '#' in each word
#Function to find #

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt  
#remove # from Description

import re

train['Description'] = np.vectorize(remove_pattern)(train['Description'],'#[\w]*')
#Total number of words.

base_list=[]

for i in train['Description']:

    i=str(i)

    for j in i.split():

        base_list.append(j)

#Set of Unique words

base_uniq = set(base_list)

#checking for Special Charcters

x_words_Special=[]

for i in base_uniq:

    if i.isalpha()==False:

        x_words_Special.append(i)

print(set(x_words_Special),len(set(x_words_Special)))

#So we now need to check and replace for '#' in each word
#Total number of unique words.

base_list=[]

for i in train['Description']:

    i=str(i)

    for j in i.split():

        base_list.append(j)

#Set of Unique words

base_uniq = set(base_list)



x_words=[]

for i in base_uniq:

    if i.isalpha()==True:

        x_words.append(i)

len(set(x_words))



print("Total number of words: {} and Total number of unique words {}, out of which {} are valid".format(len(base_list),len(base_uniq),len(set(x_words)))) 
#Removing Punctuations, Numbers, and Special Characters

train['Description'] = train['Description'].str.replace("[^a-zA-Z#]", " ")
train.head(2)
#Unique values for Browser_Used

train['Browser_Used'].unique()
train['Browser_code'] = train['Browser_Used'].map({'Google Chrome':1,'Firefox':2,'Mozilla':3,'InternetExplorer':4,'Edge':5,'Mozilla Firefox':6,'Internet Explorer':7,

                                                  'Chrome':8,'IE':9,'Opera':10,'Safari':11})
#Unique values for Device_Used

train['Device_Used'].unique()
train['Device_code'] = train['Device_Used'].map({'Desktop':1,'Tablet':2,'Mobile':3})
train.head()
#Keeping only the necessary features

train_Desc_Only = train[['Description','Is_Response','sentiment','Browser_code','Device_code']]

train_Desc_Only.head()
#Removing Short Words(Words of the length 3 and less)

train_Desc_Only['Description'] = train_Desc_Only['Description'].apply(lambda x: ' '.join([w for w in x.split() if len(x) > 3]))
stopwords = set(stopwords.words('english'))

train_Desc_Only['Description'] = train_Desc_Only['Description'].apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))
#Adding a new feature 'text_length'

train_Desc_Only['text_length'] = train_Desc_Only['Description'].apply(len)
train_Desc_Only.head()
#Setting the style for seaborn

sns.set_style('white')
#Using FacetGrid to compare the variation of size of the text

g = sns.FacetGrid(train_Desc_Only,col='Is_Response')

g.map(plt.hist,'text_length',bins = 50)

#Hence we can say that generally longer texts lead too 'Good' Review
#BoxPlot of the text_length field and the Is_Response field

sns.boxplot(x='Is_Response',y='text_length',data=train_Desc_Only)

#this shows that text length cannot be used as a useful feature to predict the Resoponse since there are many values for text_length in the outliers
#CountPlot of number of occurences of Good and Bad

sns.countplot(x='Is_Response',data = train_Desc_Only,palette='rainbow')

#This depicts that the number of responses for Good far exceeds the number responses for Bad
#Using FacetGrid to compare the variation of Device Code

g = sns.FacetGrid(train_Desc_Only,col='Is_Response')

g.map(plt.hist,'Device_code',bins = 50)

#Here we can see that we cannot decide the Response value from the Device_code feature
#BoxPlot of the text_length field and the Is_Response field

sns.boxplot(x='Is_Response',y='Device_code',data=train_Desc_Only)
#Using FacetGrid to compare the variation of Browser code 

g = sns.FacetGrid(train_Desc_Only,col='Is_Response')

g.map(plt.hist,'Browser_code',bins = 50)

#Here we can see that we cannot decide the Response value from the Device_code feature
#We will now use Group By to get the mean values of text_length column

Response = train_Desc_Only.groupby('Is_Response').mean()

Response
#Correlation between the variables

Response.corr()
#Correlation Map

sns.heatmap(Response.corr(),cmap='coolwarm',annot = True)
Ant = train_Desc_Only
Ant.info()
#Splitting into x and y

x = Ant['Description']

y = Ant['Is_Response']
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
#Using CountVectorize object to transform x

x = cv.fit_transform(x)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .3,random_state = 101)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipe = Pipeline([('bow',CountVectorizer()),

                ('tfidf',TfidfTransformer()),('model',MultinomialNB())])

#bow : bag of words
#Splitting into x and y

x = Ant['Description']

y = Ant['Is_Response']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .3,random_state = 101)
pipe.fit(X_train,y_train)
predictions = pipe.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
#Trying now with Bernoullis NB



from sklearn.naive_bayes import BernoulliNB



pipe = Pipeline([('bow',CountVectorizer()),

                ('tfidf',TfidfTransformer()),('model',BernoulliNB())])

#bow : bag of words



#Splitting into x and y

x = Ant['Description']

y = Ant['Is_Response']



## Train Test Split



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .3,random_state = 101)



## Fitting the pipeline to the training data



pipe.fit(X_train,y_train)



## Predictions and Evaluation



predictions = pipe.predict(X_test)



from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
private = test[0:2628]

public = test[2628:]

#Trying now with Bernoullis NB



from sklearn.naive_bayes import BernoulliNB



pipe = Pipeline([('bow',CountVectorizer()),

                ('tfidf',TfidfTransformer()),('model',BernoulliNB())])

#bow : bag of words



#Splitting into x and y

X_train = Ant['Description']

y_train = Ant['Is_Response']



X_test = private['Description']



## Fitting the pipeline to the training data



pipe.fit(X_train,y_train)



## Predictions and Evaluation



private['Is_Response'] = pipe.predict(X_test)



#Total number of words.

base_list=[]

for i in test['Description']:

    i=str(i)

    for j in i.split():

        base_list.append(j)

#Set of Unique words

base_uniq = set(base_list)

print("Total number of words: {} and Total number of unique words {}".format(len(base_list),len(base_uniq))) 



#checking for alphabets

x_words=[]

for i in base_uniq:

    if i.isalpha()==True:

        x_words.append(i)

len(set(x_words))

#Still few special characters exist we will now set a probe as to which all special characters still exist



#checking further for Special Charcters

x_words_Special=[]

for i in base_uniq:

    if i.isalpha()==False:

        x_words_Special.append(i)

print(set(x_words_Special),len(set(x_words_Special)))

#So we now need to check and replace for '#' in each word



#Function to find #

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt  



#remove # from Description

import re

test['Description'] = np.vectorize(remove_pattern)(test['Description'],'""[\w]*')
test['Description'] = np.vectorize(remove_pattern)(test['Description'],'--[\w]*')
#Splitting into x and y

X_train = Ant['Description']

y_train = Ant['Is_Response']

X_test = test['Description']

### Employing Count Vectorizer

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

#Using CountVectorize object to transform x

X_train = cv.fit_transform(X_train)





## Import Naive Bayes Classifier



from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()



## Fitting Naive Bayes Classifier



nb.fit(X_train,y_train)



## Predictions and Evaluations

X_test = cv.fit_transform(X_test)

print(test.shape,X_test.shape)