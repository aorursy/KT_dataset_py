# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/yelp-csv/yelp_academic_dataset_review.csv')

df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df.isnull().sum()
df.dropna()
df.describe()
df.info()
#df.drop(df.loc[10001:1125458].index, inplace=True)

#df.shape
df['text'][0]
#df['lenght']=df['text'].apply(len)

df['length']=df['text'].str.len()
df.head()
df['length'].plot(bins=100,kind='hist');
df.describe().T
df[df['length']==1]['text'].iloc[0]
df[df['length']==5000]['text'].iloc[0]
sns.countplot(x='stars',data=df);

#sns.countplot(y='stars',data=df);
g=sns.FacetGrid(data=df,col='stars',col_wrap=5)

g.map(plt.hist,'length',bins=20,color='r');
df_1=df[df['stars']==1]

df_1.head()
df_5=df[df['stars']==5]

df_5.head()
df_1_5=pd.concat([df_1,df_5])

df_1_5.shape
df_1_5.info()
print('1-Star Review Percentage=',(len(df_1)/len(df_1_5))*100,'%')
print('5-Star Review Percentage=',(len(df_5)/len(df_1_5))*100,'%')
sns.countplot(x=df_1_5['stars'],label='Count');
import string

string.punctuation
Test='Hello Mr. Future,I am so happy to be learning AI'
Test_punc_removed=[char  for char in Test if char not in string.punctuation]

Test_punc_removed
Test_punc_removed_join=''.join(Test_punc_removed)
Test_punc_removed_join
from nltk.corpus import stopwords

stopwords.words('english')
Test_punc_removed_join_clean=[word  for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]

Test_punc_removed_join_clean
mini_challenge='Here is a mini challenge,that will teach you how to remove stopwords and puncutations'
challenge=[char for char in mini_challenge if char not in string.punctuation  ]

challenge=''.join(challenge)

challenge=[word  for word in challenge.split() if word.lower() not in stopwords.words('english')]
challenge
sample_data=['This is the first document.','This is thesecond document.','This is the third document']

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()

X=vectorizer.fit_transform(sample_data)
print(vectorizer.get_feature_names())
print(X.toarray())
mini_challenge=['Hello World','Hello Hello World','Hello World world world']

vectorizer_challenge=CountVectorizer()

X_challenge=vectorizer_challenge.fit_transform(mini_challenge)

print(X_challenge.toarray())
df_1_5
df_1_5 = df_1_5.reset_index()

df_1_5.shape
#df_1_5.drop(df_1_5.loc[0:516818].index, inplace=True)   # Considering only first 10000 reviews

#df_1_5.shape
def message_cleaning(message):

    Test_punc_removed = [char for char in message if char not in string.punctuation ]

    Test_punc_removed_join=''.join(Test_punc_removed)

    Test_punc_removed_join_clean=[ word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]

    return Test_punc_removed_join_clean
df_clean=df_1_5['text'].apply(message_cleaning)
"""#test_strs = ['THIS IS A TEST!', 'another test', 'JUS!*(*UDFLJ)']

df = pd.DataFrame(df_1_5, columns=['text'])

df_clean = df.apply(lambda x: message_cleaning(x.text), axis=1)"""
"""#test_strs = ['THIS IS A TEST!', 'another test', 'JUS!*(*UDFLJ)']

df = pd.DataFrame(test_strs, columns=['text'])

df['new_text'] = df.apply(lambda x: clean(x.text), axis=1)"""
#df_clean=df_1_5['text'].apply(message_cleaning)
print(df_clean[0]) # cleaned up review 
print(df_1_5['text'][0]) # Original review
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(analyzer=message_cleaning)

df_countvectorizer=vectorizer.fit_transform(df_1_5['text'])
print(vectorizer.get_feature_names())
print(df_countvectorizer.toarray())
df_countvectorizer.shape
from sklearn.naive_bayes import MultinomialNB

NB_classifier=MultinomialNB()

label=df_1_5['stars'].values
df_1_5['stars'].values
NB_classifier.fit(df_countvectorizer,label)
testing_sample=['amazing food! highly recommended']

testing_sample_countvectorizer=vectorizer.transform(testing_sample)

test_predict=NB_classifier.predict(testing_sample_countvectorizer)

test_predict
X=df_countvectorizer

X.shape
y=label

y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
y_train.shape
from sklearn.naive_bayes import MultinomialNB

NB_classifier=MultinomialNB()

NB_classifier.fit(X_train,y_train)
from sklearn.metrics import classification_report,confusion_matrix

y_predict_train=NB_classifier.predict(X_train)

y_predict_train
cm=confusion_matrix(y_train,y_predict_train)

sns.heatmap(cm,annot=True)
y_predict_test=NB_classifier.predict(X_test)

y_predict_test

cm=confusion_matrix(y_test,y_predict_test)

sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_predict_test))