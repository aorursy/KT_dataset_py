# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import re

import string



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/google-quest-challenge/train.csv')

test=pd.read_csv('../input/google-quest-challenge/test.csv')

sample_submission=pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
train.head()
train.columns
test.columns
sample_submission.columns
train_col=test.columns[1:]

target_col=sample_submission.columns[1:]
train[train_col].head()
train[target_col].head()
len(target_col)
#distplot of train dataframe's target_col

f, axes=plt.subplots(6,5, figsize=(20,20))

ax=axes.ravel()



for i, col in enumerate(target_col):

    sns.distplot(train[col], ax=ax[i])
#check the null_value of train, test dataframe

print(train.isnull().sum(),'\n','-'*50)

print(test.isnull().sum())
#check the unique value in train, test dataframe

for i in train_col:

    print(i,' : ', len(train[i].value_counts()))

print("-"*30)

for i in train_col:

    print(i,' : ', len(test[i].value_counts()))
print(np.shape(train)[0])

train[train_col].head()
#in train_col,i think question_user_name, answer_user_name column is not useful so drop this columns

train_col=[i for i in train_col if i not in ['question_user_name','answer_user_name']]
train[train_col].head()
train['answer'][4]
def lower(text):#lowercase translation

    return text.lower()



def remove_number(text):#remove_number

    new_text=re.sub('[0-9]+','',text)

    return new_text



def remove_punctuation(text):#remove_punctuation

    new_text=''.join(c for c in text if c not in string.punctuation)

    return new_text



def remove_special_character(text):#remove special character

    new_text=re.sub(r'https://','',text)

    new_text=re.sub(r'http://','',new_text)

    new_text=re.sub(r'\n',' ',new_text)

    return new_text
train_col
for i in ['question_title','question_body','answer']:

    

    train[i]=train[i].apply(lambda i:remove_special_character(i))

    test[i]=test[i].apply(lambda i:remove_special_character(i))

    

    train[i]=train[i].apply(lambda i:lower(i))

    test[i]=test[i].apply(lambda i:lower(i))

    

    train[i]=train[i].apply(lambda i:remove_number(i))

    test[i]=test[i].apply(lambda i:remove_number(i))

    

    train[i]=train[i].apply(lambda i:remove_punctuation(i))

    test[i]=test[i].apply(lambda i:remove_punctuation(i))
train[train_col].head()
def remove_anoter_character(text):

    new_text=re.sub('/',' ',text)

    new_text=re.sub('\.',' ',new_text)

    new_text=re.sub('-',' ',new_text)

    return new_text



for i in ['question_user_page','answer_user_page','url','host']:

    

    train[i]=train[i].apply(lambda i:remove_special_character(i))

    test[i]=test[i].apply(lambda i:remove_special_character(i))

    

    train[i]=train[i].apply(lambda i:remove_number(i))

    test[i]=test[i].apply(lambda i:remove_number(i))

    

    train[i]=train[i].apply(lambda i:remove_anoter_character(i))

    test[i]=test[i].apply(lambda i:remove_anoter_character(i))
train[train_col].head()
#make tokenizing without category column



from nltk.tokenize import word_tokenize

def tokenizer(text):

    return word_tokenize(text)



for i in [col for col in train_col if col not in ['category']]:

    train[i]=train[i].apply(lambda i:tokenizer(i))

    test[i]=test[i].apply(lambda i:tokenizer(i))



train[train_col].head()
#remove 'com', 'users' in user_pages, host columns

def remove_com_users(text):

    new_text=[i for i in text if i not in ['com','users']]

    return new_text



for i in ['question_user_page','answer_user_page','host']:

    train[i]=train[i].apply(lambda i:remove_com_users(i))

    test[i]=test[i].apply(lambda i:remove_com_users(i))



train[train_col].head()
#remove stopword, normalization

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer, PorterStemmer



def remove_stopwords(text):#remove stopwords

    new_text=[i for i in text if i not in stopwords.words('english')]

    return new_text



def trans_Lemmatize(text):

    return [WordNetLemmatizer().lemmatize(w) for w in text]



def trans_stem(text):

    return [PorterStemmer().stem(w) for w in text]





for i in ['question_title','question_body','answer','url']:

    train[i]=train[i].apply(lambda i:remove_stopwords(i))

    test[i]=test[i].apply(lambda i:remove_stopwords(i))

    

    train[i]=train[i].apply(lambda i:trans_Lemmatize(i))

    test[i]=test[i].apply(lambda i:trans_Lemmatize(i))

    

train[train_col].head()
#again join the list for CountVector

def join_list(text):

    return ' '.join(text)



for i in [col for col in train_col if col not in ['category']]:

    train[i]=train[i].apply(lambda i:join_list(i))

    test[i]=test[i].apply(lambda i:join_list(i))

    

train[train_col].head()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



CT=ColumnTransformer([('onehotencoding', OneHotEncoder(), ['category']),

                     ('dropout','drop',['question_title', 'question_body', 'question_user_page', 'answer', 'answer_user_page', 'url', 'host'])])



CT.fit(train[train_col])

X_train_category=CT.transform(train[train_col])

X_test_category=CT.transform(test[train_col])





cv=CountVectorizer(min_df=5)



X_train_1=cv.fit_transform(train['question_title'])

X_test_1=cv.transform(test['question_title'])



X_train_2=cv.fit_transform(train['question_body'])

X_test_2=cv.transform(test['question_body'])



X_train_3=cv.fit_transform(train['question_user_page'])

X_test_3=cv.transform(test['question_user_page'])



X_train_4=cv.fit_transform(train['answer'])

X_test_4=cv.transform(test['answer'])



X_train_5=cv.fit_transform(train['answer_user_page'])

X_test_5=cv.transform(test['answer_user_page'])



X_train_6=cv.fit_transform(train['url'])

X_test_6=cv.transform(test['url'])



X_train_7=cv.fit_transform(train['host'])

X_test_7=cv.transform(test['host'])
from scipy import sparse



X_train=sparse.hstack((X_train_1,

               X_train_2,

               X_train_3,

               X_train_4,

               X_train_5,

               X_train_6,

               X_train_7,

               X_train_category))



X_test=sparse.hstack((X_test_1,

               X_test_2,

               X_test_3,

               X_test_4,

               X_test_5,

               X_test_6,

               X_test_7,

               X_test_category))
target_col
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor



forest=RandomForestRegressor()

forest.fit(X_train, train[target_col])
predict=forest.predict(X_test)



v=np.shape(predict)[0]

h=np.shape(predict)[1]



for i in range(v):

    for j in range(h):

        if predict[i][j]>=1:predict[i][j]=0.9999

        elif predict[i][j]<=0:predict[i][j]=0.0001



sub=pd.read_csv('../input/google-quest-challenge/sample_submission.csv')

sub[target_col]=predict
sub.to_csv('submission.csv', index=False)