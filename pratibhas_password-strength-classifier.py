#import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#load data

#if there are any bad lines in csv data, drop them.

df=pd.read_csv('../input/password-strength-classifier-dataset/data.csv',',',error_bad_lines=False,engine='python')
df.head()
df.shape
#if null values present, drop them.

df=df.dropna(axis=0)
df.shape
df.head(10)
#Shuffle data

from sklearn.utils import shuffle

df1=shuffle(df)
df1.head()
#reset index

df1=df1.reset_index(drop=True)
x=df1['password']

y=df1['strength']
sns.countplot(y,data=df1)
df1.groupby(['strength']).count()/len(df1)
#Let us make a list of characters of password

def word(password):

    character=[]

    for i in password:

        character.append(i)

    return character
#convert password into vectors

from sklearn.feature_extraction.text import TfidfVectorizer

vector=TfidfVectorizer(tokenizer=word)

x_vec=vector.fit_transform(x)
#dictionary

vector.vocabulary_
#getting  tf-idf vector for first password



feature_names=vector.get_feature_names()

first_password=x_vec[0]

vec=pd.DataFrame(first_password.T.todense(),index=feature_names,columns=['tfidf'])

vec.sort_values(by=['tfidf'],ascending=False)
#split the data into train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_vec,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
classifier=[]

classifier.append(LogisticRegression(multi_class='ovr'))

classifier.append(LogisticRegression(multi_class='multinomial',solver='newton-cg'))

classifier.append(xgb.XGBClassifier())

classifier.append(MultinomialNB())
#result

result=[]

for model in classifier:

    a=model.fit(x_train,y_train)

    result.append(a.score(x_test,y_test))
result1=pd.DataFrame({'score':result,

                      'algorithms':['logistic_regr_ovr',

                                    'logistic_regr_mutinomial',

                                    'xgboost','naive bayes']})
a=sns.barplot('score','algorithms',data=result1)

a.set_label('accuracy')

a.set_title('cross-val-score')
#prediction

x_pred=np.array(['123abc'])

x_pred=vector.transform(x_pred)

model=xgb.XGBClassifier()

model.fit(x_train,y_train)

y_pred=model.predict(x_pred)

y_pred