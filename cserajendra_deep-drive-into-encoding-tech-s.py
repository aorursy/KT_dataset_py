# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load data ets in to note book

df_test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

df_train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

df_sub = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

print('test data set',df_test.shape)

print('train data set',df_train.shape)

df_sub.shape
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

df_train.head()
# deep look into bin lables...



bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

# loop to get column and the count of plots

for n, col in enumerate(df_train[bin_cols]): 

    plt.figure(n)

    sns.countplot(x=col, data=df_train, hue='target', palette='husl')
# let's converting the bin_3 and bin_4 into 0,1 

df_train['bin_3'] = df_train['bin_3'].replace(to_replace=['F', 'T'], value=['0', '1']).astype(int)

df_train['bin_4'] = df_train['bin_4'].replace(to_replace=['Y', 'N'], value=['1', '0']).astype(int)

# test data set

df_test['bin_3'] = df_test['bin_3'].replace(to_replace=['F', 'T'], value=['0', '1']).astype(int)

df_test['bin_4'] = df_test['bin_4'].replace(to_replace=['Y', 'N'], value=['1', '0']).astype(int)
# checking the data frame

df_train.head(2)
#Drop ID and seperate target variable

target = df_train['target']

train_id = df_train['id']

test_id = df_test['id']

df_train.drop(['target', 'id'], axis=1, inplace=True)

df_test.drop('id', axis=1, inplace=True)



print(df_train.shape)

print(df_test.shape)
# let's look at Nominal feartures..

nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

from sklearn import model_selection, preprocessing, metrics

le = preprocessing.LabelEncoder()

traintest = pd.concat([df_train, df_test])

for i in nom_cols:

    print("The number of unique values in {} column is : {}".format(i, df_train[i].nunique()) )

for col in nom_cols:

    traintest[col] = le.fit_transform(traintest[col])



train_le = traintest.iloc[:df_train.shape[0], :]

test_le = traintest.iloc[df_train.shape[0]:, :]



print(train_le.shape)

print(test_le.shape)
train_le.head()
# nominal encoding with onehotencoder...

from sklearn.preprocessing import OneHotEncoder

OHE=OneHotEncoder()

train_ohe1 = OHE.fit_transform(df_train)

test_ohe1 = OHE.fit_transform(df_test)



print(train_ohe1.shape)

print(train_ohe1.dtype)

print(test_ohe1.shape)

print(test_ohe1.dtype)
# ordinal feature encoding technics...

ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



for i in ord_cols:

    print("The number of unique values in {} column is : {}".format(i, df_train[i].nunique()) )

    print("The unique values in {} column is : \n {}".format(i, df_train[i].value_counts()[:5]))

    print('\n')
mapper_ord_1 = {'Novice': 1, 'Contributor': 2, 'Expert': 3, 'Master': 4, 'Grandmaster': 5}



mapper_ord_2 = {'Freezing': 1, 'Cold': 2, 'Warm': 3, 'Hot': 4,'Boiling Hot': 5, 'Lava Hot': 6}



mapper_ord_3 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 

                'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15}



mapper_ord_4 = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 

                'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,

                'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 

                'W': 23, 'X': 24, 'Y': 25, 'Z': 26}



for col, mapper in zip(['ord_1', 'ord_2', 'ord_3', 'ord_4'], [mapper_ord_1, mapper_ord_2, mapper_ord_3, mapper_ord_4]):

    df_train[col+'_oe'] = df_train[col].replace(mapper)

    df_test[col+'_oe'] = df_test[col].replace(mapper)

    df_train.drop(col, axis=1, inplace=True)

    df_test.drop(col, axis=1, inplace=True)
# ord_5, we have high cardinality

from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories='auto')

encoder.fit(df_train.ord_5.values.reshape(-1, 1))

df_train.ord_5 = encoder.transform(df_train.ord_5.values.reshape(-1, 1))

df_test.ord_5 = encoder.transform(df_test.ord_5.values.reshape(-1, 1))
df_train.ord_5[:5]
df_train[['ord_1_oe','ord_2_oe','ord_3_oe','ord_4_oe','ord_5','ord_0']].info()
def logistic(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=32,test_size=0.2)

    lr=LogisticRegression()

    lr.fit(X_train,y_train)

    y_pre=lr.predict(X_test)

    print('Accuracy : ',accuracy_score(y_test,y_pre))
logistic(train_ohe1,target)


x_train_ohe,x_test_ohe,y_train_ohe,y_test_ohe=train_test_split(train_ohe1,target,random_state=42,test_size=0.2)
lr=LogisticRegression()

lr.fit(x_train,y_train)

y_pre=lr.predict(x)

print('Accuracy : ',accuracy_score(y_test,y_pre))
df_sub['target'] = y_pre

df_sub.to_csv('lgb_model.csv', index=False)

len(df_sub)
df_sub.shape
df_sub['target'].shape