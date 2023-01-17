# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv')
df.groupby('Category').size()
sns.countplot(df['Category'],label="Count")

plt.show()
df.isna().sum()
X=df.drop(['Category'],axis=1)  #Seperating the label as y and the rest of attributes in X

y=df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
df_test=pd.read_csv('../input/test.csv')

Predictions=logreg.predict(df_test)

df_test['Category']=Predictions

Final_Dataframe=df_test[['Id','Category']]

Final_Dataframe.to_csv('Submission_1.csv',index=False)