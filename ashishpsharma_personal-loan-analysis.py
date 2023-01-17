# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sklearn as sklear

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_excel('/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')

df.head()
df.count()
df.describe().transpose()
df.isnull().any()
df.drop_duplicates().count()
df['Age'].unique()
sns.pairplot(df)
df[df['Experience']<0]
dfExp = df.loc[df['Experience'] >0]

mylist=df.loc[df['Experience']<0]['ID'].tolist()
for id in mylist:

    age=df.loc[np.where(df['ID']==id)]['Age'].tolist()[0]

    education=df.loc[np.where(df['ID']==id)]['Education'].tolist()[0]

    df_filtered=dfExp[(dfExp.Age==age) & (dfExp.Education==education)]

    exp=df_filtered['Experience'].median()

    df.loc[df.loc[np.where(df['ID']==id)].index,'Experience']=exp
df.loc[df['Experience'] <0]
df.describe().transpose()
sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=df)
sns.barplot(x='Education',y='Income',hue='Personal Loan',data=df)
from sklearn.model_selection import train_test_split

train_set, test_set=train_test_split(df.drop(['ID','Experience'], axis=1), test_size=0.3 , random_state=100)
train_labels = train_set.pop('Personal Loan')

test_labels = test_set.pop('Personal Loan')
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

d_model=DecisionTreeClassifier(criterion = 'gini',max_depth=3)

d_model.fit(train_set,train_labels)
d_model.score(train_set,train_labels)
y_predict=d_model.predict(test_set)

y_predict