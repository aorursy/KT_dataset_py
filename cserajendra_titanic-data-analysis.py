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
# loading the data

import pandas as pd

import seaborn as sns

df = pd.read_csv('/kaggle/input/titanic/test_data.csv')

df.head()
# describe 

df.describe()
# it will check count of the survived data.... and for ML model this the dependent lable for the we will guess imbalansed or unbalansed

sns.countplot(x='Survived',data=df,saturation=0.90)
# count of the Gender M, F

sns.countplot(x = 'Sex',data = df)
# relation of the all data set

sns.boxplot(data=df)
# Calculate correlations, how lables or correlated to target lable...

corr = df.corr()

 

# Heatmap

sns.heatmap(corr)
# distribution among Age..

sns.distplot(df.Age)
# distribution among Fare lable

sns.distplot(df.Fare)


# Density Plot between sex and age

sns.kdeplot(df.Sex, df.Age)
# it will show the two defferent lable importance..

sns.jointplot(x='Fare', y='Age', data=df)
# count of family size lable..

sns.countplot(x = 'Family_size',data = df)
# count of titile

sns.countplot(x = 'Title_1',data = df)
# count of Pclass1

sns.countplot(x = 'Pclass_1',data = df)
sns.countplot(x = 'Pclass_2',data =df)
sns.countplot(x = 'Pclass_3',data =df)
# count of Emb_3

sns.countplot(x = 'Emb_3',data =df)
sns.countplot(x = 'Emb_2',data = df)
sns.countplot(x = 'Emb_3',data = df)
# lables split

y = df['Survived']

x = df.drop(['Survived'], axis = 1)
# Titanic data is ready for predictions so i want apply some ML Algos on this....

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
# apply for the logistic regreesion on it....

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(x_train,y_train)

print(lr.predict(x_test).mean())

print(lr.score(x,y))
# svm techniq

from sklearn.svm import SVC

svc = SVC().fit(x_train,y_train)

print(svc.predict(x_test).mean())

print(svc.score(x,y))
# DecisionTree Tecchnic

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier().fit(x_train,y_train)

print(dtc.predict(x_test).mean())

print(dtc.score(x,y))
# RandomForest techniq..

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier().fit(x_train,y_train)

print(rfc.predict(x_test).mean())

print(rfc.score(x,y))