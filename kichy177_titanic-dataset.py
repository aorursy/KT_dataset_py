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
#ML imports

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('/kaggle/input/train.csv')
test_df = pd.read_csv('/kaggle/input/test.csv')
df.head()
df.dtypes
df.info()
#Fresh start

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.subplot2grid((5,5),(0,0), colspan = 2, rowspan = 2)

df['Survived'].value_counts(normalize=True).plot(kind='bar', alpha = 1)

plt.title('% Survived')



plt.subplot2grid((5,5), (0,3), colspan = 2, rowspan = 2)

df.Survived[df['Sex'] == 'male'].value_counts(normalize = True).plot(kind='bar', alpha = 1)

plt.title('% Men Survived')



plt.subplot2grid((5,5), (3,0), colspan = 2, rowspan = 2)

df.Survived[df['Sex']=='female'].value_counts(normalize=True).plot(kind='bar', alpha = 1, color ='#FA0193')

plt.title('% Women Survived')



plt.subplot2grid((5,5), (3,3), colspan = 2, rowspan = 2)

df.Sex[df['Survived'] == 1].value_counts(normalize= True).plot(kind='bar', alpha = 1, color = ['#FA0193', '#000000'])

plt.title('% Gender Survived')
#Data cleaning. clearing out the categorical cols and replacing with integers

df1 = df[['Survived','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']]

x_test = test_df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']]
df1['Sex'] = df1['Sex'].map({'male':0,'female':1})

x_test['Sex'] = x_test['Sex'].map({'male':0,'female':1})
df1['Embarked'] = df1['Embarked'].map({'S':1,'C':2,'Q':3})

x_test['Embarked'] = x_test['Embarked'].map({'S':1,'C':2,'Q':3})
df1.drop('Survived', axis = 1)
df1.fillna({'Age':df1['Age'].dropna().median(), 'Embarked':1}, inplace=True)

x_test.fillna({'Age':x_test['Age'].dropna().median(), 'Embarked':1}, inplace=True)
df1.info()
x_test.info()
x_test.Fare.fillna(x_test['Fare'].dropna().median(), inplace=True)
features = df1[['Sex', 'Age', 'Parch', 'SibSp', 'Fare', 'Embarked', 'Pclass']]

test_features = x_test[['Sex', 'Age', 'Parch', 'SibSp', 'Fare', 'Embarked', 'Pclass']]
target = df1['Survived']
#correlation heatmap

corr = df1.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





fig = plt.figure(figsize=(12, 8))

sns.heatmap(corr, mask=mask, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#correlation heatmap easier version

corr = df1.corr()

sns.heatmap(corr)
#selecting columns that have <= 0.9 correlation

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_columns = df1.columns[columns]
#SVC Algo

from sklearn.svm import SVC
svc = SVC()
classifier = svc.fit(features, target)
classifier.score(features, target)
#for polynomial preprocessing

from sklearn import preprocessing
poly = preprocessing.PolynomialFeatures(degree=2)

poly_features = poly.fit_transform(features)

poly_test_features = poly.fit_transform(test_features)
classifier = svc.fit(poly_features, target)
classifier.score(poly_features, target)
#To check for overfitting, model is split and run cv times and mean accuracy is checked

from sklearn import model_selection
scores = model_selection.cross_val_score(svc, poly_features, target, scoring = 'accuracy', cv=50)
scores.mean()
#Random Forest Classifier Algo

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf1 = rf.fit(poly_features, target)
rf1.score(poly_features, target)
#The score might be overfitting

#Find the mean of scores from 50 samples

#without polynomial features

scores = model_selection.cross_val_score(rf, features, target, scoring = 'accuracy', cv=50)

scores

scores.mean()
#With polynomial features

scores = model_selection.cross_val_score(rf, poly_features, target, scoring = 'accuracy', cv=50)

print(scores)

scores.mean()
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
lr1 = log_reg.fit(poly_features, target)
lr1.score(poly_features, target)
scores = model_selection.cross_val_score(log_reg, poly_features, target, scoring = 'accuracy', cv=50)

print(scores)

scores.mean()
y_pred = lr1.predict(poly_test_features)
submission = pd.DataFrame({

    "PassengerId" : test_df["PassengerId"],

    "Survived" : y_pred

})
submission.shape
submission.to_csv('gender_submission.csv', index=False)
os.getcwd()