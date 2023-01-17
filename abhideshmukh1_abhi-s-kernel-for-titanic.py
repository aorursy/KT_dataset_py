# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dftrain = pd.read_csv("../input/train.csv")
dftrain.head()
dftrain.info()
"""
Cleaning the dataset
"""

train = dftrain[["PassengerId", "Survived","Pclass","Sex","Age","SibSp","Parch"]]
train.notnull().all()
train[["PassengerId", "Pclass","Sex", "Age"]].replace(0, np.nan)
train.dropna(subset=train.columns, inplace=True)
train.drop_duplicates(subset='PassengerId', keep='first')
print('\ninfo\n')
train.info()
print('\nhead\n')
train.head()
"""EDA"""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set(color_codes=True)
plt.figure(figsize=(15,20))
plt.subplot(2,3,1)
sns.distplot(train.Survived)

plt.subplot(2,3,2)
sns.distplot(train.Pclass)

plt.subplot(2,3,3)
sns.distplot(train.Age)

plt.subplot(2,3,4)
sns.distplot(train.SibSp)

plt.subplot(2,3,5)
sns.distplot(train.Parch)

sns.boxplot(data = train[['Survived', 'Pclass', 'SibSp', 'Parch']])
train.boxplot('Age','Sex')
sns.countplot(x='Survived', data=train)
sns.countplot(x='Sex', data=train)
sns.pairplot(train)

sns.lmplot(x = 'SibSp', y = 'Age', hue='Sex', data=train )
sns.diverging_palette(200,0,as_cmap=True)
sns.heatmap(train.corr(), annot=True, fmt='.2f')
"""Getting the test data and imputing"""

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
dftest = pd.read_csv("../input/test.csv")

test = dftest[["Pclass","Age","SibSp","Parch"]] #Remove sex since it is a string. Needs encoding here
test.notnull().all()
#test.drop_duplicates(subset='PassengerId', keep='first')

test_imputed = my_imputer.fit_transform(test)
#print('\nimputed\n')
#print(test_imputed)
#print('\ntest\n')
#print(test)

test_X = test_imputed

#Splitting the data
from sklearn.model_selection import train_test_split

train_X = train[['Pclass', 'Age', 'SibSp', 'Parch']] #Remove sex since it is a string. Needs encoding here
train_y = train['Survived']

#The model - RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

rfg = RandomForestClassifier(random_state=1)
rfg.fit(train_X, train_y)
pred_y = rfg.predict(test_X)
print(pred_y)
print(len(pred_y))
"""for i in range(len(pred_y)):
    if pred_y[i] < 0.5:
        pred_y[i] = 0
    else: 
        pred_y[i] = 1
Survived = pred_y
print(Survived)"""
result = pd.DataFrame({'PassengerId':dftest['PassengerId'], 'Survived': pred_y})

print(result)
filename = 'Titanic Prediction Abhi.csv'
result.to_csv(filename, index=False)
print('Saved file: ' + filename)
