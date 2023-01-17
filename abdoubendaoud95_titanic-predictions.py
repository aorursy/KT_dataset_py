import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')
#read the data

data = pd.DataFrame(pd.read_csv('../input/train.csv'))
data.shape
data.head() 
data.describe()
data.info()
#see the numerical and categorical data , note that we only feed numerical data  to the algorithm

numerical_data   = data.select_dtypes(include = [np.number]).columns
categorical_data = data.select_dtypes(include= [np.object]).columns
numerical_data
categorical_data
#great way to see all missing data at once

missing_values = data.isnull().sum().sort_values(ascending = False)

percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])


data.Age.tail()
#filling the missing points in age and fare columns 

data['Age'] = data['Age'].interpolate()
data['Fare'] = data['Fare'].interpolate()
data.Age.isnull().sum() 
data.Fare.isnull().sum()
#done !!
#encoding the categorical_data is a common way to do things 

data['Sex'] = data['Sex'].map({'male': 1, 'female': 2})
#this are the columns that i consider as factors for survival

factors = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']

plt.figure(figsize=(10,8))

sns.heatmap((data[factors].corr().filter(items = ['Survived'])),annot=True, robust=True)

data = data.drop(columns = ['Cabin','Ticket','Embarked','Name'] , axis = 1)
data.shape
outliers = data[factors]

plt.figure(figsize=(10,8))

sns.boxplot(data = outliers)
plt.show()

plt.scatter(data['Survived'],data['Fare'],edgecolors = "r")


from scipy import stats

data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
#spliting the data
x = data.values
x = np.delete(x ,1 , axis=1)

y = data['Survived'].values
#creating train,test  data 

from sklearn.model_selection import train_test_split

x_train, x_test , y_train , y_test = train_test_split( x, y , test_size = 0.3, random_state = 0 )
#for decisionTree 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(min_samples_split= 200)
clf.fit(x_train,y_train)

prd = clf.predict(x_test)

acc = accuracy_score(y_test , prd)
print('accuracy is % : ', (acc*100))

#for RandomForest

from sklearn import ensemble

clf1 = ensemble.RandomForestClassifier(n_estimators=250)
clf1.fit (x_train, y_train)

print('accuracy is % : ', (clf1.score (x_test, y_test))*100)
clf2 = ensemble.GradientBoostingClassifier(n_estimators=200)

clf2.fit(x_train,y_train)

clf2.score(x_test,y_test)

print('accuracy is % : ', (clf2.score (x_test, y_test))*100)