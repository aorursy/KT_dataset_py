# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train.describe()
test.describe()
print('The number of rows in train is - ' , train.shape[0])

print('The number of columns in train is - ' , train.shape[1])

print('The number of rows in test is - ' , test.shape[0])

print('The number of columns in test is - ' , test.shape[1])



print(train.columns.values) #checking the names of the collumns to determine which ones can be dropped

train.head()
train.info()



train.isnull().sum().sort_values(ascending = True) #to find missing values
test.isnull().sum().sort_values(ascending = True)
train.hist(bins = 20 , figsize= (12,16)) 
#finding corelation between  and survival rate

graph = sns.FacetGrid(train, col='Survived')

graph.map(plt.hist, 'Sex', bins=20)

#finding corelation between passenger gender and survival rate

g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
graph = sns.FacetGrid(train, col='Survived')

graph.map(plt.hist, 'Pclass', label = 'white', bins=20)
graph = sns.FacetGrid(train, col='Survived')

graph.map(plt.hist, 'Fare', label = 'white', bins=20)
#to fill missing values with the mean of age column

#train['Age'] = train['Age'].fillna(train['Age'].mean())
#dropping features

#dropping age because filling in missing values with mean returned poor score



train.drop('Name',axis=1,inplace=True)

train.drop('Cabin',axis=1,inplace=True)

train.drop('Ticket',axis=1,inplace=True)

train.drop('Age',axis=1,inplace=True)





test.drop('Name',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)

test.drop('Ticket',axis=1,inplace=True)

test.drop('Age',axis=1,inplace=True)
print(train.head())

print(test.head())
#separating features and target variable

features = ['Pclass','Sex','SibSp','Parch','Embarked']

target = 'Survived'
print(train[features].head())

print(test[features].head())
#to get integer values for string variables

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])

#checking to see if null values are gone

X_test.isnull().sum().sort_values(ascending = True)
#importing models

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

model1 = DecisionTreeClassifier()

model2 = RandomForestClassifier()

model3 = KNeighborsClassifier()

           
model1.fit(X,train[target])
predictions = model1.predict(X_test)

predictions
model2.fit(X,train[target])
predictions2 = model2.predict(X_test)

predictions2
model3.fit(X,train[target])
predictions3 = model3.predict(X_test)

predictions3
from sklearn.model_selection import cross_val_predict

DTC_predictions = cross_val_predict(DecisionTreeClassifier(),X , train[target], cv=5 )
DTC_predictions
from sklearn import metrics

from sklearn.metrics import mean_absolute_error

DTC_MAE_accuracy = metrics.mean_absolute_error(train[target], DTC_predictions)

print("Cross-Predicted MAE Accuracy for DTC:",DTC_MAE_accuracy)
RFC_predictions = cross_val_predict(RandomForestClassifier(),X , train[target], cv=5 )
RFC_predictions
RFC_MAE_accuracy = metrics.mean_absolute_error(train[target], RFC_predictions)

print("Cross-Predicted MAE Accuracy for RFC:",RFC_MAE_accuracy)
KNC_predictions = cross_val_predict(KNeighborsClassifier(),X , train[target], cv=5 )
KNC_predictions
KNC_MAE_accuracy = metrics.mean_absolute_error(train[target], KNC_predictions)

print("Cross-Predicted MAE Accuracy for KNC:",KNC_MAE_accuracy)
#grid search on random forest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth' : [4, 5, 6], 'n_estimators': [50, 100]}



clf = RandomForestClassifier()

cv = StratifiedKFold(n_splits=5)



grid_s = GridSearchCV(clf, scoring='accuracy', param_grid=param_grid, cv=cv)



model = grid_s



model.fit(X, train[target])

prediction = model.predict(X_test)
prediction
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':prediction})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)