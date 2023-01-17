# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataframe = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv('/kaggle/input/titanic/test.csv')



dataframe.info()

print(dataframe.columns.values)
sns.countplot(dataframe['Survived'], hue = dataframe['Sex'])

Dead, lives = dataframe.Survived.value_counts()

male, female = dataframe.Sex.value_counts()

print("Percentage of Male on ship:", round(male/(male+female)*100) )

print("Percentage of Female on ship:", round(female/(male+female)*100 ))
from matplotlib import pyplot as plt

plt.figure(figsize=(40,8))

sns.countplot(dataframe['Age'], hue = dataframe['Survived'])
#find out how many classes on ship

dataframe.Pclass.unique()

#so there are 3 classes on ship

dataframe.Pclass.value_counts()

#in which 1st, 2nd and 3rd class has 216, 184 and 491 respectively 

#The graph shows the most people died in 3rd class which is obvious from the

#number of people who bought 3rd class tickets are high

sns.countplot(dataframe['Pclass'], hue = dataframe['Survived'])

t_p = dataframe.groupby('Pclass')['Survived']

print(t_p.sum())
#The Embarked class does not give much info other than S class embarkment has ppl from all different classes

sns.countplot(dataframe['Embarked'], hue = dataframe['Survived'])
sns.distplot(dataframe['Fare'])

dataframe['Fare'].describe()
#find the null values in different columns first 

#Find the null values if any in our DataFrame

dataframe.isnull().values.any()

dataframe.isnull().sum()



#for test dataset

test.isnull().sum()
dataframe['Age'].fillna(round(dataframe['Age'].mean()), inplace = True)

test['Fare'].fillna(test['Fare'].mean(), inplace = True)

#doing the same for Test

test['Age'].fillna(round(test['Age'].mean()), inplace = True)

dataframe.isnull().sum()

test.isnull().sum()
dataframe['Age'].head(10)
import seaborn as sns

correlations = dataframe[dataframe.columns].corr(method='pearson')

sns.heatmap(correlations, cmap="YlGnBu", annot = True)
import heapq



print('Absolute overall correlations')

print('-' * 30)

correlations_abs_sum = correlations[correlations.columns].abs().sum()

print(correlations_abs_sum, '\n')



print('Weakest correlations')

print('-' * 30)

print(correlations_abs_sum.nsmallest(4))
train_set = dataframe.drop( ['Name','Cabin', 'Ticket','PassengerId', ], axis = 1)

test_set = test.drop( ['Name','Cabin', 'Ticket', 'PassengerId', ], axis = 1)

test_set.isnull().sum()
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace = True)

#train_set = train_set.dropna()

#test_set = test_set.dropna()
train_set.head()
test_set.head()
y = train_set.iloc[:, 0].values

X = train_set.iloc[:, train_set.columns != 'Survived'].values

print(X[0])
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

print(X[2])
#for test

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')

test_set = np.array(ct.fit_transform(test_set))

print(test_set[1])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:, 4 ] = le.fit_transform(X[:,4])



print(X[1])

#for test set

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

test_set[:, 4] = le.fit_transform(test_set[:,4])

print(test_set[2])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20 )
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test =sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

#from sklearn.linear_model import Ridge



rfc=RandomForestClassifier()

parameters= {'n_estimators':[ 100,200,300,400, 600],

             'max_depth':[3,4,6,7],

             'criterion':['entropy','gini']

    }



rfc=GridSearchCV(rfc, param_grid=parameters, cv = 5)

rfc.fit(X_train,y_train)

print("The best value of leanring rate is: ",rfc.best_params_, )
#RandomForest



from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(criterion= 'gini', n_estimators = 100 ,max_depth = 6, random_state = 0)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

Random_forest_acc= accuracy_score(y_test, y_pred)

print('acc = ', Random_forest_acc )

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

LogisticReg_acc= accuracy_score(y_test, y_pred)

print('acc = ', LogisticReg_acc )
from sklearn.svm import SVC

model = SVC(kernel = 'rbf', random_state = 0)

model.fit(X, y)

y_pred = model.predict(X_test)





from sklearn.metrics import accuracy_score, confusion_matrix

SVC_acc = accuracy_score(y_test, y_pred)

print('acc = ', SVC_acc )
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X, y)

y_pred = model.predict(X_test)





from sklearn.metrics import accuracy_score, confusion_matrix

Gaussian_acc = accuracy_score(y_test, y_pred)

print('acc = ', Gaussian_acc )
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)





from sklearn.metrics import accuracy_score, confusion_matrix

DT_acc = accuracy_score(y_test, y_pred)

print('acc = ', DT_acc )

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

#from sklearn.linear_model import Ridge



gbc=GradientBoostingClassifier()

parameters= {'n_estimators':[ 50,100,200,300, ],

             'max_depth':[3,4,6,7]

    }



gbreg=GridSearchCV(gbc, param_grid=parameters, cv = 5 )

gbreg.fit(X_train,y_train)

print("The best value of leanring rate is: ",gbreg.best_params_, )
from sklearn.ensemble import GradientBoostingClassifier

model_gb = GradientBoostingClassifier(n_estimators = 100, max_depth =4, random_state = 42)

model_gb.fit(X_train, y_train)

y_pred = model_gb.predict(X_test)





from sklearn.metrics import accuracy_score, confusion_matrix

GB_acc = accuracy_score(y_test, y_pred)

print('acc = ', GB_acc )
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)





y_pred = classifier.predict(X_test)

xgb_acc = accuracy_score(y_pred, y_test)

print('acc=',xgb_acc)
print('RF_acc=', Random_forest_acc)

print('Logistic_acc=', LogisticReg_acc)

print('SVC_acc=', SVC_acc)

print('Gaussian_acc=', Gaussian_acc)

print('DecisionTree_acc=', DT_acc)

print('GradBoost_acc=', GB_acc)

print('XGBoost_acc=', xgb_acc)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(criterion= 'gini', n_estimators = 100 ,max_depth = 6, random_state = 0)

rf_model.fit(X, y)





final_pred = rf_model.predict(test_set)



final_pred
survivors = pd.DataFrame(final_pred, columns = ['Survived'])

len(survivors)

survivors.insert(0, 'PassengerId', test['PassengerId'], True)

survivors
survivors.to_csv('Submission.csv', index = False)