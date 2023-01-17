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

%matplotlib inline

import numpy as np

import pandas as pd

import re as re



train = pd.read_csv('../input/titanic/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/titanic/test.csv' , header = 0, dtype={'Age': np.float64})

full_data = [train, test]



# Any results you write to the current directory are saved as output.
print(train.head())
print (test.info())
train['Survived'].unique()
train['Sex'].replace('female',0, inplace=True)

train['Sex'].replace('male', 1,inplace=True)

train['Embarked'].replace('S',0, inplace=True)

train['Embarked'].replace('Q', 1,inplace=True)

train['Embarked'].replace('C', 2,inplace=True)



test['Sex'].replace('female',0, inplace=True)

test['Sex'].replace('male', 1,inplace=True)

test['Embarked'].replace('S',0, inplace=True)

test['Embarked'].replace('Q', 1,inplace=True)

test['Embarked'].replace('C', 2,inplace=True)
test['Sex'].value_counts()
test=test.drop(['Name' , 'Ticket' ,'Cabin'], axis=1)

test.describe()

train=train.drop(['Name' , 'Ticket' ,'Cabin'], axis=1)

train.describe()
train.describe().T
test.describe().T
print(test['Age'].unique())
test['Age'] = test['Age'].replace(np.nan, np.random.randint(test['Age'].min(),test['Age'].max()))

test['Fare'] = test['Fare'].replace(np.nan, np.random.randint(test['Fare'].min(),test['Fare'].max()))
test.describe().T
print(test['Embarked'].isnull().values.any())

print(test['Age'].isnull().values.any())



print(train['Embarked'].isnull().values.any())

print(train['Age'].isnull().values.any())
#Xtest = test[np.isfinite(test['Age'])]

#Xtest = Xtest[np.isfinite(Xtest['Embarked'])]

#Xtest = Xtest[np.isfinite(Xtest['Fare'])]



Xtrain = train[np.isfinite(train['Age'])]

Xtrain = Xtrain[np.isfinite(Xtrain['Embarked'])]



Xtrain.describe().T
print(test['Embarked'].isnull().values.any())

print(test['Age'].isnull().values.any())

print(test['Fare'].isnull().values.any())

print(test['Embarked'].isnull().values.any())

print(test['Age'].isnull().values.any())
X_train = Xtrain.drop(['PassengerId'], axis=1).values

X_test  = test.drop(['PassengerId'], axis=1).values

print(X_test.shape)

print(X_train.shape)
X = X_train[0::, 1::]

y = X_train[0::, 0]
from sklearn import preprocessing



#saved_cols = df2.columns

#x = df2.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_train_scaled = min_max_scaler.fit_transform(X_train)

x_test_scaled = min_max_scaler.fit_transform(X_test)



#df = pd.DataFrame(x_scaled)

#df.columns = saved_cols 
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = x_train_scaled[0::, 1::]

y = x_train_scaled[0::, 0].astype(int)



print(y[1])

acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train_t, X_test_t = X[train_index], X[test_index]

	y_train_t, y_test_t = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train_t, y_train_t)

		train_predictions = clf.predict(X_test_t)

		acc = accuracy_score(y_test_t, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
candidate_classifier = GradientBoostingClassifier()

candidate_classifier.fit(x_train_scaled[0::, 1::], x_train_scaled[0::, 0].astype(int))

result = candidate_classifier.predict(x_test_scaled)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = test['PassengerId']

submission_df['Survived'] = candidate_classifier.predict(x_test_scaled)

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)