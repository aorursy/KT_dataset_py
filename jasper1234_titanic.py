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
import pandas as pd

import numpy as np

import sklearn

from sklearn import linear_model

from sklearn import neighbors

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

import seaborn as sns

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_complete = train_data.fillna(train_data.mean())



test_complete = test_data.fillna(test_data.mean())
le = preprocessing.LabelEncoder()

train_complete["Sex"] = le.fit_transform(train_complete["Sex"])

le = preprocessing.LabelEncoder()

test_complete["Sex"] = le.fit_transform(test_complete["Sex"])
train_complete = train_complete.replace('S', value = 0)

train_complete = train_complete.replace('C', value = 1)

train_complete = train_complete.replace('Q', value = 2)

train_complete['Embarked'] = train_complete['Embarked'].fillna(3)



test_complete = test_complete.replace('S', value = 0)

test_complete = test_complete.replace('C', value = 1)

test_complete = test_complete.replace('Q', value = 2)

test_complete['Embarked'] = test_complete['Embarked'].fillna(3)
train_features = ['Pclass', 'Sex', 'SibSp', 'Fare', 'Age','Parch','Embarked',]

X0 = train_complete[train_features]

X1 = X0.iloc[:594]

X2 = X0.iloc[594:]



XTest = test_complete[train_features]
train_survived = ['Survived']

y0 = train_complete[train_survived]

y1 = y0.iloc[:594]

y2 = y0.iloc[594:]

corr = X0.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(X0.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(X0.columns)

ax.set_yticklabels(X0.columns)

plt.show()
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X0,y0)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X0.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
Titanic_model = RandomForestClassifier(n_estimators=1200, max_depth=10, random_state= 1)

Titanic_model.fit(X0, y0.values.ravel())





Test_pred = (Titanic_model.predict(XTest))
print(Test_pred)
#mean_squared_error(y2, Test_pred)
#mean_absolute_error(y2, Test_pred)
#from pprint import pprint

#print('Parameters currently in use:\n')

#pprint(Titanic_model.get_params())
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
#rf_random = RandomizedSearchCV(estimator = Titanic_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

#rf_random.fit(X1, y1)
#rf_random.best_params_
#pred_round = [round(num) for num in Test_pred]

#pred_int = [int(num) for num in pred_round]

#print (pred_int)
#mean_absolute_error(y2, pred_round)
#from sklearn.metrics import classification_report, confusion_matrix

#print(confusion_matrix(y2, pred_round))

#print(classification_report(y2, pred_round))
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Test_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")