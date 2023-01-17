# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')



dt_train = train_data.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1)

dt_train.head()
print('Number of Nan value in the dataset = '+ str(dt_train.isnull().sum().sum()))

print('Number of Nan value in the Age feature = '+ str(dt_train['Age'].isnull().sum().sum()))

print('Number of Nan value in the embarked feature = '+ str(dt_train['Embarked'].isnull().sum().sum()))

dt_train.describe()

dt_train.nunique()
# Replace by the median age of each class group

# class 1 the mean age is 37 year old

# class 2 the mean age is 29 year old

# class 3 the mean age is 24 year old



dt_train.groupby('Pclass')['Age'].transform(

    lambda grp: print(np.round(np.nanmedian(grp)))

)



dt_train['Age']= dt_train.groupby('Pclass')['Age'].transform(

    lambda grp: grp.fillna(np.round(np.nanmedian(grp)))

)

print('Number of Nan value in the Age feature = '+ str(dt_train['Age'].isnull().sum().sum()))



dt_train.head()
dt_train = pd.get_dummies(dt_train, drop_first= True)

dt_train.head()
corr = abs(dt_train.corr())

#corr.style.background_gradient(cmap='coolwarm')

sns.heatmap(corr, annot=True) 
sns.pairplot(dt_train, hue = 'Survived', vars = ['Pclass', 'Age', 'SibSp', 'Sex_male'])


sns.countplot(dt_train['Survived'], label = "Count") 
from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

cols = dt_train.columns

np_scaled = min_max_scaler.fit_transform(dt_train)

dt_train = pd.DataFrame(np_scaled, columns = cols)

dt_train.describe()
from sklearn.model_selection import train_test_split

y = dt_train['Survived']

X = dt_train.drop(['Survived'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

X_train.describe()
from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier



# The "accuracy" scoring is proportional to the number of correct classifications

model = RandomForestClassifier() 

rfecv = RFECV(estimator = model, step = 1, cv = 5, scoring = 'accuracy')

rfecv = rfecv.fit(X_train, y_train)



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X_train.columns[rfecv.support_])
X_train = X_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_S']]

X_test = X_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_S']]

#from sklearn.svm import SVC 

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix



rfc = RandomForestClassifier(oob_score = True) 

param_grid = {'n_estimators':[5,10,20,100, 250],

             'max_depth':[2,4,8,16,32,None]} 

grid = GridSearchCV(estimator=rfc, param_grid = param_grid, cv=5 )

grid.fit(X_train,y_train)

grid.best_estimator_
grid_predictions = grid.predict(X_test)

cm = confusion_matrix(y_test, grid_predictions)

sns.heatmap(cm, annot=True)

print(classification_report(y_test,grid_predictions))
X_train = dt_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_S']]

y_train = dt_train['Survived']



data_test = pd.read_csv('/kaggle/input/titanic/test.csv')

X_test = data_test.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1)



X_test['Age']= X_test.groupby('Pclass')['Age'].transform(

    lambda grp: grp.fillna(np.round(np.nanmedian(grp)))

)



X_test['Fare']= X_test.groupby('Pclass')['Fare'].transform(

    lambda grp: grp.fillna(np.round(np.nanmedian(grp)))

)



X_test = pd.get_dummies(X_test, drop_first= True)



cols = X_test.columns

np_scaled = min_max_scaler.fit_transform(X_test)

X_test = pd.DataFrame(np_scaled, columns = cols)

X_test = X_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_S']]

print('Number of Nan value in the dataset = '+ str(X_test.isnull().sum().sum()))

model = grid.best_estimator_# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                      # max_depth=4, max_features='auto', max_leaf_nodes=None,

                      # min_impurity_decrease=0.0, min_impurity_split=None,

                      # min_samples_leaf=1, min_samples_split=2,

                      # min_weight_fraction_leaf=0.0, n_estimators=250,

                      # n_jobs=None, oob_score=True, random_state=None,

                      # verbose=0, warm_start=False)



model.fit(X_train, y_train)

predictions = model.predict(X_test)

predictions= [ 1 if y>=0.5 else 0 for y in predictions]



output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})

print(output)

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")