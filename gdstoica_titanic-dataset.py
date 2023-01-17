# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')



train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 1

train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 2

train_data = train_data.drop('Name', 1)

# train_data = train_data.drop('Ticket', 1)

train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 1

train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 2

train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 3

train_data["Embarked"].fillna(train_data["Embarked"].mean(), inplace = True) 

train_data = train_data.astype({"Embarked":float})

print(train_data.info())
train_data.describe()
sns.countplot(x='Survived', data=train_data);
#men survival histogram

hist = train_data[train_data['Sex']==1].hist(column='Survived', bins=2)

#female survival histogram

hist = train_data[train_data['Sex']==2].hist(column='Survived', bins=2)
#male survivors distribution per age

#drop nan values

train_data_with_age = train_data.dropna(subset=['Age'])

male_survivors= train_data_with_age[(train_data_with_age['Sex']==1) & (train_data_with_age['Survived']==1)]

print(male_survivors.head())

male_survivors_age = male_survivors['Age']

hist = male_survivors_age.hist(bins=10)
#males not-survivors distribution per age

male_not_survivors = train_data_with_age[(train_data_with_age['Sex']==1) & (train_data_with_age['Survived']==0)]

hist = male_not_survivors['Age'].hist(bins=10)
#male survivors distribution per seat class

male_survivors_pclass = male_survivors['Pclass']

hist = male_survivors_pclass.hist(bins=3)
#males non-survivors distribution per seat class

hist = male_not_survivors['Pclass'].hist(bins=3)

print(male_not_survivors.describe())
female_survivors= train_data_with_age[(train_data_with_age['Sex']==2) & (train_data_with_age['Survived']==1)]

hist = female_survivors['Pclass'].hist(bins=3)
female_non_survivors= train_data_with_age[(train_data_with_age['Sex']==2) & (train_data_with_age['Survived']==0)]

hist = female_non_survivors['Pclass'].hist(bins=3)
# fill NaN age data with mean

avg_age = train_data['Age'].mean()

train_data["Age"].fillna(avg_age, inplace = True) 

train_data.drop('Ticket', 1, inplace=True)

train_data.drop('Cabin', 1, inplace=True)

print(train_data.head())

corr = train_data.corr()

sns.heatmap(corr)
# train_data.drop('PassengerId', 1, inplace=True)

# train_data.drop('Embarked', 1, inplace=True)

# train_data.drop('Age', 1, inplace=True)
#Create a new feature for Parch and SipSp 

train_data['relatives'] = train_data['Parch'] + train_data['SibSp']

train_data.loc[train_data['relatives']>0, 'Accompanied'] = 1

train_data.loc[train_data['relatives']==0, 'Accompanied'] = 0

train_data.drop('SibSp',1, inplace=True)

train_data.drop('Parch',1, inplace=True)

train_data.drop('relatives',1, inplace=True)

print(train_data.head())
# train_data.drop('Fare', 1, inplace=True)
import numpy as np

from sklearn.preprocessing import StandardScaler



X = train_data.drop('Survived', axis=1)

y = np.array(train_data['Survived'])



scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)



X = np.array(scaled_X)



from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=42)



for train_index, test_index in sss.split(X,y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix



clf = RandomForestClassifier(n_estimators=130)

clf.fit(X_train, y_train)

y_pred3 = clf.predict(X_test)



print(confusion_matrix(y_test, y_pred3))

print(classification_report(y_test, y_pred3))
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import classification_report, confusion_matrix

# from sklearn.model_selection import GridSearchCV



# parameters = {'n_estimators':[130, 145, 200], 'max_depth':[1, 2, 4, 8, 16, 32], 'max_features':[3, 4], 'criterion':['gini', 'entropy'], 'min_samples_split':[2, 3] }

# model = RandomForestClassifier()

# clf  = GridSearchCV(model, parameters, cv=None)

# clf.fit(X_train, y_train)

# print(clf.best_estimator_)
y_pred3 = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred3))

print(classification_report(y_test, y_pred3))



mat = confusion_matrix(y_test, y_pred3)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')

plt.xlabel('true label')

plt.ylabel('predicted label');
print(clf.get_params)
#Gradient Boosting

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingClassifier



parameters = {'learning_rate':[0.01, 0.015, 0.02, 0.03], 'n_estimators':[90, 100, 120, 130, 140, 150], 'random_state':[42]}

model = GradientBoostingClassifier()

clf  = GridSearchCV(model, parameters, cv=None)

clf.fit(X_train, y_train)

print(clf.best_estimator_)

y_pred5 = clf.predict(X_test)

print(classification_report(y_test, y_pred5))

#Gradient Boosting with early stopping



from sklearn.metrics import mean_squared_error



gbrc = GradientBoostingClassifier(learning_rate=0.02, n_estimators=120, random_state=42)

gbrc.fit(X_train, y_train)

errors = [mean_squared_error(y_test, y_pred)

          for y_pred in gbrc.staged_predict(X_test)]

bst_n_estimators = np.argmin(errors) + 1

print(bst_n_estimators)

gbrc_best = GradientBoostingClassifier(learning_rate=0.02,n_estimators=bst_n_estimators)

gbrc_best.fit(X_train, y_train)

y_pred6 = gbrc_best.predict(X_test)

print(classification_report(y_test, y_pred6))
import xgboost



xgb_clf = xgboost.XGBClassifier(random_state=42)

xgb_clf.fit(X_train, y_train)

y_pred5 = xgb_clf.predict(X_test)

print(classification_report(y_test, y_pred5))