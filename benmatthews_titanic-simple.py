%matplotlib inline

import pandas as pd 

from matplotlib.pyplot import figure, plot, xlabel,ylabel, title, show, bar

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV, StratifiedKFold

import seaborn as sns

import lightgbm as lgb

sns.set()
root = '../'

train_data = pd.read_csv(root + 'input/train.csv')

train_data.head()
plt.figure()

missing_data = train_data.isna().sum().divide(train_data.count()) * 100

missing_data.plot(kind='bar')

plt.figure()

sns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.figure()

sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.figure()

sns.countplot(x='Embarked', hue='Survived', data=train_data)
plt.figure()

sns.countplot(x='SibSp', hue='Survived', data=train_data)
plt.figure()

sns.countplot(x='Parch', hue='Survived', data=train_data)
# constant representation of string categories

MALE = 0

FEMALE = 1
# Gender categorical data -> must normally be converted to numeric values

train_data = train_data.replace('male', MALE) .replace('female', FEMALE)



# Embarking -> categorical data

train_data['Embarked'] = train_data['Embarked'].replace('S', 1).replace('C', 2).replace('Q', 3).fillna(0)



# 0.8

train_data_features = train_data[['Sex','Pclass', 'Embarked']]



# 0.7

#train_data_features = train_data[['Sex']]



# 0.6

#train_data_features = train_data[['Embarked']]



train_data_labels =   train_data[['Survived']].get_values().ravel()
x_train, x_test, y_train, y_test = train_test_split(

    train_data_features,

    train_data_labels,

    test_size=0.3,

    random_state=0

)
clf = svm.SVC(gamma='scale')

clf.fit(x_train, y_train)
cross_val_score(clf, x_test, y_test, cv=3)


params = {

    'learning_rate': 0.01, 

    'num_leaves': 6,

    'num_boost_round': 50,

    'reg_lambda': 1,

    'verbosity': -1,

    'objective': 'binary',

}







clf = lgb.LGBMClassifier(**params).fit(x_train, y_train)

cross_val_score(clf, x_test, y_test, cv=3)


