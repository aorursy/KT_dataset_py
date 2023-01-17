# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
basic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

basic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
print(basic_train.shape)

basic_train.head()
print(basic_test.shape)

basic_test.head()
basic_train.info()
basic_train.isnull().sum()
basic_test.info()
basic_test.isnull().sum()
basic_train.Sex.value_counts()/len(basic_train)
sns.barplot(x='Sex', y='Survived',data=basic_train)

plt.show()
basic_train.Embarked.value_counts()/len(basic_train)
sns.lineplot(x='Embarked', y='Survived', data=basic_train)

plt.show()
sns.FacetGrid(basic_train, hue='Survived', height=8).map(sns.distplot,'Age').set_axis_labels('Age','Survived').add_legend()

plt.show()
sns.barplot(x='Pclass',y='Survived' ,data=basic_train)

plt.show()
# Change NaN by Mode in Embarked column

basic_train.Embarked = basic_train.Embarked.fillna(basic_train.Embarked.mode()[0])

basic_test.Embarked = basic_test.Embarked.fillna(basic_test.Embarked.mode()[0])
# Only test data has missing values for fare

pclass = basic_test.loc[basic_test.Fare.isnull(), 'Pclass'].values[0]

median_fare = basic_test.loc[basic_test.Pclass== pclass, 'Fare'].median()

basic_test.loc[basic_test.Fare.isnull(), 'Fare'] = median_fare
# Extracting first letter of the cabin string and mapping it in both data sets

basic_train.Cabin = basic_train.Cabin.str[0]

basic_test.Cabin = basic_test.Cabin.str[0]



cabin_map = {'C':3, 'E':5, 'G':7, 'D':4, 'A':1, 'B':2, 'F':6, 'T':8}

basic_train.replace({'Cabin': cabin_map}, inplace=True)

basic_test.replace({'Cabin': cabin_map}, inplace=True)



# Since the cabin has a very high percentage of missing values, we directly replace NaN with 0

basic_train.Cabin = basic_train.Cabin.fillna(0)

basic_test.Cabin = basic_test.Cabin.fillna(0)
basic_train['Family Size'] = basic_train.SibSp + basic_train.Parch + 1

basic_test['Family Size'] = basic_test.SibSp + basic_test.Parch + 1
basic_train['title'] = basic_train.Name.str.split(',', expand=True)[1].str.split('.',expand=True)[0]

basic_test['title'] = basic_test.Name.str.split(',', expand=True)[1].str.split('.',expand=True)[0]
# Mapping the Title values.

mapping = {' Mr': 1, ' Mrs': 2, ' Dona':2, ' Miss':3, ' Master':4, ' Don':1, ' Rev':5, ' Dr':6, ' Mme':3,

           ' Ms':3, ' Major':1, ' Lady':2, ' Sir':1, ' Mlle':3, ' Col':1, ' Capt':1,

           ' the Countess':2, ' Jonkheer':1}

basic_train.replace({'title': mapping}, inplace=True)

basic_test.replace({'title': mapping}, inplace=True)
for title, age in basic_train.groupby('title')['Age'].median().iteritems():

    basic_train.loc[(basic_train['title']==title) & (basic_train['Age'].isnull()), 'Age'] = age

for title, age in basic_test.groupby('title')['Age'].median().iteritems():

    basic_test.loc[(basic_test['title']==title) & (basic_test['Age'].isnull()), 'Age'] = age
basic_train['FpP'] = basic_train['Fare']/(basic_train['Family Size'])

basic_test['FpP'] = basic_test['Fare']/(basic_test['Family Size'])
# Mapping the gender column

sex_map = {'male': 1, 'female':0}

basic_train.replace({'Sex': sex_map}, inplace=True)

basic_test.replace({'Sex': sex_map}, inplace=True)
# Mapping the Embarked (Port) column

port_map = {'S':1, 'C':2, 'Q':3}

basic_train.replace({'Embarked': port_map}, inplace=True)

basic_test.replace({'Embarked': port_map}, inplace=True)
# Survived is our target column. Hence separating it from our data frame.

target = basic_train['Survived']
basic_train
#dropping non-useful (non-features) columns from the dataframe 

train_df=basic_train.drop(['SibSp', 'Parch', 'Name', 'Ticket','Survived','Fare'], axis = 1)

test_df=basic_test.drop(['SibSp', 'Parch', 'Name', 'Ticket','Fare'], axis = 1)
train_df
plt.subplots(figsize=(15,10))

sns.heatmap(train_df.corr(),annot=True,cmap='Greens')

plt.show()
X = train_df[['PassengerId', 'Pclass','Sex', 'Age','Cabin', 'title','Family Size', 'FpP', 'Embarked']]

y = target

X_test = test_df[['PassengerId', 'Pclass','Sex', 'Age','Cabin', 'title','Family Size','FpP', 'Embarked']]
sc = StandardScaler()

X = sc.fit_transform(X)

X_test = sc.transform(X_test)
# random forest model creation

rfc = RandomForestClassifier()

# rfc = RandomForestClassifier(criterion = 'entropy', max_depth = 8, n_estimators = 500, random_state = 42)

rfc.fit(X,y)

# predictions

rfc_predict = rfc.predict(X_test)
# rfc_predict
# gradient booster

gbr = GradientBoostingClassifier(max_depth=4, max_features='auto', n_estimators=100)

gbr.fit(X,y)

# prediction

gbr_predict=gbr.predict(X_test)
dt = DecisionTreeClassifier(max_depth=4, max_features='auto')

dt.fit(X,y)

# prediction

dt_predict = dt.predict(X_test)
lr = LogisticRegression()

lr.fit(X,y)

# prediction

lr_predict=lr.predict(X_test)
param_grid = {

#     'n_estimators': [50, 200, 500, 1000],

    'max_features': ['auto'],

    'max_depth': [4,5,6, 7, 8]

}
CV = GridSearchCV(estimator = dt, param_grid = param_grid, cv = 5)

CV.fit(X, y)

CV.best_estimator_


rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')

gbr_cv_score = cross_val_score(gbr, X, y, cv=10, scoring='roc_auc')

dt_cv_score = cross_val_score(dt, X, y, cv=10, scoring='roc_auc')

lr_cv_score = cross_val_score(lr, X, y, cv=10, scoring='roc_auc')
print("=== Mean AUC Score ===")

print("Mean AUC Score - Gradient Booster: ", gbr_cv_score.mean())

print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

print("Mean AUC Score - Decision Tree: ", dt_cv_score.mean())

print("Mean AUC Score - Logistic Regression: ", lr_cv_score.mean())
ids = basic_test['PassengerId']

submit = pd.DataFrame()

submit['PassengerId'] = ids

submit['Survived'] = rfc_predict
submit
# submit.Survived.value_counts()
submit.to_csv('submission17.csv',index=False)