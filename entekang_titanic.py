import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.metrics import accuracy_score
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
trdata = train_data.copy()

tesdata = test_data.copy()

dsets = [trdata,tesdata]

print(trdata.head())

print(trdata.info())

print(tesdata.info())
trdata.describe(include='all')
trdata.groupby('Sex').Survived.mean()
trdata.groupby('Pclass')['Survived'].mean()
trdata.groupby('Embarked')['Survived'].mean()
sns.countplot(x='Survived', data=trdata, hue='Embarked')
print(train_data.groupby('SibSp')['Survived'].mean())

print(trdata.groupby('Parch').Survived.mean())
sns.catplot(x='Survived', hue='Sex', data=trdata, kind='count', col='Pclass')
sns.catplot(x='Survived', y='Fare', data=train_data)
g=sns.FacetGrid(trdata, col='Survived')

g.map(plt.hist, 'Age')
# fill in the missing values in age and fare with the median value. 

# the median is used as there are a few outliers for these features.

# there is an 80 year old on board, along with a passenger who paid $512.



age_med=trdata['Age'].median()

fare_med=trdata['Fare'].median()

tesdata['Fare'].fillna(fare_med, inplace=True)



# impute median

for dset in dsets:

    dset['Age'].fillna(age_med, inplace=True)

    

# create bins for age

age_bins = [-np.inf, 20, 40, 60, np.inf]

fare_bins = [-np.inf, 128.082, 256.165, 384.247, np.inf]

labs = [0, 1, 2, 3]







for dset in dsets:

    dset['Agebin'] = pd.cut(dset['Age'], bins=age_bins, labels=labs)

    dset['Farebin'] = pd.cut(dset['Fare'], bins=fare_bins, labels=labs)

    dset['Agebin'].astype('int64')

    dset['Farebin'].astype('int64')

    

    

print(trdata['Agebin'].unique())

print(trdata['Farebin'].unique())



# sns.countplot(x='Survived', data=trdata, hue='Agebin')

# plt.figure()

# sns.countplot(x='Survived', data=trdata, hue='Farebin')
trdata.groupby('Agebin')['Survived'].mean()
trdata.groupby('Farebin')['Survived'].sum()
trdata.groupby('Farebin')['Survived'].mean()
# create a family feature



for dset in dsets:

    dset['Family'] = dset['SibSp'] + dset['Parch'] + 1



trdata.groupby('Family')['Survived'].mean()
# create an 'Alone' feature 

# Alone = 1 if the passenger is onboard by him/herself

for dset in dsets:

    dset['Alone']=1

    dset.loc[dset['Family'] > 1, 'Alone'] = 0

    

trdata.groupby('Alone')['Survived'].mean()
mode_emb = trdata['Embarked'].mode()[0]

trdata['Embarked'].fillna(mode_emb, inplace=True)
emb_dummy_tr = pd.get_dummies(trdata['Embarked'], drop_first=True)

sex_dummy_tr = pd.get_dummies(trdata['Sex'], drop_first=True)

emb_dummy_te = pd.get_dummies(tesdata['Embarked'], drop_first=True)

sex_dummy_te = pd.get_dummies(tesdata['Sex'], drop_first=True)



trdata_enc = pd.concat([trdata, emb_dummy_tr, sex_dummy_tr], axis=1)

tesdata_enc = pd.concat([tesdata, emb_dummy_te, sex_dummy_te], axis=1)



trdata_enc.head()
drop_col = ['PassengerId', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Family']

dset_enc = [trdata_enc, tesdata_enc]

for dset in dset_enc:

    dset.drop(drop_col, axis=1, inplace=True)

    

trdata_enc.head()
tesdata_enc.head()
y_train = trdata_enc.Survived

X_train = trdata_enc.drop('Survived', axis=1)

X_test = tesdata_enc
rs = 99

lr = LogisticRegression(random_state=rs)

knn = KNN()

dt=DecisionTreeClassifier(random_state=rs)

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbors', knn), ('Decision Tree', dt)]



for clfname, clf in classifiers:

    # fit the model

    clf.fit(X_train, y_train)

    

    # predict

    pred = clf.predict(X_test)

    

    # score of the model 

    print('The score of {} is {:.4f}'.format(clfname, clf.score(X_train, y_train)))
vc=VotingClassifier(estimators=classifiers)

vc.fit(X_train, y_train)

vc_pred = vc.predict(X_test)

print('The score of the Voting classifier is: {:.4f}'.format(vc.score(X_train, y_train)))

rf=RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print('The score of the Random Forest classifier is: {:.4f}'.format(rf.score(X_train, y_train)))

gb=GradientBoostingClassifier()

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

print('The score of the Gradient Boosting classifier is: {:.4f}'.format(gb.score(X_train, y_train)))
output = pd.DataFrame({'PassengerId': tesdata.PassengerId, 'Survived': rf_pred})

output.to_csv('my_submission.csv', index=False)

print("Submission was successfully saved!")