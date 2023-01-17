import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler,Normalizer

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.model_selection import cross_val_score

from catboost import CatBoostClassifier

from sklearn.ensemble import AdaBoostClassifier,VotingClassifier,GradientBoostingClassifier,RandomForestClassifier
#Loading the data

train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')

datasets = [train,test]
train.head()
#check for missing values

train.isnull().sum()
train.Ticket
train.Cabin
#fill in missing data according to sex and Pclass of the training data

for set in datasets:

    set.Age = train.groupby(['Sex', 'Pclass']).Age.transform(lambda x: x.fillna(x.mean()))

    set.Fare = train.groupby(['Sex', 'Pclass']).Fare.transform(lambda x: x.fillna(x.mean()))

    #set.Embarked = train.groupby(['Sex', 'Pclass']).Embarked.transform(lambda x: x.fillna(x.mode()[0]))
#Create new Features    

for set in datasets:

    set["FSize"] = set.SibSp+set.Parch + 1 #family size

    set['Title'] = set['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    set['IsAlone'] = 1

    set.IsAlone.loc[set['FSize'] > 1 ] = 0

    title_names = (set['Title'].value_counts() < 10) #reduce the amount of titles

    set.Title = set.Title.apply(lambda x:'Rare' if title_names.loc[x] == True else x)
target = train.Survived

ids = test.PassengerId

train = train.drop(['Cabin','Survived','Name','PassengerId','Ticket','Embarked'],axis=1)

test = test.drop(['Cabin','Name','PassengerId','Ticket','Embarked'],axis=1)

train.head()
train = pd.get_dummies(train)

test = pd.get_dummies(test)
#to check if something else catches my eyes

corr = train.corr()

fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(corr, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(30, 300, n=300),square=True,ax=ax)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.2)
model_ab = AdaBoostClassifier(algorithm='SAMME', base_estimator=None)

model_ab.fit(X_train,y_train)

print(model_ab.score(X_train,y_train))

print(model_ab.score(X_test,y_test))
model_rfc = RandomForestClassifier(bootstrap=True,criterion='entropy',

                                   max_depth=20, max_features='auto', max_leaf_nodes=None,

                                   min_impurity_decrease=0.0, min_impurity_split=None,

                                   min_samples_leaf=7, min_samples_split=10,

                                   min_weight_fraction_leaf=0.0, n_estimators=200,

                                   n_jobs=None, oob_score=False, random_state=None,

                                   verbose=0, warm_start=False)

model_rfc.fit(X_train,y_train)

print(model_rfc.score(X_train,y_train))

print(model_rfc.score(X_test,y_test))
model_cat=CatBoostClassifier(depth=3, iterations= 250, l2_leaf_reg=5, learning_rate=0.03)

model_cat.fit(X_train,y_train)

print(model_cat.score(X_train,y_train))

print(model_cat.score(X_test,y_test))
vc_model = VotingClassifier(estimators=[('model_ab', model_ab), ('model_rfc', model_rfc),("model_cat",model_cat)], voting='soft')
vc_model.fit(X_train,y_train)

print(vc_model.score(X_train,y_train))

print(vc_model.score(X_test,y_test))
vc_model.fit(train,target)

print(vc_model.score(train,target))
test['Survived'] = vc_model.predict(test)
test["PassengerId"] = ids
submit = test[['PassengerId','Survived']]

submit.to_csv("submit.csv", index=False)