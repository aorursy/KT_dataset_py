import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# tools
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn import model_selection

import xgboost as xgb

# ensemble of classifiers
from mlxtend.classifier import StackingClassifier

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.head()
data = pd.concat([train, test])
train_size = train.shape[0] # 891
test_size = test.shape[0] # 418
data[train_size-3:train_size+3]
data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(data['Title'], data['Pclass'])
age_ref = data.groupby('Title').Age.mean()
data = data.assign(
    Age = data.apply(lambda r: r.Age if pd.notnull(r.Age) else age_ref[r.Title] , axis=1)
)
del age_ref
a = sns.FacetGrid(data[data.Sex=='male'], hue = 'Survived', aspect=6)
a.map(sns.kdeplot, 'Age', shade=True)
a.set(xlim=(0 , data['Age'].max()))
a.add_legend()

a = sns.FacetGrid(data[data.Sex=='female'], hue = 'Survived', aspect=6)
a.map(sns.kdeplot, 'Age', shade=True)
a.set(xlim=(0 , data['Age'].max()))
a.add_legend()
data['AgeBand'] = pd.cut(data['Age'], 5, labels=range(5)).astype(int)
data[['AgeBand', 'Survived']].groupby(['AgeBand']).agg(['count','mean'])
data['Title'] = data['Title'].replace(['Don', 'Capt', 'Col', 'Major', 'Sir', 'Jonkheer', 'Rev', 'Dr'], 'Honored')
data['Title'] = data['Title'].replace(['Lady', 'Dona', 'Mme', 'Countess'], 'Mrs')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data[['Title', 'Survived']].groupby(['Title']).agg(['count','mean'])
data['Fare'] = data['Fare'].fillna(13.30)
a = sns.FacetGrid(data, hue = 'Survived', aspect=6)
a.map(sns.kdeplot, 'Fare', shade=True)
a.set(xlim=(0 , 200))#data['Fare'].max())) # data['Fare'].max()
a.add_legend()
data['FareBand'] = 0
data.loc[(data.Fare > 0) & (data.Fare <= 7.5), 'FareBand'] = 1
data.loc[(data.Fare > 7.5) & (data.Fare <= 12.5), 'FareBand'] = 2
data.loc[(data.Fare > 12.5) & (data.Fare <= 17), 'FareBand'] = 3
data.loc[(data.Fare > 17) & (data.Fare <= 29), 'FareBand'] = 4
data.loc[data.Fare > 29, 'FareBand'] = 5
data[['FareBand', 'Survived']].groupby(['FareBand']).agg(['count','mean'])
data['Embarked'] = data['Embarked'].fillna('S')
data[['Embarked', 'Survived']].groupby(['Embarked']).agg(['count','mean'])
#data['Deck'] = data['Cabin'].str.slice(0,1).fillna('T') # hack: T = undefined + one passanger
data['DeckCode'] = (data['Cabin']
                        .str.slice(0,1)
                        .map({
                            'C':1, 
                            'E':2, 
                            'G':3,
                            'D':4, 
                            'A':5, 
                            'B':6, 
                            'F':7, 
                            #'T':8 #to rare
                        })
                        .fillna(0)
                        .astype(int))
data[['DeckCode', 'Survived']].groupby(['DeckCode']).agg(['count','mean'])
data['Room'] = (data['Cabin']
                    .str.slice(1,5).str.extract('([0-9]+)', expand=False)
                    .fillna(0)
                    .astype(int))

data['RoomBand'] = 0
data.loc[(data.Room > 0) & (data.Room <= 20), 'RoomBand'] = 1
data.loc[(data.Room > 20) & (data.Room <= 40), 'RoomBand'] = 2
data.loc[(data.Room > 40) & (data.Room <= 80), 'RoomBand'] = 3
data.loc[data.Room > 80, 'RoomBand'] = 4

data[['RoomBand', 'Survived']].groupby(['RoomBand']).agg(['count','mean'])
data.loc[data.Ticket=='LINE', 'Ticket'] = 'LINE1'
data['Odd'] = (data['Ticket']
                   .str.slice(-1) # last symbol
                   .astype(int)
                   .map(lambda x: x % 2 == 0)
                   .astype(int)
              )
data[['Odd', 'Survived']].groupby(['Odd']).agg(['count','mean'])
data['FamilySize'] = (data['SibSp'] + data['Parch']).astype(int)
data[['FamilySize', 'Survived']].groupby(['FamilySize']).agg(['count','mean'])
data[['Sex', 'Survived']].groupby(['Sex']).agg(['count','mean'])
data['FamilySizeBand'] = 0
data.loc[(data.FamilySize == 1), 'FamilySizeBand'] = 1
data.loc[(data.FamilySize == 2), 'FamilySizeBand'] = 2
data.loc[(data.FamilySize == 3), 'FamilySizeBand'] = 2
data.loc[data.FamilySize > 3, 'FamilySizeBand'] = 2
data[['FamilySizeBand', 'Survived']].groupby(['FamilySizeBand']).agg(['count','mean'])
data['IsAlone'] = (data['SibSp'] + data['Parch'] == 0).astype(int)
data[['IsAlone', 'Survived']].groupby(['IsAlone']).agg(['count','mean'])
data['SexCode'] = LabelEncoder().fit_transform(data['Sex'])
data['TitleCode'] = LabelEncoder().fit_transform(data['Title'])
data['EmbarkedCode'] = LabelEncoder().fit_transform(data['Embarked'])
features = data[:train_size][[
    'Survived',
    'Pclass',
    'SexCode',
    'TitleCode',
    'FamilySize',
    'FamilySizeBand',
    'SibSp',
    'Parch',
    'IsAlone',
    'Age',
    'AgeBand',
    'Fare',
    'FareBand',    
    'EmbarkedCode',
    'DeckCode',
    'Room',
    'RoomBand',
    'Odd'
]]
plt.figure(figsize=(20,18))
sns.heatmap(features.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=plt.cm.RdBu, annot=True)
cols = [
    'Pclass',
    'Sex',
    'FamilySize',
    #'FamilySizeBand',
    'SibSp',
    'Parch',
    'IsAlone',
    #'Age',
    'AgeBand',
    'Fare',
    #'FareBand',
    'Title',
    #'Embarked',
    'DeckCode',
    #'Room',
    'RoomBand',
    #'Odd',
]
X_train = data[:train_size][cols]
Y_train = data[:train_size]['Survived'].astype(int)
X_test = data[train_size:][cols]

print(X_train.shape, Y_train.shape, X_test.shape)
X_train.head()
one_hot_features = [
    #'Pclass',
    'Sex',    
    #'FamilySizeBand',
    'AgeBand',
    #'FareBand',
    'Title',
    #'Embarked',
    'DeckCode',
    'RoomBand',
    #'Odd'
]
X_train = pd.get_dummies(X_train, columns = one_hot_features)
X_test = pd.get_dummies(X_test, columns = one_hot_features)

print(X_train.shape, Y_train.shape, X_test.shape)
# sclr = StandardScaler()
# X_train = sclr.fit_transform(X_train)
# X_test = sclr.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)
svc = SVC(C=9)
svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
print(svc.score(X_train, Y_train))

scores = model_selection.cross_val_score(svc, X_train, Y_train, cv=5, scoring='accuracy')
print(scores)
print("Kfold on SVC: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
print(knn.score(X_train, Y_train))

scores = model_selection.cross_val_score(knn, X_train, Y_train, cv=5, scoring='accuracy')
print(scores)
print("Kfold on KNeighborsClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
gaussian.score(X_train, Y_train)
perceptron = Perceptron(max_iter=6)
perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
perceptron.score(X_train, Y_train)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
decision_tree.score(X_train, Y_train)
sgd = SGDClassifier(max_iter=3)
sgd.fit(X_train, Y_train)
# Y_pred = sgd.predict(X_test)
sgd.score(X_train, Y_train)
linear_svc = LinearSVC(C=0.07)
linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
print(linear_svc.score(X_train, Y_train))

scores = model_selection.cross_val_score(linear_svc, X_train, Y_train, cv=5, scoring='accuracy')
print(scores)
print("Kfold on LinearSVC: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
rf_params = {
    'n_estimators': 700,
    'max_depth': 5,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'max_features':'auto',
    'oob_score':True,
}

random_forest = RandomForestClassifier(**rf_params)
random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))

scores = model_selection.cross_val_score(random_forest, X_train, Y_train, cv=5, scoring='accuracy')
print(scores)
print("Kfold on RandomForestClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
'''
parameters = {
    'n_estimators':[280,320],
    'max_depth':[4,5,6,7,8,9,10,11,12],
    'gamma':[1,2,3],
    #'max_delta_step':[0,1,2],
    'min_child_weight':[1,2,3], 
    #'colsample_bytree':[0.55,0.65],
    'learning_rate':[0.1,0.2,0.3],
    'subsample':[1,0.9,0.8],
    'base_score':[0.5]
}

grid = model_selection.GridSearchCV(xgb.XGBClassifier(), parameters, cv=5)
grid.fit(X_train, Y_train)
print(grid.best_score_)

xg_boost = grid.best_estimator_
xg_boost
'''
xg_boost = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.65, gamma=2, learning_rate=0.3, max_delta_step=1,
       max_depth=4, min_child_weight=2, missing=None, n_estimators=280,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
xg_boost.fit(X_train, Y_train)
Y_pred = xg_boost.predict(X_test)
print(xg_boost.score(X_train, Y_train))

scores = model_selection.cross_val_score(xg_boost, X_train, Y_train, cv=5, scoring='accuracy')
print(scores)
print("Kfold on XGBClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
#xgb.to_graphviz(xg_boost)
'''
lr = LogisticRegression()
sclf = StackingClassifier(
    classifiers=[clf1.best_estimator_, clf2.best_estimator_, clf3.best_estimator_],
    #classifiers=[logreg, svc, knn, gaussian, perceptron, decision_tree, sgd, linear_svc, random_forest],
    meta_classifier=lr,
    #use_probas=True,
    #average_probas=False
)

scores = model_selection.cross_val_score(sclf, X_train, Y_train, cv=5, scoring='accuracy')
print("Kfold on StackingClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

sclf.fit(X_train, Y_train)
'''
feature_importance = list(zip(X_train.columns.values, xg_boost.feature_importances_))
feature_importance.sort(key=lambda x:x[1])
feature_importance
submission = pd.DataFrame({
    "PassengerId": data[train_size:]["PassengerId"], 
    "Survived": Y_pred 
})
submission.to_csv('submission.csv', index=False)