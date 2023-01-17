import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style(style='darkgrid')
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.info()     # Missing values in Age, Cabin, Embarked
test.info()   # Missing values in Age, Cabin, Fare
# Function to find missing values
def miss(table):
    miss = table.isnull().sum()
    return miss[miss>0]
# Function to build stacked-bar chart
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
sns.countplot(train.Survived)
f,ax = plt.subplots(2,1,figsize=(20,15))
sns.countplot(train.Age,ax=ax[0])
plt.xticks(rotation=90)

sns.countplot(train.Age,data=train,hue='Survived',ax=ax[1])
plt.xticks(rotation=90)
bar_chart('Pclass')
bar_chart('Sex')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')

train_test_data = [train, test] # combining train and test dataset
# print(train_test_data)
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train
t1 = train.Title.unique()
t2 = test.Title.unique()
print(t1)
print(t2)
plt.subplots(2,1,figsize=(20,6))
plt.subplot(121)
t = train.Title.value_counts().reset_index()
sns.barplot(x='Title',y='index',data=t)

plt.subplot(122)
sns.countplot(y='Title',data=train,hue='Survived')
mp = {'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Don':4, 'Rev':4, 'Dr':4, 'Mme':4, 'Ms':4,
       'Major':4, 'Lady':4, 'Sir':4, 'Mlle':4, 'Col':4, 'Capt':4, 'Countess':4, 'Jonkheer':4,'Dona':4}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(mp)
train
mp = {'male':0,'female':1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(mp)
train
bar_chart('Sex')
# Filling missing values based on the Title of the name
train['Age'].fillna(train.groupby(['Title'])['Age'].transform('median'),inplace=True)
test['Age'].fillna(test.groupby(['Title'])['Age'].transform('median'),inplace=True)
train
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(0)

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
# Replacing cabin values by there first letter
# train_test_data = [train, test] # combining train and test dataset
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
train
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
train.columns
# Mapping Different cabins with a numerical value
mp = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(mp)
train
# Filling missing values based on the Pclass
train["Cabin"].fillna((train.groupby("Pclass")["Cabin"].transform("median")), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train
bar_chart('Embarked')
# Since most of them are from Southampton (S), we can replace it with S
train['Embarked'].fillna('S',inplace=True)
test['Embarked'].fillna('S',inplace=True)
mp = {'S':0,'C':1,'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(mp)
train
Train = train.copy(deep=True)
Test = test.copy(deep=True)
# Sex
# Male:0, Female:1
Train_Test = [Train,Test]
for dataset in Train_Test:
    dataset['Sex_Male'] = [1 if x == 0 else 0 for x in dataset.Sex]
    dataset['Sex_Female'] = [1 if x == 1 else 0 for x in dataset.Sex]
    dataset.drop('Sex',axis=1,inplace=True)
Train
# Embarked
# 'S':0,'C':1,'Q':2
Train_Test = [Train,Test]
for dataset in Train_Test:
    dataset['Embarked_S'] = [1 if x == 0 else 0 for x in dataset.Embarked]
    dataset['Embarked_C'] = [1 if x == 1 else 0 for x in dataset.Embarked]
    dataset['Embarked_Q'] = [1 if x == 2 else 0 for x in dataset.Embarked]
    dataset.drop('Embarked',axis=1,inplace=True)
Train
# Pclass
Train_Test = [Train,Test]
for dataset in Train_Test:
    dataset['Pclass_1'] = [1 if x == 1 else 0 for x in dataset.Pclass]
    dataset['Pclass_2'] = [1 if x == 2 else 0 for x in dataset.Pclass]
    dataset['Pclass_3'] = [1 if x == 3 else 0 for x in dataset.Pclass]
    dataset.drop('Pclass',axis=1,inplace=True)
Train
# Cabin
Train_Test = [Train,Test]
for dataset in Train_Test:
    dataset['Cabin_1'] = [1 if x == 1 else 0 for x in dataset.Cabin]
    dataset['Cabin_2'] = [1 if x == 2 else 0 for x in dataset.Cabin]
    dataset['Cabin_3'] = [1 if x == 3 else 0 for x in dataset.Cabin]
    dataset['Cabin_4'] = [1 if x == 4 else 0 for x in dataset.Cabin]
    dataset['Cabin_5'] = [1 if x == 5 else 0 for x in dataset.Cabin]
    dataset['Cabin_5.5'] = [1 if x == 5.5 else 0 for x in dataset.Cabin]
    dataset['Cabin_6'] = [1 if x == 6 else 0 for x in dataset.Cabin]
    dataset['Cabin_7'] = [1 if x == 7 else 0 for x in dataset.Cabin]
    dataset['Cabin_8'] = [1 if x == 8 else 0 for x in dataset.Cabin]
    dataset.drop('Cabin',axis=1,inplace=True)
Train
# Title
Train_Test = [Train,Test]
for dataset in Train_Test:
    dataset['Title_0'] = [1 if x == 0 else 0 for x in dataset.Title]
    dataset['Title_1'] = [1 if x == 1 else 0 for x in dataset.Title]
    dataset['Title_2'] = [1 if x == 2 else 0 for x in dataset.Title]
    dataset['Title_3'] = [1 if x == 3 else 0 for x in dataset.Title]
    dataset['Title_4'] = [1 if x == 4 else 0 for x in dataset.Title]
    dataset.drop('Title',axis=1,inplace=True)
Train
Train['Fare'].skew(),Test['Fare'].skew()
np.power(Train['Fare'],0.22)
np.power(Train['Fare'],0.22).skew(),np.power(Test['Fare'],0.22).skew()
for dataset in Train_Test:
    dataset['Fare'] = np.power(dataset['Fare'],0.22)
Train
Train.head().T
Test.head().T

predictor_set1 = ['Pclass','Sex', 'Age', 'Fare', 'Cabin', 'Embarked', 'Title', 'FamilySize']
predictor_set2 = ['Age', 'Fare', 'FamilySize', 'Sex_Male', 'Sex_Female', 'Embarked_S', 'Title_0', 'Title_1', 
                  'Title_2', 'Title_3', 'Title_4', 'Embarked_C', 'Embarked_Q', 'Pclass_1', 'Pclass_2', 'Pclass_3', 
                  'Cabin_1', 'Cabin_2', 'Cabin_3', 'Cabin_4', 'Cabin_5', 'Cabin_5.5', 'Cabin_6', 'Cabin_7', 'Cabin_8']
target = ['Survived']
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=123)

scoring = 'accuracy'
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(max_iter=1000)
score = cross_val_score(lg, train[predictor_set1].values, train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
score = cross_val_score(lg, Train[predictor_set2].values, Train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
score = cross_val_score(dt, train[predictor_set1].values, train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
score = cross_val_score(dt, Train[predictor_set2].values, Train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

score = cross_val_score(dt, train[predictor_set1].values, train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
score = cross_val_score(nb, Train[predictor_set2].values, Train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, max_depth=6)

score = cross_val_score(dt, train[predictor_set1].values, train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.model_selection import GridSearchCV
parameters = {'bootstrap':['True','False'],
              'max_depth': [3, 5, 7, 9],
              'n_estimators': [10, 50, 100, 300, 500, 1000]} #number of trees, change it to 1000 for better results


grid_search_rf_1 = GridSearchCV(rf, parameters, n_jobs=5, 
                   cv= KFold(n_splits=5, shuffle=True, random_state=123), 
                   scoring='accuracy', return_train_score=True,
                   verbose=2, refit=True)

grid_result = grid_search_rf_1.fit(train[predictor_set1].values, train[target].values.ravel())

# #trust your CV!
print("Best score: %f using \n%s" % (grid_result.best_score_, grid_result.best_params_))
score = cross_val_score(rf, Train[predictor_set2].values, Train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.model_selection import GridSearchCV
parameters = {'bootstrap':['True','False'],
              'max_depth': [3, 5, 7, 9],
              'n_estimators': [10, 50, 100, 300, 500, 1000]} #number of trees, change it to 1000 for better results


grid_search_rf_2 = GridSearchCV(rf, parameters, n_jobs=5, 
                   cv= KFold(n_splits=5, shuffle=True, random_state=123), 
                   scoring='accuracy', return_train_score=True,
                   verbose=2, refit=True)

grid_result = grid_search_rf_2.fit(Train[predictor_set2].values, Train[target].values.ravel())

# #trust your CV!
print("Best score: %f using \n%s" % (grid_result.best_score_, grid_result.best_params_))
import xgboost as xgb
xgb = xgb.XGBClassifier()
score = cross_val_score(xgb, train[predictor_set1].values, train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.model_selection import GridSearchCV
parameters = { 'objective':['binary:logistic'],
              'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.8], #so called `eta` value
              'max_depth': [3, 5, 7, 9],
              'min_child_weight': [1, 5, 11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10, 50, 100, 300, 500], #number of trees, change it to 1000 for better results
              'seed': [123]}


grid_search_xgb_1 = GridSearchCV(xgb, parameters, n_jobs=5, 
                   cv= KFold(n_splits=5, shuffle=True, random_state=123), 
                   scoring='accuracy', return_train_score=True,
                   verbose=2, refit=True)

grid_result = grid_search_xgb_1.fit(train[predictor_set1].values, train[target].values.ravel())

# #trust your CV!
print("Best score: %f using \n%s" % (grid_result.best_score_, grid_result.best_params_))
score = cross_val_score(xgb, Train[predictor_set2].values, Train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.model_selection import GridSearchCV
parameters = { 'objective':['binary:logistic'],
              'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.8], #so called `eta` value
              'max_depth': [3, 5, 7, 9],
              'min_child_weight': [1, 5, 11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10, 50, 100, 300, 500], #number of trees, change it to 1000 for better results
              'seed': [123]}


grid_search_xgb_2 = GridSearchCV(xgb, parameters, n_jobs=5, 
                   cv= KFold(n_splits=5, shuffle=True, random_state=123), 
                   scoring='accuracy', return_train_score=True,
                   verbose=2, refit=True)

grid_result = grid_search_xgb_2.fit(Train[predictor_set2].values, Train[target].values.ravel())

# #trust your CV!
print("Best: %f using \n%s" % (grid_result.best_score_, grid_result.best_params_))

import lightgbm as lgb
lgb = lgb.LGBMClassifier()

score = cross_val_score(lgb, train[predictor_set1].values, train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.model_selection import GridSearchCV
parameters = { 'objective':['binary'],
              'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.8], #so called `eta` value
              'max_depth': [3, 5, 7, 9],
              'min_child_weight': [1, 5, 11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10, 50, 100, 300, 500], #number of trees, change it to 1000 for better results
              'seed': [123]}


grid_search_lgb_1 = GridSearchCV(lgb, parameters, n_jobs=5, 
                   cv= KFold(n_splits=5, shuffle=True, random_state=123), 
                   scoring='accuracy', return_train_score=True,
                   verbose=2, refit=True)

grid_result = grid_search_lgb_1.fit(train[predictor_set1].values, train[target].values.ravel())

# #trust your CV!
print("Best score: %f using \n%s" % (grid_result.best_score_, grid_result.best_params_))
score = cross_val_score(lgb, Train[predictor_set2].values, Train[target].values.ravel(), cv=k_fold, n_jobs=1,\
                                                                scoring=scoring)
print(score)
print('Accuracy:',round(np.mean(score)*100,2))
from sklearn.model_selection import GridSearchCV
parameters = { 'objective':['binary'],
              'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.8], #so called `eta` value
              'max_depth': [3, 5, 7, 9],
              'min_child_weight': [1, 5, 11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10, 50, 100, 300, 500], #number of trees, change it to 1000 for better results
              'seed': [123]}


grid_search_lgb_2 = GridSearchCV(lgb, parameters, n_jobs=5, 
                   cv= KFold(n_splits=5, shuffle=True, random_state=123), 
                   scoring='accuracy', return_train_score=True,
                   verbose=2, refit=True)

grid_result = grid_search_lgb_2.fit(Train[predictor_set2].values, Train[target].values.ravel())

# #trust your CV!
print("Best score: %f using \n%s" % (grid_result.best_score_, grid_result.best_params_))

from sklearn import metrics
import pickle
pickle.dump(grid_search_rf_1,open('grid_search_rf_1.sav','wb'))
pickle.dump(grid_search_rf_1,open('grid_search_rf_2.sav','wb'))

pickle.dump(grid_search_xgb_1,open('grid_search_xgb_1.sav','wb'))
pickle.dump(grid_search_xgb_1,open('grid_search_xgb_2.sav','wb'))

pickle.dump(grid_search_lgb_1,open('grid_search_lgb_1.sav','wb'))
pickle.dump(grid_search_lgb_1,open('grid_search_lgb_2.sav','wb'))
# pickle.dump(feat_imp,open('grid_search_lgb_1.sav','wb'))
lgb = pickle.load(open('grid_search_lgb_1.sav','rb'))
pred = lgb.predict(train[predictor_set1].values)
print('Accuracy score of train data set 1 with Logistic Regression: ',metrics.accuracy_score(train[target].values,Train_pred['']))
print(train.shape,test.shape)
print(Train.shape,Test.shape)
del Train_pred
del Train_score
del Test_pred
del Test_score
Train_pred = pd.DataFrame(train['PassengerId'])

Train_pred['y_pred_train_lg_1'] = lg.fit(train[predictor_set1].values,train[target].values.ravel()).\
                                        predict(train[predictor_set1].values)
print('Accuracy score of with set 1 with Logistic Regression: ',metrics.accuracy_score(train[target].values,Train_pred['y_pred_train_lg_1']))
Train_pred['y_pred_train_lg_2'] = lg.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
                                        predict(Train[predictor_set2].values)
print('Accuracy score of with set 2 with Logistic Regression: ',metrics.accuracy_score(Train[target].values,Train_pred['y_pred_train_lg_2']))

# Train_pred['y_pred_train_dt_1'] = dt.fit(train[predictor_set1].values,train[target].values.ravel()).\
#                                         predict(train[predictor_set1].values)
# print('Accuracy score of with set 1 with Decision Tree: ',metrics.accuracy_score(train[target].values,Train_pred['y_pred_train_dt_1']))
# Train_pred['y_pred_train_dt_2'] = dt.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
#                                         predict(Train[predictor_set2].values)
# print('Accuracy score of with set 2 with Decision Tree: ',metrics.accuracy_score(Train[target].values,Train_pred['y_pred_train_dt_2']))

Train_pred['y_pred_train_nb_1'] = nb.fit(train[predictor_set1].values,train[target].values.ravel()).\
                                        predict(train[predictor_set1].values)
print('Accuracy score of with set 1 with Naive Bayes: ',metrics.accuracy_score(train[target].values,Train_pred['y_pred_train_nb_1']))
Train_pred['y_pred_train_nb_2'] = nb.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
                                        predict(Train[predictor_set2].values)
print('Accuracy score of with set 2 with Naive Bayes: ',metrics.accuracy_score(Train[target].values,Train_pred['y_pred_train_nb_2']))

Train_pred['y_pred_train_rf_1'] = grid_search_rf_1.predict(train[predictor_set1].values)
print('Accuracy score of with set 1 with Random Forest: ',metrics.accuracy_score(train[target].values,Train_pred['y_pred_train_rf_1']))
Train_pred['y_pred_train_rf_2'] = grid_search_rf_2.predict(Train[predictor_set2].values)
print('Accuracy score of with set 2 with Random Forest: ',metrics.accuracy_score(Train[target].values,Train_pred['y_pred_train_rf_2']))

Train_pred['y_pred_train_xgb_1'] = grid_search_xgb_1.predict(train[predictor_set1].values)
print('Accuracy score of with set 1 with XG Boost: ',metrics.accuracy_score(train[target].values,Train_pred['y_pred_train_xgb_1']))
Train_pred['y_pred_train_xgb_2'] = grid_search_xgb_2.predict(Train[predictor_set2].values)
print('Accuracy score of with set 2 with  XG Boost: ',metrics.accuracy_score(Train[target].values,Train_pred['y_pred_train_xgb_2']))

Train_pred['y_pred_train_lgb_1'] = grid_search_lgb_1.predict(train[predictor_set1].values)
print('Accuracy score of with set 1 with Light GBM: ',metrics.accuracy_score(train[target].values,Train_pred['y_pred_train_lgb_1']))
Train_pred['y_pred_train_lgb_2'] = grid_search_lgb_2.predict(Train[predictor_set2].values)
print('Accuracy score of with set 2 with Light GBM: ',metrics.accuracy_score(Train[target].values,Train_pred['y_pred_train_lgb_2']))

Train_pred
Train_score = pd.DataFrame(train['PassengerId'])

Train_score['y_score_train_lg_1'] = lg.fit(train[predictor_set1].values,train[target].values.ravel()).\
                                        predict_proba(train[predictor_set1].values)[:,1]
print('AUC score of with set 1 with Logistic Regression: ',metrics.roc_auc_score(train[target].values,Train_score['y_score_train_lg_1']))
Train_score['y_score_train_lg_2'] = lg.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
                                        predict_proba(Train[predictor_set2].values)[:,1]
print('AUC score of with set 2 with Logistic Regression: ',metrics.roc_auc_score(Train[target].values,Train_score['y_score_train_lg_2']))

# Train_score['y_score_train_dt_1'] = dt.fit(train[predictor_set1].values,train[target].values.ravel()).\
#                                         predict_proba(train[predictor_set1].values)[:,1]
# print('AUC score of with set 1 with Decision Tree: ',metrics.roc_auc_score(train[target].values,Train_score['y_score_train_dt_1']))
# Train_score['y_score_train_dt_2'] = dt.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
#                                         predict_proba(Train[predictor_set2].values)[:,1]
# print('AUC score of with set 2 with Decision Tree: ',metrics.roc_auc_score(Train[target].values,Train_score['y_score_train_dt_2']))

Train_score['y_score_train_nb_1'] = nb.fit(train[predictor_set1].values,train[target].values.ravel()).\
                                        predict_proba(train[predictor_set1].values)[:,1]
print('AUC score of with set 1 with Naive Bayes: ',metrics.roc_auc_score(train[target].values,Train_score['y_score_train_nb_1']))
Train_score['y_score_train_nb_2'] = nb.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
                                        predict_proba(Train[predictor_set2].values)[:,1]
print('AUC score of with set 2 with Naive Bayes: ',metrics.roc_auc_score(Train[target].values,Train_score['y_score_train_nb_2']))

Train_score['y_score_train_rf_1'] = grid_search_rf_1.predict_proba(train[predictor_set1].values)[:,1]
print('AUC score of with set 1 with Random Forest: ',metrics.roc_auc_score(train[target].values,Train_score['y_score_train_rf_1']))
Train_score['y_score_train_rf_2'] = grid_search_rf_2.predict_proba(Train[predictor_set2].values)[:,1]
print('AUC score of with set 2 with Random Forest: ',metrics.roc_auc_score(Train[target].values,Train_score['y_score_train_rf_2']))

Train_score['y_score_train_xgb_1'] = grid_search_xgb_1.predict_proba(train[predictor_set1].values)[:,1]
print('AUC score of with set 1 with XG Boost: ',metrics.roc_auc_score(train[target].values,Train_score['y_score_train_xgb_1']))
Train_score['y_score_train_xgb_2'] = grid_search_xgb_2.predict_proba(Train[predictor_set2].values)[:,1]
print('AUC score of with set 2 with  XG Boost: ',metrics.roc_auc_score(Train[target].values,Train_score['y_score_train_xgb_2']))

Train_score['y_score_train_lgb_1'] = grid_search_lgb_1.predict_proba(train[predictor_set1].values)[:,1]
print('AUC score of with set 1 with Light GBM: ',metrics.roc_auc_score(train[target].values,Train_score['y_score_train_lgb_1']))
Train_score['y_score_train_lgb_2'] = grid_search_lgb_2.predict_proba(Train[predictor_set2].values)[:,1]
print('AUC score of with set 2 with Light GBM: ',metrics.roc_auc_score(Train[target].values,Train_score['y_score_train_lgb_2']))

Train_score
Test_pred = pd.DataFrame(Test['PassengerId'])

Test_pred['y_pred_Test_lg_1'] = lg.fit(train[predictor_set1].values,train[target].values.ravel()).\
                                        predict(test[predictor_set1].values)
Test_pred['y_pred_Test_lg_2'] = lg.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
                                        predict(Test[predictor_set2].values)

# Test_pred['y_pred_Test_dt_1'] = dt.fit(train[predictor_set1].values,train[target].values.ravel()).\
#                                         predict(test[predictor_set1].values)
# Test_pred['y_pred_Test_dt_2'] = dt.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
#                                         predict(Test[predictor_set2].values)

Test_pred['y_pred_Test_nb_1'] = nb.fit(train[predictor_set1].values,train[target].values.ravel()).\
                                        predict(test[predictor_set1].values)
Test_pred['y_pred_Test_nb_2'] = nb.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
                                        predict(Test[predictor_set2].values)

Test_pred['y_pred_Test_rf_1'] = grid_search_rf_1.predict(test[predictor_set1].values)
Test_pred['y_pred_Test_rf_2'] = grid_search_rf_2.predict(Test[predictor_set2].values)

Test_pred['y_pred_Test_xgb_1'] = grid_search_xgb_1.predict(test[predictor_set1].values)
Test_pred['y_pred_Test_xgb_2'] = grid_search_xgb_2.predict(Test[predictor_set2].values)

Test_pred['y_pred_Test_lgb_1'] = grid_search_lgb_1.predict(test[predictor_set1].values)
Test_pred['y_pred_Test_lgb_2'] = grid_search_lgb_2.predict(Test[predictor_set2].values)

Test_pred
Test_score = pd.DataFrame(Test['PassengerId'])

Test_score['y_pred_Test_lg_1'] = lg.fit(train[predictor_set1].values,train[target].values.ravel()).\
                                        predict_proba(test[predictor_set1].values)[:,1]
Test_score['y_pred_Test_lg_2'] = lg.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
                                        predict_proba(Test[predictor_set2].values)[:,1]

# Test_score['y_pred_Test_dt_1'] = dt.fit(train[predictor_set1].values,train[target].values.ravel()).\
#                                         predict_proba(test[predictor_set1].values)[:,1]
# Test_score['y_pred_Test_dt_2'] = dt.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
#                                         predict_proba(Test[predictor_set2].values)[:,1]

Test_score['y_pred_Test_nb_1'] = nb.fit(train[predictor_set1].values,train[target].values.ravel()).\
                                        predict_proba(test[predictor_set1].values)[:,1]
Test_score['y_pred_Test_nb_2'] = nb.fit(Train[predictor_set2].values,Train[target].values.ravel()).\
                                        predict_proba(Test[predictor_set2].values)[:,1]

Test_score['y_pred_Test_rf_1'] = grid_search_rf_1.predict_proba(test[predictor_set1].values)[:,1]
Test_score['y_pred_Test_rf_2'] = grid_search_rf_2.predict_proba(Test[predictor_set2].values)[:,1]

Test_score['y_pred_Test_xgb_1'] = grid_search_xgb_1.predict_proba(test[predictor_set1].values)[:,1]
Test_score['y_pred_Test_xgb_2'] = grid_search_xgb_2.predict_proba(Test[predictor_set2].values)[:,1]

Test_score['y_pred_Test_lgb_1'] = grid_search_lgb_1.predict_proba(test[predictor_set1].values)[:,1]
Test_score['y_pred_Test_lgb_2'] = grid_search_lgb_2.predict_proba(Test[predictor_set2].values)[:,1]

Test_score
Train_pred.columns
sub_pred_train = Train_pred[['PassengerId','y_pred_train_lg_2','y_pred_train_nb_1','y_pred_train_rf_1','y_pred_train_xgb_2','y_pred_train_lgb_1']]
sub_pred_train['target'] = train[target]
sub_pred_train
sub_pred_train['sum'] = sub_pred_train.iloc[:,1] + sub_pred_train.iloc[:,2] + sub_pred_train.iloc[:,3] +\
                            sub_pred_train.iloc[:,4] + sub_pred_train.iloc[:,5]
sub_pred_train['check2'] = [1 if x>=2 else 0 for x in sub_pred_train['sum']]
sub_pred_train['check3'] = [1 if x>=3 else 0 for x in sub_pred_train['sum']]
sub_pred_train
print(metrics.accuracy_score(sub_pred_train['target'],sub_pred_train['y_pred_train_lg_2']))
print(metrics.accuracy_score(sub_pred_train['target'],sub_pred_train['y_pred_train_nb_1']))
print(metrics.accuracy_score(sub_pred_train['target'],sub_pred_train['y_pred_train_rf_1']))
print(metrics.accuracy_score(sub_pred_train['target'],sub_pred_train['y_pred_train_xgb_2']))
print(metrics.accuracy_score(sub_pred_train['target'],sub_pred_train['y_pred_train_lgb_1']))
print(metrics.accuracy_score(sub_pred_train['target'],sub_pred_train['check2']))
print(metrics.accuracy_score(sub_pred_train['target'],sub_pred_train['check3']))
my_submission = pd.DataFrame({'PassengerId': Test_pred['PassengerId'], 'survived': Test_pred['y_pred_Test_rf_1']})
my_submission.to_csv('my_submission.csv', index=False)
my_submission
sub.Survived.value_counts(normalize=True)
sub.survived.value_counts(normalize=True)
# my_submission = pd.DataFrame({'PassengerId': test.Id, 'Survived': prediction})
# my_submission.to_csv('my_submission.csv', index=False)