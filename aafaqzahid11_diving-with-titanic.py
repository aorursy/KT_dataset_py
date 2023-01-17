import pandas as pd

import numpy as np
import os
import matplotlib.pyplot as plt

import seaborn as sns
from plotly import __version__

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
import re
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_test.head()
df_train.info()
df_train.describe()
sns.barplot(x = 'Pclass', y = 'Survived', data = df_train )
sns.countplot(x='Sex', hue = 'Survived',data=df_train)
sns.violinplot(x="Pclass", y="Survived",hue ="Sex" ,data=df_train,palette='rainbow')
sns.heatmap(df_train.drop(['PassengerId', 'Name'], axis =1).fillna(0).corr())
df_train['Sex'] = df_train['Sex'].apply(lambda x: 1 if x == 'male' else 2)
df_test['Sex'] = df_test['Sex'].apply(lambda x: 1 if x == 'male' else 2)
df_train['Salutation'] = df_train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df_test['Salutation'] = df_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df_train['Salutation'].unique()
def replace_titles(x):

    title=x['Salutation']

    

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Mr', 'Master', 'Sir']:

        return 1

    elif title in ['the Countess', 'Mme', 'Mrs', 'Lady']:

        return 2

    elif title in ['Mlle', 'Ms', 'Miss']:

        return 3

    elif title =='Dr':

        if x['Sex']=='Male':

            return 1

        else:

            return 2

    else:

        return title
salutation = {'Don':1, 'Major':2, 'Capt':3, 'Jonkheer':4, 

              'Rev':5, 'Col':6, 'Mr':7, 'Master':8, 'Sir':9,

              'the Countess':10, 'Mme':11, 'Mrs':12, 'Lady':13,

              'Mlle':14, 'Ms':15, 'Miss':16,'Dr':17, 'Dona':1}
df_train['Salutation'] = df_train['Salutation'].apply(lambda x:salutation[x] )
df_test['Salutation'] = df_test['Salutation'].apply(lambda x:salutation[x] )
df_train['Embarked'].unique()
embarked = {'S':1,'C':2, 'Q':3}
df_train['Embarked'] = df_train['Embarked'].apply(lambda x: embarked[x] if x in embarked else x)
df_test['Embarked'] = df_test['Embarked'].apply(lambda x: embarked[x] if x in embarked else x)
df_train['CabinCount'] = df_train['Cabin'].apply(lambda x: x if type(x) is float else len(x.split()))
df_test['CabinCount'] = df_test['Cabin'].apply(lambda x: x if type(x) is float else len(x.split()))
df_train['Cabin'] = df_train['Cabin'].apply(lambda x: x if type(x) is float else re.split('(\d+)',x)[0])
df_test['Cabin'] = df_test['Cabin'].apply(lambda x: x if type(x) is float else re.split('(\d+)',x)[0])
df_train['Cabin'].unique()
cabins = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'F G':6, 'F E':6}
df_train['Cabin'] = df_train['Cabin'].apply(lambda x: cabins[x] if x in cabins else x)
df_test['Cabin'] = df_test['Cabin'].apply(lambda x: cabins[x] if x in cabins else x)
df_train.head()
df_train.fillna(0 , inplace=True)
df_test.fillna(0 , inplace=True)
import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(df_train.drop(['Name','Ticket','Age', 'Embarked', 'Survived', 'Cabin', 'CabinCount', 'PassengerId'],axis=1), method='ward'))
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(df_train.drop(['Name','Ticket','Age', 'Embarked', 'Survived', 'Cabin', 'CabinCount', 'PassengerId'],axis=1))

    wcss.append(kmeans.inertia_)



plt.plot(range(1,11), wcss)

plt.show()

from sklearn.cluster import AgglomerativeClustering

agc = AgglomerativeClustering(n_clusters=4)

df_train['Cluster'] = agc.fit_predict(df_train.drop(['Name','Ticket','Age', 'Embarked', 'Survived', 'Cabin', 'CabinCount', 'PassengerId'],axis=1))
df_test['Cluster'] = agc.fit_predict(df_test.drop(['Name','Ticket','Age', 'Embarked', 'Cabin', 'CabinCount', 'PassengerId'],axis=1))
df_groupby = df_train.groupby(by=df_train['Cluster']).count()
df_groupby_test = df_test.groupby(by=df_train['Cluster']).count()
df_groupby.head()
df_groupby = df_train.groupby(by=df_train['Cluster']).mean()
df_groupby_test = df_test.groupby(by=df_train['Cluster']).mean()
df_groupby.reset_index(inplace=True)
#df_groupby_test.reset_index(inplace=True)
df_groupby_test.head()
def fill_missing(x):

    

    if x['Cabin'] == 0.0:

        x['Cabin'] = float(round(df_groupby[df_groupby['Cluster'] == x['Cluster']]['Cabin']))

    if x['Age'] == 0.0:

        x['Age'] = float(round(df_groupby[df_groupby['Cluster'] == x['Cluster']]['Age']))

    if x['CabinCount'] == 0.0:

        x['CabinCount'] = float(round(df_groupby[df_groupby['Cluster'] == x['Cluster']]['CabinCount']))

    if x['Embarked'] == 0.0:

        x['Embarked'] = float(round(df_groupby[df_groupby['Cluster'] == x['Cluster']]['Embarked']))

    return x
df_train = df_train.apply(fill_missing, axis=1)
df_test = df_test.apply(fill_missing, axis=1)
df_train.head()
df_train['CabinCount'].value_counts()
df_train.groupby(by = df_train['Pclass']).sum()
df_train['Age'].min()
df_train['Age'].max()
def assing_age_group(x):

    if x < 10:

        return 1

    elif x >= 10 and x < 18:

        return 2

    elif x >= 18 and x <25:

        return 3

    elif x >= 25 and x < 30:

        return 4

    elif x >= 30 and x < 40:

        return 5

    elif x >= 40 and x < 50:

        return 6

    elif x >= 50 and x < 60:

        return 7

    elif x >= 60:

        return 8
df_train['AgeGroup'] = df_train['Age'].apply(assing_age_group)
df_test['AgeGroup'] = df_test['Age'].apply(assing_age_group)
sns.heatmap(df_train.drop(['PassengerId', 'Name', 'Ticket'], axis =1).fillna(0).corr())
df_train['Family_Size']=df_train['SibSp']+df_train['Parch']
df_test['Family_Size']=df_test['SibSp']+df_test['Parch']
sns.pairplot(df_train)
df_train['Fare_Per_Person']=df_train['Fare']/(df_train['Family_Size']+1)
df_test['Fare_Per_Person']=df_test['Fare']/(df_train['Family_Size']+1)
sns.heatmap(df_train.drop(['PassengerId', 'Name', 'Ticket'], axis =1).fillna(0).corr())
df_train.drop(['Ticket','Name', 'Cluster', 'Age'], inplace=True, axis = 1)

df_train.head()
df_test.drop(['Ticket','Name', 'Cluster', 'Age'], inplace=True, axis = 1)

df_test.head()
X = df_train.iloc[:, 2:14].values

y = df_train.iloc[:, 1].values
X_main_test =  df_test.iloc[:, 1:13].values
from sklearn.preprocessing import OneHotEncoder
classhotEncoder = OneHotEncoder(categorical_features=[0])

X = classhotEncoder.fit_transform(X).toarray()
X = X[:,1:]
classhotEncoder = OneHotEncoder(categorical_features=[0])

X_main_test = classhotEncoder.fit_transform(X_main_test).toarray()
X_main_test = X_main_test[:,1:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
sc = StandardScaler()

X = sc.fit_transform(X)

X_main_test = sc.fit_transform(X_main_test)
accuracy_results = pd.DataFrame(columns=['Algorithm','Results'])
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix , classification_report, accuracy_score

cm = confusion_matrix(y_test, y_pred)
accuracy_results = accuracy_results.append(pd.DataFrame({'Algorithm':['Logistic_Regression'],'Results':[accuracy_score(y_test, y_pred)]}))
print(cm)

print('\n')

print(classification_report(y_test,y_pred.round()))
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_results = accuracy_results.append(pd.DataFrame({'Algorithm':['SVC'],'Results':[accuracy_score(y_test, y_pred)]}))
print(cm)

print('\n')

print(classification_report(y_test,y_pred.round()))
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_results = accuracy_results.append(pd.DataFrame({'Algorithm':['Kernel_SVM'],'Results':[accuracy_score(y_test, y_pred)]}))
print(cm)

print('\n')

print(classification_report(y_test,y_pred.round()))
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_results = accuracy_results.append(pd.DataFrame({'Algorithm':['Naive_Bayes'],'Results':[accuracy_score(y_test, y_pred)]}))
print(cm)

print('\n')

print(classification_report(y_test,y_pred.round()))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100,criterion='gini')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_results = accuracy_results.append(pd.DataFrame({'Algorithm':['Random_Forest'],'Results':[accuracy_score(y_test, y_pred)]}))
print(cm)

print('\n')

print(classification_report(y_test,y_pred.round()))
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=13, units=7, kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.03))
classifier.add(Dense(activation="relu", units=7, kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.03))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train.reshape(-1,1), batch_size = 10, epochs = 300)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
accuracy_results = accuracy_results.append(pd.DataFrame({'Algorithm':['ANN'],'Results':[accuracy_score(y_test, y_pred)]}))
cm = confusion_matrix(y_test, y_pred)

print(cm)

print('\n')

print(classification_report(y_test,y_pred.round()))
from xgboost import XGBClassifier

import xgboost as xgb

classifier = XGBClassifier(n_estimators=100)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

print('\n')

print(classification_report(y_test,y_pred.round()))
accuracy_results = accuracy_results.append(pd.DataFrame({'Algorithm':['XGBoost'],'Results':[accuracy_score(y_test, y_pred)]}))
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)

accuracies.mean()
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [1, 10, 100, 1000], 'criterion': ['entropy', 'gini']}]



classifier = RandomForestClassifier(n_estimators=100,criterion='gini')

classifier.fit(X_train, y_train)



grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X, y)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print(best_accuracy)

print(best_parameters)
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
def modelfit(alg, xtrain, xtest, ytrain, ytest, predictors, cv_folds=5, early_stopping_rounds=50):

    

    alg.fit(xtrain, ytrain,eval_metric='auc')

        

    dtrain_predictions = alg.predict(xtest)

    dtrain_predprob = alg.predict_proba(xtest)[:,1]

        

    print ("\nModel Report")

    print ("Accuracy : %.4g" % accuracy_score(ytest, dtrain_predictions))

    print (dtrain_predprob.shape)

    print ("AUC Score (Train): %f" % roc_auc_score(ytest, dtrain_predprob))

                    

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')
predictors = [x for x in X]

xgb_child = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=7,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

modelfit(xgb_child, X_train, X_test, y_train, y_test.reshape(-1,1), predictors)
param_child = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

gsearch_child = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_child, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch_child.fit(X_train,y_train)

gsearch_child.cv_results_, gsearch_child.best_params_, gsearch_child.best_score_
param_child_2 = {

 'max_depth':[3,4,5,6],

 'min_child_weight':[4,5,6]

}

gsearch_child_2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,

 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_child_2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch_child_2.fit(X_train,y_train)

gsearch_child_2.cv_results_, gsearch_child_2.best_params_, gsearch_child_2.best_score_
param_test2b = {

 'min_child_weight':[4,5,6,8,10,12]

}

gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=3,

 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2b.fit(X_train,y_train)

gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_
param_test2b = {

 'min_child_weight':[3,4,8,12,14,16],

 'max_depth':[2,3,4,5,6]

}

gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,

 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2b.fit(X_train,y_train)

gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_
param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=3,

 min_child_weight=12, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(X_train,y_train)

gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
xgb2 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=3,

 min_child_weight=12,

 gamma=0.0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

modelfit(xgb2, X_train, X_test, y_train, y_test, predictors)
param_test4 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,

 min_child_weight=12, gamma=0.0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(X_train,y_train)

gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_
param_test6 = {

 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

}

gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,

 min_child_weight=12, gamma=0.0, subsample=0.8, colsample_bytree=0.9,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch6.fit(X_train,y_train)

gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_
param_test7 = {

 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]

}

gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=300, max_depth=3,

 min_child_weight=12, gamma=0.0, subsample=0.8, colsample_bytree=0.9,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch7.fit(X_train, y_train)

gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_
xgb_final = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=3,

 min_child_weight=12,

 gamma=0.0,

 reg_alpha = 0.001,

 subsample=0.8,

 colsample_bytree=0.9,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

modelfit(xgb_final, X_train, X_test, y_train, y_test.reshape(-1,1), predictors)
def build_classifier(optimizer):

    classifier = Sequential()

    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32, 35, 40],

              'epochs': [100, 200, 500],

              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)

grid_search = grid_search.fit(X, y)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_
print(best_accuracy)

print(best_parameters)
df_results = pd.DataFrame()
classifier = RandomForestClassifier(n_estimators=100,criterion='gini')

classifier.fit(X_train, y_train)

df_results['RandomForest'] = classifier.predict(X_test)
classifier = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=3,

 min_child_weight=12,

 gamma=0.0,

 reg_alpha = 0.001,

 subsample=0.8,

 colsample_bytree=0.9,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

classifier.fit(X_train, y_train)

df_results['XGBoost'] = classifier.predict(X_test)
classifier = Sequential()

classifier.add(Dense(activation="relu", input_dim=13, units=7, kernel_initializer="uniform"))

classifier.add(Dropout(rate=0.03))

classifier.add(Dense(activation="relu", units=7, kernel_initializer="uniform"))

classifier.add(Dropout(rate=0.03))

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train.reshape(-1,1), batch_size = 25, epochs = 500)
df_results['ANN'] = (classifier.predict(X_test) > 0.5)
df_results['Ensemble'] = (df_results.mean(axis=1) > 0.5)
cm = confusion_matrix(y_test, df_results['Ensemble'])

print(cm)

print('\n')

print(classification_report(y_test,df_results['Ensemble']))
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,df_results['Ensemble'])
df_results = pd.DataFrame()
classifier = RandomForestClassifier(n_estimators=100,criterion='gini')

classifier.fit(X, y)

df_results['RandomForest'] = classifier.predict(X_main_test)
classifier = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=3,

 min_child_weight=12,

 gamma=0.0,

 reg_alpha = 0.001,

 subsample=0.8,

 colsample_bytree=0.9,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

classifier.fit(X, y)

df_results['XGBoost'] = classifier.predict(X_main_test)
classifier = Sequential()

classifier.add(Dense(activation="relu", input_dim=13, units=7, kernel_initializer="uniform"))

classifier.add(Dropout(rate=0.03))

classifier.add(Dense(activation="relu", units=7, kernel_initializer="uniform"))

classifier.add(Dropout(rate=0.03))

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X, y.reshape(-1,1), batch_size = 25, epochs = 500)
df_results['Ensemble'] = (df_results.mean(axis=1) > 0.5)
df_test['Survived'] = df_results['Ensemble']
df_test['Survived'] = df_test['Survived'].apply(lambda x: int(x))
df_test.drop(['Pclass','Sex','Parch', 'SibSp', 'Fare', 'Cabin', 'Embarked', 'Salutation', 'CabinCount', 'AgeGroup','Family_Size','Fare_Per_Person'], axis =1, inplace=True)
df_test.to_csv('test_results.csv', index=False)