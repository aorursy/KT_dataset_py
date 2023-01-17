###This Python 3 environment comes with many helpful analytics libraries installed

#It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#PLOTTING GRAPH

import matplotlib.pyplot as plt

import seaborn as sns



#BUILDING CLASSIFIERS

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('../input/train.csv')
df_train.head()
df_train.isna().sum().reset_index
df_train.isna().sum().reset_index
df_train['Embarked']=df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
df_train['Cabin']=df_train['Cabin'].transform(lambda x:0 if pd.isnull(x) else 1)
age_avg= df_train['Age'].mean()

age_std= df_train['Age'].std()



age_null = df_train['Age'].isnull().sum()

age_assign=np.random.randint(age_avg - age_std,age_avg + age_std,size=age_null)



df_train['Age'][np.isnan(df_train['Age'])] = age_assign

df_train['Age']=df_train['Age'].astype(int)
df_train.head()
df_train['Sex'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_train['Sex'],data=df_train,hue='Survived')

ax.set_title('Number of Male /Female survived', fontsize=16)

ax.set_xlabel('Gender',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_train['Pclass'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_train['Pclass'],data=df_train,hue='Survived')

ax.set_title('Passengers survived based on Class', fontsize=16)

ax.set_xlabel('Ticket class',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_train['Cabin'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_train['Cabin'],data=df_train,hue='Survived')

ax.set_title('Passengers survived based on Cabin availability', fontsize=16)

ax.set_xlabel('Cabin Availability',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_train['Embarked'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_train['Embarked'],data=df_train,hue='Survived')

ax.set_title('Passengers survived based on Ports Embarked', fontsize=16)

ax.set_xlabel('Ports Embarked',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_train['Embarked'].unique()

df_train['Embarked']=df_train['Embarked'].map({'C': 0, 'Q':1, 'S': 2})
df_train['Sex'].unique()

df_train['Sex']=df_train['Sex'].map({'male': 0, 'female': 1})
pd.qcut(df_train['Age'],4).unique()
df_train.loc[df_train['Age']<= 21.0,'Age'] = 0

df_train.loc[(df_train['Age'] > 21.0) & (df_train['Age'] <= 29.0),'Age'] = 1

df_train.loc[(df_train['Age'] > 29.0) & (df_train['Age'] <= 38.0),'Age'] = 2

df_train.loc[df_train['Age'] > 38.0,'Age'] = 3
df_train['Age'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_train['Age'],data=df_train,hue='Survived')

ax.set_title('Passengers survived based on Age', fontsize=16)

ax.set_xlabel('Age',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
pd.qcut(df_train['Fare'],4).unique()
df_train.loc[df_train['Fare']<= 21.0,'Fare'] = 0

df_train.loc[(df_train['Fare'] > 21.0) & (df_train['Fare'] <= 29.0),'Fare'] = 1

df_train.loc[(df_train['Fare'] > 29.0) & (df_train['Fare'] <= 38.0),'Fare'] = 2

df_train.loc[df_train['Fare'] > 38.0,'Fare'] = 3

df_train['Fare']=df_train['Fare'].astype(int)
df_train['Fare'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_train['Fare'],data=df_train,hue='Survived')

ax.set_title('Passengers survived based on Fare paid', fontsize=16)

ax.set_xlabel('Ticket Fare',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_train.drop('Ticket', inplace=True, axis=1)
df_train['Salutation']= df_train['Name'].transform(lambda x: x.split(',')[1].split('.')[0])

df_train['Salutation']=df_train['Salutation'].transform(lambda x: x.strip())
df_train.head()
df_train['Salutation'].unique()
df_train['Salutation']= df_train['Salutation'].replace('Mme','Mrs')

df_train['Salutation']= df_train['Salutation'].replace(['Ms','Mlle'],'Miss')

df_train['Salutation']= df_train['Salutation'].replace(['Lady','Don','Rev','Dr','Major','Sir','Col','Capt','the Countess','Jonkheer'],'Others')
df_train['Salutation'].unique()
df_train['Salutation'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_train['Salutation'],data=df_train,hue='Survived')

ax.set_title('Passengers survived based on Salutation', fontsize=16)

ax.set_xlabel('Salutation',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_train['Salutation']=df_train['Salutation'].map({'Mr': 0, 'Mrs': 1, 'Master':2, 'Miss': 3, 'Others':4})
df_train.drop('Name', inplace=True, axis=1)
df_train['LoneTraveller']=0

for index, item in df_train.iterrows():

    if item['Parch']==0 and item['SibSp']==0:

        df_train.loc[index,'LoneTraveller'] = 1  # doubt1
df_train['LoneTraveller'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_train['LoneTraveller'],data=df_train,hue='Survived')

ax.set_title('number of Travellers survived', fontsize=16)

ax.set_xlabel('Traveller type with 0-Accompanied passengers and 1- lone traveller',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_train.head(20)
df_test=pd.read_csv('../input/test.csv')
df_test.head()
df_test.isna().sum().reset_index()
age_avg= df_test['Age'].mean()

age_std= df_test['Age'].std()

age_null = df_test['Age'].isna().sum()

age_assign= np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null)

df_test['Age'][np.isnan(df_test['Age'])]=age_assign # doubt2

df_test['Age']=df_test['Age'].astype(int)
df_test['Fare']=df_test['Fare'].fillna(df_test['Fare'].mean())
df_test['Cabin']=df_test['Cabin'].transform(lambda x:0 if pd.isnull(x) else 1)
df_test['Sex'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_test['Sex'],data=df_test)

ax.set_title('Number of Male /Female survived', fontsize=16)

ax.set_xlabel('Gender',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_test['Pclass'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_test['Pclass'],data=df_test)

ax.set_title('Number of passengers survived based on Ticket Class', fontsize=16)

ax.set_xlabel('Ticket Class',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_test['Cabin'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_test['Cabin'],data=df_test)

ax.set_title('Number of passengers based on Cabin Availability', fontsize=16)

ax.set_xlabel('Cabin Availability with 0-No cabin, 1- Cabin',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_test['Embarked'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_test['Embarked'],data=df_test)

ax.set_title('Number of passengers based on ports of embarkation', fontsize=16)

ax.set_xlabel('ports of embarkation',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_test['Embarked'].unique()
df_test['Embarked']=df_test['Embarked'].map({'C': 0, 'Q':1, 'S': 2})
df_test['Sex'].unique()
df_test['Sex']=df_test['Sex'].map({'male': 0, 'female': 1})
pd.qcut(df_test['Fare'],4).unique()
df_test.loc[df_test['Fare']<= 7.896,'Fare'] = 0

df_test.loc[(df_test['Fare'] > 7.896) & (df_test['Fare'] <= 14.454),'Fare'] = 1

df_test.loc[(df_test['Fare'] > 14.454) & (df_test['Fare'] <= 31.5),'Fare'] = 2

df_test.loc[df_test['Fare'] > 31.5,'Fare'] = 3

df_test['Fare']=df_test['Fare'].astype(int)
df_test['Fare'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_test['Fare'],data=df_test)

ax.set_title('Number of passengers based on Ticket Fare', fontsize=16)

ax.set_xlabel('Ticket fare',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
pd.qcut(df_test['Age'],4).unique()
df_test.loc[df_test['Age']<= 21.0,'Age'] = 0

df_test.loc[(df_test['Age'] > 21.0) & (df_test['Age'] <= 28.0),'Age'] = 1

df_test.loc[(df_test['Age'] > 28.0) & (df_test['Age'] <= 38.0),'Age'] = 2

df_test.loc[df_test['Age'] > 38.0,'Age'] = 3
df_test['Age'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_test['Age'],data=df_test)

ax.set_title('Number of passengers based on Age', fontsize=16)

ax.set_xlabel('Passenger Age',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_test.drop('Ticket', inplace=True, axis=1)
df_test['Salutation']= df_test['Name'].transform(lambda x: x.split(',')[1].split('.')[0])

df_test['Salutation']= df_test['Salutation'].transform(lambda x: x.strip())
df_test['Salutation']= df_test['Salutation'].replace('Mme','Mrs')

df_test['Salutation']= df_test['Salutation'].replace(['Ms','Mlle'],'Miss')

df_test['Salutation']= df_test['Salutation'].replace(['Lady','Don','Rev','Dr','Major','Sir','Col','Capt','the Countess','Jonkheer','Dona'],'Others')
df_test['Salutation'].unique()
df_test['Salutation'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_test['Salutation'],data=df_test)

ax.set_title('Passengers survived based on Salutation', fontsize=16)

ax.set_xlabel('Salutation',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_test['Salutation']=df_test['Salutation'].map({'Mr': 0, 'Mrs': 1, 'Master':2, 'Miss': 3, 'Others':4})
df_test.drop('Name', inplace=True, axis=1)
df_test['LoneTraveller']=0

for index, item in df_test.iterrows():

    if item['Parch']==0 and item['SibSp']==0:

        df_test.loc[index,'LoneTraveller']=1
df_test['LoneTraveller'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x=df_test['LoneTraveller'],data=df_test)

ax.set_title('number of Travellers survived', fontsize=16)

ax.set_xlabel('Traveller type with 0-Accompanied passengers and 1- lone traveller',fontsize=12)

ax.set_ylabel('No. of passengers Survived',fontsize=12)

plt.show()
df_test.head(20)
PassengerId=df_test['PassengerId'].ravel()

df_test.drop('PassengerId', inplace=True, axis=1)

df_train.drop('PassengerId', inplace=True, axis=1)
df_train.head()
df_test.head()
#input

X= df_train.iloc[:,1:]



#Output

Y= df_train['Survived'].ravel()

X.shape, Y.shape
fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(X.corr(), annot=True)

ax.set_title('Correlation of training data')

plt.show()
fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(df_test.corr(), annot=True)

ax.set_title('Correlation of actual test data')

plt.show()
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.4, random_state=0)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, make_scorer

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC
accuracy_prediction= make_scorer(accuracy_score)
'''rfc=RandomForestClassifier()

rf_parm={   "n_estimators": [100, 300, 500, 1000],

    "bootstrap": [True, False],

    "criterion": ['gini', 'entropy'],

    "warm_start": [True, False],

    "max_depth": [2, 4, 6],

    "max_features": ['sqrt', 'log2'],

    "min_samples_split": [2, 4, 6],

    "min_samples_leaf": [2, 4, 6]

}



grid_search=GridSearchCV(estimator=rfc, param_grid=rf_parm, scoring=accuracy_prediction)

grid_search.fit(X_train,Y_train)



grid_search.best_estimator_'''
#Best prediction for RandomForestClassifier:

rf_classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=4, max_features='sqrt', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=2, min_samples_split=4,

            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

'''ETC = ExtraTreesClassifier()

xt_parm = {

    "n_estimators":[100, 300, 500, 1000],

    "bootstrap": [True, False],

    "criterion": ['gini', 'entropy'],

    "warm_start": [True, False],

    "max_depth": [2, 4, 6],

    "max_features": ['sqrt', 'log2'],

    "min_samples_split": [2, 4, 6],

    "min_samples_leaf": [2, 4, 6]

}



grid_search=GridSearchCV(estimator=ETC, param_grid=xt_parm, scoring=accuracy_prediction)

grid_search.fit(X_train,Y_train)



grid_search.best_estimator_'''
#Best Prediction for ExtraTreeClassifier:

et_classifier = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',

           max_depth=6, max_features='log2', max_leaf_nodes=None,

           min_impurity_decrease=0.0, min_impurity_split=None,

           min_samples_leaf=2, min_samples_split=6,

           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,

           oob_score=False, random_state=None, verbose=0, warm_start=False)
'''abc = AdaBoostClassifier()

abc_parm = {

    "n_estimators":[100, 300, 500, 1000],

    "learning_rate": [0.1, 0.3, 0.5, 0.75, 1]

}



grid_search=GridSearchCV(estimator=abc, param_grid=abc_parm, scoring=accuracy_prediction)

grid_search.fit(X_train,Y_train)



grid_search.best_estimator_'''
#Best Prediction for AdBoostClassifier:

ab_classifier = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,

          learning_rate=0.1, n_estimators=100, random_state=None)
'''GBC = GradientBoostingClassifier()

gb_parm = {

    "n_estimators":[100, 300, 500, 1000],

    "learning_rate": [0.1, 0.3, 0.5, 0.75, 1],

    "warm_start": [True, False],

    "max_depth": [2, 4, 6],

    "max_features": ['sqrt', 'log2'],

    "min_samples_split": [2, 4, 6],

    "min_samples_leaf": [2, 4, 6]

}



grid_search=GridSearchCV(estimator=GBC, param_grid=gb_parm, scoring=accuracy_prediction)

grid_search.fit(X_train,Y_train)



grid_search.best_estimator_'''
#Best Prediction for GradientBoostingClassifier:

gb_classifier = GradientBoostingClassifier(criterion='friedman_mse', init=None,

              learning_rate=0.1, loss='deviance', max_depth=2,

              max_features='log2', max_leaf_nodes=None,

              min_impurity_decrease=0.0, min_impurity_split=None,

              min_samples_leaf=2, min_samples_split=6,

              min_weight_fraction_leaf=0.0, n_estimators=100,

              n_iter_no_change=None, presort='auto', random_state=None,

              subsample=1.0, tol=0.0001, validation_fraction=0.1,

              verbose=0, warm_start=True)
'''svc = SVC()

sv_parm = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [0.01, 0.1, 1, 10, 100]},

                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]



grid_search=GridSearchCV(estimator=svc, param_grid=sv_parm, scoring=accuracy_prediction)

grid_search.fit(X_train,Y_train)



grid_search.best_estimator_'''
#Best Prediction for SupportVectorClassifier:

sv_classifier = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)
rf_classifier.fit(X_train,Y_train)

et_classifier.fit(X_train,Y_train)

ab_classifier.fit(X_train,Y_train)

gb_classifier.fit(X_train,Y_train)

sv_classifier.fit(X_train,Y_train)
rfc_rank= rf_classifier.feature_importances_

etc_rank= et_classifier.feature_importances_

abc_rank= ab_classifier.feature_importances_

gbc_rank= gb_classifier.feature_importances_
df_important_features= pd.DataFrame({

    "features": X.columns,

    "Random_forest": rfc_rank,

    "Extra Trees": etc_rank,

    "Adaboost": abc_rank,

    "Gradientboost": gbc_rank

    })

df_important_features
fig, ax= plt.subplots(figsize=(8,8))

sns.barplot(x=df_important_features['features'],y=df_important_features['Random_forest'])

ax.set_title('Importance of features in Random forest classifier', fontsize=12)

ax.set_xlabel('Feature importance',fontsize=10)

ax.set_ylabel('Rank',fontsize=10)

plt.show()
fig, ax= plt.subplots(figsize=(8,8))

sns.barplot(x=df_important_features['features'],y=df_important_features['Extra Trees'])

ax.set_title('Importance of features in Extra Trees classifier', fontsize=12)

ax.set_xlabel('Feature importance',fontsize=10)

ax.set_ylabel('Rank',fontsize=10)

plt.show()
fig, ax= plt.subplots(figsize=(8,8))

sns.barplot(x=df_important_features['features'],y=df_important_features['Adaboost'])

ax.set_title('Importance of features in Ada Boost classifier', fontsize=12)

ax.set_xlabel('Feature importance',fontsize=10)

ax.set_ylabel('Rank',fontsize=10)

plt.show()
fig, ax= plt.subplots(figsize=(8,8))

sns.barplot(x=df_important_features['features'],y=df_important_features['Gradientboost'])

ax.set_title('Importance of features in Gradient Boost classifier', fontsize=12)

ax.set_xlabel('Feature importance',fontsize=10)

ax.set_ylabel('Rank',fontsize=10)

plt.show()
rf_pred= rf_classifier.predict(X_test)

print('Random Forest classifier Accuracy: {0:.2f}'.format(accuracy_score(Y_test, rf_pred)*100))

et_pred= et_classifier.predict(X_test)

print('Extra Trees classifier Accuracy: {0:.2f}'.format(accuracy_score(Y_test, et_pred)*100))

ab_pred= ab_classifier.predict(X_test)

print('AdaBoost classifier Accuracy: {0:.2f}'.format(accuracy_score(Y_test, ab_pred)*100))

gb_pred= gb_classifier.predict(X_test)

print('Gradient Boost classifier Accuracy: {0:.2f}'.format(accuracy_score(Y_test, gb_pred)*100))

sv_pred= sv_classifier.predict(X_test)

print('Support Vector classifier Accuracy: {0:.2f}'.format(accuracy_score(Y_test, sv_pred)*100))
from sklearn.model_selection import KFold
def Kfold_pred(clf, X, Y):

    Outcomes = []    

    Kfold=KFold(n_splits=5, random_state=0, shuffle=False)

    for i, (train_index, test_index) in enumerate(Kfold.split(X)):

        X_train, X_test= X.values[train_index], X.values[test_index]

        Y_train, Y_test= Y[train_index], Y[test_index] # already an array, hence .values not required

        clf.fit(X_train, Y_train)

        y_pred= clf.predict(X_test)

        acc= accuracy_score(y_pred, Y_test)

        Outcomes.append(acc)

        print('Fold {0} accuracy {1:.2f}'.format(i,acc))

    mean_accuracy =  np.mean(Outcomes)

    return mean_accuracy
rf_predict = Kfold_pred(rf_classifier, X, Y)

print('Random Forest classifiers 5 fold mean accuracy : {0: .2f}'.format(rf_predict))

et_predict = Kfold_pred(et_classifier, X, Y)

print('Extra Trees classifiers 5 fold mean accuracy : {0: .2f}'.format(et_predict))

ab_predict = Kfold_pred(ab_classifier, X, Y)

print('Ada Boost classifiers 5 fold mean accuracy : {0: .2f}'.format(ab_predict))

gb_predict = Kfold_pred(gb_classifier, X, Y)

print('Gradient Boost classifiers 5 fold mean accuracy : {0: .2f}'.format(gb_predict))

sv_predict = Kfold_pred(sv_classifier, X, Y)

print('Support Vector classifiers 5 fold mean accuracy : {0: .2f}'.format(sv_predict))
def oof_predict(clf, X,Y, df_test):

    Kfold=KFold(n_splits=5, random_state=0, shuffle=True)

    oof_train=np.zeros(X.shape[0])

    oof_test=np.zeros(df_test.shape[0])

    oof_test_kf=np.empty((Kfold.get_n_splits(),df_test.shape[0]))

    

    for i, (train_index, test_index) in enumerate(Kfold.split(X)):

        X_train, X_test= X.values[train_index], X.values[test_index]

        Y_train, Y_test= Y[train_index], Y[test_index]

        clf.fit(X_train, Y_train)

        oof_train[test_index]= clf.predict(X_test)

        oof_test_kf[i,:]=clf.predict(df_test)

    oof_test=oof_test_kf.mean(axis=0)

    return oof_train, oof_test
rf_oof_train, rf_oof_test = oof_predict(rf_classifier,X,Y,df_test)

et_oof_train, et_oof_test = oof_predict(et_classifier,X,Y,df_test)

ab_oof_train, ab_oof_test = oof_predict(ab_classifier,X,Y,df_test)

gb_oof_train, gb_oof_test = oof_predict(gb_classifier,X,Y,df_test)

sv_oof_train, sv_oof_test = oof_predict(sv_classifier,X,Y,df_test)
final_train= pd.DataFrame({

    "Random_forest" : rf_oof_train,

    "Extra_Trees" : et_oof_train,

    "AdaBoost": ab_oof_train,

    "Gradient Boost": gb_oof_train,

    "Support Vector": sv_oof_train

})
final_train.shape
fig, ax= plt.subplots(figsize=(8,8))

sns.heatmap(final_train.corr(), annot=True)

ax.set_title('Correlation of cross validated classifiers for training data', fontsize=12)

plt.show()
final_test= pd.DataFrame({

    "Random_forest": rf_oof_test,

    "Extra_Trees" : et_oof_test,

    "AdaBoost": ab_oof_test,

    "Gradient Boost": gb_oof_test,

    "Support Vector": sv_oof_test

})
final_test.shape
import xgboost as xgb

xg_clf=xgb.XGBClassifier(

n_estimators=2000,

max_depth=4,

min_child_weight=2,

gamma=0.9,

subsample=0.8,

colsample_bytree=0.8,

objective='binary:logistic',

n_thread=-1,

scale_pos_weight=1)
xg_clf.fit(final_train,Y)

final_prediction=xg_clf.predict(final_test)

output=pd.DataFrame({

    "PassengerId": PassengerId,

    "Survived": final_prediction.astype(int)

})

output.head(20)