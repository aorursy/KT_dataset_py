# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(10)
test.head(10)
train["Ticket_first"] = [i[0] for i in train["Ticket"]]
train.dtypes
train.groupby("Ticket_first").mean()["Survived"].plot.bar()
test["Ticket_first"] = [i[0] for i in test["Ticket"]]
test["Ticket_first"].value_counts()
def group_as_per_ticket(ticket):
    a=''
    if ticket == '1':
        a="High"
    elif ticket == '2':
        a="Med"
    elif ticket == '3':
        a="Low"
    elif ticket == '4':
        a="Low"
    elif ticket == '5':
        a='Low'
    elif ticket == '6':
        a='Low' 
    elif ticket == '7':
        a='Low'  
    elif ticket == '8':
        a='Low'   
    elif ticket == '9':
        a='Low' 
    elif ticket == 'A':
        a='Low'
    elif ticket == 'C':
        a='Med'
    elif ticket == 'F':
        a='High'
    elif ticket == 'L':
        a='Low'
    elif ticket == 'P':
        a='High'
    elif ticket == 'S':
        a='Med'
    elif ticket == 'W':
        a='Low'
    return a

train["Ticket_type"] = train["Ticket_first"].map(group_as_per_ticket)
test["Ticket_type"] = test["Ticket_first"].map(group_as_per_ticket)
train.groupby('Ticket_type').mean()["Survived"]
#print(train.info())
#print('*'*40)
#print(test.info())
train[train["Embarked"].isnull()]
train.describe()
train[train["Pclass"]==1].mean()["Fare"]
train[(train["Sex"]=='female')&(train["Pclass"]==1)&(train["Survived"]==1)&(train["Fare"]>80)]["Embarked"].value_counts()
train[(train["Sex"]=='female')&(train["Pclass"]==1)&(train["Survived"]==1)&(train["Fare"]==80)]
#train.info()
train = train.drop(index=[61,829])
#train.info()
train[train["Age"].isnull()]["Survived"].value_counts()
train.corr()["Age"]
train.plot.hexbin(x="Age",y="SibSp",gridsize=15)
new_data_train = train.copy()
new_data_test = test.copy()

# make new columns indicating what will be imputed
cols_with_missing_train = (col for col in new_data_train.columns if new_data_train[col].isnull().any())
cols_with_missing_test = (col for col in new_data_test.columns if new_data_test[col].isnull().any())

for col in cols_with_missing_train:
    new_data_train[col + '_was_missing'] = new_data_train[col].isnull()

for col in cols_with_missing_test:
    new_data_test[col + '_was_missing'] = new_data_test[col].isnull()
    
train["Age_was_missing"] = new_data_train["Age_was_missing"]
test["Age_was_missing"] = new_data_test["Age_was_missing"]
train.info()
#print(train.info())
#print('*'*40)
#print(test.info())
test[test["Fare"].isnull()]
test[(test["Pclass"]==3) & (test["Sex"]=='male') & (test["Embarked"]=='S')].mean()["Fare"]
values = {'Fare':12.71887}
test[test["Fare"].isnull()] = test[test["Fare"].isnull()].fillna(value=values)
#print(train.info())
#print('*'*40)
#print(test.info())
train["Cabin"] = train["Cabin"].fillna("N")
test["Cabin"] = test["Cabin"].fillna("N")
train.Cabin = [i[0] for i in train.Cabin]
test.Cabin = [i[0] for i in test.Cabin]
train["Cabin"].value_counts()
train.groupby("Cabin").mean()["Survived"]
def group_as_per_cabin(cabin):
    a=''
    if cabin == 'B':
        a="High"
    elif cabin == 'D':
        a="High"
    elif cabin == 'E':
        a="High"
    elif cabin == 'A':
        a="Medium"
    elif cabin == 'C':
        a='Med'
    elif cabin == 'F':
        a='Med' 
    elif cabin == 'G':
        a='Med'  
    elif cabin == 'N':
        a='Low'   
    elif cabin == 'T':
        a='Low'
    else:
        a='New'
    return a

train["Cabin"] = train["Cabin"].map(group_as_per_cabin)
test["Cabin"] = test["Cabin"].map(group_as_per_cabin)
train.groupby("Cabin").mean()["Survived"]
train = pd.get_dummies(train,columns=["Cabin","Embarked","Age_was_missing","Sex","Ticket_type","Pclass"])
test = pd.get_dummies(test,columns=["Cabin","Embarked","Age_was_missing","Sex","Ticket_type","Pclass"])
train = train.drop(columns=['Ticket','Ticket_first'])
test = test.drop(columns=['Ticket','Ticket_first'])
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1
def accompanied(size):
    a = ''
    if size < 2:
        a="alone"
    elif size < 3:
        a="couple"
    elif size < 4:
        a="small_family"
    elif size < 5:
        a="family"
    elif size < 6:
        a="large_family"
    elif size < 7:
        a="extended_family"
    elif size < 8:
        a="joint_family"
    else:
        a="all"
    return a
    
train["accompanied"] = train["family_size"].map(accompanied)
test["accompanied"] = test["family_size"].map(accompanied)
train = pd.get_dummies(train,columns=["accompanied"])
test = pd.get_dummies(test,columns=["accompanied"])
TrainData = train.copy()
TestData = test.copy()
TrainData["title"] = [i.split(',')[1].split('.')[0] for i in TrainData.Name]
TestData["title"] = [i.split(',')[1].split('.')[0] for i in TestData.Name]
TrainData["title"].value_counts()
TrainData.groupby('title').mean()
TestData["title"].value_counts()
TrainData["title"].dtypes
TrainData["title"] = [i.replace('Ms', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Mlle', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Mme', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Dr', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Col', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Major', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Don', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Jonkheer', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Sir', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Lady', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Capt', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('the Countess', 'rare') for i in TrainData.title]
TrainData["title"] = [i.replace('Rev', 'rare') for i in TrainData.title]

TestData["title"] = [i.replace('Ms', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Mlle', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Mme', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Dr', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Col', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Major', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Dona', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Jonkheer', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Sir', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Lady', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Capt', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('the Countess', 'rare') for i in TestData.title]
TestData["title"] = [i.replace('Rev', 'rare') for i in TestData.title]

TrainData["title"].value_counts()
TestData["title"].value_counts()
TrainData = pd.get_dummies(TrainData,columns=["title"])
TestData = pd.get_dummies(TestData,columns=["title"])
TrainData = TrainData.drop(columns=['PassengerId','Name','Survived'])
TestData = TestData.drop(columns=['PassengerId','Name'])
print(TrainData.info())
print('*'*40)
print(TestData.info())
#print(TrainData.info())
#print('*'*40)
#print(TestData.info())
age_train = TrainData[TrainData["Age"].notnull()]
age_impute_TrainData = TrainData[TrainData["Age"].isnull()]
Y_age_train = age_train["Age"]
X_age_train = age_train.drop(columns=["Age"])

age_test_train = TestData[TestData["Age"].notnull()]
age_impute_TestData = TestData[TestData["Age"].isnull()]
Y_age_test = age_test_train["Age"]
X_age_test = age_test_train.drop(columns=["Age"])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_age_train[["Fare","family_size"]] = sc.fit_transform(X_age_train[["Fare","family_size"]])
age_impute_TrainData[["Fare","family_size"]] = sc.transform(age_impute_TrainData[["Fare","family_size"]])

X_age_test[["Fare","family_size"]] = sc.fit_transform(X_age_test[["Fare","family_size"]])
age_impute_TestData[["Fare","family_size"]] = sc.transform(age_impute_TestData[["Fare","family_size"]])
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100],'gamma':(0.001,'auto')}
svr = SVR()
clf1 = GridSearchCV(svr, parameters,cv=5)
clf2 = GridSearchCV(svr, parameters,cv=5)

clf1.fit(X_age_train,Y_age_train)
clf2.fit(X_age_test,Y_age_test)
age_impute_TrainData = age_impute_TrainData.drop(columns=["Age"])
age_impute_TestData = age_impute_TestData.drop(columns=["Age"])
predicted_train_age = clf1.predict(age_impute_TrainData)
predicted_test_age = clf2.predict(age_impute_TestData)
TrainData.loc[TrainData.Age.isnull(), "Age"] = predicted_train_age
TestData.loc[TestData.Age.isnull(), "Age"] = predicted_test_age
print(TrainData.info())
print('*'*40)
print(TestData.info())
def age_group_fun(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 5: 
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'Young_Adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior_citizen'
    else:
        a = 'old'
    return a
        
## Applying "age_group_fun" function to the "Age" column.
TrainData['age_group'] = TrainData['Age'].map(age_group_fun)
TestData['age_group'] = TestData["Age"].map(age_group_fun)
TrainData["is_child"] = [True if i<18 else False for i in TrainData["Age"]]
TestData["is_child"] = [True if i<18 else False for i in TestData["Age"]]
#TrainData.info()
TrainData = pd.get_dummies(TrainData,columns=["age_group", "is_child"])
TestData = pd.get_dummies(TestData,columns=["age_group", "is_child"])
#print(TrainData.info())
#print('*'*40)
#print(TestData.info())
train.plot.hexbin(x="Fare",y="Survived",gridsize=15)
train.groupby("Survived")["Fare"].describe()
train['Fare'].describe()
def fare_groups(fare):
    a=''
    if fare <= 125:
        a="low"
    elif fare <= 250:
        a="middle"
    elif fare <= 375:
        a="upper middle"
    elif fare <= 500:
        a="upper"
    else:
        a="luxury"
    return a

TrainData["Fare_status"] = TrainData["Fare"].map(fare_groups)
TestData["Fare_status"] = TestData["Fare"].map(fare_groups)
TrainData = pd.get_dummies(TrainData, columns=["Fare_status"])
TestData = pd.get_dummies(TestData, columns=["Fare_status"])
print(TrainData.info())
print('*'*40)
print(TestData.info())
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
TrainData[["Age","Fare","family_size"]] = sc.fit_transform(TrainData[["Age","Fare","family_size"]])
TestData[["Age","Fare","family_size"]] = sc.transform(TestData[["Age","Fare","family_size"]])
#print(TrainData.info())
#print('*'*40)
#print(TestData.info())
from sklearn.decomposition import PCA
pca = PCA(n_components = 15)

pca.fit_transform(TrainData)
pca.transform(TestData)
from sklearn.model_selection import train_test_split
X_train_tot, X_test, y_train_tot, y_test = train_test_split(TrainData, train["Survived"], test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_tot, y_train_tot, test_size=0.2, random_state=42)
from sklearn.metrics import accuracy_score
##############################  Logistic Regression W/O GridSearchCV   ###############################
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train,y_train)
log_reg.fit(X_train,y_train)

Y_test_Pred = log_reg.predict(X_test)
acc_log_reg = accuracy_score(Y_test_Pred,y_test)
print(acc_log_reg)
##############################   Ridge Regression    #####################################
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train,y_train)

Y_test_Pred = ridge.predict(X_test)
Y_test_Pred[Y_test_Pred >= 0.5] = 1
Y_test_Pred[Y_test_Pred < 0.5] = 0
acc_ridge = accuracy_score(Y_test_Pred,y_test)
print(acc_ridge)
###################################  Logistic Regression using GridSearchCV  ######################################
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
parameters = {'solver':('lbfgs','newton-cg','sag','liblinear','saga')}
logR = LogisticRegression(random_state = 0,multi_class = 'ovr')
cross_validation = StratifiedKFold(n_splits=5)
logRGCV = GridSearchCV(logR, parameters,cv = cross_validation)
logRGCV.fit(X_train,y_train)

Y_test_Pred = logRGCV.predict(X_test)
acc_log_reg_grid = accuracy_score(Y_test_Pred,y_test)
print(acc_log_reg_grid)
##################################################### K Neighbours #############################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors':[1,100]}
neigh = KNeighborsClassifier()
cross_validation = StratifiedKFold(n_splits=5)
knn = GridSearchCV(neigh, parameters,cv=cross_validation)
knn.fit(X_train,y_train)

Y_test_Pred = knn.predict(X_test)
acc_knn = accuracy_score(Y_test_Pred,y_test)
print(acc_knn)
#################################################### Random Forest ###################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
parameter_grid = {
             'max_depth' : [4, 6, 8],
             'n_estimators': [10, 50,100],
             'max_features': ['sqrt', 'auto', 'log2'],
             'min_samples_split': [0.001,0.003,0.01],
             'min_samples_leaf': [1, 3, 10],
             'bootstrap': [True,False],
             }
forest = RandomForestClassifier()
cross_validation = StratifiedKFold(n_splits=5)
rdclf = GridSearchCV(forest,scoring='accuracy',param_grid=parameter_grid,cv=cross_validation)
rdclf.fit(X_train,y_train)

Y_test_Pred = rdclf.predict(X_test)
acc_forest = accuracy_score(Y_test_Pred,y_test)
print(acc_forest)
#################################################### AdaBoost ########################################
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state = 0)
parameters= {'n_estimators':[10,1000]}
cross_validation = StratifiedKFold(n_splits=5)
adaB = GridSearchCV(ada, parameters, cv = cross_validation)
adaB.fit(X_train,y_train)

Y_test_Pred = adaB.predict(X_test)
acc_adaboost = accuracy_score(Y_test_Pred,y_test)
print(acc_adaboost)
########################### Decision Tree ######################################
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(random_state=0)
parameter_grid = {
             'max_depth' : [4, 6, 8],
             'max_features': ['sqrt', 'auto', 'log2'],
             'min_samples_split': [0.001,0.003,0.01],
             'min_samples_leaf': [1, 3, 10],
             }
cross_validation = StratifiedKFold(n_splits=5)
dcT = GridSearchCV(dct,scoring='accuracy',param_grid=parameter_grid,cv=cross_validation)
dcT.fit(X_train,y_train)

Y_test_Pred = dcT.predict(X_test)
acc_dTree = accuracy_score(Y_test_Pred,y_test)
print(acc_dTree)
############################ SVC ##############################################
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100],'gamma':(0.001,'auto')}
svc = SVC()
cross_validation = StratifiedKFold(n_splits=5)
svcClf = GridSearchCV(svc, parameters,cv=cross_validation)
svcClf.fit(X_train,y_train)

Y_test_Pred = svcClf.predict(X_test)
acc_svc = accuracy_score(Y_test_Pred,y_test)
print(acc_svc)
########################### Gradient Boosting Classifier################################################
from sklearn.ensemble import GradientBoostingClassifier
gdb = GradientBoostingClassifier()
clf= GradientBoostingClassifier()
parameter_grid = {
             'max_depth' : [4, 6, 8],
             'n_estimators': [10, 50,100],
             'max_features': ['sqrt', 'auto', 'log2'],
             'min_samples_split': [0.001,0.003,0.01],
             'min_samples_leaf': [1, 3, 10]
             }
cross_validation = StratifiedKFold(n_splits=5)
gdB = GridSearchCV(gdb,scoring='accuracy',param_grid=parameter_grid,cv=cross_validation)
gdB.fit(X_train,y_train)

Y_test_Pred = gdB.predict(X_test)
acc_gboost = accuracy_score(Y_test_Pred,y_test)
print(acc_gboost)
########################### XG Boost ###########################################
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)

Y_test_Pred = xgb.predict(X_test)
acc_xgB = accuracy_score(Y_test_Pred,y_test)
print(acc_xgB)
########################## Extra Trees Classifier #################################
from sklearn.ensemble import ExtraTreesClassifier
trees = ExtraTreesClassifier()
parameter_grid = {
             'max_depth' : [4, 6, 8],
             'n_estimators': [10, 50,100],
             'max_features': ['sqrt', 'auto', 'log2'],
             'min_samples_split': [0.001,0.003,0.01],
             'min_samples_leaf': [1, 3, 10],
             'bootstrap': [True,False],
             }
cross_validation = StratifiedKFold(n_splits=5)
extT = GridSearchCV(trees,scoring='accuracy',param_grid=parameter_grid,cv=cross_validation)
extT.fit(X_train,y_train)

Y_test_Pred = extT.predict(X_test)
acc_exTree = accuracy_score(Y_test_Pred,y_test)
print(acc_exTree)
########################## Voting Classifier ####################################
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

svc = SVC()
logreg = LogisticRegression()
randomforest = RandomForestClassifier()
gradient = GradientBoostingClassifier()
dectree = DecisionTreeClassifier()
knn = KNeighborsClassifier()
XGBClassifier = XGBClassifier()
ExtraTreesClassifier = ExtraTreesClassifier()

votClf = VotingClassifier(estimators=[
   ('logreg',logreg), 
  ('random_forest', randomforest),
   ('gradient_boosting', gradient),
  ('decision_tree',dectree),  
   ('knn',knn),
   ('XGB Classifier', XGBClassifier),
   ('ExtraTreesClassifier', ExtraTreesClassifier)], voting='soft')

votClf.fit(X_train,y_train)

Y_test_Pred = votClf.predict(X_test)
acc_voting = accuracy_score(Y_test_Pred,y_test)
print(acc_voting)
#y_test_pred = clf.predict(X_val)
#y_test_pred[y_test_pred >= 0.5] = 1
#y_test_pred[y_test_pred < 0.5] = 0
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test_pred,y_val)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Logistic Regression with Grid', 'XGBoost', 
              'Gradient Boosting', 'AdaBoost', 'ExtraTrees Classifier', 
               'Ridge Regression', 'Voting Classifier', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log_reg, 
              acc_forest, acc_log_reg_grid, acc_xgB, 
              acc_gboost, acc_adaboost, acc_exTree, acc_ridge, acc_voting, acc_dTree]})
models.sort_values(by='Score', ascending=False)
Y_pred = svcClf.predict(TestData)
Y_pred = pd.Series(Y_pred)
FINAL = pd.concat([test["PassengerId"],Y_pred],axis=1)
FINAL.to_csv("submission.csv", encoding='utf-8', index=False)