import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

import warnings

warnings.filterwarnings("ignore")



sns.set(style='white', context='notebook', palette='deep')
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head(10)
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head(10)
train_data.info()
train_data.describe()
train_data.shape
test_data.info()
test_data.describe()
test_data.shape
train_data.isnull().sum()
test_data.isnull().sum()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train_data.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test_data.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
train_data.hist(bins=20,figsize=(20,15))

plt.show()
test_data.hist(bins=20,figsize=(20,15))

plt.show()
sex = sns.barplot(x="Sex",y="Survived",data=train_data)

sex = sex.set_ylabel("Survival Probability")
train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()
sns.barplot(x='Embarked',y='Survived',data=train_data,color='blue',ci=None);
plt.figure(figsize=(10,10))

sns.heatmap(train_data.corr(), annot=True)# vmin = 0, vmax = +1)

plt.show()
sibps = sns.factorplot(x="SibSp",y="Survived",ci=None,data=train_data,kind="bar", size = 6 , 

palette = "muted")

sibps.despine(left=True)

sibps = sibps.set_ylabels("survival probability")
age =sns.distplot(train_data['Age'][(train_data['Survived']==0) &(train_data['Age'].notnull())],color='Red',kde=False)

age =sns.distplot(train_data['Age'][(train_data['Survived']==1) &(train_data['Age'].notnull())],color='Green',kde=False)

age.legend(['Dead','Survived'])

age.set_ylabel('Frequency');
fill_age = list(train_data["Age"][train_data["Age"].isnull()].index)



for i in fill_age : # looping over the age null indexes

    age_med = train_data["Age"].median() # finding the median age

    age_pred = train_data["Age"][((train_data['SibSp'] == train_data.iloc[i]["SibSp"]) & (train_data['Parch'] == train_data.iloc[i]["Parch"]) & (train_data['Pclass'] == train_data.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        train_data['Age'].iloc[i] = age_pred

    else :

        train_data['Age'].iloc[i] = age_med
fill_age = list(test_data["Age"][test_data["Age"].isnull()].index)



for i in fill_age :

    age_med = test_data["Age"].median()

    age_pred = test_data["Age"][((test_data['SibSp'] == test_data.iloc[i]["SibSp"]) & (test_data['Parch'] == test_data.iloc[i]["Parch"]) & (test_data['Pclass'] == test_data.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        test_data['Age'].iloc[i] = age_pred

    else :

        test_data['Age'].iloc[i] = age_med
train_data['Embarked']=train_data['Embarked'].fillna('S')
train_data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
test_data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
train_data["Cabin"]=train_data["Cabin"].map(lambda x: 0 if pd.isnull(x) else 1)
test_data["Cabin"]=test_data["Cabin"].map(lambda x: 0 if pd.isnull(x) else 1)
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female":1})
test_data["Sex"] = test_data["Sex"].map({"male": 0, "female":1})
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train_data.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test_data.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
train_data.drop(columns=['Name','Ticket'],inplace=True)

test_data.drop(columns=['Name','Ticket'],inplace=True)
train_data.dtypes
test_data.dtypes
train_data.head()
X_train= train_data[['Pclass','Sex','Age','Cabin','Embarked']]

y_train= train_data['Survived']

X_test =test_data[['Pclass','Sex','Age','Cabin','Embarked']]

scaler = StandardScaler()

_x = scaler.fit(X_train)



_tx = scaler.fit(X_test)

X_train_scaled = pd.DataFrame(_x.transform(X_train),columns=X_train.columns)
X_test_scaled = pd.DataFrame(_tx.transform(X_test),columns=X_test.columns)
knn_model= KNeighborsClassifier()



knn_model.fit(X_train_scaled,y_train) 
knn_model.score(X_train_scaled,y_train)
cross_val_score(knn_model,X_train_scaled,y_train).mean()
random_state = 101

 

model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'SVC', 

              'RandomForestClassifier', 'XGBClassifier', 'ExtraTreesClassifier'

              , 'GradientBoostingClassifier','AdaBoostClassifier','GaussianNB','SVM']

models = [ ('LogisticRegression',LogisticRegression(random_state=random_state)),

          ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),

          ('SVC', SVC(random_state=random_state)),

          ('RandomForestClassifier',RandomForestClassifier(random_state=42)),

          ('ExtraTreesClassifier',ExtraTreesClassifier(random_state=random_state)),

          ('GradientBoostingClassifier',GradientBoostingClassifier(random_state=random_state)),

          ('AdaBoostClassifier',AdaBoostClassifier(random_state=random_state)),

          ('GaussianNB',GaussianNB()),

          ('SVM ',svm.SVC())

         ]

model_accuracy = []

for m,model in models:

    print (m , ':')

    model.fit(X_train_scaled, y_train)

    accuracy = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()

    model_accuracy.append(accuracy)

    print(accuracy)

    print('\n')
kfold = StratifiedKFold(n_splits=10)
random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train_scaled, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(X_train_scaled,y_train)



ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [None],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(X_train_scaled,y_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [None],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose = 1)



gsRFC.fit(X_train_scaled,y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train_scaled,y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
# predicted = gsGBC.predict(test_data[['Pclass','Sex','Age','Cabin','Embarked']])
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsSVMC.fit(X_train_scaled,y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
pred1 = gsSVMC.predict(X_test_scaled)
test_data.shape
test_data['Survived'] = pred1
subm = test_data[['PassengerId','Survived']]
subm.to_csv('submit.csv',index=False)