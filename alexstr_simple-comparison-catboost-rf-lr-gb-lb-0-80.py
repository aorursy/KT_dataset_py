# The goal of this kernel is the just a comparison of methods of most popular classification methods(no stacking)

# methods 

# 1) LogisticRegression

# 2) RandomForest

# 3) GradientBoosting

# 4) CatBoost
%matplotlib inline

from sklearn.feature_extraction import DictVectorizer as DV

from matplotlib import pyplot as plt

import sklearn as sk

import pandas as pd

import numpy as np

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

from sklearn import cross_validation

from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.grid_search import GridSearchCV

data_train=pd.read_csv('../input/train.csv')

print ("size of train - ",data_train.shape)

data_test=pd.read_csv('../input/test.csv')

print ("size of test -",data_test.shape)

all_data=pd.concat([data_train,data_test])

print ("size of all data -",all_data.shape)

data_train.head(5)
# lets check how many absent values do we have

data_train.isnull().sum(axis=0)
print(data_train[data_train["Survived"]==1].describe())
print(data_train[data_train["Survived"]==0][data_train["Sex"]=="male"].describe())
#Show how dependent survival of Age and Pclass

data_train.plot(x='Survived',y='Pclass',kind='hist')

data_train.plot(x='Survived',y='Age',kind='hist')
# delet unuseful all_data 

all_data_del=all_data.drop(["Name","Ticket","PassengerId","Cabin","Embarked"],axis=1)
# delete unuseful  train data

data_train_p=data_train.drop(["Name","Ticket","PassengerId","Cabin","Embarked"],axis=1)
# delete unuseful  test data

y_test=data_test.drop(["Name","Ticket","PassengerId","Cabin","Embarked"],axis=1)
# calculate mean Age by all data set

mean_Age = all_data_del["Age"].mean()

print (mean_Age,all_data_del["Age"].shape)
# calculate mean Fare by all data set

mean_Fare = all_data_del["Fare"].mean()

print (mean_Fare,all_data_del["Fare"].shape)
#give everything to a numeric format and change NaN on mean

# for train data

#Sex

people={"male":1,"female":0}

data_train_p["Sex"]=data_train_p["Sex"].apply(lambda s: people.get(s))



#Age

data_train_p["Age"]=data_train_p["Age"].fillna(mean_Age)
# for TEST data

#Sex

y_test["Sex"]=y_test["Sex"].apply(lambda s: people.get(s))



#Age

y_test["Age"]=y_test["Age"].fillna(mean_Age)



#Fare

y_test["Fare"]=y_test["Fare"].fillna(mean_Fare)
# For all data

#Sex

all_data_del["Sex"]=all_data["Sex"].apply(lambda s: people.get(s))



#Age

all_data_del["Age"]=all_data_del["Age"].fillna(mean_Age)



#Fare

all_data_del["Fare"]=all_data_del["Fare"].fillna(mean_Fare)
#check test data

for i in y_test:

    print (y_test[i].shape, i)
y_test.dropna().shape
#check trein data

for i in data_train_p:

    print (data_train_p[i].shape, i)
data_train_p.dropna().shape
data_train_p.head(5)
# Sex / Survived

sns.pairplot(data_train_p, vars=["Sex"], hue="Survived", size=5,dropna=True)
# Fare / Survived

sns.pairplot(data_train_p, vars=["Fare"], hue="Survived", size=5,dropna=True)
sns.pairplot(data_train_p, vars=["Age", "Pclass","Sex","Fare"], hue="Survived", dropna=True)
y_train=data_train_p["Survived"]
x_train=data_train_p.drop(["Survived"],axis=1)
c_v=5

vol={}
# methods

# 1) LogisticRegression

# 2) RandomForest

# 3) GradientBoosting

# 4) CatBoost
LRestimator=LogisticRegression(random_state=0)

#from the box

scores_lr = cross_validation.cross_val_score(LRestimator, x_train, y_train, cv = c_v)
print (scores_lr.mean())
%%time

param_grid_LR={"C":[0.01,0.05,0.1],"penalty":["l1","l2"]}

grid_LR=GridSearchCV(LRestimator,param_grid_LR,cv=c_v)

grid_LR.fit(x_train,y_train)
vol["LogisticRegression"] = grid_LR.best_score_

print ("best params:",grid_LR.best_params_)

print ("best score:",grid_LR.best_score_)

RFestimator=RandomForestClassifier(random_state=0,n_jobs=-1)

#from the box

scores_rf = cross_validation.cross_val_score(RFestimator,x_train,y_train,cv=c_v)
print (scores_rf.mean())
%%time

param_gird_RF={"max_depth":[2,3,5,10,12,15],"n_estimators":[70,80,100,150,200,300,400,500],"criterion":["entropy","gini"]}

grid_RF=GridSearchCV(RFestimator,param_gird_RF,cv=c_v)

grid_RF.fit(x_train,y_train)
vol["RandomForest"]=grid_RF.best_score_

print ("best params:",grid_RF.best_params_)

print ("best score:",grid_RF.best_score_)
GBestimator=GradientBoostingClassifier(random_state=0)

#from the box

scores_boost=cross_validation.cross_val_score(GBestimator,x_train,y_train,cv=c_v)
scores_boost.mean()
%%time

param_grid_GB={"max_depth":[2,3,5],"n_estimators":[50,80,100,200,250,300,350,450,600,500],"learning_rate": [0.01, 0.02, 0.05]}

grid_GB=GridSearchCV(GBestimator,param_grid_GB,cv=c_v)

grid_GB.fit(x_train,y_train)
vol["Gboost"]=grid_GB.best_score_

print ("best parpams:",grid_GB.best_params_)

print ("best score:",grid_GB.best_score_)
# Pay attention catboost himself processes categorical features

data_train.dtypes
data_train.fillna(-999, inplace=True)
data_test.fillna(-999, inplace=True)
X = data_train.drop('Survived', axis=1)

y = data_train.Survived
categorical_features_indices = np.where(X.dtypes != np.float)[0]
Catmodel = CatBoostClassifier(

    custom_loss=['Accuracy'],

    random_seed=42

)
from sklearn.model_selection import train_test_split



X_train_c, X_validation, y_train_c, y_validation = train_test_split(X, y, train_size=0.8, random_state=1234)
# It is worth noting that the model has a pleasant visualization :)

Catmodel.fit(

    X_train_c, y_train_c,

    cat_features=categorical_features_indices,

    eval_set=(X_validation, y_validation),

     verbose=True,

    plot=True

)
cv_data = cv(

    Catmodel.get_params(),

    Pool(X, label=y, cat_features=categorical_features_indices),

)
vol["Catboost"]=np.max(cv_data['Accuracy_test_avg'])

print( 'Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(

    np.max(cv_data['Accuracy_test_avg']),

    cv_data['Accuracy_test_stddev'][np.argmax(cv_data['Accuracy_test_avg'])],

    np.argmax(cv_data['Accuracy_test_avg'])

))
Catmodel = CatBoostClassifier(

    iterations=193,

    eval_metric='Accuracy',

    random_seed=42

)

Catmodel.fit(

    X_train_c, y_train_c,

    cat_features=categorical_features_indices,

    eval_set=(X_validation, y_validation),

)
pd.DataFrame.from_dict(data = vol, orient='index').plot(kind='bar', legend=False)
print ("LogisticRegression AUC:",roc_auc_score(y_train,grid_LR.predict_proba(x_train)[:,1]))
fpr,tpr,thr=roc_curve(y_train,grid_LR.predict_proba(x_train)[:,1])

plt.plot(fpr,tpr)

plt.grid()

plt.xlabel("false positive rate")

plt.ylabel("true positive rate")
print ("RandomForest AUC:",roc_auc_score(y_train,grid_RF.predict_proba(x_train)[:,1]))
fpr,tpr,thr=roc_curve(y_train,grid_RF.predict_proba(x_train)[:,1])

plt.plot(fpr,tpr)

plt.grid()

plt.xlabel("false positive rate")

plt.ylabel("true positive rate")
print ("Gboost AUC:",roc_auc_score(y_train,grid_GB.predict_proba(x_train)[:,1]))
fpr,tpr,thr=roc_curve(y_train,grid_GB.predict_proba(x_train)[:,1])

plt.plot(fpr,tpr)

plt.grid()

plt.xlabel("false positive rate")

plt.ylabel("true positive rate")
print ("Catboost AUC:",roc_auc_score(y_train,Catmodel.predict_proba(X)[:,1]))
fpr,tpr,thr=roc_curve(y_train,Catmodel.predict_proba(X)[:,1])

plt.plot(fpr,tpr)

plt.grid()

plt.xlabel("false positive rate")

plt.ylabel("true positive rate")
predict = Catmodel.predict(data_test).astype(int)
submission = pd.DataFrame({

    "PassengerId": data_test["PassengerId"],

    "Survived": predict})
submission.to_csv("titanic-submission.csv", index=False)
# LB 0.80861 to improve LB change other parameters catboost.

# I'm going to show it in the next kernel.

# Thanks for reading 