# importing important modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
test.info()
train.info()
def explore(feature):
    pd.crosstab(train.Survived,train[feature]).plot(kind="bar",stacked=True)
# it shows that survival of women is more probable
explore("Sex")
# people embarked from south-ampton have more probablity to die 

explore("Embarked")
# only using the first letter of the respectice cabins
train["Cabin"]=train.Cabin.str[0]
# extraction of titles in the name columns
# function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# to get the titles from the name feature
for i in [train,test]:
    i["titles"]=i.Name.apply(get_title)
# assigning rare_titles and some corrections to be made
for i in [train,test]:
    i.loc[i.titles=="Ms","titles"]="Miss"
    i.loc[i.titles=="Mme","titles"]="Mrs"
    i.loc[i.titles=="Mlle","titles"]="Miss"
    i.loc[i.titles.isin(['Master', 'Dr', 'Rev', 'Col', 'Major','Countess', 'Lady', 'Sir', 'Jonkheer', 'Don',"Dona", 'Capt']),"titles"]="rare_titles"
# some feature engineering 
for i in [train,test]:
    i["fsize"]=i.SibSp+i.Parch
# treatment of null values for Age feature using similar case imputation
for i in [train,test]:
    i.Age.fillna(i.groupby("titles")["Age"].transform("median"),inplace=True)
    
train.head() 
train.info()
train.loc[train.Embarked.isnull()]
# finding the most similar case for first missing embarked instance
train.loc[(train.Sex==1)&(train.Age<=40)&(train.titles=="Miss")&(train.fsize==0)&(train.Cabin=="B")].Embarked.value_counts()
# from the results S is more probable to be the embarked place
train.loc[61,"Embarked"]="S"
# finding the most similar case for second missing embarked instance
train.loc[(train.Sex==1)&(train.Age<=40)&(train.titles=="Mrs")&(train.fsize==0)&(train.Cabin=="B")].Embarked.value_counts()
# from the results S is more probable to be the embarked place
train.loc[829,"Embarked"]="C"
test.loc[test.Fare.isnull()]
test.loc[(test.Pclass==3)&(test.Sex==0)&(test.Embarked=="S")&(test.Age>40)]
# filling the missing value for Fare columns in test-set from the above results
test.Fare.fillna(7.87,inplace=True)
test.info()
# now we are only left with filling the missing values of the Cabins,let's do it.
# importing random forest module fron sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

# data preparation for model building imputation for missing Cabin feature values

train1=train.drop(['PassengerId',"SibSp","Parch",'Name',"Survived", 'Ticket'],axis=1)
train1=pd.get_dummies(train1,columns=["Embarked","titles","Sex"],drop_first=True)
train_X=train1.loc[train1.Cabin.notnull()].drop("Cabin",axis=1)
train_Y=train1.loc[train1.Cabin.notnull()].Cabin
# initialization of the grid search that will help us to find the optimum values of the different parameters

k_range=[90,100,120,150]
lead_=[2,3,4]
params=dict(n_estimators=k_range,min_samples_leaf=lead_)
rf=RandomForestClassifier()
grid=GridSearchCV(rf,params,cv=20,scoring="accuracy")

grid.fit(train_X,train_Y)
grid.grid_scores_
rf=grid.best_estimator_
rf.fit(train_X,train_Y)
# preparing the data-set for prediction of missing Cabin feature values in Training data-set
train_tX=train1.loc[train1.Cabin.isnull()].drop("Cabin",axis=1)
train_tX.head()
# predicting the null values for the training set
pred=rf.predict(train_tX)
#new_train=train.loc[train.Cabin.isnull()]
train_tX["Cabin"]=pred

# final combination of both the sets of null Cabin values and non-null cabin values
new_train=pd.concat([train1.loc[train1.Cabin.notnull()],train_tX],axis=0).sort_index()
# combining the Survived column which was initially dropped to predict Cabin null values
new_train["Survived"]=train.Survived
# finding Cabin values for test data
# data preparation for model building imputation
test1=test.loc[test.Cabin.notnull()].drop(['PassengerId',"SibSp","Parch",'Name', 'Ticket'],axis=1)
test1=pd.get_dummies(test1,columns=["Embarked","titles","Sex"],drop_first=True)
test_X=test1.drop("Cabin",axis=1)
test_Y=test1.Cabin
test_X.info()
# for building the model for predicting the missing values of the Cabin feature of the test-set, i have combined the training and test data-set to increase the data-set for it.
large_X=pd.concat([train_X,test_X],axis=0).sort_index()
large_y=pd.concat([train_Y,test_Y],axis=0).sort_index().str[0]


k_range=[90,100,120,150]
lead_=[2,3,4]
params=dict(n_estimators=k_range,min_samples_leaf=lead_)
rf=RandomForestClassifier()
grid=GridSearchCV(rf,params,cv=20,scoring="accuracy")

grid.fit(large_X,large_y)
grid.grid_scores_
rf=grid.best_estimator_
rf.fit(large_X,large_y)
# Combination of test data-set of both null and non-null values for Cabin feature after missing value treatment
test2=test.loc[test.Cabin.isnull()].drop(['PassengerId',"SibSp","Parch",'Name', 'Ticket'],axis=1)
test2=pd.get_dummies(test2,columns=["Embarked","titles","Sex"],drop_first=True)
test2_X=test2.drop("Cabin",axis=1)

pred=rf.predict(test2_X)
#new_test=test.loc[test.Cabin.isnull()]
test2_X["Cabin"]=pred
# final combination of both the test sets of null Cabin values and non-null cabin values
new_test=pd.concat([test1,test2_X],axis=0)
new_test.Cabin=new_test.Cabin.str[0]
new_test=new_test.sort_index()

# final preparation for the training of the model
newtrain_Y=new_train.Survived
newtrain_X=new_train.drop("Survived",axis=1)
newtrain_X=pd.get_dummies(newtrain_X,columns=["Cabin"],drop_first=True)
#train33_X=train33_X.drop(["Name","PassengerId","Ticket"],axis=1)
newtrain_X
# using Grid search we can find the optimized value of the alogrithm parameters

k_range=[300,250,200]
lead_=[2,3,4,5,6,7]
params=dict(n_estimators=k_range,min_samples_leaf=lead_)
rf=RandomForestClassifier()
grid=GridSearchCV(rf,params,cv=50,scoring="accuracy")
grid.fit(newtrain_X,newtrain_Y)
grid.grid_scores_
# initialization of the predictive model with best modelling parameters
rf=grid.best_estimator_
rf.fit(newtrain_X,newtrain_Y)
newtest_X=pd.get_dummies(new_test,columns=["Cabin"],drop_first=True)
newtest_X["Cabin_T"]=0
newtest_X=newtest_X.sort_index()
# final prediction for the test data-set given
pred=rf.predict(newtest_X)

# preparing file for submission
pd.DataFrame([test.PassengerId,pd.Series(pred)]).T
