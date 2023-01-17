import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

tita=pd.read_csv('../input/train.csv')
tita.head()
tita.isnull().sum()
tita[['Pclass','Survived','SibSp','Parch']]=tita[['Pclass','Survived','SibSp','Parch']].astype('str')
tita.shape
tita.info()
tita=tita.drop(['Cabin','Name','Embarked','PassengerId','Ticket'],axis=1)

tita=tita.dropna()

tita.shape
tita.Sex=tita.Sex.replace({'male':1,'female':0})
from sklearn.utils import resample





df_Parch_0 = resample(tita[tita['Parch']=='0'],n_samples=500,replace=True,random_state=1)

df_Parch_1 = resample(tita[tita['Parch']=='1'],n_samples=500,replace=True,random_state=1)



df_Parch_2 = resample(tita[tita['Parch']=='2'],n_samples=500,replace=True,random_state=1)



df_Parch_3 = resample(tita[tita['Parch']=='3'],n_samples=500,replace=True,random_state=1)



df_Parch_4 = resample(tita[tita['Parch']=='4'],n_samples=500,replace=True,random_state=1)



df_Parch_5 = resample(tita[tita['Parch']=='5'],n_samples=500,replace=True,random_state=1)

df_Parch_6 = resample(tita[tita['Parch']=='6'],n_samples=500,replace=True,random_state=1)

#df_survived_1=resample(tita[tita['Survived']=='1'],n_samples=500,replace=True,random_state=1)

#df_survived_1=resample(tita[tita['Survived']=='0'],n_samples=500,replace=True,random_state=1)





tita=pd.concat([df_Parch_0,df_Parch_1,df_Parch_2,df_Parch_3,df_Parch_4,df_Parch_5,df_Parch_6])
from sklearn.model_selection import train_test_split



first,second=train_test_split(tita,test_size=0.3,random_state=22)
#from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

ss.fit(first[['Age','Fare']])

first[['Age','Fare']]=ss.transform(first[['Age','Fare']])

second[['Age','Fare']]=ss.transform(second[['Age','Fare']])

first=pd.get_dummies(first)

second=pd.get_dummies(second)

first=first.drop(['Survived_0','SibSp_0','SibSp_5','Parch_0'],axis=1)

second=second.drop(['Survived_0','SibSp_0','SibSp_5','Parch_0'],axis=1)

print('shape of first and second',first.shape,second.shape)
x=first.drop('Survived_1',axis=1)

y=first['Survived_1']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=22,stratify=y)

## code to find the best model for the dataset



## Before using this template you should find the hyperparameters for the KNN 

## and Decision and random Forest the add those parameters in the model below 



## After selecting the best model we can use KFold CV to check for the bias error and variance error



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from prettytable import PrettyTable

from sklearn import metrics



report= PrettyTable()

report.field_names=['Model name','Accuracy_score','Precision_score','Recall_score','F1_score']





regressor=['LogisticRegression','KNN','DecisionTreeClassifier','RandomForestClassifier']

accuracy=[]

precision=[]

recall=[]

f1_score=[]



for regressor in regressor:

    if regressor=='LogisticRegression':

        model1=LogisticRegression()

        model1.fit(xtrain,ytrain)

        log_pred=pd.DataFrame(model1.predict(xtest))

        #Evaluation metrics

        report.add_row([regressor,

                    metrics.accuracy_score(ytest,log_pred),

                    metrics.precision_score(ytest,log_pred,average='weighted'),

                    metrics.recall_score(ytest,log_pred,average='weighted'),

                    metrics.f1_score(ytest,log_pred,average='weighted')])

        

    elif regressor=='KNN': 

        model2=KNeighborsClassifier(n_neighbors=3)

        model2.fit(xtrain,ytrain)

        knn_pred=model2.predict(xtest)

        #Evaluation metrics

        report.add_row([regressor,

                    metrics.accuracy_score(ytest,knn_pred),

                    metrics.precision_score(ytest,knn_pred,average='weighted'),

                    metrics.recall_score(ytest,knn_pred,average='weighted'),

                    metrics.f1_score(ytest,knn_pred,average='weighted')])

    elif regressor=='DecisionTreeClassifier':

        model3=DecisionTreeClassifier(criterion='entropy')

        model3.fit(xtrain,ytrain)

        dec_pred=model3.predict(xtest)

        #Evaluation metrics

        report.add_row([regressor,

                    metrics.accuracy_score(ytest,dec_pred),

                    metrics.precision_score(ytest,dec_pred,average='weighted'),

                    metrics.recall_score(ytest,dec_pred,average='weighted'),

                    metrics.f1_score(ytest,dec_pred,average='weighted')])

        

    elif regressor=='RandomForestClassifier':

        model4=RandomForestClassifier(criterion='gini')

        model4.fit(xtrain,ytrain)

        random_pred=model4.predict(xtest)

        #Evaluation metrics

        report.add_row([regressor,

                    metrics.accuracy_score(ytest,random_pred),

                    metrics.precision_score(ytest,random_pred,average='weighted'),

                    metrics.recall_score(ytest,random_pred,average='weighted'),

                    metrics.f1_score(ytest,random_pred,average='weighted')])

print(report)

## Code to find the best hyper parameters for the KNN and DecisionTree and Random forest



from sklearn.model_selection import RandomizedSearchCV

## code to find the best model for the dataset





best_par= PrettyTable()

best_par.field_names=['Model name','Best Parameters','Best Score']





regressor=['KNN','DecisionTreeClassifier','RandomForestClassifier']





for regressor in regressor:

    if regressor=='KNN': 

        grid1={'n_neighbors': np.arange(1,50),'p': np.arange(1,50)}

        ran_search1=RandomizedSearchCV(model2,grid1,cv=3)

        ran_search1.fit(xtrain,ytrain)

        best_par.add_row([regressor,

                          ran_search1.best_params_,

                          ran_search1.best_score_])

    elif regressor=='DecisionTreeClassifier':

        

        

        grid2={'criterion':['gini','entropy'],'max_depth': np.arange(2,10),'max_leaf_nodes':np.arange(2,10),'min_samples_leaf':np.arange(2,10)}

        ran_search2=RandomizedSearchCV(model3,grid2,cv=3)

        ran_search2.fit(xtrain,ytrain)

        best_par.add_row([regressor,

                          ran_search2.best_params_,

                          ran_search2.best_score_])

        

    elif regressor=='RandomForestClassifier':

        

        

        grid3={'criterion':['gini','entropy'],'n_estimators':np.arange(1,100),'max_features':np.arange(1,10)}

        ran_search3=RandomizedSearchCV(model4,grid3,cv=3)

        ran_search3.fit(xtrain,ytrain)

        best_par.add_row([regressor,

                          ran_search3.best_params_,

                          ran_search3.best_score_])

        

print(best_par)



from sklearn.model_selection import KFold



kf=KFold(n_splits=10)

random_final=PrettyTable()

random_final.field_names=['Model','Accuracy','Precision','Recall','F1_score']

accuracy=[]

precision=[]

recall=[]

f1_score=[]

for train,test in kf.split(xtrain,ytrain):

    

    xtrain1,xtest1=xtrain.iloc[train,:],xtrain.iloc[test,:]

    ytrain1=ytrain.iloc[train]

    ytest1=ytrain.iloc[test]

    

    model4=RandomForestClassifier(n_estimators=15,max_features=7,criterion='gini')

    model4.fit(xtrain1,ytrain1)

    random_pred=model4.predict(xtest1)

    #Evaluation metrics

    accuracy.append(metrics.accuracy_score(ytest1,random_pred))

    precision.append(metrics.precision_score(ytest1,random_pred,average='weighted'))

    recall.append(metrics.recall_score(ytest1,random_pred,average='weighted'))

    f1_score.append(metrics.f1_score(ytest1,random_pred,average='weighted'))

random_final.add_row([regressor,np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1_score)])

                        





print(random_final)



metrics.confusion_matrix(ytest1,random_pred)
x_sec=first.drop('Survived_1',axis=1)

y_sec=first['Survived_1']
from sklearn.model_selection import KFold



kf=KFold(n_splits=10)

random_final=PrettyTable()

random_final.field_names=['Model','Accuracy','Precision','Recall','F1_score']

accuracy=[]

precision=[]

recall=[]

f1_score=[]

for train,test in kf.split(x_sec,y_sec):

    

    xtrain1,xtest1=x_sec.iloc[train,:],x_sec.iloc[test,:]

    ytrain1=y_sec.iloc[train]

    ytest1=y_sec.iloc[test]

    

    model4=RandomForestClassifier(n_estimators=63,max_features=7,criterion='entropy')

    model4.fit(xtrain1,ytrain1)

    random_pred=model4.predict(xtest1)

    #Evaluation metrics

    accuracy.append(metrics.accuracy_score(ytest1,random_pred))

    precision.append(metrics.precision_score(ytest1,random_pred,average='weighted'))

    recall.append(metrics.recall_score(ytest1,random_pred,average='weighted'))

    f1_score.append(metrics.f1_score(ytest1,random_pred,average='weighted'))

random_final.add_row([regressor,np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1_score)])

                        





print(random_final)



metrics.confusion_matrix(ytest1,random_pred)
test_df=pd.read_csv('../input/test.csv')

test_tita=test_df

test_tita.head()
test_tita1=test_tita
test_tita.isnull().sum()
test_tita.describe()
test_tita['Age']=test_tita['Age'].fillna(test_tita['Age'].median())

test_tita['Fare']=test_tita['Fare'].fillna(test_tita['Fare'].median())



test_tita[['Pclass','SibSp','Parch']]=test_tita[['Pclass','SibSp','Parch']].astype('str')



test_tita.shape



test_tita.info()



test_tita=test_tita.drop(['Cabin','Name','Embarked','PassengerId','Ticket'],axis=1)

test_tita=test_tita.dropna()

test_tita.shape



test_tita.Sex=test_tita.Sex.replace({'male':1,'female':0})
test_tita=pd.get_dummies(test_tita)
test_tita.shape
test_tita.head()
xtest.head()
test_tita.head()
test_tita=test_tita.drop(['SibSp_0','SibSp_5','Parch_0','SibSp_8','Parch_9'],axis=1)



test_tita.shape
ypred=model4.predict(test_tita)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": ypred

    })

submission.head()
submission.to_csv('tita_sub.csv',index=False)