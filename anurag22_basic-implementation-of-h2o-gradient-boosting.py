import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandasql import sqldf
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
data_train_all = pd.read_csv("../input/train.csv")
data_test_all=pd.read_csv("../input/test.csv")
data_test_all_pred=pd.read_csv("../input/test.csv")      ## This dataframe will be used in the last . This Dataframe doesn't have much significance
data_train_all.head()
data_train_all.info()
data_test_all.info()
import re
data_train_all['Title']= data_train_all.Name.apply(lambda a:re.search(' ([A-Z][a-z]+)\.',a).group(1))
data_test_all['Title'] = data_test_all.Name.apply(lambda a:re.search(' ([A-Z][a-z]+)\.',a).group(1))
data_train_all['Title'].head()
data_train_all.Title.value_counts()
data_test_all.Title.value_counts()
data_train_all['Title'] = data_train_all['Title'].replace(['Mlle','Ms'],'Miss')  
data_train_all['Title'] = data_train_all['Title'].replace('Mme','Mrs')
data_train_all['Title'] = data_train_all['Title'].replace(['Capt','Col','Major'],'Army')
data_train_all['Title'] = data_train_all['Title'].replace(['Countess','Don','Dona','Jonkheer','Lady','Sir'],'Noble')
data_test_all['Title'] = data_test_all['Title'].replace(['Mlle','Ms'],'Miss')  
data_test_all['Title'] = data_test_all['Title'].replace('Mme','Mrs')
data_test_all['Title'] = data_test_all['Title'].replace(['Capt','Col','Major'],'Army')
data_test_all['Title'] = data_test_all['Title'].replace(['Countess','Don','Dona','Jonkheer','Lady','Sir'],'Noble')
data_train_all['Title'].value_counts()
data_test_all['Title'].value_counts()
data_train_all=data_train_all.drop(columns='Ticket')  ## we could have also used (inplace= True) and then we need not do data_train _all= data_train_all.drop(xxxxxx)
data_test_all=data_test_all.drop(columns='Ticket')
f= lambda x: str(x)[0]
data_train_all.Cabin=data_train_all.Cabin.apply(f)
data_train_all['Cabin']=data_train_all['Cabin'].replace(['T'],'n')
data_test_all.Cabin=data_test_all.Cabin.apply(f)
data_train_all.Cabin.value_counts()
data_test_all.Cabin.value_counts()
data_train_all.groupby(['Cabin'])['Survived'].sum()
data_train_all.groupby(['Sex'])['Age'].median()
data_train_all.Age.median()
data_train_all.groupby(['Pclass']).Age.mean()
## we will be replaceing Age by median value
data_train_all.Age=data_train_all.Age.fillna(data_train_all.Age.median())
data_test_all.Age=data_test_all.Age.fillna(data_train_all.Age.median())
# Binning by quantile
#data_train_all.Age=pd.qcut(data_train_all.Age, q=4, labels=False)
# Binning by fixed Interval
data_train_all.Age=pd.cut(data_train_all.Age, bins=[0,20,40,60,80,100],right=True, labels=False, retbins=0, include_lowest=1)
data_test_all.Age=pd.cut(data_test_all.Age, bins=[0,20,40,60,80,100],right=True, labels=False, retbins=0, include_lowest=1)
data_test_all.Age.hist()
data_train_all.Fare.min()
data_train_all.Fare=pd.cut(data_train_all.Fare, bins=[0,10,20,30,40,50,100,600],right=True, labels=False, retbins=0, include_lowest=1)
data_train_all.Fare.value_counts()
data_train_all.Fare.hist(bins=20)
data_test_all.Fare=pd.cut(data_test_all.Fare, bins=[0,10,20,30,40,50,100,600],right=True, labels=False, retbins=0, include_lowest=1)
data_test_all.Fare.hist(bins=20)
data_train_all.info()
data_test_all.info()
### Missing Value Treatment of Column Fare in Test Set
data_test_all.Fare.fillna(0,inplace=True)
data_test_all.Fare.value_counts()
data_train_all.Embarked.value_counts()
data_train_all.Embarked.fillna('S', inplace=True)
data_train_all.drop(columns=['Name','PassengerId'],inplace=True)
data_train_all.head()
data_test_all.drop(columns=['Name','PassengerId'],inplace=True)
data_test_all.info()
type(data_train_all.Title[2])
## Create the dummy variables
data_train_all=pd.get_dummies(data_train_all,drop_first=False)  ## In case of categorical variable you dont need to drop one dummy variable.
data_test_all=pd.get_dummies(data_test_all,drop_first=False) 
data_train_all.head()
data_test_all.head()
data_train_all.info()
## Column Family
data_train_all['Family']=data_train_all['SibSp']+data_train_all['Parch']
data_test_all['Family']=data_test_all['SibSp']+data_test_all['Parch']
data_train_all.drop(columns=['SibSp','Parch'],inplace=True)
data_test_all.drop(columns=['SibSp','Parch'],inplace=True)
data_test_all.Family.value_counts()
data_test_all.info()
data_train_all.Fare=data_train_all.Fare.astype(float)
data_train_all.info()
import h2o
h2o.init()
data_train_h2o=h2o.H2OFrame(data_train_all)
data_test_h2o=h2o.H2OFrame(data_test_all)
type(data_test_h2o)
data_train_h2o['Survived']=data_train_h2o['Survived'].asfactor()    ## Converting Target Variable as Factor
from h2o.estimators.gbm import H2OGradientBoostingEstimator  # import gbm estimator
model = H2OGradientBoostingEstimator(## more trees is better if the learning rate is small enough 
  ## here, use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            

  ## smaller learning rate is better (this is a good value for most datasets, but see below for annealing)
  learn_rate = 0.01,                                                         

  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 

  ## sample 80% of rows per tree
  sample_rate = 0.8,                                                       

  ## sample 80% of columns per split
  col_sample_rate = 0.8,                                                   

  ## fix a random number generator seed for reproducibility
  seed = 1234,                                                             

  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10, nfolds=5, max_depth=3)   ## Instantiating the class
model.train(x=data_train_h2o.names[1:],y=data_train_h2o.names[0], training_frame=data_train_h2o, model_id="GBM_Titanic",
            validation_frame=data_train_h2o)
dir(model)
model.cross_validation_metrics_summary()
model.params
f=model.predict(test_data=data_test_h2o)
f=f.as_data_frame()             ## Converting Predicted Results to Python Dataframe
submission_H2O = pd.DataFrame({'PassengerId':data_test_all_pred['PassengerId'],'Survived':f['predict']})
## submission_H2O.to_csv('D:/Titanic/Titanic Predictions_H2O.csv',index=False)
