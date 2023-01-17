import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold   
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

cancer_d=pd.read_csv("../input/data.csv")
cancer_d.info()
cancer_d.describe()
cancer_d.corr()
cancer_d.drop('id',axis=1,inplace=True)
cancer_d.drop('Unnamed: 32',axis=1,inplace=True)
cancer_d.corr()
cancer_d['diagnosis'] = cancer_d['diagnosis'].map({'M':1,'B':0})
cancer_d.corr()
ms.matrix(cancer_d)
fts=list(cancer_d.columns[1:11])

cancer_d.describe()
from sklearn.model_selection import train_test_split

train,test = train_test_split(cancer_d, test_size=0.30, 
                                                    random_state=101)
train.head()
test.head()
def classification_model(model, train,test, predictors, outcome):
  model.fit(train[predictors],train[outcome])
  predictions = model.predict(test[predictors])
  res= metrics.accuracy_score(predictions,test[outcome])
  print(res)
  model.fit(train[predictors],train[outcome]) 

model = RandomForestClassifier()
outcome_var='diagnosis'
classification_model(model,train,test,fts,'diagnosis')

