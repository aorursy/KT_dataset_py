import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
test.head()
train.isnull().sum().sort_values(ascending=False)
train.rez_esc.fillna(train.rez_esc.mean(),inplace=True)
train.v18q1.fillna(train.v18q1.mean(),inplace=True)
train.v2a1.fillna(train.v2a1.mean(),inplace=True)
train.meaneduc.fillna(train.meaneduc.mean(),inplace=True)
train.SQBmeaned.fillna(train.SQBmeaned.mean(),inplace=True)
test.isnull().sum().sort_values(ascending=False)
test.rez_esc.fillna(test.rez_esc.mean(),inplace=True)
test.v18q1.fillna(test.v18q1.mean(),inplace=True)
test.v2a1.fillna(test.v2a1.mean(),inplace=True)
test.meaneduc.fillna(test.meaneduc.mean(),inplace=True)
test.SQBmeaned.fillna(test.SQBmeaned.mean(),inplace=True)
test.dtypes
intcols=train.select_dtypes(include=["int64"])
floatcols=train.select_dtypes(include=["float64"])
objectcols=train.select_dtypes(include=["object"])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
intcols1=intcols.apply(le.fit_transform)
objectcols1=objectcols.apply(le.fit_transform)
train1=pd.concat([intcols1,objectcols1,floatcols],axis=1)
intcols=test.select_dtypes(include=["int64"])
floatcols=test.select_dtypes(include=["float64"])
objectcols=test.select_dtypes(include=["object"])
intcols1=intcols.apply(le.fit_transform)
objectcols1=objectcols.apply(le.fit_transform)
test1=pd.concat([intcols1,objectcols1,floatcols],axis=1)
test1.dtypes
x=train1.drop(["Target"],axis=1)
y=train1.Target
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfcmodel=rfc.fit(x,y)
rfcmodel.score(x,y)
predict=rfcmodel.predict(test1)
predict
accuracy=round(rfcmodel.score(x,y)*100,2)
accuracy
sub=pd.read_csv('../input/sample_submission.csv')
sub.to_csv('sample_submission.csv',index=False)

