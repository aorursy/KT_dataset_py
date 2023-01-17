import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv')
testdf=pd.read_csv('../input/test.csv')
print('-------Training data info------------>')
print(df.info())
print('-------Testing data info------------>')
print(testdf.info())
#lets get all the numerical features.
numCols=df.select_dtypes(include=['float64']).columns #Getting the column name of type float64
numdata=df[numCols].copy()
dataCorr=numdata.corr()

plt.figure(figsize=(20,8))
sns.heatmap(dataCorr,annot=True)
plt.show()
#Lets find out the features where correlation is geater than 0.75.
col_corr=set()
for i in range(len(dataCorr.columns)):
    for j in range(i):
        if abs(dataCorr.iloc[i,j])>.75:
            col_corr.add(dataCorr.columns[i])
print(col_corr) 
#len(df)
#plt.figure(figsize=(20,6))
#plt.hist(df.loss,bins=100)
#plt.xlim(df.loss.min())
#plt.show()
#[col for col in df.columns if df[col].isnull().sum()>0]
print(df.isnull().sum().max())
print(testdf.isnull().sum().max())
#vt=VarianceThreshold(threshold=0)
#vt.fit(df)
colNames=df.select_dtypes(include='object').columns
colNames
df=pd.get_dummies(df,columns=colNames,drop_first=True)
testdf=pd.get_dummies(testdf,columns=colNames,drop_first=True)
print('-------Training data info------------>')
print(df.info())
print('-------Testing data info------------>')
print(testdf.info())
#After creating dummies variable for train and test data column numbers are not matching.
#We need to remove the uncommon columns
#Lets find out.
trainCols=df.columns
testCols=testdf.columns
#find test columns not present in train data
c1=[col for col in testCols if col not in trainCols]
#find train columns not present in test data
c2=[col for col in trainCols if col not in testCols]
print('Number of columns not present in train data-->',len(c1))
print('Number of columns not present in test data-->',len(c2))
y=df['loss']
#Lets delete 37 columns from test data and 75 columns from train data
testdf=testdf.drop(columns=c1, axis=1)
df=df.drop(columns=c2,axis=1)
print(testdf.shape)
print(df.shape)
#vt=VarianceThreshold(threshold=.01)
#vt.fit(df)
#print(collections.Counter(vt.get_support()))
#print(collections.Counter(vt.fit(testdf).get_support()))

#if there is no false it means there is no counstant features.
#Dropping the highly correlated features from training data and test data.
testdf=testdf.drop(list(col_corr)+['id'],axis=1)
X=df.drop(list(col_corr)+['id'],axis=1)
#y=df['loss']
print(X.shape)
print(testdf.shape)
#spliting data.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.1,random_state=1)
y_train=np.log1p(y_train)
ss=StandardScaler()
ss.fit(X_train)
#rb=RobustScaler(quantile_range=(5.0,80.0))
#rb.fit(X_train)
#sfm=SelectFromModel(LinearRegression(),threshold='median')
#sfm.fit(ss.transform(X_train),y_train)
#collections.Counter(sfm.get_support())
#Selected features are 228
#X_train_tran=sfm.transform(X_train)
#X_test_tran=sfm.transform(X_test)
#testdata_tran=sfm.transform(testdf)
from xgboost import XGBRegressor
xg=XGBRegressor(n_estimators=1000,learning_rate=.1,n_jobs=-1)
xg.fit(ss.transform(X_train),y_train)
pred=xg.predict(ss.transform(X_test))
print(mean_absolute_error(y_test,np.expm1(pred)))
#mae 1164.0271622 highly correlated feature with each other ('cont10', 'cont11', 'cont6', 'cont9', 'cont13', 'cont12').
#mae 1164.14458746 only with StandardScaler
#mae 1164.64735593 with RobustScaler with quantile_range=(20.0, 80.0)
#mae 1451.61430774 with 228 feature selection.
#mae 1335.64901659 with 519 feature selection
testPredict=xg.predict(ss.transform(testdf))
sample=pd.read_csv('../input/sample_submission.csv')
sample['loss']=np.expm1(testPredict)
sample.head()
sample.to_csv('Submission.csv',index=False)
