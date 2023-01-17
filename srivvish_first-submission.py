from fastai.imports import *
from fastai.structured import *
import pandas as pd
import numpy as np
import os
os.listdir("../input/train/")
df=pd.read_csv('../input/train/Train.csv',low_memory=False,parse_dates=["saledate"])
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(df.head())
df=df.sort_values('saledate').copy()
## It is necessary to sort time series data.
#If your dataset has a time piece in it (as is in Blue Book competition), 
#you would likely want to predict future prices/values/etc. 
#What Kaggle did was to give us data representing a particular date range in the training set, 
#and then the test set presented a future set of dates that wasn’t represented in the training set. 
#So we need to create a validation set that has the same properties:
df['SalePrice']=np.log(df['SalePrice'])
df['saledate'].head()
df1=df.copy() ## making a copy of df
#convert object into categorical
for i in df.select_dtypes(include='object').columns:
    df[i]=df[i].astype('category')
    
## Find columns with more than 80% missing values
#s=pd.DataFrame((df.isnull().sum().sort_values(ascending=False))/df.shape[0],columns=['Percentage'])
#s.reset_index(inplace=True)
#replaceC=s[s['Percentage'] < 0.80]['index']
## replaced missing columns
#df2=df[np.array(replaceC)]
df2=df.copy()
#Fill numeric columns with mean
#intC=df2.select_dtypes(include=['int64','float64']).isna().sum().index
d=pd.DataFrame(df2.select_dtypes(include=['int64','float64']).isna().sum(),columns=['Count'])
intC=d[d['Count']>0].index
for c in np.array(intC):
    df2[c+'_na'] = df2[c].isnull()
    df2[c]=df2[c].fillna(df[c].mean())
df.shape,df2.shape
# Fill cateogorical columns with cat.codes
intC=df2.select_dtypes(include='category').columns
#intC=d2[d2['Count']>0].index
for c in intC:
    df2[c]=df2[c].cat.codes+1 # codes of missing data is -1
## Check missing
df2.isna().sum().sum()
display_all(df2.head())
add_datepart(df2, 'saledate')
##Above function is from fastai.structured.py
X=df2.drop(['SalePrice'],axis=1)
Y=df2['SalePrice']
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(X,Y)
rfr.score(X,Y)
import math
def rmse(x,y): return math.sqrt(((x-y)**2).mean())
rmse(rfr.predict(X), Y)

# We can't use the below function as we don't want to split the dataset randomly. 
#from sklearn.model_selection import train_test_split
#X_Train,X_Test,Y_Train,Y_Test= train_test_split(X,Y,test_size=1200,random_state=10)
#rr=RandomForestRegressor()
def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_valid=12000
n_trn = len(df2)-n_valid
raw_train, raw_valid = split_vals(df, n_trn)
X_Train, X_Test = split_vals(X, n_trn)
Y_Train, Y_Test = split_vals(Y, n_trn)
print(X_Train.shape,X_Test.shape,Y_Train.shape,Y_Test.shape)
def score(model,X_Train,Y_Train,X_Test,Y_Test):
    if hasattr(model, 'oob_score_'): 
        print("Training Score : "+ str(model.score(X_Train,Y_Train))  + " Test Score : " + str(model.score(X_Test,Y_Test)) + " OOb score : " + str(model.oob_score_) + " RMSE Training : " + str(rmse(model.predict(X_Train), Y_Train)) + " RMSE Test : " + str(rmse(model.predict(X_Test), Y_Test)) )
    else:
        print("Training Score : "+ str(model.score(X_Train,Y_Train))  + " Test Score : " + str(model.score(X_Test,Y_Test)) + " RMSE Training : " + str(rmse(model.predict(X_Train), Y_Train)) + " RMSE Test : " + str(rmse(model.predict(X_Test), Y_Test)))
    
rfr=RandomForestRegressor()
%time rfr.fit(X_Train,Y_Train)
score(rfr,X_Train,Y_Train,X_Test,Y_Test)
set_rf_samples(20000) ## Changes Scikit learn's random forests to give each tree a random sample of n random rows.
m = RandomForestRegressor(n_estimators=20,n_jobs=-1, oob_score=True)
%time m.fit(X_Train, Y_Train)
score(m,X_Train,Y_Train,X_Test,Y_Test)
reset_rf_samples() ## reset Random forest samples
# Using min_samples_leaf and max_features
m = RandomForestRegressor(n_estimators=40,n_jobs=-1, oob_score=True, min_samples_leaf=3,max_features=0.5)
%time m.fit(X_Train, Y_Train)
score(m,X_Train,Y_Train,X_Test,Y_Test)
m = RandomForestRegressor(n_estimators=60,n_jobs=-1, oob_score=True, min_samples_leaf=5,max_features=0.5)
%time m.fit(X_Train, Y_Train)
score(m,X_Train,Y_Train,X_Test,Y_Test)