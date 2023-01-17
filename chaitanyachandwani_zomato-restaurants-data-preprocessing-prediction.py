import numpy as np
import pandas as pd
pd.set_option("display.max_column",80)
pd.set_option("display.max_row",1000)
pd.set_option('display.width', 1000)
import matplotlib as plt
data=pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')
data.isnull().sum()
data.dtypes
data['rate'].unique()         #"NEW" and "-" are dropped later
zomato=data.drop(['phone','url','dish_liked'],axis=1)
zomato
import re
zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].str.replace('\W', '').astype(float)
zomato                        # replacing commas (1,000) into empty string (1000) and coverting into float
zomato['approx_cost(for two people)'].dtypes
zomato.isnull().sum()
zomato['rest_type'] = zomato['rest_type'].fillna(zomato['rest_type'].mode()[0])
zomato['cuisines'] = zomato['cuisines'].fillna(zomato['cuisines'].mode()[0])
zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].fillna(zomato['approx_cost(for two people)'].mode()[0])
zomato
zomato=zomato[zomato['rate']!="NEW"]#       #drop rows having "NEW" and "-" values
zomato=zomato[zomato['rate']!="-"]

zomato['rate']=zomato['rate'].str.replace('\W5','').astype(float) #using regex
zomato
zomato.isnull().sum()
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy="mean")
imp
zomato.iloc[:,4:5]=imp.fit_transform(zomato.iloc[:,4:5])
zomato
zomato.isnull().sum()
zomato.dropna(inplace=True)
zomato.isnull().any()
zomato.info()
X=zomato.iloc[:,[0,1,2,3,5,6,7,8,9,10,11,12,13]]
X
y=zomato.iloc[:,4]
y
from sklearn.preprocessing import LabelEncoder
categorical_feature_mask = X.dtypes==object    #filters categorical features using boolean mask
categorical_cols = X.columns[categorical_feature_mask].tolist()
categorical_cols
lb=LabelEncoder()
X[categorical_cols] = X[categorical_cols].apply(lambda x: lb.fit_transform(x))
X
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.ensemble import BaggingRegressor
br=BaggingRegressor(n_estimators=100,random_state=42)
model=br.fit(Xtrain,ytrain)
ypred=model.predict(Xtest)
ypred
from sklearn.metrics import r2_score
r2_score(ytest,ypred)
