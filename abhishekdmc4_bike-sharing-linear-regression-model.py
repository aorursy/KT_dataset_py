import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

#pd.set_option('display.max_columns',125)
#pd.set_option('display.max_rows',200)
sharing=pd.read_csv("../input/bike-sharing/day.csv")
sharing.head(10)
sharing.shape
sharing.info()
sharing.describe()
# just to be sure checking null values
sharing.isnull().sum()
# no null found
sharing.nunique().sort_values()
var_drop=['casual','registered','dteday','instant','atemp'] #variables to drop
sharing=sharing.drop(var_drop,axis=1)
sharing.head()
sharing.weathersit=sharing.weathersit.map({1: 'Clear', 2: 'Mist + Cloudy' , 3: 'Light Snow', 4: 'Heavy Rain'})
sharing.season=sharing.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})
sharing.weekday=sharing.weekday.map({0:'Sunday', 1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thrusday',5:'Friday',6:'Saturday'})
sharing.mnth=sharing.mnth.map({1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September'
                              ,10:'October',11:'November',12:'December'})
sharing.head()
sharing.info()
#all the non-binary columns are now object type
# for continous Numericalvariables
num_vars=['temp','hum','windspeed','cnt']
L=len(num_vars)
plt.figure(figsize=(10,10))
for i in range(1,L+1):
    plt.subplot((L//2)+1,2,i)
    sharing[num_vars[i-1]].plot.box()
#Looks like windspeed has some outliers
sharing.windspeed.quantile([0.0,0.25,0.50,0.75,0.95,1])
# 0.75 and 1.00 quantile as high gap, but as the unit of windspeed
# is not given, keeping this as it is.
# for Categorical variables
cat_vars=['season','yr','mnth','holiday','weekday','workingday','weathersit']
plt.figure(figsize=(25,25))
L=len(cat_vars)
for i in range(1,L):
    plt.subplot(L//2,2,i)
    sharing[cat_vars[i-1]].value_counts().plot.bar()
    plt.title(cat_vars[i-1])
    
# holiday and workingday has a very imbalanced ratio
# cnt against Categorical variables
cat_vars=['season','yr','mnth','holiday','weekday','workingday','weathersit']
plt.figure(figsize=(30,20))
L=len(cat_vars)
for i in range(1,L):
    plt.subplot(L//2,2,i)
    #sharing[cat_vars[i-1]].value_counts().plot.bar()
    sns.boxplot(x=cat_vars[i-1],y='cnt',data=sharing)    
# cnt against numerical variables
num_vars=['temp','hum','windspeed']
plt.figure(figsize=(10,12))
L=len(num_vars)
for i in range(1,L+1):
    plt.subplot((L//2)+1,2,i)
    sns.scatterplot(x='cnt',y=num_vars[i-1],data=sharing)
sharing.corr()
plt.figure(figsize=(10,12))
sns.heatmap(data=sharing.corr(),annot=True,cmap='YlGnBu')
# dummyfying all the categorical variables
to_dummy_vars=['season','mnth','weekday','weathersit']
for i in to_dummy_vars:
    buffer=pd.get_dummies(sharing[i],drop_first=True)
    sharing=pd.concat([sharing,buffer],axis=1)
    sharing=sharing.drop(i,axis=1)
sharing.info()
#all the categorical variables converted into object/uint8
#now we can proceed
#splitting Train, Test in 70:30
sharing_train, sharing_test = train_test_split(sharing,train_size=0.7,random_state=100)
scaler=MinMaxScaler()
num_vars=['temp','hum','windspeed','cnt']
sharing_train[num_vars]=scaler.fit_transform(sharing_train[num_vars])
sharing_train.head()
# shape of test train split
print(sharing_train.shape)
print(sharing_test.shape)
#seperating dependent and independent variables 
y_train=sharing_train.pop('cnt')
X_train=sharing_train
lm=LinearRegression()
lm.fit(X_train,y_train)
#using RFE to reduce to 15 columns
rfe=RFE(lm,15)
rfe=rfe.fit(X_train,y_train)
#checking the list RFE has selected
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#reducing non-impartant columns in training data
col=X_train.columns[rfe.support_]
X_train_rfe=X_train[col]
X_train_rfe.head()
X_train_sm=sm.add_constant(X_train_rfe)
lr_model=sm.OLS(y_train,X_train_sm).fit()
lr_model.summary()
#P-value of holiday is very big
#but also checking VIF

VIF= pd.DataFrame()
VIF['features']=X_train_rfe.columns
VIF['vif']=[vif(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
VIF.vif=round(VIF.vif,2)
VIF.sort_values(by='vif',ascending=False,inplace=True)
VIF
# p-value shows that holiday is insignificant
# as p-value removal takes higher precedence than VIF
# dropping holiday and modelling again
X_train_rfe=X_train_rfe.drop('holiday',axis=1)
X_train_sm=sm.add_constant(X_train_rfe)

lr_model=sm.OLS(y_train,X_train_sm).fit()
lr_model.summary()
# all p-vals are less than 0.05
# let's check VIF now

VIF= pd.DataFrame()
VIF['features']=X_train_rfe.columns
VIF['vif']=[vif(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
VIF.vif=round(VIF.vif,2)
VIF.sort_values(by='vif',ascending=False,inplace=True)
VIF
# dropping hum as it has high vif and modelling again
X_train_rfe=X_train_rfe.drop('hum',axis=1)
X_train_sm=sm.add_constant(X_train_rfe)

lr_model=sm.OLS(y_train,X_train_sm).fit()
lr_model.summary()
# let's check VIF now

VIF= pd.DataFrame()
VIF['features']=X_train_rfe.columns
VIF['vif']=[vif(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
VIF.vif=round(VIF.vif,2)
VIF.sort_values(by='vif',ascending=False,inplace=True)
VIF
# as workingday has higher VIF than 2
# dropping it and re-modelling
X_train_rfe=X_train_rfe.drop('workingday',axis=1)
X_train_sm=sm.add_constant(X_train_rfe)

lr_model=sm.OLS(y_train,X_train_sm).fit()
lr_model.summary()
# let's check VIF now

VIF= pd.DataFrame()
VIF['features']=X_train_rfe.columns
VIF['vif']=[vif(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
VIF.vif=round(VIF.vif,2)
VIF.sort_values(by='vif',ascending=False,inplace=True)
VIF
# as Saturday has high p-value than 0.05
# dropping it and re-modelling
X_train_rfe=X_train_rfe.drop('Saturday',axis=1)
X_train_sm=sm.add_constant(X_train_rfe)

lr_model=sm.OLS(y_train,X_train_sm).fit()
lr_model.summary()
# let's check VIF now

VIF= pd.DataFrame()
VIF['features']=X_train_rfe.columns
VIF['vif']=[vif(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
VIF.vif=round(VIF.vif,2)
VIF.sort_values(by='vif',ascending=False,inplace=True)
VIF
# temp has a VIF greater than 5
# but as we know from EDA it's a important variable
# in buisness terms temp has higher correlation with cnt
# so dropping the next highest VIF ie windspeed and re-modelling
X_train_rfe=X_train_rfe.drop('windspeed',axis=1)
X_train_sm=sm.add_constant(X_train_rfe)

lr_model=sm.OLS(y_train,X_train_sm).fit()
lr_model.summary()
# let's check VIF now

VIF= pd.DataFrame()
VIF['features']=X_train_rfe.columns
VIF['vif']=[vif(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
VIF.vif=round(VIF.vif,2)
VIF.sort_values(by='vif',ascending=False,inplace=True)
VIF
#this is the final Model and its R2 score
print(lr_model.summary())

y_train_pred = lr_model.predict(X_train_sm)
#***************************************************************************************
r2_score=round(r2_score(y_true= y_train,y_pred= y_train_pred),4)
#***************************************************************************************
print(f"\n R^2 of train set is: {r2_score}")
y_train_pred = lr_model.predict(X_train_sm)
res = y_train - y_train_pred
sns.distplot(res)
plt.show()
# Scaling numerical variables
num_vars=['temp','hum','windspeed','cnt']
sharing_test[num_vars]=scaler.transform(sharing_test[num_vars])
sharing_test.head()
# Making X and Y test sets
y_test = sharing_test.pop('cnt')
X_test = sharing_test
X_test.head()
# dropping variables from X_test which were dropped in Model building
vars=['yr', 'temp', 'spring', 'summer', 'winter', 'July',
       'September', 'Sunday', 'Light Snow', 'Mist + Cloudy']
X_test=X_test[vars]
X_test_sm=sm.add_constant(X_test)
X_test_sm.head()
y_test_pred = lr_model.predict(X_test_sm)
#***************************************************************************************
from sklearn.metrics import r2_score
r2_score_test=round(r2_score(y_true= y_test,y_pred= y_test_pred),4)
#***************************************************************************************
print(f"R^2 of test set is: {r2_score_test}")
Best_fit_line=lr_model.params.reset_index()
Best_fit_line.columns=['Variable','Coefficient']
Best_fit_line.sort_values(by='Coefficient',ascending=False)