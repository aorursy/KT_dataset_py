# Loading Modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Data Loading 

dh = pd.read_csv('../input/air-quality-data-in-india/city_hour.csv')
dh.head(5)
dd = pd.read_csv('../input/air-quality-data-in-india/city_day.csv')
dd.head(5)
# Basic Dataset Characteristics

print(dh.shape)
print(dd.shape)

dh_col = ['City', 'Datetime','NO', 'NO2', 'NOx', 'NH3', 'CO','SO2', 'O3']
dd_col = ['City', 'Date','NO', 'NO2', 'NOx', 'NH3', 'CO','SO2', 'O3']
dh1 = dh[dh_col]
dd1 = dd[dd_col]
print(dh1.shape)
print(dd1.shape)
print(dh1.info())
print(dd1.info())
# Deal with Missing values

dh_col1 = ['NO', 'NO2', 'NOx', 'NH3', 'CO','SO2', 'O3']
dd_col1 = ['NO', 'NO2', 'NOx', 'NH3', 'CO','SO2', 'O3']

for i in dh_col1:
    a = dh1[i].median()
    dh1[i].replace(np.nan , a,inplace =  True)

for i in dd_col1:
    b = dd1[i].median()
    dd1[i].replace(np.nan , b,inplace =  True)   
      
        
# Basic Distribution Data plotting

sns.distplot(dh1['NO'], hist = False,label="NO_Hour")
sns.distplot(dd1['NO'], hist = False,label="NO_Day")
sns.distplot(dh1['NO2'], hist = False,label="NO2_Hour")
sns.distplot(dd1['NO2'], hist = False,label="NO2_Day")
sns.distplot(dh1['NH3'], hist = False,label="NH3_Hour")
sns.distplot(dd1['NH3'], hist = False,label="NH3_Day")
sns.distplot(dh1['NOx'], hist = False,label="NOx_Hour")
sns.distplot(dd1['NOx'], hist = False,label="NOx_Day")
plt.figure(figsize=(10,10))
sns.distplot(dh1['CO'], hist = False,label="CO_Hour")
sns.distplot(dd1['CO'], hist = False,label="CO_Day")
sns.distplot(dh1['SO2'], hist = False,label="SO_Hour")
sns.distplot(dd1['SO2'], hist = False,label="SO_Day")
sns.distplot(dh1['O3'], hist = False,label="O3_Hour")
sns.distplot(dd1['O3'], hist = False,label="O3_Day")
# Dealing with city Ahmedabad
dh_Ahem = dh1[dh1['City']=='Ahmedabad']
print(dh_Ahem.shape)
sns.distplot(dh_Ahem['NO'], hist = False,label="NO_Hour")
corrMatrix = dh_Ahem.corr()
#print(dh_Ahem['NO'].corr())
sns.heatmap(corrMatrix, annot=True)
plt.show()
# Pair plots
sns.pairplot(dh_Ahem)
# Box Plots

plt.figure(figsize=(30,20))
sns.boxplot(y='NO', x='City', data=dh1)
dh_Ahem.sort_values(by=['Datetime'] , inplace  =  True)
dh_Ahem.tail(5)
# Creating Seasonal Data  
df = pd.DataFrame(columns = ['Season'])
le = len(dh_Ahem)
for i in range(0,le):
    a  =  dh_Ahem.loc[i,'Datetime']
    l = a.split('-')
    if(l[1]=='12' or l[1]=='01' or l[1]=='02'):
        df.loc[i,'Season'] = 'Winter'
    elif(l[1]=='03' or l[1]=='04' or l[1]=='05'):
        df.loc[i,'Season'] = 'Summer'
    elif(l[1]=='06' or l[1]=='07' or l[1]=='08'):
        df.loc[i,'Season'] = 'Rainy'
    elif(l[1]=='09' or l[1]=='10' or l[1]=='11'):
        df.loc[i,'Season'] = 'Autumn'        
print(df.shape)

print(dh_Ahem.shape)
dh_AhemS = pd.concat([dh_Ahem, df], axis =1)
print(dh_AhemS.shape)
# Sesonaltiy plots
plt.figure(figsize=(10,10))
sns.boxplot(y='NO', x='Season', data=dh_AhemS)
plt.figure(figsize=(10,10))
sns.violinplot(y="NO", x="Season", data=dh_AhemS)
dh_AhemS.describe()
dh_AhemS = pd.get_dummies(dh_AhemS , columns=['Season'], prefix = ['Season'])
dh_AhemS.head(5)
X  =  dh_AhemS[['NO2','NOx' , 'NH3' , 'CO' , 'SO2' ,'O3' ,'Season_Autumn' , 'Season_Rainy' , 'Season_Summer',
                'Season_Winter']]
y = dh_AhemS[['NO']]
# Time based splitting 

from sklearn.model_selection import TimeSeriesSplit

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tscv = TimeSeriesSplit(n_splits=4)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

     #To get the indices 
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(X_train.shape)    
print((X_train.shape[0]/(X_train.shape[0] + X_test.shape[0]))*100)    
    
# XGBoost Regressor
from sklearn.preprocessing import PolynomialFeatures
import xgboost 
'''for i in range(6,7):
    pf = PolynomialFeatures(i)
    Xtrain_poly = pf.fit_transform(X_train)
    Xtest_poly = pf.fit_transform(X_test)

    gm  = xgboost.XGBRegressor(colsample_bytree=0.35,
                 gamma=0,                 
                 learning_rate=0.004,
                 max_depth=3,
                 n_estimators=3000,                                                                  
                 reg_alpha=0.60,
                 reg_lambda=0.80,
                 subsample=0.7,
                 seed=42)
    gm.fit(Xtrain_poly,y_train)
    y_pred = gm.predict(Xtest_poly)'''
'''print(y_test.shape)
y_test = np.squeeze(np.asarray(y_test)).flatten()
y_pred = np.squeeze(np.asarray(y_pred)).flatten()
print(y_pred.shape)
print(y_test.shape)'''


# Result is 0.26056434501208664
import math

def rmsle(ypred, ytest):
    #assert len(ytest) == len(ypred)
    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest))**2))

#print(rmsle(y_test, y_pred))
# Linear Regression
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(X_train,y_train)
y_pred = lreg.predict(X_test)
print(y_pred.shape)
y_test = np.squeeze(np.asarray(y_test)).flatten()
y_pred = np.squeeze(np.asarray(y_pred)).flatten()
print(y_pred)
print(y_test)
# Calculating R2 
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
sns.residplot(y_pred, y_test, lowess=True, color="g")
# Distribution Plot of Predicted Values and Actual Values
sns.distplot(y_pred, hist = False,label="Pred")
sns.distplot(y_test, hist = False,label="Test")