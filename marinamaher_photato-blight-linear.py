import pandas as pd
import os
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
df_total = pd.DataFrame()

for file in os.listdir(r"../input/potatoblight"):
    if file.endswith('.xls'):
        excel_file = pd.ExcelFile("../input/potatoblight/" + str(file))
        sheets = excel_file.sheet_names
        for sheet in sheets: # loop through sheets inside an Excel file
            df = excel_file.parse(sheet_name = sheet)
            df_total = df_total.append(df)          
            
print(df_total.shape)
print("NA ? ",df_total.isnull().values.any())
df_total.head(1)
lastDS=np.zeros(df_total.shape[0])
last3DS=np.zeros(df_total.shape[0])

for i in range(df_total.shape[0]):
    lastDS[i]=df_total["Daily blight obsrv."].iloc[i-1]
    last3DS[i]=(df_total["Daily blight obsrv."].iloc[i-1]+df_total["Daily blight obsrv."].iloc[i-2]+df_total["Daily blight obsrv."].iloc[i-3])/3
    
lastDS[0]=0
last3DS[0]=last3DS[1]=last3DS[2]=0

df_total["LastDS"]=lastDS
df_total["Last3DS"]=last3DS
df_total["month"]= pd.DatetimeIndex(df_total['Date']).month
y=df_total["Daily blight obsrv."]
X=df_total.drop(['Date','jd','Tmin>8','Tmax<25','Rain','Count of IP requirements',"Daily blight obsrv."], axis=1)
X.head(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) 
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(X_train.shape, y_train.shape)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100 / np.mean(y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) 

print(X_train.shape, y_train.shape)

iso = IsolationForest(contamination=0.1,random_state=42)
yhat = iso.fit_predict(X_train)
mask = yhat != -1

X_train, y_train = X_train[mask], y_train[mask]

print(X_train.shape, y_train.shape)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(X_train.shape, y_train.shape)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100 / np.mean(y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) 
print(X_train.shape, y_train.shape)

ee = EllipticEnvelope(contamination=0.1,random_state=42)
yhat = ee.fit_predict(X_train)
mask = yhat != -1
X_train, y_train = X_train[mask], y_train[mask]
print(X_train.shape, y_train.shape)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100 / np.mean(y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) 
print(X_train.shape, y_train.shape)

lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
mask = yhat != -1
X_train, y_train = X_train[mask], y_train[mask]

print(X_train.shape, y_train.shape)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100 / np.mean(y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) 
print(X_train.shape, y_train.shape)


ee = OneClassSVM(nu=0.08)
yhat = ee.fit_predict(X_train)

mask = yhat != -1
X_train, y_train = X_train[mask], y_train[mask]
print(X_train.shape, y_train.shape)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100 / np.mean(y_test))

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
plt.scatter(X_pca,y, alpha=0.8 ,color='b')

ee = OneClassSVM(nu=0.08)
yhat = ee.fit_predict(X_pca)

mask = yhat != -1
X_train,y2 = X_pca[mask],y[mask]
plt.scatter(X_train,y2, alpha=0.8,color='r')

y=df_total["Daily blight obsrv."]
X=df_total.drop(['Date','jd','Tmin>8','Tmax<25','Rain','Count of IP requirements',"Daily blight obsrv."], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=4752) 
#iso = IsolationForest(contamination=0.01,random_state=42)
#yhat = iso.fit_predict(X_train)

ee = OneClassSVM(nu=0.09)
yhat = ee.fit_predict(X_train)

#ee = EllipticEnvelope(contamination=0.01,random_state=42)
#yhat = ee.fit_predict(X_train)

#lof = LocalOutlierFactor()
#yhat = lof.fit_predict(X_train)

mask = yhat != -1
X_train, y_train = X_train[mask], y_train[mask]

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100 / np.mean(y_test))
y=df_total["Daily blight obsrv."]
X=df_total.drop(['Date','jd','Tmin>8','Tmax<25','Rain','Count of IP requirements',"Daily blight obsrv."], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) 

r1 = LinearRegression()
r2 = RandomForestRegressor(n_estimators=9, random_state=1)
r3 = ExtraTreesRegressor(n_estimators=9, random_state=1)

er = VotingRegressor([('lr', r1), ('rf', r2), ('ef', r3)])
er.fit(X_train, y_train)
y_pred = er.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100 / np.mean(y_test))
col_names = df_total.columns.values.tolist()
col_names.pop(0)

df_total2=df_total.drop(['Date'], axis=1)

scaler = MinMaxScaler()
df_total2 = scaler.fit_transform(df_total2)
df_total2=pd.DataFrame(df_total2)
df_total2.columns =col_names
y=df_total2["Daily blight obsrv."]
X=df_total2.drop(['jd','Tmin>8','Tmax<25','Rain','Count of IP requirements',"Daily blight obsrv."], axis=1)
X.head(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) 
er.fit(X_train, y_train)
y_pred = er.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100 / np.mean(y_test))

ee = OneClassSVM(nu=0.17)
yhat = ee.fit_predict(X)

mask = yhat != -1
X2, y2 = X[mask], y[mask]

cv = cross_val_score(regressor, X2, y2, cv=7,scoring=make_scorer(metrics.mean_squared_error))
np.sqrt(np.sum(cv)/len(cv))*100/ np.mean(y_test)

