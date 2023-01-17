import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%matplotlib inline
dftrain=pd.read_csv('../input/radiation/train.csv')
dftest=pd.read_csv('../input/radiation/test.csv')
dftrain.head(15)
#renaming the wind direction column
dftrain = dftrain.rename(columns={'WindDirection(Degrees)': 'wd'})
dftest = dftest.rename(columns={'WindDirection(Degrees)': 'wd'})
dftrain.shape
dftrain.info()
dftrain.isna().sum()
dftrain.corr()["Radiation"]
dftrain.drop(['UNIXTime'],axis=1,inplace=True)
dftest.drop(['UNIXTime'],axis=1,inplace=True)
dftrain.info()
#Filling float with mean and string with median
dftrain['Temperature'].fillna(dftrain['Temperature'].mean(), inplace=True)
dftrain['Pressure'].fillna(dftrain['Pressure'].mean(), inplace=True)
dftrain['Humidity'].fillna(dftrain['Humidity'].mean(), inplace=True)
dftrain['wd'].fillna(dftrain['wd'].mean(), inplace=True)
dftrain['Speed'].fillna(dftrain['Speed'].mean(), inplace=True)
dftrain['Temperature'].fillna(dftrain['Temperature'].mean(), inplace=True)
dftrain['Radiation'].fillna(dftrain['Radiation'].mean(), inplace=True)
#Filling Objects(strings) with the mode
dftrain['TimeSunRise'].fillna(dftrain['TimeSunRise'].mode()[0], inplace=True)
dftrain['TimeSunSet'].fillna(dftrain['TimeSunSet'].mode()[0], inplace=True)
dftrain['Time'].fillna(dftrain['Time'].mode()[0], inplace=True)
dftrain['Data'].fillna(dftrain['Data'].mode()[0], inplace=True)
dftrain.isna().sum()
#doing the same in test dftest
dftest['Temperature'].fillna(dftest['Temperature'].mean(), inplace=True)
dftest['Pressure'].fillna(dftest['Pressure'].mean(), inplace=True)
dftest['Humidity'].fillna(dftest['Humidity'].mean(), inplace=True)
dftest['wd'].fillna(dftest['wd'].mean(), inplace=True)
dftest['Speed'].fillna(dftest['Speed'].mean(), inplace=True)
dftest['Temperature'].fillna(dftest['Temperature'].mean(), inplace=True)

#Filling Objects(strings) with the mode
dftest['TimeSunRise'].fillna(dftest['TimeSunRise'].mode()[0], inplace=True)
dftest['TimeSunSet'].fillna(dftest['TimeSunSet'].mode()[0], inplace=True)
dftest['Time'].fillna(dftest['Time'].mode()[0], inplace=True)
dftest['Data'].fillna(dftest['Data'].mode()[0], inplace=True)
dftest.isna().sum()

sns.boxplot(x=dftrain["Temperature"])
Q1 = dftrain['Temperature'].quantile(0.25)
Q3 = dftrain['Temperature'].quantile(0.75)
IQR = Q3 - Q1   
filter = (dftrain['Temperature'] >= Q1 - 1.5 * IQR) & (dftrain['Temperature'] <= Q3 + 1.5 *IQR)
dftrain=dftrain[(dftrain['Temperature'] >= Q1 - 1.5 * IQR) & (dftrain['Temperature'] <= Q3 + 1.5 *IQR)]
sns.boxplot(x=dftrain["Temperature"])
sns.boxplot(x=dftrain["Pressure"])
Q1 = dftrain['Pressure'].quantile(0.25)
Q3 = dftrain['Pressure'].quantile(0.75)
IQR = Q3 - Q1     

filter = (dftrain['Pressure'] >= Q1 - 1.5 * IQR) & (dftrain['Pressure'] <= Q3 + 1.5 *IQR)
dftrain=dftrain[(dftrain['Pressure'] >= Q1 - 1.5 * IQR) & (dftrain['Pressure'] <= Q3 + 1.5 *IQR)]
sns.boxplot(x=dftrain["Pressure"])
sns.boxplot(x=dftrain["Humidity"])
#Replacing humidity outliers with mean.
Q1 = dftrain['Humidity'].quantile(0.25)
Q3 = dftrain['Humidity'].quantile(0.75)
IQR = Q3 - Q1     

a=Q1 - 1.5 * IQR
a

dftrain.loc[dftrain.Humidity <a , 'Humidity'] = dftrain['Humidity'].mean()
sns.boxplot(x=dftrain["Humidity"])
sns.boxplot(x=dftrain["wd"])
#Replacing Winddir outliers with mean.
Q1 = dftrain['wd'].quantile(0.25)
Q3 = dftrain['wd'].quantile(0.75)
IQR = Q3 - Q1     

a=Q3 + 1.5 * IQR
a

dftrain.loc[dftrain.wd>a , 'wd'] = dftrain['wd'].mean()
sns.boxplot(x=dftrain["wd"])
sns.boxplot(x=dftrain["Speed"])
#Replacing Speed outliers with mean.
Q1 = dftrain['Speed'].quantile(0.25)
Q3 = dftrain['Speed'].quantile(0.75)
IQR = Q3 - Q1     

a=Q3 + 1.5 * IQR
a

dftrain.loc[dftrain.Speed>a , 'Speed'] = dftrain['Speed'].mean()
sns.boxplot(x=dftrain["Speed"])
dftrain.corr()["Radiation"]
dftrain.corr()
y=dftrain['Radiation']
X=dftrain[['Temperature','Pressure','Humidity','wd','Speed']]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
para_grid = {'n_estimators': [100,150,200,250,300],
               'max_depth': [2,3,4,5],
               'min_samples_split': [4,5,6],
               'min_samples_leaf': [3,4,5],
               }
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
RSCV = RandomizedSearchCV(estimator=rf,param_distributions=para_grid,cv=5)
RSCV.fit(X_train,y_train)
RSCV.best_estimator_
#Random Forest
modelRF = RandomForestRegressor(max_depth=5, min_samples_leaf=5, min_samples_split=4,
                      n_estimators=250)
modelRF.fit(X,y)
modelRF.score(X,y)
dftest.head()
dftest.info()
x1=dftest[['Temperature','Pressure','Humidity','wd','Speed']]
pred=modelRF.predict(x1)
pred
new=dftest[['ID']]
new['Radiation']=pred
new
new.to_csv('submission_new_rf.csv', index=False)
