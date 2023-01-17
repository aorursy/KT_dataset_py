import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline
solar = pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')
solar.head()
solar.info()
solar['Time'] = pd.to_datetime(solar['Time'])
solar['Data'] = pd.to_datetime(solar['Data'])
solar.info()
date = solar['Data']

solar['Day'] = solar['Data'].apply(lambda date: date.day)

solar['Month'] = solar['Data'].apply(lambda date: date.month)

solar['hour'] = solar['Time'].apply(lambda date: date.hour)
solar.head()
solar = solar.drop(['UNIXTime','Time','Data','TimeSunRise','TimeSunSet'],axis=1)
solar.head()
solar_sorted = solar.pivot_table(index=['Month', 'Day','hour'],values= ['Radiation','Temperature','Pressure','Humidity','WindDirection(Degrees)','Speed'],aggfunc=np.mean)
solar_sorted
sb.pairplot(solar_sorted)
plt.figure(figsize = (12,8))

sb.heatmap(solar_sorted.corr(),cmap='coolwarm',annot=True)

plt.title('Correlations')
solar_sorted.columns
from sklearn.model_selection import train_test_split
X = solar_sorted[['Humidity', 'Pressure', 'Speed', 'Temperature', 'WindDirection(Degrees)']]

y = solar_sorted['Radiation']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.linear_model import LinearRegression
solar_linear = LinearRegression()
solar_linear.fit(X_test,y_test)
cdf =pd.DataFrame(solar_linear.coef_,X.columns,columns=['Coeffecient'])

cdf
predictions = solar_linear.predict(X_test)
plt.scatter(y_test,predictions)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,predictions))
metrics.r2_score(y_test,predictions)
from sklearn.svm import SVR
solar_svr = SVR()
solar_svr.fit(X_train,y_train)
predictions_svr = solar_svr.predict(X_test)
plt.scatter(y_test,predictions_svr)
np.sqrt(metrics.mean_squared_error(y_test,predictions_svr))
metrics.r2_score(y_test,predictions_svr)
param_grid ={'C':[1000,2500,5000,7500,10000], 'gamma':[0.01,0.001,0.0001], 'kernel':['rbf']} 

#note the values I have selcted here have been obtained through an iterative trial and error process to tune the model, and the values shown are values I have found to be optimum.
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
solar_svr2 = SVR(C=2500,gamma=0.001)
solar_svr2.fit(X_train,y_train)
predictions_svr2 = solar_svr2.predict(X_test)
plt.scatter(y_test,predictions_svr2)
np.sqrt(metrics.mean_squared_error(y_test,predictions_svr2))
metrics.r2_score(y_test,predictions_svr2)