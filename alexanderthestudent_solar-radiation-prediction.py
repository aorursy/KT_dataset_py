import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

import plotly as py

import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data_path = "../input/"

df = pd.read_csv(data_path+"SolarPrediction.csv")

df.head()
data=pd.read_csv('../input/SolarPrediction.csv')
import matplotlib.pyplot as plt

import seaborn as sns

data['Radiation'].plot()
df['Radiation'].hist()
corrmat = df.corr()

sns.heatmap(corrmat, vmax=.8, square=True)
g = sns.jointplot(x="Radiation", y="Temperature", data=df)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Temp vs. Radiation')
g = sns.jointplot(x="Radiation", y="Humidity", data=df)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Temp vs. Humidity')
#drop low radiation values

df = df[df['Radiation'] >= 10]
#Covert time to_datetime

#Add column 'hour'

df['Time_conv'] =  pd.to_datetime(df['Time'], format='%H:%M:%S')

df['hour'] = pd.to_datetime(df['Time_conv'], format='%H:%M:%S').dt.hour



#Add column 'month'

df['month'] = pd.to_datetime(df['UNIXTime'].astype(int), unit='s').dt.month



#Add column 'year'

df['year'] = pd.to_datetime(df['UNIXTime'].astype(int), unit='s').dt.year



#Duration of Day

df['total_time'] = pd.to_datetime(df['TimeSunSet'], format='%H:%M:%S').dt.hour - pd.to_datetime(df['TimeSunRise'], format='%H:%M:%S').dt.hour

df.head()
ax = plt.axes()

sns.barplot(x="hour", y='Radiation', data=df, palette="BuPu", ax = ax)

ax.set_title('Mean Radiation by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="month", y='Radiation', data=df, palette="BuPu", ax = ax, order=[9,10,11,12,1])

ax.set_title('Mean Radiation by Month')

plt.show()
ax = plt.axes()

sns.barplot(x="hour", y='Humidity', data=df, palette=("coolwarm"), ax = ax)

ax.set_title('Humidity by Hour')

plt.show()
ax = plt.axes()

sns.barplot(x="total_time", y='Radiation', data=df, palette="BuPu", ax = ax)

ax.set_title('Radiation by Total Daylight Hours')

plt.show()
from sklearn.cluster import KMeans
df = pd.read_csv(data_path+"SolarPrediction.csv")

X1 = df[['Temperature' , 'Humidity']].iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)
plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

algorithm.fit(X1)

labels = algorithm.labels_

centroids = algorithm.cluster_centers_
h = 0.02

x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1

y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 
plt.figure(1 , figsize = (15 , 7) )

plt.clf()

Z = Z.reshape(xx.shape)

plt.imshow(Z , interpolation='nearest', 

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')



plt.scatter( x = 'Temperature' ,y = 'Humidity' , data = df , c = labels , 

            s = 200 )

plt.scatter(x = centroids[: , 0] , y =  centroids[: , 1] , s = 300 , c = 'red' , alpha = 0.5)

plt.ylabel('Humidity') , plt.xlabel('Temperature')

plt.show()
y = df['Radiation']

X = df.drop(['Radiation', 'Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)
scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
lpredictions = lm.predict( X_test)

print(lpredictions)
plt.scatter(y_test,lpredictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, lpredictions))

print('MSE:', metrics.mean_squared_error(y_test, lpredictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lpredictions)))
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import RidgeCV

ridge = RidgeCV(cv=5)

ridge.fit(X_train,y_train)
print('Coefficients: \n', ridge.coef_)
rpredictions = ridge.predict( X_test)

print(rpredictions)
plt.scatter(y_test,rpredictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, rpredictions))

print('MSE:', metrics.mean_squared_error(y_test, rpredictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rpredictions)))
coeffecients = pd.DataFrame(ridge.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn import linear_model

sgd = linear_model.SGDRegressor(max_iter=1000)

sgd.fit(X_train,y_train)
print('Coefficients: \n', sgd.coef_)
spredictions = sgd.predict(X_test)

print(spredictions)
plt.scatter(y_test,spredictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, spredictions))

print('MSE:', metrics.mean_squared_error(y_test, spredictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, spredictions)))
coeffecients = pd.DataFrame(sgd.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.ensemble import AdaBoostRegressor

abreg = AdaBoostRegressor(random_state=0, n_estimators=100)

abreg.fit(X_train,y_train)
abpredictions = abreg.predict( X_test)

print(abpredictions)
plt.scatter(y_test,abpredictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, abpredictions))

print('MSE:', metrics.mean_squared_error(y_test, abpredictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, abpredictions)))
final_data=data.drop(['UNIXTime','Data','Time','TimeSunRise','TimeSunSet'],axis=1)
final_y=final_data.pop('Radiation')
final_x=final_data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final_x, final_y, test_size=0.33, random_state=42)

import xgboost as xgb
xgdmat=xgb.DMatrix(X_train,y_train)

our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':1}

final_gb=xgb.train(our_params,xgdmat)

tesdmat=xgb.DMatrix(X_test)

xpredictions=final_gb.predict(tesdmat)

print(xpredictions)
plt.scatter(y_test,xpredictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, xpredictions))

print('MSE:', metrics.mean_squared_error(y_test, xpredictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xpredictions)))
Q1 = y.quantile(0.25)

Q3 = y.quantile(0.75)

IQR = Q3 - Q1

Outliers = (y < (Q1 - 1.5 * IQR)) |(y > (Q3 + 1.5 * IQR))

from fbprophet import Prophet
df = pd.DataFrame(data)
date_rad = df[['Data', 'Radiation']] 
date_rad.plot(x='Data', y='Radiation', kind="line", rot=45)

date_rad = date_rad.rename(columns={'Data':'ds', 'Radiation':'y'})
p = Prophet()

p.fit(date_rad)
future = p.make_future_dataframe(periods=1825)

forecast = p.predict(future)
forecast.tail()
forecastplot = p.plot_components(forecast)