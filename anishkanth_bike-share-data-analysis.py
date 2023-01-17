# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
bike_df = pd.read_csv('../input/bike_share.csv',sep=',')
bike_df.shape
bike_df.head()
bike_df.info()
bike_df.isna().sum()
bike_df.duplicated().sum()
bike_df = bike_df.drop_duplicates()
import seaborn as sns

from matplotlib import pyplot
fig, ax = pyplot.subplots(figsize=(11.7, 8.27))

ax = sns.boxplot(data=bike_df)
lower_bnd = lambda x: x.quantile(0.25) - 1.5 * ( x.quantile(0.75) - x.quantile(0.25) )

upper_bnd = lambda x: x.quantile(0.75) + 1.5 * ( x.quantile(0.75) - x.quantile(0.25) )
bike_df = bike_df[(bike_df['holiday'] >= lower_bnd(bike_df['holiday'])) & (bike_df['holiday'] <= upper_bnd(bike_df['holiday'])) & (bike_df['weather'] >= lower_bnd(bike_df['weather'])) & (bike_df['weather'] <= upper_bnd(bike_df['weather'])) & (bike_df['humidity'] >= lower_bnd(bike_df['humidity'])) & (bike_df['humidity'] <= upper_bnd(bike_df['humidity']))] 
bike_df = bike_df[(bike_df['windspeed'] >= lower_bnd(bike_df['windspeed'])) & (bike_df['windspeed'] <= upper_bnd(bike_df['windspeed'])) & (bike_df['casual'] >= lower_bnd(bike_df['casual'])) & (bike_df['casual'] <= upper_bnd(bike_df['casual'])) & (bike_df['registered'] >= lower_bnd(bike_df['registered'])) & (bike_df['registered'] <= upper_bnd(bike_df['registered']))] 
bike_df = bike_df[(bike_df['count'] >= lower_bnd(bike_df['count'])) & (bike_df['count'] <= upper_bnd(bike_df['count']))]
bike_df.shape
list(bike_df)
bike_df.apply(lambda x: len(x.unique()))
# sns.pairplot(data=bike_df,hue='season')
# sns.pairplot(data=bike_df,hue='workingday')
# sns.pairplot(data=bike_df,hue='weather')
bike_df.corr()
from sklearn.linear_model import LinearRegression  

from sklearn.model_selection import train_test_split 

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
modelinput_cas = bike_df.drop(columns=['holiday','casual','registered','count'],axis=1)

modeloutput_cas = bike_df['casual']
X_train,X_test,Y_train, Y_test = train_test_split(modelinput_cas,modeloutput_cas,test_size=0.3,random_state=123)
lm = LinearRegression()

lm.fit(X_train,Y_train)

Y_train_predict = lm.predict(X_train)

Y_test_predict = lm.predict(X_test)

print("-----------------------casual--------------------------")

print("MSE Train:",mean_squared_error(Y_train, Y_train_predict))

print("MSE Test:",mean_squared_error(Y_test, Y_test_predict))

print("RMSE Train:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("RMSE Test:",np.sqrt(mean_squared_error(Y_test, Y_test_predict)))

print('MAE Train', mean_absolute_error(Y_train, Y_train_predict))

print('MAE Test', mean_absolute_error(Y_test, Y_test_predict))

print('R2 Train',r2_score(Y_train, Y_train_predict))

print('R2 Test',r2_score(Y_test, Y_test_predict))
modelinput_reg = bike_df.drop(columns=['holiday','casual','registered','count'],axis=1)

modeloutput_reg = bike_df['registered']
X_train,X_test,Y_train, Y_test = train_test_split(modelinput_reg,modeloutput_reg,test_size=0.3,random_state=123)
lm = LinearRegression()

lm.fit(X_train,Y_train)

Y_train_predict = lm.predict(X_train)

Y_test_predict = lm.predict(X_test)

print("---------------------registered-------------------------")

print("MSE Train:",mean_squared_error(Y_train, Y_train_predict))

print("MSE Test:",mean_squared_error(Y_test, Y_test_predict))

print("RMSE Train:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("RMSE Test:",np.sqrt(mean_squared_error(Y_test, Y_test_predict)))

print('MAE Train', mean_absolute_error(Y_train, Y_train_predict))

print('MAE Test', mean_absolute_error(Y_test, Y_test_predict))

print('R2 Train',r2_score(Y_train, Y_train_predict))

print('R2 Test',r2_score(Y_test, Y_test_predict))
modelinput_tot = bike_df.drop(columns=['holiday','casual','registered','count'],axis=1)

modeloutput_tot = bike_df['count']
from sklearn import preprocessing

modelinput_tot = preprocessing.StandardScaler().fit(modelinput_tot).transform(modelinput_tot.astype(float))
X_train,X_test,Y_train, Y_test = train_test_split(modelinput_tot,modeloutput_tot,test_size=0.3,random_state=123)
lm = LinearRegression()

lm.fit(X_train,Y_train)

Y_train_predict = lm.predict(X_train)

Y_test_predict = lm.predict(X_test)

print("-----------------------Total---------------------------")

print("MSE Train:",mean_squared_error(Y_train, Y_train_predict))

print("MSE Test:",mean_squared_error(Y_test, Y_test_predict))

print("RMSE Train:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("RMSE Test:",np.sqrt(mean_squared_error(Y_test, Y_test_predict)))

print('MAE Train', mean_absolute_error(Y_train, Y_train_predict))

print('MAE Test', mean_absolute_error(Y_test, Y_test_predict))

print('R2 Train',r2_score(Y_train, Y_train_predict))

print('R2 Test',r2_score(Y_test, Y_test_predict))
from sklearn.preprocessing import Imputer

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score

for K in range(50):

    K_value = K + 1

    neigh = KNeighborsRegressor(n_neighbors=K_value,weights='uniform',algorithm='auto')

    neigh.fit(X_train, Y_train)

    y_pred=neigh.predict(X_test)

    print("Accuracy is",r2_score(Y_test, y_pred)*100,"% for K-Value",K_value)
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import PolynomialFeatures

dt = DecisionTreeRegressor(max_depth=6) 

dt.fit(X_train, Y_train)

y_pred = dt.predict(X_test)

print(r2_score(Y_test, y_pred)*100)
for i in range(1, 30):

    print('Accuracy score using max_depth =', i, end = ': ')

    dt = DecisionTreeRegressor(max_depth=i)

    dt.fit(X_train, Y_train)

    y_pred = dt.predict(X_test)

    print(r2_score(Y_test, y_pred)*100)
for i in ['auto','sqrt','log2',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:

    print('Accuracy score using max_features =', i, end = ': ')

    dt = DecisionTreeRegressor(max_depth=6,max_features=i)

    dt.fit(X_train, Y_train)

    y_pred = dt.predict(X_test)

    print(r2_score(Y_test, y_pred)*100)
for i in range(2, 40):

    print('Accuracy score using min_samples_split =', i, end = ': ')

    dt = DecisionTreeRegressor(max_depth=6,max_features=0.5,min_samples_split=i)

    dt.fit(X_train, Y_train)

    y_pred = dt.predict(X_test)

    print(r2_score(Y_test, y_pred)*100)
mean_squared_error(Y_test, y_pred)