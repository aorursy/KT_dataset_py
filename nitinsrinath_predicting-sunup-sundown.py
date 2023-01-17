import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
sun=pd.read_csv('sundata.csv')
sun=sun.drop('City', axis=1)

for i in sun.index:
    temp=sun.Date[i].split('-')
    sun.at[i, 'day']=int(temp[2])
    sun.at[i, 'month']=int(temp[1])
    sun.at[i, 'year']=int(temp[0])
    temp2=sun.Sunup[i].split(':')
    sun.at[i, 'sunrise']=(int(temp2[0])*60)+int(temp2[1])
    temp3=sun.Sundown[i].split(':')
    sun.at[i, 'sunset']=(int(temp3[0])*60)+int(temp3[1])

sun=sun[['day', 'month', 'year', 'sunrise', 'sunset']]

sun['DayOfYear']=30*(sun.month-1)+sun.day
sun.head()
sun.corr()
plt.plot(sun.index, sun.sunrise, label='sunrise')
plt.plot(sun.index, sun.sunset, label='sunsset')
plt.xlabel('day')
plt.ylabel('sunrise/sunset time')
plt.legend()
dayofyear=pd.DataFrame(sun.groupby('DayOfYear')['sunrise'].mean())
dayofyear['sunset']=pd.DataFrame(sun.groupby('DayOfYear')['sunset'].mean())['sunset']
plt.plot(dayofyear.index, dayofyear.sunrise, label='sunrise')
plt.plot(dayofyear.index, dayofyear.sunset, label='sunset')
plt.xlabel('day of the year')
plt.ylabel('sunrise/sunset time')
plt.legend()
sun['sunupdown']=10000*sun.sunrise+sun.sunset
plt.scatter(sun.DayOfYear, sun.sunrise)
plt.scatter(sun.DayOfYear, sun.sunset)
X=sun[['day', 'month']]
Y=sun[['sunrise', 'sunset']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
Yrise_train=Y_train['sunrise']
Yrise_test=Y_test['sunrise']
Yset_train=Y_train['sunset']
Yset_test=Y_test['sunset']
NNtest=pd.DataFrame()
for i in range(1, 16):
    temp=tuple([2 for j in range(i)])
    NNR = MLPRegressor(hidden_layer_sizes=temp, random_state=0)
    NNR.fit(X_train, Yrise_train)
    y_train_pred=NNR.predict(X_train)
    y_test_pred=NNR.predict(X_test)
    NNtest.at[i, 'TrainAcc']=r2_score(Yrise_train, y_train_pred)
    NNtest.at[i, 'TestAcc']=r2_score(Yrise_test, y_test_pred)
    print(i, end=' ')
plt.plot(NNtest.index, NNtest.TrainAcc, label='Train Acc')
plt.plot(NNtest.index, NNtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Number of layers')
plt.ylabel('Accuracy')
plt.grid()
RFtest=pd.DataFrame()
for i in range(1, 16):
    RFR = RandomForestRegressor(max_depth=i, random_state=0)
    RFR.fit(X_train, Yrise_train)
    y_train_pred=RFR.predict(X_train)
    RFtest.at[i, 'TrainAcc']=r2_score(Yrise_train, y_train_pred)
    RFtest.at[i, 'TestAcc']=r2_score(Yrise_test, y_test_pred)
    y_test_pred=RFR.predict(X_test)
    print(i, end=' ')
plt.plot(RFtest.index, RFtest.TrainAcc, label='Train Acc')
plt.plot(RFtest.index, RFtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.grid()
RFtest=pd.DataFrame()
for i in range(1, 16):
    RFR = RandomForestRegressor(max_depth=i, random_state=0)
    RFR.fit(X_train, Yset_train)
    y_train_pred=RFR.predict(X_train)
    RFtest.at[i, 'TrainAcc']=r2_score(Yset_train, y_train_pred)
    RFtest.at[i, 'TestAcc']=r2_score(Yset_test, y_test_pred)
    y_test_pred=RFR.predict(X_test)
    print(i, end=' ')
plt.plot(RFtest.index, RFtest.TrainAcc, label='Train Acc')
plt.plot(RFtest.index, RFtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.grid()
RFR_rise = RandomForestRegressor(max_depth=10, random_state=0)

RFR_rise.fit(X_train, Yrise_train)
y_pred=RFR_rise.predict(X_test)
print('R2=', r2_score(Yrise_test, y_pred))
print('MSE=', mean_squared_error(Yrise_test, y_pred))
RFR_set = RandomForestRegressor(max_depth=10, random_state=0)

RFR_set.fit(X_train, Yset_train)
y_pred=RFR_set.predict(X_test)
print('R2=', r2_score(Yset_test, y_pred))
print('MSE=', mean_squared_error(Yset_test, y_pred))
cvc=cross_val_score(RFR, X, Y, cv=10)
plt.boxplot(cvc)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
RFR.fit(X_train, Y_train)
y_train_pred=RFR.predict(X_train)
y_test_pred=RFR.predict(X_test)
print(r2_score(Y_train, y_train_pred))
print(r2_score(Y_test, y_test_pred))
temp=X.sample(frac=1).head(10)
sample=temp.copy()
sample['sunrise_pred']=RFR_rise.predict(temp)
sample['sunset_pred']=RFR_set.predict(temp)
for i in sample.index:
    temp=sample.sunrise_pred[i]/60
    temp2=int(60*(temp-int(temp)))
    sample.at[i, 'sunrise']=str(int(temp))+':'+str(temp2)
    temp=sample.sunset_pred[i]/60
    temp2=int(60*(temp-int(temp)))
    sample.at[i, 'sunset']=str(int(temp))+':'+str(temp2)
sample[['day', 'month', 'sunrise', 'sunset']]
