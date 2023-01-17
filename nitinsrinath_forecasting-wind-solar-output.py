import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
energy=pd.read_csv('Energy.csv')
energy=energy.rename(columns={'Time':'time', 'Solar Power (MW)': 'sun', 'Windspeed (mph)':'wind'})

for i in range(len(energy.index)-1):
    if np.isnan(energy.wind[i]):
        energy.at[i, 'wind']=(energy.wind[i-1]+energy.wind[i+1])/2
energy=energy.dropna()

for i in energy.index:
    energy.at[i, 'month']=int(energy.time[i].split('/')[0])
    energy.at[i, 'day']=int(energy.time[i].split('/')[1])
    temp=energy.time[i].split(' ')[1]
    energy.at[i, 'hour']=int(temp.split(':')[0])
    energy.at[i, 'min']=int(temp.split(':')[1])

energy['dayofyr']=energy['day']+(30*(energy['month']-1))
energy['timeofday']=60*energy['hour']+energy['min']

timestats=pd.DataFrame(energy.groupby('timeofday')['sun'].mean())
timestats['suntotal']=pd.DataFrame(energy.groupby('timeofday')['sun'].sum())['sun']
timestats['windavg']=pd.DataFrame(energy.groupby('timeofday')['wind'].mean())['wind']
timestats['windtotal']=pd.DataFrame(energy.groupby('timeofday')['wind'].sum())['wind']

daystats=pd.DataFrame(energy.groupby('dayofyr')['sun'].mean())
daystats['suntotal']=pd.DataFrame(energy.groupby('dayofyr')['sun'].sum())['sun']
daystats['windavg']=pd.DataFrame(energy.groupby('dayofyr')['wind'].mean())['wind']
daystats['windtotal']=pd.DataFrame(energy.groupby('dayofyr')['wind'].sum())['wind']

energy['windpower_kW']=0.16*(energy.wind**3)/1000
energy['sunpower_kW']=0.4*energy.sun*3600*1000/3840
energy['singleturbine_kW']=0.01328*60*60*3.33*3.33*(energy.wind**3)/(365*24)
energy['combined']=(energy.sunpower_kW*0.9*3400/3600)+energy.singleturbine_kW
plt.plot(timestats.index, timestats.sun)
plt.xlabel('time of day (min)')
plt.ylabel('avg sun output (MW)')
plt.plot(daystats.index, daystats.sun)
plt.xlabel('day of year')
plt.ylabel('avg sun output (MW)')
plt.plot(timestats.index, timestats.windavg)
plt.xlabel('time of day (min)')
plt.ylabel('avg wind speed (mph)')
plt.plot(daystats.index, daystats.windavg)
plt.xlabel('day of year')
plt.ylabel('avg wind speed (mph)')
energy.describe()[['windpower_kW', 'sunpower_kW', 'singleturbine_kW', 'combined']]
energy[['windpower_kW', 'sunpower_kW', 'singleturbine_kW', 'combined']].hist(figsize=(10,5))
X=energy[['dayofyr', 'timeofday']]
Y=energy['sunpower_kW']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
PRTest=pd.DataFrame()
PR = LinearRegression()
for i in range(1, 15):
    pf= PolynomialFeatures(degree=i)
    X_train_poly = pf.fit_transform(X_train)
    X_test_poly=pf.fit_transform(X_test)
    PR.fit(X_train_poly, Y_train)
    y_pred_test = PR.predict(X_test_poly)
    y_pred_train=PR.predict(X_train_poly)
    PRTest.at[i, 'TrainAcc']=r2_score(Y_train, y_pred_train)
    PRTest.at[i, 'TestAcc']=r2_score(Y_test, y_pred_test)
plt.plot(PRTest.index, PRTest.TrainAcc, label='Train Acc')
plt.plot(PRTest.index, PRTest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.grid()
pf= PolynomialFeatures(degree=8)
X_train_poly = pf.fit_transform(X_train)
X_test_poly=pf.fit_transform(X_test)
PR.fit(X_train_poly, Y_train)
y_pred = PR.predict(X_test_poly)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
KNtest=pd.DataFrame()
for i in range(1, 41):
    KNR = KNeighborsRegressor(n_neighbors=i)
    KNR.fit(X_train, Y_train)
    y_train_pred=KNR.predict(X_train)
    y_test_pred=KNR.predict(X_test)
    KNtest.at[i, 'TrainAcc']=r2_score(Y_train, y_train_pred)
    KNtest.at[i, 'TestAcc']=r2_score(Y_test, y_test_pred)
plt.plot(KNtest.index, KNtest.TrainAcc, label='Train Acc')
plt.plot(KNtest.index, KNtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.grid()
KNR = KNeighborsRegressor(n_neighbors=12)

KNR.fit(X_train, Y_train)
y_pred=KNR.predict(X_test)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
RFtest=pd.DataFrame()
for i in range(1, 25):
    RFR = RandomForestRegressor(max_depth=i, random_state=0)
    RFR.fit(X_train, Y_train)
    y_train_pred=RFR.predict(X_train)
    y_test_pred=RFR.predict(X_test)
    RFtest.at[i, 'TrainAcc']=r2_score(Y_train, y_train_pred)
    RFtest.at[i, 'TestAcc']=r2_score(Y_test, y_test_pred)
plt.plot(RFtest.index, RFtest.TrainAcc, label='Train Acc')
plt.plot(RFtest.index, RFtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.grid()
RFR = RandomForestRegressor(max_depth=25, random_state=0)

RFR.fit(X_train, Y_train)
y_pred=RFR.predict(X_test)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
RFR.fit(X_train, Y_train)
y_train_pred=RFR.predict(X_train)
y_test_pred=RFR.predict(X_test)
print(r2_score(Y_train, y_train_pred))
print(r2_score(Y_test, y_test_pred))
X=energy[['dayofyr', 'timeofday']]
Y=energy['singleturbine_kW']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
PRTest=pd.DataFrame()
PR = LinearRegression()
for i in range(1, 15):
    pf= PolynomialFeatures(degree=i)
    X_train_poly = pf.fit_transform(X_train)
    X_test_poly=pf.fit_transform(X_test)
    PR.fit(X_train_poly, Y_train)
    y_pred_test = PR.predict(X_test_poly)
    y_pred_train=PR.predict(X_train_poly)
    PRTest.at[i, 'TrainAcc']=r2_score(Y_train, y_pred_train)
    PRTest.at[i, 'TestAcc']=r2_score(Y_test, y_pred_test)
plt.plot(PRTest.index, PRTest.TrainAcc, label='Train Acc')
plt.plot(PRTest.index, PRTest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.grid()
pf= PolynomialFeatures(degree=6)
X_train_poly = pf.fit_transform(X_train)
X_test_poly=pf.fit_transform(X_test)
PR.fit(X_train_poly, Y_train)
y_pred = PR.predict(X_test_poly)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
KNtest=pd.DataFrame()
for i in range(1, 41):
    KNR = KNeighborsRegressor(n_neighbors=i)
    KNR.fit(X_train, Y_train)
    y_train_pred=KNR.predict(X_train)
    y_test_pred=KNR.predict(X_test)
    KNtest.at[i, 'TrainAcc']=r2_score(Y_train, y_train_pred)
    KNtest.at[i, 'TestAcc']=r2_score(Y_test, y_test_pred)
plt.plot(KNtest.index, KNtest.TrainAcc, label='Train Acc')
plt.plot(KNtest.index, KNtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.grid()
KNR = KNeighborsRegressor(n_neighbors=12)

KNR.fit(X_train, Y_train)
y_pred=KNR.predict(X_test)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
RFtest=pd.DataFrame()
for i in range(1, 25):
    RFR = RandomForestRegressor(max_depth=i, random_state=0)
    RFR.fit(X_train, Y_train)
    y_train_pred=RFR.predict(X_train)
    y_test_pred=RFR.predict(X_test)
    RFtest.at[i, 'TrainAcc']=r2_score(Y_train, y_train_pred)
    RFtest.at[i, 'TestAcc']=r2_score(Y_test, y_test_pred)
plt.plot(RFtest.index, RFtest.TrainAcc, label='Train Acc')
plt.plot(RFtest.index, RFtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.grid()
RFR = RandomForestRegressor(max_depth=20, random_state=0)

RFR.fit(X_train, Y_train)
y_pred=RFR.predict(X_test)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
RFR.fit(X_train, Y_train)
y_train_pred=RFR.predict(X_train)
y_test_pred=RFR.predict(X_test)
print(r2_score(Y_train, y_train_pred))
print(r2_score(Y_test, y_test_pred))
RFR_wind=RandomForestRegressor(max_depth=20, random_state=0)
RFR_sun=RandomForestRegressor(max_depth=25, random_state=0)


X=energy[['dayofyr', 'timeofday']]
Ywind=energy['singleturbine_kW']
Ysun=energy['sunpower_kW']

RFR_wind.fit(X, Ywind)
RFR_sun.fit(X, Ysun)

temp=X.sample(frac=1).head(10)
sample=temp.copy()
sample['sunpower_kW']=RFR_sun.predict(temp)
sample['windpower_kW']=RFR_wind.predict(temp)
sample['combined_kW']=(0.9*3400*sample.sunpower_kW/3600)+sample.windpower_kW

for i in sample.index:
    temp1=[int(energy.dayofyr[i]/30)+1, int(energy.dayofyr[i]%30)]
    temp2=[int(energy.timeofday[i]/60), int(energy.timeofday[i]%60)]
    sample.at[i, 'timestamp']=str(temp1[0])+'/'+str(temp1[1])+' '+str(temp2[0])+':'+str(temp2[1])
sample[['timestamp', 'sunpower_kW', 'windpower_kW', 'combined_kW']]
cross_val_score(RFR_wind, X, Ywind)
