# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import scipy.stats
#Question No 1
# Null Hypothesis :  Mu>20000
#Alternate Hypothesis : Mu<=20000
# Single tail hypothesis 
Xbar=18050
Mu=20000
sigma=155
n=30
s_n=155/(30**0.5)
Zvalue=(Xbar-Mu)/s_n
print('Zvalue',Zvalue)
print('Zcriticalvalue',scipy.stats.norm(0,1).ppf(.05))
#Zvalue is very low compared to Zcritical value, so null hypothesis rejected 

#Question 2
import scipy.stats as stats
Mag1=[15.75,11.55,11.16,9.92,9.23,8.20]
Mag2=[12.63,11.46,10.77,9.93,9.87,9.42]
Mag3=[9.27,8.28,8.15,6.37,6.37,5.66]

stats.f_oneway(Mag1,Mag2,Mag3)
#Null Hypothesis , mu1=mu2=mu3
#p value is less than .05, implies null hypothesis is rejected with 95% confidence

#Question3
import scipy.stats as stats
Males=[80,75,95,55,60,70,75,72,80,65]
Females=[60,70,50,85,45,60,80,65,70,62,77,82]
stats.ttest_ind(Males,Females)
#Null Hyptohesis, Mu1=Mu2
#pvalue is more than 0.01 , implies null hypothesis cant be rejected with 99% confidence
#Question 4
import scipy.stats as stats
#variance=4y2
#n=100
#Xbar=1.3y
#Mu=1.5y
#alpha =5%
#sigma=2y

s_n=2/10
Z_value=(1.3-1.5)/s_n  #y wil be cancelled out while calculating Z value 
Z_criticalvalue=scipy.stats.norm(0,1).ppf(.025)
print('Zvalue',Z_value)
print('Zcriticalvalue',Z_criticalvalue)
#Zvalue is more than Zcrticial Value, implies null hypothesis cant be rejected 

#Question 5
df=pd.read_csv('../input/datasets1/Precipitation.csv')
df['diff']= df.drop(['Years'],axis=1).max(axis=1)-df.drop(['Years'],axis=1).min(axis=1) #taking difference of maximum and minimum and storing it in an array
df1=df.drop(['Years'],axis=1) 
variable=list(df1) #all the column nmames 
for i in variable:
    df[i].fillna(df['diff'],inplace=True) #repalcing NaN values for each columns.........Question 5 a
    
#Tranforming data into time series data n
df_A=df.select(lambda col: col.endswith('A'), axis=1)
df_B=df.select(lambda col: col.endswith('B'), axis=1)
val = pd.melt(df_A.T)
dates = np.arange('1970-01', '2009-01', dtype='datetime64[M]')
df_A = pd.DataFrame(val.iloc[:,1].values, index = dates,columns=['A'])
val = pd.melt(df_B.T)
dates = np.arange('1970-01', '2009-01', dtype='datetime64[M]')
df_B = pd.DataFrame(val.iloc[:,1].values, index = dates,columns=['B'])



#SMA with window 25...................Question 5 b
sma1 = df_A.rolling(window=25,center = False).mean()
plt.figure(1)
plt.plot(df_A, label = "Original data")
plt.plot(sma1, label = "Smoothed with SMA")
plt.legend(loc = "best")
plt.xlabel("Year")
plt.ylabel("Precipitation A")


sma2 = df_B.rolling(window=25,center = False).mean()
plt.figure(2)
plt.plot(df_B, label = "Original data")
plt.plot(sma2, label = "Smoothed with SMA")
plt.legend(loc = "best")
plt.xlabel("Year")
plt.ylabel("Precipitation B")




#Question no 5C
sma1['year']=sma1.index.year   #save the year to a column
sma2['year']=sma2.index.year
y_A=sma1.groupby(['year']).mean()
y_B=sma2.groupby(['year']).mean()# yearly average of the data  #Question no 5 c
sma1=sma1.drop(['year'],axis=1)
sam2=sma2.drop(['year'],axis=1)
#Question no 5D
import statsmodels.api as sm
from linearmodels import PanelOLS
prec_mean=(y_A['A']+y_B['B'])/2
prec_mean=prec_mean.to_frame()
Y=prec_mean.groupby('year').mean()
Y=Y.dropna()
Y=Y.reset_index()
X=pd.read_csv('../input/datasets4/Rainfall.csv')
X['SMA rainfalls']=X['Rainfall in mm'].rolling(window=2,center = False).mean()
X=X.dropna()
X=X.reset_index(drop=True)

mod = sm.OLS(Y[0],X['SMA rainfalls'])#.........OLS regression is carried out between the SMA values of both rainfall and percipitation data and this model is used to predict the precipitation for 1970
res = mod.fit()
print(res.summary())
value=res.predict(5)
value#..................................predicted value of precipitation 5mm rainfall Question 5D

#Question 6
import matplotlib.pyplot as plt
ts_data=pd.read_csv('../input/datasets3/ts_data.csv')
dates = np.arange('2004-01', '2016-01', dtype='datetime64[M]')
ts_data_1 = pd.DataFrame(ts_data.iloc[:,0].values, index = dates,columns=['Sales Company 1'])
ts_data_2 = pd.DataFrame(ts_data.iloc[:,1].values, index = dates,columns=['Sales Company 2'])
ts_data_1['Sales Company 1']=ts_data_1['Sales Company 1'].astype('float64')
ts_data_2['Sales Company 2']=ts_data_2['Sales Company 2'].astype('float64')



#SMA plot for mean with a window 4 for company 1
sma3 = ts_data_1.rolling(window=4,center = False).mean()
plt.figure(1)
plt.plot(ts_data_1, label = "Original data")
plt.plot(sma3,label = "Smoothed with SMA")
plt.legend(loc = "best")
plt.xlabel("Year")
plt.ylabel("Sales of company 1")
#SMA plot for mean with a window 4 for compamy 2
sma4= ts_data_2.rolling(window=4,center = False).mean()
plt.figure(2)
plt.plot(ts_data_2, label = "Original data")
plt.plot(sma4,label = "Smoothed with SMA")
plt.legend(loc = "best")
plt.xlabel("Year")
plt.ylabel("Sales of company 2")








# evaluate an ARIMA model for a given order (p,d,q)
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float64')
    best_aic, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    model=ARIMA(dataset, order=order)
                    results=model.fit()
                    aic=results.aic
                    if aic < best_aic:
                        best_aic, best_cfg = aic, order
                    print('ARIMA %s aic=%.3f' % (order,aic))
                except:
                    continue
    print('Best ARIMA %s MSE=%.3f' % (best_cfg, best_aic))
    return best_cfg

p_values = [0, 1, 2, 4]
d_values = range(0, 5)
q_values = range(0, 5)
ts_data_1
order1=evaluate_models(ts_data_1, p_values, d_values, q_values)  
order2=evaluate_models(ts_data_2, p_values, d_values, q_values)



#MODELS for both time series data 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
mod1= ARIMA(ts_data_1, order=order1)
mod2= ARIMA(ts_data_2, order=order2)

results1 = mod1.fit()
results2 = mod2.fit()

print('Time series 1 ARIMA  model ',results1.summary().tables[1])
print('Time series 2 ARIMA  model',results2.summary().tables[1])




#Rolling Forecast ARIMA Model for TS_data1
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
X = ts_data_1.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=order1)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.title('Rolling forecast for ARIMA Model for company 1')
pyplot.plot(predictions, color='red')
pyplot.show()
#Rolling Forecast ARIMA Model for TS_data2
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
X = ts_data_2.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=order2)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.title('Rolling forecast for ARIMA Model for company 2')
pyplot.plot(predictions, color='red')
pyplot.show()
#Question 7 
import pandas as pd
from scipy import stats
from scipy.stats import t
n=25
df7=pd.read_csv('../input/datasets2/student.csv')
mean=df7['Pocket money'].mean()
std=df7['Pocket money'].std()/((len(df7['Pocket money']))**0.5)
R = t.interval(0.90, n-1, loc=mean, scale=std)
print(R) #.................................Question 7 a

#.....Question 7b
mean1=df7['Internet usage'].mean()
std1=df7['Internet usage'].std()/((len(df7['Internet usage']))**0.5)
tcritical=stats.t.ppf(.05, n-1)
t_value=(mean1-60)/std1 #.............T value Null hypothesis Mu>60, Alternate hypothis Mu<=60 single tail t test
t_value 
tcritical
print('T statistics',t_value)
print('T critical',tcritical)

#we reject the null hypothesis because t statistics is less than t critical
