# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import itertools

import sklearn

import scipy

import seaborn as sns

from matplotlib import pyplot as plt

import squarify

import matplotlib.ticker as ticker

import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.stattools import adfuller

import statsmodels.api as sm

from scipy.spatial.distance import euclidean

import sys

from sklearn.preprocessing import MinMaxScaler

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



Rawdata = pd.read_csv('../input/crimes-in-boston/crime.csv',encoding='latin-1')

# Drop "INCIDENT_NUMBER" colume, we are not going to use it in our analysis.

Rawdata.drop("INCIDENT_NUMBER",axis=1, inplace=True) 

# Split 'OCCURRED_ON_DATE' colume into 'DATE' and 'Time'. 'Date' will give us the exact date of the crime

Rawdata[["DATE","TIME"]]=Rawdata['OCCURRED_ON_DATE'].str.split(" ",expand=True) 
Rawdata.info()
# plot line chart

def lineplt(x,y,xlabel,ylabel,title,size,tick_spacing):

    fig,ax=plt.subplots(figsize = size)

    plt.plot(x,y)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    plt.xlabel(xlabel,fontsize = 15)

    plt.ylabel(ylabel,fontsize = 15)

    plt.title(title,fontsize = 20)

    plt.show()



# Create 2 columes DateFrame

def createdf(c1,d1,c2,d2):

    dic = {c1:d1,c2:d2}

    df = pd.DataFrame(dic)

    return df



# Plot histogram

def plthis(d,bin, title):

    plt.figure(figsize=(10,8))

    plt.hist(d, bins=bin)

    plt.title(title, fontsize = 20)

    plt.show()
# Put Date and Count into a new Dataframe

c = createdf("Date",Rawdata["DATE"].value_counts().index,"Count",Rawdata["DATE"].value_counts())



# c is the total number of crimes per day

c.head(5)
plthis(c["Count"],50, "Crimes Count Distribution")
print('skewness is ' + str(c['Count'].skew()))

print('kurtosis is ' + str(c['Count'].kurt()))
bin=pd.cut(c["Count"],50)

fre= createdf("Bin",bin.value_counts().index,"Count",bin.value_counts())

fre_sort = fre.sort_values(by = "Bin", ascending = True)
(_,p) = scipy.stats.shapiro(fre_sort["Count"])

print('p-value is ' + str(p))
(_,p) = scipy.stats.kstest(fre_sort["Count"],'norm')

print('p-value is ' + str(p))
c=c.sort_values(by="Date",ascending = True)

lineplt(c["Date"],c["Count"],"Date","Count","Crimes by Time",(20,15),80)
fig = plt.figure(figsize=(16,16))

ax1 = fig.add_subplot(411)

fig = plot_acf(c["Count"],lags=200,ax=ax1)

plt.title('Autocorrelation Lag=200')

ax2 = fig.add_subplot(412)

fig = plot_pacf(c["Count"],lags=200,ax=ax2)

plt.title('Partial Autocorrelation Lag=200')

ax3 = fig.add_subplot(413)

fig = plot_acf(c["Count"],lags=15,ax=ax3)

plt.title('Autocorrelation Lag=15')

ax4 = fig.add_subplot(414)

fig = plot_pacf(c["Count"],lags=15,ax=ax4)

plt.title('Partial Autocorrelation Lag=15')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,

                wspace=None, hspace=0.5)

plt.show()

res = sm.tsa.seasonal_decompose(c['Count'],freq=12,model="additive")

# # original = res

trend = res.trend

seasonal = res.seasonal

residual = res.resid



fig,ax=plt.subplots(figsize = (20,15))

ax1 = fig.add_subplot(411)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(80))

ax1.plot(c['Count'], label='Original')

ax1.legend(loc='best')

ax2 = fig.add_subplot(412)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(80))

ax2.plot(trend, label='Trend')

ax2.legend(loc='best')

ax3 = fig.add_subplot(413)

ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))

ax3.plot(seasonal[:100],label='Seasonality')

ax3.legend(loc='best')

ax4 = fig.add_subplot(414)

ax4.xaxis.set_major_locator(ticker.MultipleLocator(80))

ax4.plot(residual, label='Residuals')

ax4.legend(loc='best')

plt.tight_layout()

def test_stationarity(series,mlag = 365, lag = None,):

    print('ADF Test Result')

    res = adfuller(series, maxlag = mlag, autolag = lag)

    output = pd.Series(res[0:4],index = ['Test Statistic', 'p value', 'used lag', 'Number of observations used'])

    for key, value in res[4].items():

        output['Critical Value ' + key] = value

    print(output)
test_stationarity(c['Count'],lag = 'AIC')
d1 = c.copy()

d1['Count'] = d1['Count'].diff(1)

d1 = d1.dropna()

lineplt(d1["Date"],d1["Count"],"Date","Count","Crimes by Time",(20,15),80)

print('Average= '+str(d1['Count'].mean()))

print('Std= ' + str(d1['Count'].std()))

print('SE= ' + str(d1['Count'].std()/math.sqrt(len(d1))))

print(test_stationarity(d1['Count'],lag = 'AIC'))
fig_2 = plt.figure(figsize=(16,8))

ax1_2 = fig_2.add_subplot(211)

fig_2 = plot_acf(d1["Count"],lags=15,ax=ax1_2)

ax2_2 = fig_2.add_subplot(212)

fig_2 = plot_pacf(d1["Count"],lags=15,ax=ax2_2)
timeseries = c['Count']

p,d,q = (4,1,2)

arma_mod = ARMA(timeseries,(p,d,q)).fit()

summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))

print(summary)
predict_data = arma_mod.predict(start='2016-07-01', end='2017-07-01', dynamic = False)

timeseries.index = pd.DatetimeIndex(timeseries.index)

fig, ax = plt.subplots(figsize=(20, 15))

ax = timeseries.plot(ax=ax)

predict_data.plot(ax=ax)

plt.show()
p = d = q = range(0, 2)

 

# Generate all different combinations of p, q and q triplets

pdq = list(itertools.product(p, d, q))

 

# Generate all different combinations of seasonal p, q and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

 

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
res = pd.DataFrame(columns = ['order', 'seasonal_order', 'AIC'])
warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = SARIMAX(c['Count'],order=param,seasonal_order=param_seasonal) 

            results = mod.fit()

            data = {'order': param, 'seasonal_order': param_seasonal, 'AIC':results.aic}

#             print('ARIMA{}x{}7 - AIC:{}'.format(param, param_seasonal, results.aic))

            res = res.append(data,ignore_index=True)

        except:

            continue

res = res.sort_values(by = 'AIC', ascending = True)

print(res.head(5))
model=SARIMAX(c['Count'], order=(1,1,1), seasonal_order=(1,1,1, 7)).fit()

summary = model.summary()

print(summary)

# print(c['Count'].index.inferred_freq)

model.plot_diagnostics(figsize=(15, 12))

plt.show()
predict_data = model.predict(start='2016-07-01', end='2017-07-01', dynamic = False)

timeseries.index = pd.DatetimeIndex(timeseries.index)

fig, ax = plt.subplots(figsize=(20, 15))

ax = timeseries.plot(ax=ax)

predict_data.plot(ax=ax)

plt.show()
# Get forecast 30 steps ahead in future

pred_uc = model.get_forecast(steps=30)



# Get confidence intervals of forecasts

pred_ci = pred_uc.conf_int()
ax = c['Count'][-60:].plot(label='observed', figsize=(15, 10))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Counts')

 

plt.legend()

plt.show()
week = createdf("Week",Rawdata["DAY_OF_WEEK"].value_counts().index,"Count",Rawdata["DAY_OF_WEEK"].value_counts())

week=week.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.bar(week["Week"] , week["Count"], width=0.3)

plt.ylim(36000, 50000)

plt.title('Crimes by WeekDay')

plt.show()
target = Rawdata[(Rawdata['DATE'] > "2016-07-01") & (Rawdata['DATE'] < "2017-08-01")]

target = target.sort_values(by="DATE",ascending = True)
t1 = createdf("Date",target["DATE"].value_counts().index,"Count",target["DATE"].value_counts())

t1 = t1.sort_values(by="Date",ascending = True)

lineplt(t1["Date"],t1["Count"],"Date","Count","Crimes by Time(2016-07-01~2017-08-01)",(15,8),80)
test_stationarity(t1['Count'],mlag = 180,lag='AIC')
target.info()
print(target["DISTRICT"].unique())
# target = target.dropna()

fig,ax = plt.subplots(figsize =(15,40))

# ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

i = 0

for dis in target["DISTRICT"].unique():

    if dis is not np.nan :

        i += 1

        da = target[target["DISTRICT"] == dis]

        d = createdf("Date",da["DATE"].value_counts().index,"Count",da["DATE"].value_counts())

        d = d.sort_values(by="Date",ascending = True)

        fig.add_subplot(12,1,i)

        plt.plot(d["Date"],d["Count"])     

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,

                wspace=None, hspace=0.4)

        ax=plt.gca()

        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

        plt.title(dis,fontsize = 20)

plt.show()
def featureScaling(arr):

    scaler = MinMaxScaler(feature_range=(0, 1))

    result = scaler.fit_transform(arr)

    return result
t1["Count"]=featureScaling(t1["Count"].values.reshape(-1,1))

lineplt(t1["Date"],t1["Count"],"Date","Count","Crimes by Time(2016-07-01~2017-08-01)",(15,8),80)
def dtw(seq1,seq2): #动态时间规整：形参为时间序列seq1,seq2 

    m1=len(seq1)

    m2=len(seq2)

    #初始化距离矩阵

    distance=np.zeros(shape=(m1,m2)) #m1行,m2列的距离矩阵

    for i in range(m1):

        for j in range(m2):

            distance[i,j]=(seq1[i]-seq2[j])**2 #一维数组元素之间的欧式距离的平方    

    #构建一个与矩阵d相同大小累积距离矩阵的D

    D=np.zeros(shape=(m1,m2))

    D[0,0]=distance[0,0] #第一个元素和距离矩阵保持一致

    for i in range(1,m1): #累积距离矩阵的左边界

        D[i,0]=distance[i,0]+D[i-1,0]

    for j in range(1,m2):#累积距离矩阵的上边界

        D[0,j]=distance[0,j]+D[0,j-1]

    for i in range(1,m1):

        for j in range(1,m2):

            D[i,j]=distance[i,j]+np.min([D[i-1,j-1],D[i-1,j],D[i,j-1]])

    return D[m1-1,m2-1] #函数返回值为最小动态规划路径
for dis in target["DISTRICT"].unique():

    if dis is not np.nan :

        da = target[target["DISTRICT"] == dis]

        d = createdf("Date",da["DATE"].value_counts().index,"Count",da["DATE"].value_counts())

        d = d.sort_values(by="Date",ascending = True)

        d["Count"]=featureScaling(d["Count"].values.reshape(-1,1))

        print(dis + ' distance: ' + str(dtw(t1["Count"],d["Count"])))
t2 = createdf("District",target['DISTRICT'].value_counts(dropna=False).index,"Count",target['DISTRICT'].value_counts(dropna=False))
t2["Count"].sum()

t2['Percent'] = t2["Count"]/t2["Count"].sum()
fig = plt.figure(figsize=(16,8))

plot = squarify.plot(sizes = t2["Percent"], # 指定绘图数据

                     label = t2["District"], # 指定标签

                     alpha = 0.6, # 指定透明度

                     value = t2["Percent"].apply(lambda x: format(x, '.2%')) , # 添加数值标签

                     edgecolor = 'white', # 设置边界框为白色

                     linewidth =3 # 设置边框宽度为3

                    )

plot.set_title('Crimes by Districts',fontdict = {'fontsize':25})

plt.show()
Rawdata.Lat.replace(-1, None, inplace=True)

Rawdata.Long.replace(-1, None, inplace=True)
fig = plt.figure(figsize=(16,8))

sns.scatterplot(x='Lat',

               y='Long',

                hue='DISTRICT',

                alpha=0.01,

               data=Rawdata)

plt.legend(loc=2)