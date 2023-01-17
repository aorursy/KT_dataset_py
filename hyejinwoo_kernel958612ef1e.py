# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in

import matplotlib.pyplot as plt

import datetime

from datetime import datetime

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import missingno as msno



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



# Any results you write to the current directory are saved as output.
df_train.head(3)
df_train.info()
df_test.info()
df_test['date']  = df_test.datetime.apply(lambda x: x.split()[0])

df_test['hour'] = df_test.datetime.apply(lambda x: x.split()[1].split(':')[0])

df_test['weekday'] = df_test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())

df_test['month'] = df_test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)

df_test.drop(['datetime'], axis=1, inplace=True)

df_test['year']=[d.split('-')[0] for d in df_test.date]

df_test['day']=[d.split('-')[2] for d in df_test.date]
df_train['date']  = df_train.datetime.apply(lambda x: x.split()[0])

df_train['hour'] = df_train.datetime.apply(lambda x: x.split()[1].split(':')[0])
df_train['weekday'] = df_train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())

df_train['month'] = df_train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)
df_train.drop(['datetime'], axis=1, inplace=True)
df_train['year']=[d.split('-')[0] for d in df_train.date]

df_train['day']=[d.split('-')[2] for d in df_train.date]
df_train.head(3)
time1 = df_train[df_train['hour'].isin(['00', '01', '02'])]

time2 = df_train[df_train['hour'].isin(['03', '04', '05'])]

time3 = df_train[df_train['hour'].isin(['06', '07', '08'])]

time4 = df_train[df_train['hour'].isin(['09', '10', '11'])]

time5 = df_train[df_train['hour'].isin(['12', '13', '14'])]

time6 = df_train[df_train['hour'].isin(['15', '16', '17'])]

time7 = df_train[df_train['hour'].isin(['18', '19', '20'])]

time8 = df_train[df_train['hour'].isin(['21', '22', '23'])]
#checktime 변수 생성

df_train['checktime'] = None
df_train.loc[df_train["hour"] == '00', "checktime"] = '1'

df_train.loc[df_train["hour"] == '01', "checktime"] = '1'

df_train.loc[df_train["hour"] == '02', "checktime"] = '1'



df_train.loc[df_train["hour"] == '03', "checktime"] = '2'

df_train.loc[df_train["hour"] == '04', "checktime"] = '2'

df_train.loc[df_train["hour"] == '05', "checktime"] = '2'



df_train.loc[df_train["hour"] == '06', "checktime"] = '3'

df_train.loc[df_train["hour"] == '07', "checktime"] = '3'

df_train.loc[df_train["hour"] == '08', "checktime"] = '3'



df_train.loc[df_train["hour"] == '09', "checktime"] = '4'

df_train.loc[df_train["hour"] == '10', "checktime"] = '4'

df_train.loc[df_train["hour"] == '11', "checktime"] = '4'



df_train.loc[df_train["hour"] == '12', "checktime"] = '5'

df_train.loc[df_train["hour"] == '13', "checktime"] = '5'

df_train.loc[df_train["hour"] == '14', "checktime"] = '5'



df_train.loc[df_train["hour"] == '15', "checktime"] = '6'

df_train.loc[df_train["hour"] == '16', "checktime"] = '6'

df_train.loc[df_train["hour"] == '17', "checktime"] = '6'



df_train.loc[df_train["hour"] == '18', "checktime"] = '7'

df_train.loc[df_train["hour"] == '19', "checktime"] = '7'

df_train.loc[df_train["hour"] == '20', "checktime"] = '7'



df_train.loc[df_train["hour"] == '21', "checktime"] = '8'

df_train.loc[df_train["hour"] == '22', "checktime"] = '8'

df_train.loc[df_train["hour"] == '23', "checktime"] = '8'
df_train.head(100)
time1 = time1['count']

time2 = time2['count']

time3 = time3['count']

time4 = time4['count']

time5 = time5['count']

time6 = time6['count']

time7 = time7['count']

time8 = time8['count']
sum1 = 0

for i in time1:

    sum1 += i

sum2 = 0

for i in time2:

    sum2 += i

sum3 = 0

for i in time3:

    sum3 += i

sum4 = 0

for i in time4:

    sum4 += i

sum5 = 0

for i in time5:

    sum5 += i

sum6 = 0

for i in time6:

    sum6 += i

sum7 = 0

for i in time7:

    sum7 += i

sum8 = 0

for i in time8:

    sum8 += i
#hour을 3시간대 별로 나누어 그린 그래프

y1_value = (sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)

x_name=('time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8')

n_groups = len(x_name)

index = np.arange(n_groups)



plt.bar(index, y1_value, tick_label=x_name, align='center')



plt.xlabel('24hour / 8')

plt.ylabel('count')

plt.title('3hours count')

#plt.xlim( -1, n_groups)

#plt.ylim( 0, 400)plt.show()
df_train.info()
df_train['season'] = df_train['season'].astype(str)

df_train['weather'] = df_train['weather'].astype(str)

df_train['hour'] = df_train['hour'].astype(str)

df_train['weekday'] = df_train['weekday'].astype(str)

df_train['month'] = df_train['month'].astype(str)

df_train['workingday'] = df_train['workingday'].astype(str)

df_train['holiday'] = df_train['holiday'].astype(str)
df_train.info()
fig, axes = plt.subplots(nrows = 2)

fig.set_size_inches(18,30)



sns.pointplot(data=df_train, x="hour", y="count", ax=axes[0])

sns.pointplot(data=df_train, x="checktime", y="count", ax=axes[1])
df_train[df_train.columns[1:]].corr()['count'][:]
# 등록자와 비등록자 count와 시간대별 이용이 다름을 나타내는 그래프

figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



sns.barplot(data=df_train, x="hour", y="casual", ax=ax1)

sns.barplot(data=df_train, x="hour", y="registered", ax=ax2)
# workingday 가 0일때와 1일때 count한 그래프

figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



sns.barplot(data=df_train.loc[df_train["workingday"] == '0'], x="hour", y="count", ax=ax1)

sns.barplot(data=df_train.loc[df_train["workingday"] == '1'], x="hour", y="count", ax=ax2)
figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



sns.barplot(data=df_train.loc[df_train["workingday"] == '0'], x="hour", y="casual", ax=ax1)

sns.barplot(data=df_train.loc[df_train["workingday"] == '0'], x="hour", y="registered", ax=ax2)
figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



sns.barplot(data=df_train.loc[df_train["workingday"] == '1'], x="hour", y="casual", ax=ax1)

sns.barplot(data=df_train.loc[df_train["workingday"] == '1'], x="hour", y="registered", ax=ax2)
fig, axes = plt.subplots(nrows = 6)

fig.set_size_inches(18,30)



sns.pointplot(data=df_train, x="hour", y="count", ax=axes[0])

sns.pointplot(data=df_train, x="hour", y="count", hue = "workingday", ax=axes[1])

sns.pointplot(data=df_train, x="hour", y="count", hue = "holiday", ax=axes[2])

sns.pointplot(data=df_train, x="hour", y="count", hue = "weekday", ax=axes[3])

sns.pointplot(data=df_train, x="hour", y="count", hue = "weather", ax=axes[4])

sns.pointplot(data=df_train, x="hour", y="count", hue = "season", ax=axes[5])
goodweather = df_train.loc[df_train["weather"] == '1']

sosoweather = df_train.loc[df_train["weather"] == '2']

#sosoweather = sosoweather.loc[sosoweather["hour"] == '07']

goodweather = goodweather.loc[goodweather["hour"] == '08']

sosoweather = sosoweather.loc[sosoweather["hour"] == '08']



figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



sns.barplot(data=sosoweather, x="workingday", y="count", ax=ax1)

sns.barplot(data=goodweather, x="workingday", y="count", ax=ax2)
figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



sns.barplot(data=sosoweather, x="season", y="count", ax=ax1)

sns.barplot(data=goodweather, x="season", y="count", ax=ax2)
figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



sns.barplot(data=sosoweather, x="workingday", y="count", hue = "temp", ax=ax1)

sns.barplot(data=goodweather, x="workingday", y="count", hue = "temp", ax=ax2)
figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



#sns.barplot(data=sosoweather, x="humidity", y="count", hue = "atemp", ax=ax1)

#sns.barplot(data=goodweather, x="humidity", y="count", hue = "atemp", ax=ax2)
df_train.info()
df_test['season'] = df_train['season'].astype(str)

df_test['weather'] = df_train['weather'].astype(str)

df_test['hour'] = df_train['hour'].astype(str)

df_test['weekday'] = df_train['weekday'].astype(str)

df_test['month'] = df_train['month'].astype(str)

df_test['workingday'] = df_train['workingday'].astype(str)

df_test['holiday'] = df_train['holiday'].astype(str)
df_test.info()
df_test.head(1)
del df_test['date']
df_train.head(1)
del df_train['date']

del df_train['casual']

del df_train['registered']

del df_train['checktime']
# 모델

'''from sklearn.model_selection import KFold'''
'''yLabels = df_train['count']

del df_train['count']

dataTrain = df_train

dataTest = df_test'''
'''def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))'''
'''from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)



# Initialize logistic regression model

lModel = LinearRegression()



# Train the model

yLabelsLog = np.log1p(yLabels)

lModel.fit(X = dataTrain,y = yLabelsLog)



# Make predictions

preds = lModel.predict(X= dataTrain)

print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))'''
'''ridge_m_ = Ridge()

ridge_params_ = { 'max_iter':[3000],'alpha':[0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}

rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

grid_ridge_m = GridSearchCV( ridge_m_,

                          ridge_params_,

                          scoring = rmsle_scorer,

                          cv=5)

yLabelsLog = np.log1p(yLabels)

grid_ridge_m.fit( dataTrain, yLabelsLog )

preds = grid_ridge_m.predict(X= dataTrain)

print (grid_ridge_m.best_params_)

print ("RMSLE Value For Ridge Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))'''
'''lasso_m_ = Lasso()



alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])

lasso_params_ = { 'max_iter':[3000],'alpha':alpha}



grid_lasso_m = GridSearchCV( lasso_m_,lasso_params_,scoring = rmsle_scorer,cv=5)

yLabelsLog = np.log1p(yLabels)

grid_lasso_m.fit( dataTrain, yLabelsLog )

preds = grid_lasso_m.predict(X= dataTrain)

print (grid_lasso_m.best_params_)

print ("RMSLE Value For Lasso Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))'''