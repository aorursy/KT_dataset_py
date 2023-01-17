# IMPORTS

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)
%%time

# LOAD TRAIN DATA

train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
df = train[train['Country_Region']=='Pakistan']

Pakistan_data = df.copy()

Pakistan_data= Pakistan_data[Pakistan_data.ConfirmedCases > 0.0]

Pakistan_data.reset_index(inplace=True)

Pakistan_data.drop(columns= ['index','Id','Province_State'],inplace=True)

Pakistan_data['ConfirmedCases'] = Pakistan_data['ConfirmedCases'].astype(int) 

Pakistan_data['Fatalities'] = Pakistan_data['Fatalities'].astype(int) 

Pakistan_data.head()
month_day_list = []

for date in Pakistan_data['Date']:

    month_day_list.append(date.split('2020-0')[1])



Pakistan_data['Month_Day'] = month_day_list

Pakistan_data.head()
def Calculate_Table ( X_train ):

    # CALCULATE EXPANSION TABLE

    diff_conf, conf_old = [], 0 

    diff_fat, fat_old = [], 0

    dd_conf, dc_old = [], 0

    dd_fat, df_old = [], 0

    ratios = []

    for row in X_train.values:

        diff_conf.append(row[2]-conf_old)

        conf_old = row[2]

        diff_fat.append(row[3]-fat_old)

        fat_old = row[3]

        dd_conf.append(diff_conf[-1]-dc_old)

        dc_old = diff_conf[-1]

        dd_fat.append(diff_fat[-1]-df_old)

        df_old = diff_fat[-1]

        ratios.append(fat_old / conf_old)

        ratio = fat_old / conf_old

        



    return diff_conf, conf_old, diff_fat, fat_old, dd_conf, dc_old, dd_fat, df_old, ratios, ratio
def populate_df_features(X_train,diff_conf, diff_fat, dd_conf, dd_fat, ratios):    

    # POPULATE DATAFRAME FEATURES

    pd.options.mode.chained_assignment = None  # default='warn'

    X_train['diff_confirmed'] = diff_conf

    X_train['diff_fatalities'] = diff_fat

    X_train['dd_confirmed'] = dd_conf

    X_train['dd_fatalities'] = dd_fat

    X_train['ratios'] = ratios

    return X_train
def fill_nan ( variable):

    if math.isnan(variable):

        return 0

    else:

        return variable
def Cal_Series_Avg(X_train,ratio):

    # CALCULATE SERIES AVERAGES

    d_c = fill_nan( X_train.diff_confirmed[X_train.diff_confirmed != 0].mean() )

    dd_c = fill_nan( X_train.dd_confirmed[X_train.dd_confirmed != 0].mean() )

    d_f = fill_nan( X_train.diff_fatalities[X_train.diff_fatalities != 0].mean() )

    dd_f = fill_nan( X_train.dd_fatalities[X_train.dd_fatalities != 0].mean() )

    rate = fill_nan( X_train.ratios[X_train.ratios != 0].mean() )

    print("rate: %.2f ratio: %.2f" %(rate,ratio))

    print("d_c: %.2f, dd_c: %.2f, d_f: %.2f, dd_f: %.2f "%(d_c, dd_c, d_f, dd_f))

    rate = max(rate,ratio)

    return d_c, dd_c, d_f, dd_f, rate
def apply_taylor(train, d_c, dd_c, d_f, dd_f, rate):

    # ITERATE TAYLOR SERIES

    

    pred_c, pred_f = [],[]

    for i in range(1, 34):

        pred_c.append(int( ( train.ConfirmedCases[len(train)-1] + d_c*i + 0.5*dd_c*(i**2)) ) )

        pred_f.append(pred_c[-1]*rate )

    return pred_c, pred_f


diff_conf, conf_old, diff_fat, fat_old, dd_conf, dc_old, dd_fat, df_old, ratios, ratio = Calculate_Table(Pakistan_data)



Pakistan_data = populate_df_features(Pakistan_data,diff_conf, diff_fat, dd_conf, dd_fat, ratios)



d_c, dd_c, d_f, dd_f, rate = Cal_Series_Avg(Pakistan_data, ratio)



pc, pf = apply_taylor(Pakistan_data, d_c, dd_c, d_f, dd_f, rate)
len(pc), len(pf)
Pakistan_data.iloc[:,:]
Pakistan_data.shape
dates = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')

dates.drop(columns= ['ForecastId','Province_State','Country_Region'],inplace=True)

dates = dates.iloc[10:43,:]



pd_list = []

for date in dates.Date:

    pd_list.append(date.split('2020-0')[1])



Date_list = list(Pakistan_data.Month_Day.copy())

Date_list.extend(pd_list)

#Date_list
plt.figure(figsize=(25,6))



tpc = list(Pakistan_data.ConfirmedCases.copy())

tpc.extend(pc)

plt.plot(Date_list,tpc,'r',linestyle='dashed',label='Prediction')

plt.plot(Pakistan_data.ConfirmedCases,label='Confirmed')

plt.xlabel("Days passed since first Confirmed",fontdict={'fontsize': 18})

plt.ylabel("Total Confirmed Cases",fontdict={'fontsize': 18})

plt.legend(fontsize= 18)

plt.title('Pakistan COVID19 Confirmed Cases',fontdict={'fontsize': 28})

plt.xticks(rotation = 45)

#plt.yscale('log')

#plt.xscale('log')

plt.show()

# AR example

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.statespace.sarimax import SARIMAX

from random import random
data = Pakistan_data.ConfirmedCases



# fit model

model = AR(Pakistan_data.ConfirmedCases)

model_fit = model.fit()



# make prediction

start = len(Pakistan_data.ConfirmedCases)

yhat = model_fit.predict(start,71)

pcc= list(data[0:start])



pcc.extend(list(yhat))

print(len(pcc))
pcc2 = pcc[30:]

dl = Date_list[30:]

for i,d in enumerate(dl):

    dl[i] = d.replace("3-","Mar-")

    dl[i] = dl[i].replace("4-","Apr-")

    dl[i] = dl[i].replace("5-","May-")

cc = Pakistan_data.ConfirmedCases[30:]

plt.figure(figsize=(20,7))

plt.plot(dl,pcc2,'r',linestyle='dashed',label='Prediction')

plt.plot(dl[:9],cc,label='Confirmed')

plt.xlabel("Date",fontdict={'fontsize': 18})

plt.ylabel("Total Confirmed Cases",fontdict={'fontsize': 18})

plt.legend(fontsize= 18)

plt.title('Pakistan COVID19 Confirmed Cases (Autoregression) ',fontdict={'fontsize': 28})

plt.xticks(rotation=45)

plt.show()
models = SARIMAX(Pakistan_data.ConfirmedCases, order=(1, 0, 0), trend='t')

models_fit = models.fit()

# make prediction

start = len(Pakistan_data.ConfirmedCases)

yhats = models_fit.predict(start,71)

pccs= list(data[0:start])

pccs.extend(list(yhats))

pccs2 = pccs[30:]

dls = Date_list[30:]

for i,d in enumerate(dls):

    dls[i] = d.replace("3-","Mar-")

    dls[i] = dls[i].replace("4-","Apr-")

    dls[i] = dls[i].replace("5-","May-")

ccs = Pakistan_data.ConfirmedCases[30:]

plt.figure(figsize=(20,7))

plt.plot(dls,pccs2,'r',linestyle='dashed',label='Prediction')

plt.plot(dls[:9],ccs,label='Confirmed')

plt.xlabel("Date",fontdict={'fontsize': 18})

plt.ylabel("Total Confirmed Cases",fontdict={'fontsize': 18})

plt.legend(fontsize= 18)

plt.title('Pakistan COVID19 Confirmed Cases (SARIMAX) ',fontdict={'fontsize': 28})

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(12,6))



tpf = list(Pakistan_data.Fatalities.copy())

tpf.extend(pf)

plt.plot(tpf,'r',linestyle='dashed',label='Prediction')

plt.plot(Pakistan_data.Fatalities,label='Fatalities')

plt.xlabel("Days passed since first Confirmed",fontdict={'fontsize': 18})

plt.ylabel("Total Fatalities",fontdict={'fontsize': 18})

plt.legend(fontsize= 18)

plt.title('Pakistan COVID19 Fatalities',fontdict={'fontsize': 28})

plt.xticks(rotation=45)

plt.show()

data = Pakistan_data.Fatalities

data_len = len(data)

to_predict = range(data_len+1, data_len+30)

# fit model

model = AR(Pakistan_data.Fatalities)

model_fit = model.fit()

# make prediction

yhat = model_fit.predict(start,71)

pfc= list(data[0:start])

pfc.extend(list(yhat))

#print(pfc)
plt.figure(figsize=(12,6))



plt.plot(pfc,'r',linestyle='dashed',label='Prediction')

plt.plot(Pakistan_data.Fatalities,label='Fatalities')

plt.xlabel("Days passed since first Confirmed",fontdict={'fontsize': 18})

plt.ylabel("Total Fatalities",fontdict={'fontsize': 18})

plt.legend(fontsize= 18)

plt.title('Pakistan COVID19 Fatalities  (Autoregression) ',fontdict={'fontsize': 28})

#plt.yscale('log')

#plt.xscale('log')

plt.show()