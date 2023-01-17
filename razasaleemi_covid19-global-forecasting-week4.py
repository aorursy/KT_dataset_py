# IMPORTS

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# AR example

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.statespace.sarimax import SARIMAX

from random import random



import math
%%time

# LOAD TRAIN DATA

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
train.tail()
c = list()

for i,x in enumerate(train['Province_State']):

    if x is not np.nan:

        c.append(x+ ' - '+train['Country_Region'][i])

    else:

        c.append(train['Country_Region'][i])

        

print(len(c))
# SCRUB DATA

junk =['Id','Date','Province_State']

train.drop(junk, axis=1, inplace=True)
train['Country_Region'] = c
train['ConfirmedCases'] = train['ConfirmedCases'].astype(int) 

train['Fatalities'] = train['Fatalities'].astype(int) 
train.head()
end = 84

country_list = train['Country_Region'][0::end]

print(len(country_list))

print(country_list)
def prep_data (train):

    # PREP TRAIN DATA 

    X_train = train[train.ConfirmedCases >0]

    X_train.reset_index(inplace = True, drop = True) 

    

    train.reset_index(inplace = True, drop = True) 

    

    return (X_train, train)
def Calculate_Table ( X_train ):

    # CALCULATE EXPANSION TABLE

    diff_conf, conf_old = [], 0 

    diff_fat, fat_old = [], 0

    dd_conf, dc_old = [], 0

    dd_fat, df_old = [], 0

    ratios = []

    for row in X_train.values:

        diff_conf.append(row[1]-conf_old)

        conf_old = row[1]

        diff_fat.append(row[2]-fat_old)

        fat_old = row[2]

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

    #print("rate: %.2f ratio: %.2f" %(rate,ratio))

    rate = max(rate,ratio)

    return d_c, dd_c, d_f, dd_f, rate
def apply_taylor(train, d_c, dd_c, d_f, dd_f, rate, end):

    # ITERATE TAYLOR SERIES

    

    pred_c, pred_f = list(train.ConfirmedCases.loc[end-12:end-1].astype(int)), list(train.Fatalities.loc[end-12:end-1].astype(int))

    #pred_c, pred_f = list(train.ConfirmedCases.loc[57:58].astype(int)), list(train.Fatalities.loc[57:58].astype(int))

    for i in range(1, 32):

        pred_c.append(int( ( train.ConfirmedCases[end-1] + d_c*i + 0.5*dd_c*(i**2)) ) )

        pred_f.append(pred_c[-1]*rate )

    return pred_c, pred_f
def apply_taylor2(train, d_c, dd_c, d_f, dd_f, rate, end ):

    # ITERATE TAYLOR SERIES

    

    #pred_c, pred_f = list(train.ConfirmedCases.loc[57:69].astype(int)), list(train.Fatalities.loc[57:69].astype(int))

    pred_c, pred_f = list(train.ConfirmedCases.loc[end-12:end-11].astype(int)), list(train.Fatalities.loc[end-2:end-1].astype(int))

    for i in range(1, 42):

        pred_c.append(int( ( train.ConfirmedCases[end-1] + d_c*i + 0.5*dd_c*(i**2)) ) )

        pred_f.append(pred_c[-1]*rate )

    return pred_c, pred_f
pc = []

pf = []

pc2 = []

pf2 = []

pcS = []

pfS = []

pred_c = []

pred_f = []

pred_c2 = []

pred_f2 = []

pred_c_S = []

pred_f_S = []

for i,country in enumerate(country_list):

    country_data = train[train['Country_Region'] == country]

    X_train, country_data = prep_data(country_data)

    

    if ( len(X_train) > 0):

        diff_conf, conf_old, diff_fat, fat_old, dd_conf, dc_old, dd_fat, df_old, ratios, ratio = Calculate_Table(X_train)



        X_train = populate_df_features(X_train,diff_conf, diff_fat, dd_conf, dd_fat, ratios)



        d_c, dd_c, d_f, dd_f, rate = Cal_Series_Avg(X_train, ratio)

        #print(type(np.nan))

        pred_c, pred_f = apply_taylor(country_data, d_c, dd_c, d_f, dd_f, rate, end)

        pred_c2, pred_f2 = apply_taylor2(country_data, d_c, dd_c, d_f, dd_f, rate, end)

        

        adj= end - len(X_train.ConfirmedCases)

        if ( (end-12-adj) > 10  & len(X_train.ConfirmedCases)>2):

            model = SARIMAX(X_train.ConfirmedCases, order=(1, 0, 0), trend='t')

            model_fit = model.fit()

            # make prediction

            

            pred_c_S = list(model_fit.predict(len(X_train.ConfirmedCases),113-adj))

            my = []

            s= 43-(114-end)

            my.extend(pred_c2[0:s])

            my.extend(pred_c_S)

            pred_c_S = my.copy()

            if i==0:

                print(pred_c_S)

                print(country, len(pred_c_S), s)

            modelf = SARIMAX(X_train.Fatalities, order=(1, 0, 0), trend='t')

            modelf_fit = modelf.fit()

            # make prediction

            pred_f_S = list(modelf_fit.predict(end-adj,113-adj))

            my = []



            my.extend(pred_f2[0:s])

            my.extend(pred_f_S)

            pred_f_S = my.copy()

        else:

            pred_c_S = pred_c2

            pred_f_S = pred_f2

    else:

        #print('--Zeroing--')

        pred_c = list(np.zeros(43))

        pred_f = list(np.zeros(43))

        pred_c2 = list(np.zeros(43))

        pred_f2 = list(np.zeros(43))

        pred_c_S = list(np.zeros(43))

        pred_f_S = list(np.zeros(43))

        

    #print(country, len(pred_c_S))

    #print("------------")

    pc += pred_c

    pf += pred_f

    pc2 += pred_c2

    pf2 += pred_f2

    pcS += pred_c_S

    pfS += pred_f_S
len(pc), len(pcS), len(pc2)
pc = list(map(int, pc))

pf = list(map(int, pf))

pc2 = list(map(int, pc2))

pf2 = list(map(int, pf2))

pcS = list(map(int, pcS))

pfS = list(map(int, pfS))
import matplotlib.pyplot as plt
plt.figure(figsize= (15,6))

plt.plot(pc2[:43*4],'r')

plt.plot(pcS[:43*4])

plt.title("Confirmed")

plt.show()
plt.figure(figsize= (15,6))

plt.plot(pf2[:1000],'r')

plt.plot(pfS[:1000])

plt.title("Fatalities")

plt.show()
# WRITE SUBMISSION

my_submission = pd.DataFrame({'ForecastId': list(range(1,len(pcS)+1)), 'ConfirmedCases': pc2, 'Fatalities': pf2})

print(my_submission)

my_submission.to_csv('submission.csv', index=False)