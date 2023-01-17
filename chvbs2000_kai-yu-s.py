# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



from scipy.special import expit

from scipy.optimize import curve_fit
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv', dtype={'Id': int, 'ConfirmedCases': int, 'Fatalities': int})

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv', dtype={'ForecastId': int})

df_sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')



df_train['Date'] = pd.to_datetime(df_train['Date'])

df_test['Date'] = pd.to_datetime(df_test['Date'])

day0 = df_train['Date'].min()



df_train['days'] = (df_train['Date'] - day0).dt.days

df_test['days'] = (df_test['Date'] - day0).dt.days



df_train['unique_key'] = df_train['Country_Region'] + '_' + df_train['Province_State'].fillna('NaN')

df_test['unique_key'] = df_test['Country_Region'] + '_' + df_test['Province_State'].fillna('NaN')



display(df_train.head())

display(df_train.tail())

display(df_test.head())

display(df_sub.head())
exp_formula = lambda x, a, b, c: a * np.exp(b*(x-c))
mysigmoid = lambda x, a, b, c: a * expit(b * (x-c))
days_low = {key: 28 for key in df_train['unique_key'].unique()}

days_low.update({

    # adding manual days_low here

})
plt.figure(figsize = (6,3))

x = np.arange(df_train['days'].max()+30)

a = 10; b = 2.; c = 10

plt.plot(x, exp_formula(x, a, b, c))

plt.grid(); plt.show()



plt.figure(figsize = (6,3))

a = 10; b = .02; c = 10

plt.plot(x, exp_formula(x, a, b, c))

plt.grid(); plt.show()



df_train.loc[df_train['Country_Region']=='China']
days_low = {key: 21 for key in df_train['unique_key'].unique()}

days_low.update({

    # adding manual days_low here

})


params_confirmed = {}

for key in df_train['unique_key'].unique():

    print(key)

    df = df_train.loc[df_train['unique_key'] == key]

    if 'China' in key:

        a, b, c = df['ConfirmedCases'].max()*(2**5), 0.05, df.loc[df['ConfirmedCases']!=0, 'days'].min()+45  # initial

        mysigmoid_fixa_c = lambda x, b, c: a * expit(b * (x-c))

        try:

            params, _ = curve_fit(mysigmoid_fixa_c, df.loc[df['days'] >= 0, 'days'], df.loc[df['days'] >= 0, 'ConfirmedCases'], p0=[b, c])

            b, c = params

        except:

            print('Warning: for key {} cannot find curve, manually write one below. (a0, b0, c0) {}, {}, {}'.format(key, a, b, c))

        

        params_confirmed[key] = (a, b, c)

        plt.plot(df['days'], df['ConfirmedCases'], '*', label='actual data')

        pred = [mysigmoid_fixa_c(x,b,c) for x in range(110)]

        plt.plot(range(110), pred, label='my-prediction cuve')

        plt.show()

        

    else:

        y0 = df[df['ConfirmedCases']> 5]['ConfirmedCases'].min()

        a, b, c = y0 if not 'nan' else 5, .095, df.loc[df['ConfirmedCases']!=0, 'days'].min(),   # initial

        exp_formula_fix = lambda x, c: a * np.exp(b*(x-c))

        try:

            params, _ = curve_fit(exp_formula_fix, df.loc[df['days'] >= days_low[key], 'days'], df.loc[df['days'] >= days_low[key], 'ConfirmedCases'], p0=[c])

            c = params

        except:

            print('Warning: for key {} cannot find curve, manually write one below. (a0, b0, c0) {}, {}, {}, {}'.format(key, a, b, c)) 



        params_confirmed[key] = (a, b, c[0])

        plt.plot(df['days'], df['ConfirmedCases'], '*', label='actual data')

        pred = [exp_formula_fix(x,c[0]) for x in range(110)]

        plt.plot(range(110), pred, label='my-prediction cuve')

        plt.show()

    
# days_low (lowest day to include in the curve fitting)



days_low = {key: 0 for key in df_train['unique_key'].unique()}

days_low.update({

    # adding manual days_low here

})
multiple = 4.6  # use to linear fit 

#linear_model = lambda x, a, b: int(a*x+b)



params_fatalities = {}

for key in df_train['unique_key'].unique():

    print(key)

    df = df_train.loc[df_train['unique_key'] == key]

    y0 = df[df['Fatalities']> 10]['Fatalities'].min() if not 'nan' else 10

    start_day = df.loc[df['Fatalities']!=0, 'days'].min() if not 'nan' else 42

    a, b, c = y0, 0.0999, start_day   # initial

    

    # fix a

    exp_model_fix_f = lambda x, c: a * np.exp(b*(x-c))

    plt.plot(df['days'], df['Fatalities'], '*', label='actual data')

    fmax = df['Fatalities'].max()

    if "China" in key:

        pred = [0.5*fmax*np.log(x) for x in range(110)]

        

    else:

        if fmax < 1:

            params_fatalities[key] = [fmax, fmax*multiple]

            #pred = [fmax for x in range(69)] + [fmax + (x-69)/(99-69)*(multiple-1)*fmax for x in range(69,120)] #by sky

            pred = [fmax for x in range(69)] + [fmax + (x-69)/(99-69)*(multiple)*fmax for x in range(69,110)] 

        else:

            try:

                #fix a

                params, _ = curve_fit(exp_model_fix_f, df.loc[df['days'] >= start_day+14, 'days'], df.loc[df['days'] >= start_day+14, 'Fatalities'], p0=[c])

                c = params



            except:

                print('Warning: for key {} cannot find curve, manually write one below. (a0, b0, c0) {}, {}, {}'.format(key, a, b, c))    

#             print (c)

#             print ('\n',c[0])

            params_fatalities[key] = (a, b, c[0])

            #pred = [mysigmoid(x,a,b,c) for x in range(100)]

            pred = [exp_model_fix_f(x, c[0]) for x in range(110)] #fix a

            

    #print (a,b,c)

    plt.plot(range(110), pred, label='my-prediction cuve')

    plt.show()

    

    

    
# key = 'Barbados_NaN'

# df = df_train.loc[df_train['unique_key'] == key]

# plt.plot(df['days'], df['Fatalities'], '*', label='actual data')

# start_day = df.loc[df['Fatalities']!=0, 'days'].min() if not 'nan' else 1

# #print (df.loc[df['days'] >= 42])

# a, b, c = 5, 0.168,70

# params_fatalities[key] = (a,b,c)

# pred = [exp_formula(x,a,b,c) for x in range(100)]

# plt.plot(range(100), pred, label='my-prediction cuve')
pred_fid = []

pred_confirmed = []

pred_fatalities = []



for ind in df_test.index:

    pred_fid.append(df_test.loc[ind,'ForecastId'])

    

    key = df_test.loc[ind, 'unique_key']

    day = df_test.loc[ind, 'days']

    fmax = df_train.loc[ind, 'Fatalities'].max()

    

    # confirmed

    if key == "Diamond Princess_NaN":

        pred_confirmed.append(712)

    elif key == "China_Tibet":

        pred_confirmed.append(1)

    elif key == "China_Xinjiang":

        pred_confirmed.append(76)

    elif key == "China_Qinghai":

        pred_confirmed.append(18)

    elif key == "China_Shandong":

        pred_confirmed.append(777)

    else:

        a, b, c = params_confirmed[key]

        #print (exp_formula(day, a, b, c))

        pred_confirmed.append(exp_formula(day, a, b, c))

    

    # fatalities

    if "China" in key:

        # print (0.5*fmax*np.log(day-40))

        #pred_fatalities.append(fmax+0.5*day)

        pred_fatalities.append(0.5*fmax*np.log(day-40))

#     elif key == 'France_NaN':

#         a, b, c = 250000, 0.45, 77

#         pred_fatalities.append( exp_formula(day, a, b, c) )

#     elif key == 'Iran_NaN':

#         a, b, c = 250000, 0.45, 79

#         pred_fatalities.append( exp_formula(day, a, b, c) )

#     elif key == 'Iran_NaN':

#         a, b, c = 250000, 0.45, 79

#         pred_fatalities.append( exp_formula(day, a, b, c) )

#     elif key == "Japan_NaN":

#         a, b, c = 1000, 0.06, 115

#         pred_fatalities.append( exp_formula(day, a, b, c) )

#     elif key == "Philippines_NaN":

#         a, b, c = 26000, 0.3, 82.5

#         pred_fatalities.append( exp_formula(day, a, b, c) )

    elif len(params_fatalities[key]) == 2:

        y0, y1 = params_fatalities[key]

        pred = y0 + (day-69)/(99-69)*(y1-y0)

        pred_fatalities.append( params_fatalities[key][0] )

    else:

        a, b, c = params_fatalities[key]

        #print (exp_formula(day, a, b, c)[0])

        pred_fatalities.append( exp_formula(day, a, b, c) )

print (pred_confirmed[12])

#np.around(pred_confirmed).astype('int')

#np.around(pred_fatalities).astype('int')
# out

df_out = pd.DataFrame({'ForecastId': pred_fid, 'ConfirmedCases': np.around(pred_confirmed).astype('int'), 'Fatalities': np.around(pred_fatalities).astype('int')})

display(df_out.head(10)); display(df_out.tail(10))

df_out.to_csv('submission.csv',index=False)