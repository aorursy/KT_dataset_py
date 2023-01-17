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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from scipy.special import expit

from scipy.optimize import curve_fit
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv', dtype={'Id': int, 'ConfirmedCases': int, 'Fatalities': int})

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv', dtype={'ForecastId': int})

df_sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')



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
print(df_test['Country_Region'].unique())

display(df_test.loc[(df_test['Country_Region']=='US') & (df_test['Province_State']=='Alabama')])
mysigmoid = lambda x, a, b, c: a * expit(b * (x-c))
# days_low (lowest day to include in the curve fitting)



days_low = {key: 20 for key in df_train['unique_key'].unique()}

days_low.update({

    # adding manual days_low here

})
params_confirmed = {}

for key in df_train['unique_key'].unique():

    print(key)

    

    df = df_train.loc[df_train['unique_key'] == key]

    a, b, c = df['ConfirmedCases'].max()*2, 0.5, df.loc[df['ConfirmedCases']!=0, 'days'].min()+21  # initial

    

    try:

        params, _ = curve_fit(mysigmoid, df.loc[df['days'] >= days_low[key], 'days'], df.loc[df['days'] >= days_low[key], 'ConfirmedCases'], p0=[a, b, c])

        a, b, c = params

    except:

        print('Warning: for key {} cannot find curve, manually write one below. (a0, b0, c0) {}, {}, {}'.format(key, a, b, c))

    params_confirmed[key] = (a, b, c)



    plt.plot(df['days'], df['ConfirmedCases'], '*', label='actual data')

    pred = [mysigmoid(x,a,b,c) for x in range(100)]

    plt.plot(range(100), pred, label='my-prediction cuve')

    plt.show()
display(params_confirmed)
# Modify params_confirmed here



#key = 'Azerbaijan_NaN'

#df = df_train.loc[df_train['unique_key'] == key]

#plt.plot(df['days'], df['ConfirmedCases'], '*', label='actual data')

#a, b, c = 596, 0.5, 54.0

#params_confirmed[key] = (a,b,c)

#pred = [mysigmoid(x,a,b,c) for x in range(100)]

#plt.plot(range(100), pred, label='my-prediction cuve')
# days_low (lowest day to include in the curve fitting)



days_low = {key: 20 for key in df_train['unique_key'].unique()}

days_low.update({

    # adding manual days_low here

})
multiple = 2.5  # use to linear fit 



params_fatalities = {}

for key in df_train['unique_key'].unique():

    print(key)

    

    df = df_train.loc[df_train['unique_key'] == key]

    a, b, c = df['Fatalities'].max()*2, 0.5, df.loc[df['Fatalities']!=0, 'days'].min()+21  # initial

    

    plt.plot(df['days'], df['Fatalities'], '*', label='actual data')



    fmax = df['Fatalities'].max()

    if fmax <= 10:

        params_fatalities[key] = [fmax, fmax*multiple]

        pred = [fmax for x in range(69)] + [fmax + (x-69)/(99-69)*(multiple-1)*fmax for x in range(69,100)]

    else:

        try:

            params, _ = curve_fit(mysigmoid, df.loc[df['days'] >= days_low[key], 'days'], df.loc[df['days'] >= days_low[key], 'Fatalities'], p0=[a, b, c])

            a, b, c = params

        except:

            print('Warning: for key {} cannot find curve, manually write one below. (a0, b0, c0) {}, {}, {}'.format(key, a, b, c))    

        params_fatalities[key] = (a, b, c)

        pred = [mysigmoid(x,a,b,c) for x in range(100)]

    

    plt.plot(range(100), pred, label='my-prediction cuve')

    plt.show()
display(params_fatalities)
# Modify params_confirmed here



#key = 'Azerbaijan_NaN'

#df = df_train.loc[df_train['unique_key'] == key]

#plt.plot(df['days'], df['Fatalities'], '*', label='actual data')

#a, b, c = 596, 0.5, 54.0

#params_fatalities[key] = (a,b,c)

#pred = [mysigmoid(x,a,b,c) for x in range(100)]

#plt.plot(range(100), pred, label='my-prediction cuve')
pred_fid = []

pred_confirmed = []

pred_fatalities = []



for ind in df_test.index:

    pred_fid.append(df_test.loc[ind,'ForecastId'])

    

    key = df_test.loc[ind, 'unique_key']

    day = df_test.loc[ind, 'days']

    

    # confirmed

    a, b, c = params_confirmed[key]

    pred_confirmed.append( mysigmoid(day, a, b, c) )

    

    # fatalities

    if len(params_fatalities[key]) == 2:

        y0, y1 = params_fatalities[key]

        pred = y0 + (day-69)/(99-69)*(y1-y0)



        pred_fatalities.append( params_fatalities[key][0] )

    else:

        a, b, c = params_fatalities[key]

        pred_fatalities.append( mysigmoid(day, a, b, c) )



# out

df_out = pd.DataFrame({'ForecastId': pred_fid, 'ConfirmedCases': np.around(pred_confirmed).astype('int'), 'Fatalities': np.around(pred_fatalities).astype('int')})

display(df_out.head(10)); display(df_out.tail(10))

df_out.to_csv('submission.csv',index=False)