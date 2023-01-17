# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression

import pylab



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '../input/covid19-global-forecasting-week-2/'

train = pd.read_csv(path + 'train.csv')



column_name_map = {

    'Country_Region' : 'country',

    'Province_State' : 'state',

    'Date' : 'date',

    'ConfirmedCases' : 'cases',

    'Fatalities' : 'deaths',

}



train = train.rename(columns = column_name_map)



from_date = train['date'].min()

to_date = train['date'].max()

# print(from_date, to_date)



train['state'].fillna('-', inplace = True)



print(train.dtypes)
path = '../input/covid19-global-forecasting-week-2/'

test = pd.read_csv(path + 'test.csv')



column_name_map = {

    'Country_Region' : 'country',

    'Province_State' : 'state',

    'Date' : 'date',

    'ForecastId' : 'id',

}



test = test.rename(columns = column_name_map)



test['state'].fillna('-', inplace = True)



regions = test[['state', 'country']].drop_duplicates()

regions.head()
def logistic(xs, l, L, k, x_0):

    result = []

    for x in xs:

        xp = k*(x-x_0)

        if xp >= 0:

            result.append(l + (L-l) / ( 1. + np.exp(-xp) ) )

        else:

            result.append(l + (L-l) * np.exp(xp) / ( 1. + np.exp(xp) ) )

    return result



def date_day_diff(d1, d2):

    delta = dt.datetime.strptime(d1, "%Y-%m-%d") - dt.datetime.strptime(d2, "%Y-%m-%d")

    return delta.days
predictions_cases = []

plot = True



for index, region in regions.iterrows():

    predicted = False

    

    st = region['state']

    co = region['country']

    

    rdata = train[(train['state']==st) & (train['country']==co)]

    rtest = test[(test['state']==st) & (test['country']==co)]

    

    window = rdata[rdata['cases']>=100]['date']

    if(window.count() < 10):

        window = rdata[rdata['cases']>=10]['date']



    if(window.count() >= 10):     

        start_date = window.min()

        rdata = rdata[rdata['date']>=start_date]



        t = rdata['date'].values

        t = [float(date_day_diff(d, start_date)) for d in t]

        y = rdata['cases'].values



        try:

            bounds = ([-1e6, -1e6, 0.001, 0.0], [1e6, 1e6, 0.999, t[-1]+10]) # assumes the strongest increase is no more than 10 days away

            popt, pcov = curve_fit(logistic, t, y, bounds = bounds)



            residuals = y - logistic(t, *popt)

            ss_res = np.sum(residuals**2)

            ss_tot = np.sum((y - np.mean(y))**2)

            rs = 1 - (ss_res / ss_tot)



            if plot:

                print(st, co)

                print(popt)

                print('R squared: ', rs)



                T = np.arange(0, 60, 1).tolist()

                yfit = logistic(T, *popt)



                pylab.plot(t, y, 'o')

                pylab.plot(T, yfit)

                pylab.show()



            if rs>=0.95:

                for index, rt in rtest.iterrows():

                    tdate = rt['date']

                    prev_max = 0

                    if(tdate<=to_date):

                        ca = list(train[(train['date']==tdate) & (train['state']==st) & (train['country']==co)]['cases'].values)[0]

                        prev_max = max(prev_max, ca)

                    else:

                        ca = logistic([date_day_diff(tdate, start_date)], *popt)

                        ca = max(prev_max, ca[0])

                        prev_max = ca

                    predictions_cases.append((rt['id'], int(ca)))



                predicted = True



        except:

            pass

    

    if not predicted:

        # try linear regression with the latest 10 values



        t = rdata['date'].values

        start_date = t[0]

        t = np.array([float(date_day_diff(d, start_date)) for d in t])

        y = rdata['cases'].values

        

        linreg = LinearRegression()  

        linreg.fit(t[-10:].reshape(-1, 1) , y[-10:])

        

        m = linreg.coef_[0]

        b = linreg.intercept_

        

        if plot:

            print(st, co)

            print(linreg.intercept_, linreg.coef_)

            

            T = np.arange(0, 90, 1).tolist() 

            y_pred = [m*x+b for x in T]



            pylab.plot(t, y, 'o')

            pylab.plot(T, y_pred)

            pylab.show()

        

        for index, rt in rtest.iterrows():

            tdate = rt['date']

            prev_max = 0

            if(tdate<=to_date):

                ca = list(train[(train['date']==tdate) & (train['state']==st) & (train['country']==co)]['cases'].values)[0]

                prev_max = max(prev_max, ca)

            else:

                ca = m*date_day_diff(tdate, start_date) + b

                ca = max(ca, prev_max)

                prev_max = ca

            predictions_cases.append((rt['id'], int(ca)))
data = {

    'ForecastId': [pred[0] for pred in predictions_cases],

    'ConfirmedCases': [pred[1] for pred in predictions_cases],

}

df_cases = pd.DataFrame (data, columns = data.keys())

df_cases.head()
predictions_deaths = []

plot = True



for index, region in regions.iterrows():

    predicted = False

    

    st = region['state']

    co = region['country']

    

    rdata = train[(train['state']==st) & (train['country']==co)]

    rtest = test[(test['state']==st) & (test['country']==co)]

    

    window = rdata[rdata['deaths']>=100]['date']

    if(window.count() < 10):

        window = rdata[rdata['deaths']>=10]['date']



    if(window.count() >= 10):     

        start_date = window.min()

        rdata = rdata[rdata['date']>=start_date]



        t = rdata['date'].values

        t = [float(date_day_diff(d, start_date)) for d in t]

        y = rdata['deaths'].values



        try:

            bounds = ([-1e6, -1e6, 0.001, 0.0], [1e6, 1e6, 0.999, t[-1]+16]) # assumes the strongest increase is no more than 16 days away

            popt, pcov = curve_fit(logistic, t, y, bounds = bounds)



            residuals = y - logistic(t, *popt)

            ss_res = np.sum(residuals**2)

            ss_tot = np.sum((y - np.mean(y))**2)

            rs = 1 - (ss_res / ss_tot)



            if plot:

                print(st, co)

                print(popt)

                print('R squared: ', rs)



                T = np.arange(0, 60, 1).tolist()

                yfit = logistic(T, *popt)



                pylab.plot(t, y, 'o')

                pylab.plot(T, yfit)

                pylab.show()



            if rs>=0.95:

                for index, rt in rtest.iterrows():

                    tdate = rt['date']

                    prev_max = 0

                    if(tdate<=to_date):

                        ca = list(train[(train['date']==tdate) & (train['state']==st) & (train['country']==co)]['deaths'].values)[0]

                        prev_max = max(prev_max, ca)

                    else:

                        ca = logistic([date_day_diff(tdate, start_date)], *popt)

                        ca = max(prev_max, ca[0])

                        prev_max = ca

                    predictions_deaths.append((rt['id'], int(ca)))



                predicted = True



        except:

            pass

    

    if not predicted:

        # try linear regression with the latest 10 values



        t = rdata['date'].values

        start_date = t[0]

        t = np.array([float(date_day_diff(d, start_date)) for d in t])

        y = rdata['deaths'].values

        

        linreg = LinearRegression()  

        linreg.fit(t[-10:].reshape(-1, 1) , y[-10:])

        

        m = linreg.coef_[0]

        b = linreg.intercept_

        

        if plot:

            print(st, co)

            print(linreg.intercept_, linreg.coef_)

            

            T = np.arange(0, 90, 1).tolist() 

            y_pred = [m*x+b for x in T]



            pylab.plot(t, y, 'o')

            pylab.plot(T, y_pred)

            pylab.show()

        

        for index, rt in rtest.iterrows():

            tdate = rt['date']

            prev_max = 0

            if(tdate<=to_date):

                ca = list(train[(train['date']==tdate) & (train['state']==st) & (train['country']==co)]['deaths'].values)[0]

                prev_max = max(prev_max, ca)

            else:

                ca = m*date_day_diff(tdate, start_date) + b

                ca = max(ca, prev_max)

                prev_max = ca

            predictions_deaths.append((rt['id'], int(ca)))
data = {

    'ForecastId': [pred[0] for pred in predictions_deaths],

    'Fatalities': [pred[1] for pred in predictions_deaths],

}

df_deaths = pd.DataFrame (data, columns = data.keys())

df_deaths.head()
df_submission = df_cases.join(df_deaths.set_index('ForecastId'), on = 'ForecastId')

df_submission.to_csv('submission.csv', index=False)

df_submission.head()