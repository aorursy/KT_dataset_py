print("Read in libraries")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA

from random import random
print("read in train file")

df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv",

               usecols=['Province_State','Country_Region','Date','ConfirmedCases','Fatalities'])

print("fill blanks and add region for counting")

df.fillna(' ',inplace=True)

df['Lat']=df['Province_State']+df['Country_Region']

df.drop('Province_State',axis=1,inplace=True)

df.drop('Country_Region',axis=1,inplace=True)





countries_list=df.Lat.unique()

df1=[]

for i in countries_list:

    df1.append(df[df['Lat']==i])

print("we have "+ str(len(df1))+" regions in our dataset")



#read in test file 

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
#create the estimates assuming measurement error 

submit_confirmed=[]

submit_fatal=[]

for i in df1:

    # contrived dataset

    data = i.ConfirmedCases.astype('int32').tolist()

    # fit model

    try:

        #model = SARIMAX(data, order=(2,1,0), seasonal_order=(1,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        model = SARIMAX(data, order=(1,1,0), seasonal_order=(1,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        #model = SARIMAX(data, order=(1,1,0), seasonal_order=(0,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        #model = ARIMA(data, order=(3,1,2))

        model_fit = model.fit(disp=False)

        # make prediction

        predicted = model_fit.predict(len(data), len(data)+34)

        new=np.concatenate((np.array(data),np.array([int(num) for num in predicted])),axis=0)

        submit_confirmed.extend(list(new[-43:]))

    except:

        submit_confirmed.extend(list(data[-10:-1]))

        for j in range(34):

            submit_confirmed.append(data[-1]*2)

    

    # contrived dataset

    data = i.Fatalities.astype('int32').tolist()

    # fit model

    try:

        #model = SARIMAX(data, order=(1,0,0), seasonal_order=(0,1,1,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        model = SARIMAX(data, order=(1,1,0), seasonal_order=(1,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        #model = ARIMA(data, order=(3,1,2))

        model_fit = model.fit(disp=False)

        # make prediction

        predicted = model_fit.predict(len(data), len(data)+34)

        new=np.concatenate((np.array(data),np.array([int(num) for num in predicted])),axis=0)

        submit_fatal.extend(list(new[-43:]))

    except:

        submit_fatal.extend(list(data[-10:-1]))

        for j in range(34):

            submit_fatal.append(data[-1]*2)



#create an alternative fatality metric 

#submit_fatal = [i * .005 for i in submit_confirmed]

#print(submit_fatal)
#make the submission file 

df_submit=pd.concat([pd.Series(np.arange(1,1+len(submit_confirmed))),pd.Series(submit_confirmed),pd.Series(submit_fatal)],axis=1)

df_submit=df_submit.fillna(method='pad').astype(int)
#view submission file 

df_submit.head()

#df_submit.dtypes
#examine the test file 

test.head()
#join the submission file info to the test data set 

#rename the columns 

df_submit.rename(columns={0: 'ForecastId', 1: 'ConfirmedCases',2: 'Fatalities',}, inplace=True)



#join the two data items 

complete_test= pd.merge(test, df_submit, how="left", on="ForecastId")
#df_submit.interpolate(method='pad', xis=0, inplace=True)

#df_submit.to_csv('submission.csv',header=['ForecastId','ConfirmedCases','Fatalities'],index=False)

complete_test.to_csv('complete_test.csv',index=False)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from tqdm.notebook import tqdm

from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



%matplotlib inline

dpi = 96

plt.rcParams['figure.figsize'] = (1600/dpi, 600/dpi)

plt.style.use('ggplot')



# grabbing prepared dataset from https://www.kaggle.com/jorijnsmit/population-and-sub-continent-for-every-entity

covid = pd.read_csv('../input/covid19/covid.csv', parse_dates=['date'])



# perform same manipulations from the prepared dataset to the test set

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv', parse_dates=['Date'])

test.columns = ['id', 'province_state', 'country_region', 'date']

test['country_region'].update(test['country_region'].str.replace('Georgia', 'Sakartvelo'))

test['entity'] = test['province_state'].where(~test['province_state'].isna(), test['country_region'])

test = test.set_index('id')[['date', 'entity']]



def logistic(t, k, r, a):

    """k > 0: final epidemic size

    r > 0: infection rate

    a = (k - c_0) / c_0

    """

    

    return k / (1 + a * np.exp(-r * t))



def solve(c):

    """port from https://mathworks.com/matlabcentral/fileexchange/74411-fitvirus"""

    

    n = len(c)

    nmax = max(1, n // 2)



    for i in np.arange(1, nmax+1):

        k1 = i

        k3 = n - 1

        if (n - i) % 2 == 0:

            k3 -= 1



        k2 = (k1 + k3) // 2

        m = k2 - k1 - 1



        if k1 < 1 or k2 < 1 or k3 < 1 or m < 1:

            return None



        k1 -= 1

        k2 -= 1

        k3 -= 1



        # calculate k

        v = c[k1] * c[k2] - 2 * c[k1] * c[k3] + c[k2] * c[k3]

        if v <= 0:

            continue

        w = c[k2]**2 - c[k3] * c[k1]

        if w <= 0:

            continue

        k = c[k2] * v / w

        if k <= 0:

            continue



        # calculate r

        x = c[k3] * (c[k2] - c[k1])

        if x <= 0:

            continue

        y = c[k1] * (c[k3] - c[k2])

        if y <= 0:

            continue

        r = (1 / m) * np.log(x / y)

        if r <= 0:

            continue



        # calculate a

        z = ((c[k3] - c[k2]) * (c[k2] - c[k1])) / w

        if z <= 0:

            continue

        a = z * (x / y) ** ((k3 + 1 - m) / m)

        if a <= 0:

            continue

        

        return k, r, a



def plot_fit(x_train, y_train, x_predict, y_predict, r2):

    fig, ax = plt.subplots()

    ax.set_title(f'{subject} {r2}')

    color = 'green' if r2 > 0.99 else 'red'

    pd.Series(y_train, x_train).plot(subplots=True, style='.', color='black', legend=True, label='train')

    pd.Series(y_predict, x_predict).plot(subplots=True, style=':', color=color, legend=True, label='predict')

    plt.show()



herd_immunity = 0.7

test_ratio = 0.2



for target in ['confirmed', 'fatal']:

    for subject in tqdm(covid['entity'].unique()):

        population = covid[covid['entity'] == subject]['population'].max()



        x_train = covid[covid['entity'] == subject]['date'].dt.dayofyear.values

        y_train = covid[covid['entity'] == subject][target].values



        mask = y_train > 0

        x_train_m = x_train[mask]

        y_train_m = y_train[mask]

        

        # no point in modelling a single point or no ints at all

        if x_train_m.size < 2 or x_train_m.sum() == 0:

            continue



        x_predict = test[test['entity'] == subject]['date'].dt.dayofyear.values

        submission_size = x_predict.size

        # start calculating sigmoid at same point x_train_m starts

        x_predict = np.arange(start=x_train_m[0], stop=x_predict[-1]+1)



        params = solve(y_train_m)



        if params != None:

        #try:

            params = (max(params[0], max(y_train_m)), params[1], params[2])

            lower_bounds = (max(y_train_m), 0, 0)

            upper_bounds = (max(population * herd_immunity * test_ratio, params[0]), np.inf, np.inf)



            params, _ = curve_fit(

                logistic,

                np.arange(x_train_m.size),

                y_train_m,

                p0=params,

                bounds=(lower_bounds, upper_bounds),

                maxfev=100000

            )



            y_eval = logistic(np.arange(x_train_m.size), params[0], params[1], params[2])

            y_predict = logistic(np.arange(x_predict.size), params[0], params[1], params[2])



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'log_{target}'] = r2



        else:

            # we fit a polynomial instead

            # while forcing cumulative behaviour, i.e. never lower numbers

            # it's ugly

            # i know



            model = linear_model.LinearRegression()

#             model = Pipeline([

#                 ("polynomial_features", PolynomialFeatures(degree=2)), 

#                 ("linear_regression", linear_model.Ridge())

#             ])

            if target == 'fatal':

                # pass more features; including confirmed!

                pass

            model.fit(x_train_m.reshape(-1, 1), y_train_m)



            y_eval = model.predict(x_train_m.reshape(-1, 1))

            y_predict = model.predict(x_predict.reshape(-1, 1))

            y_predict = np.maximum.accumulate(y_predict)



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'poly_{target}'] = r2



        if target == 'confirmed' and subject in ['Hubei', 'Italy', 'New York']:

            plot_fit(x_train, y_train, x_predict, y_predict, r2)



        # assign the prediction to the test dataframe

        delta = submission_size - y_predict.size

        if delta > 0:

            filler = [100] * delta if target == 'confirmed' else [1] * delta

            y_predict = filler + y_predict.tolist()

        test.loc[test['entity'] == subject, target] = y_predict[-submission_size:]



# resulting R2 scores for logistic approach

for target in ['confirmed', 'fatal']:

    r2s = covid.groupby('entity')[f'log_{target}'].max()

    print(r2s.describe())

    print(r2s[r2s.isna()].index)



# any doubtful maxima due to regression?

for target in ['confirmed', 'fatal']:

    df = []

    for subject in covid.loc[covid[f'poly_{target}'].isna()]['entity'].unique():

        df.append(test[test['entity'] == subject][['entity', target]].max().to_dict())

    df = pd.DataFrame(df).set_index('entity')

    print(df[target].sort_values(ascending=False).fillna(0).astype('int').head(10))

    

# @TODO

# some are way too high; this is a problem!

# what are the parameters they are fitted on?



# sanity check before submitting

submission = test[['entity', 'date']].copy()

submission[['confirmed', 'fatal']] = test[['confirmed', 'fatal']].fillna(0).astype('int')

submission[submission['entity'] == 'Netherlands']



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from tqdm.notebook import tqdm

from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



%matplotlib inline

dpi = 96

plt.rcParams['figure.figsize'] = (1600/dpi, 600/dpi)

plt.style.use('ggplot')



# grabbing prepared dataset from https://www.kaggle.com/jorijnsmit/population-and-sub-continent-for-every-entity

covid = pd.read_csv('../input/covid19/covid.csv', parse_dates=['date'])



# perform same manipulations from the prepared dataset to the test set

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv', parse_dates=['Date'])

test.columns = ['id', 'province_state', 'country_region', 'date']

test['country_region'].update(test['country_region'].str.replace('Georgia', 'Sakartvelo'))

test['entity'] = test['province_state'].where(~test['province_state'].isna(), test['country_region'])

test = test.set_index('id')[['date', 'entity']]



def logistic(t, k, r, a):

    """k > 0: final epidemic size

    r > 0: infection rate

    a = (k - c_0) / c_0

    """

    

    return k / (1 + a * np.exp(-r * t))



def solve(c):

    """port from https://mathworks.com/matlabcentral/fileexchange/74411-fitvirus"""

    

    n = len(c)

    nmax = max(1, n // 2)



    for i in np.arange(1, nmax+1):

        k1 = i

        k3 = n - 1

        if (n - i) % 2 == 0:

            k3 -= 1



        k2 = (k1 + k3) // 2

        m = k2 - k1 - 1



        if k1 < 1 or k2 < 1 or k3 < 1 or m < 1:

            return None



        k1 -= 1

        k2 -= 1

        k3 -= 1



        # calculate k

        v = c[k1] * c[k2] - 2 * c[k1] * c[k3] + c[k2] * c[k3]

        if v <= 0:

            continue

        w = c[k2]**2 - c[k3] * c[k1]

        if w <= 0:

            continue

        k = c[k2] * v / w

        if k <= 0:

            continue



        # calculate r

        x = c[k3] * (c[k2] - c[k1])

        if x <= 0:

            continue

        y = c[k1] * (c[k3] - c[k2])

        if y <= 0:

            continue

        r = (1 / m) * np.log(x / y)

        if r <= 0:

            continue



        # calculate a

        z = ((c[k3] - c[k2]) * (c[k2] - c[k1])) / w

        if z <= 0:

            continue

        a = z * (x / y) ** ((k3 + 1 - m) / m)

        if a <= 0:

            continue

        

        return k, r, a



def plot_fit(x_train, y_train, x_predict, y_predict, r2):

    fig, ax = plt.subplots()

    ax.set_title(f'{subject} {r2}')

    color = 'green' if r2 > 0.99 else 'red'

    pd.Series(y_train, x_train).plot(subplots=True, style='.', color='black', legend=True, label='train')

    pd.Series(y_predict, x_predict).plot(subplots=True, style=':', color=color, legend=True, label='predict')

    plt.show()



herd_immunity = 0.7

test_ratio = 0.2



for target in ['confirmed', 'fatal']:

    for subject in tqdm(covid['entity'].unique()):

        population = covid[covid['entity'] == subject]['population'].max()



        x_train = covid[covid['entity'] == subject]['date'].dt.dayofyear.values

        y_train = covid[covid['entity'] == subject][target].values



        mask = y_train > 0

        x_train_m = x_train[mask]

        y_train_m = y_train[mask]

        

        # no point in modelling a single point or no ints at all

        if x_train_m.size < 2 or x_train_m.sum() == 0:

            continue



        x_predict = test[test['entity'] == subject]['date'].dt.dayofyear.values

        submission_size = x_predict.size

        # start calculating sigmoid at same point x_train_m starts

        x_predict = np.arange(start=x_train_m[0], stop=x_predict[-1]+1)



        params = solve(y_train_m)



        if params != None:

        #try:

            params = (max(params[0], max(y_train_m)), params[1], params[2])

            lower_bounds = (max(y_train_m), 0, 0)

            upper_bounds = (max(population * herd_immunity * test_ratio, params[0]), np.inf, np.inf)



            params, _ = curve_fit(

                logistic,

                np.arange(x_train_m.size),

                y_train_m,

                p0=params,

                bounds=(lower_bounds, upper_bounds),

                maxfev=100000

            )



            y_eval = logistic(np.arange(x_train_m.size), params[0], params[1], params[2])

            y_predict = logistic(np.arange(x_predict.size), params[0], params[1], params[2])



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'log_{target}'] = r2



        else:

            # we fit a polynomial instead

            # while forcing cumulative behaviour, i.e. never lower numbers

            # it's ugly

            # i know



            model = linear_model.LinearRegression()

#             model = Pipeline([

#                 ("polynomial_features", PolynomialFeatures(degree=2)), 

#                 ("linear_regression", linear_model.Ridge())

#             ])

            if target == 'fatal':

                # pass more features; including confirmed!

                pass

            model.fit(x_train_m.reshape(-1, 1), y_train_m)



            y_eval = model.predict(x_train_m.reshape(-1, 1))

            y_predict = model.predict(x_predict.reshape(-1, 1))

            y_predict = np.maximum.accumulate(y_predict)



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'poly_{target}'] = r2



        if target == 'confirmed' and subject in ['Hubei', 'Italy', 'New York']:

            plot_fit(x_train, y_train, x_predict, y_predict, r2)



        # assign the prediction to the test dataframe

        delta = submission_size - y_predict.size

        if delta > 0:

            filler = [100] * delta if target == 'confirmed' else [1] * delta

            y_predict = filler + y_predict.tolist()

        test.loc[test['entity'] == subject, target] = y_predict[-submission_size:]



# resulting R2 scores for logistic approach

for target in ['confirmed', 'fatal']:

    r2s = covid.groupby('entity')[f'log_{target}'].max()

    print(r2s.describe())

    print(r2s[r2s.isna()].index)



# any doubtful maxima due to regression?

for target in ['confirmed', 'fatal']:

    df = []

    for subject in covid.loc[covid[f'poly_{target}'].isna()]['entity'].unique():

        df.append(test[test['entity'] == subject][['entity', target]].max().to_dict())

    df = pd.DataFrame(df).set_index('entity')

    print(df[target].sort_values(ascending=False).fillna(0).astype('int').head(10))

    

# @TODO

# some are way too high; this is a problem!

# what are the parameters they are fitted on?



# sanity check before submitting

submission = test[['entity', 'date']].copy()

submission[['confirmed', 'fatal']] = test[['confirmed', 'fatal']].fillna(0).astype('int')

submission[submission['entity'] == 'Netherlands']







submission = submission[['confirmed', 'fatal']]

submission.index.name = 'ForecastId'

submission.columns = ['ConfirmedCases', 'Fatalities']



submission['ConfirmedCases'] = ((df_submit['ConfirmedCases'].values + submission['ConfirmedCases'].values)/2).astype(int)

submission['Fatalities'] = ((df_submit['Fatalities'].values + submission['Fatalities'].values)/2).astype(int)



submission.to_csv('submission.csv')