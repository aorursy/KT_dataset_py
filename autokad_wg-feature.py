###############################################################################

## Imports

###############################################################################

from datetime import timedelta

import numpy as np

import pandas as pd

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error

from google.cloud import bigquery

from sklearn.model_selection import KFold

from scipy.spatial.distance import cdist



from scipy.optimize import minimize

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from math import sin, cos, sqrt, atan2, radians

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_log_error

from scipy.optimize import curve_fit

from datetime import datetime

###############################################################################

## Functions

###############################################################################

def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = pd.to_datetime(df['Date']).values

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    return df



def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 10





def wg_func(params):

    a = params[0]

    r = params[1]

    c = params[2]

    hcnb = params[3]

    

    incr = [cdata[i] if i == 0 else cdata[i] - cdata[i-1] for i,item in enumerate(cdata)]

    ptlt = [(1-(a/(a+item**c))**r)*(1-hcnb) for item in ts]

    ptwt = [ptlt[i] if i == 0 else ptlt[i] - ptlt[i-1] for i,item in enumerate(ptlt)]

    return (np.sum([j * np.log(i) for i,j in zip(ptwt,incr)]) 

            + (pop*(1-hcnb) - np.max(cdata)) * np.log(1-np.max(ptlt))

            + np.log(hcnb)*pop*hcnb)*-1



def wg_func2(params):

    a = params[0]

    r = params[1]

    c = params[2]

    hcnb = .98

    

    incr = [cdata[i] if i == 0 else cdata[i] - cdata[i-1] for i,item in enumerate(cdata)]

    ptlt = [(1-(a/(a+item**c))**r)*(1-hcnb) for item in ts]

    ptwt = [ptlt[i] if i == 0 else ptlt[i] - ptlt[i-1] for i,item in enumerate(ptlt)]

    return (np.sum([j * np.log(i) for i,j in zip(ptwt,incr)]) 

            + (pop*(1-hcnb) - np.max(cdata)) * np.log(1-np.max(ptlt))

            + np.log(hcnb)*pop*hcnb)*-1





def constraint1(inputs):

    return inputs[0]





cons = ({'type': 'ineq', "fun": constraint1})



def calc_distance(lat1, lng1, lat2, lng2):

    # approximate radius of earth in km

    R = 6373.0

    

    #Python, all the trig functions use radians, not degrees

    lat1 = radians(lat1)

    lng1 = radians(lng1)

    lat2 = radians(lat2)

    lng2 = radians(lng2)

    

    dlon = lng2 - lng1

    dlat = lat2 - lat1

    

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    

    return R * c



def fill_missing_coords(df):

    ## East is postiive

    ## west is negative

    ## north is possitive

    ## south is negative

    print(' Filling in missing lat,lng')

    df.loc[df.Country_Region=='Zimbabwe', 'Lat'] = 19.0154

    df.loc[df.Country_Region=='Zimbabwe', 'Long']= 29.1549

    

    df.loc[(df.Country_Region=='Angola') & (df.Province_State==''), 'Lat'] = -11.2027

    df.loc[(df.Country_Region=='Angola') & (df.Province_State==''), 'Long']= 17.8739

    

    df.loc[(df.Country_Region=='Bahamas') & (df.Province_State==''), 'Lat'] = 25.0343

    df.loc[(df.Country_Region=='Bahamas') & (df.Province_State==''), 'Long']= -77.3963

    

    df.loc[(df.Country_Region=='Belize') & (df.Province_State==''), 'Lat'] = 17.1899

    df.loc[(df.Country_Region=='Belize') & (df.Province_State==''), 'Long']= -88.4976

    

    df.loc[(df.Country_Region=='United Kingdom') & (df.Province_State==''), 'Lat'] = 55.3781

    df.loc[(df.Country_Region=='United Kingdom') & (df.Province_State==''), 'Long']= -3.4360

    

    df.loc[(df.Country_Region=='United Kingdom') & (df.Province_State=='Isle of Man'), 'Lat'] = 54.2361

    df.loc[(df.Country_Region=='United Kingdom') & (df.Province_State=='Isle of Man'), 'Long']= -4.5481

    

    df.loc[(df.Country_Region=='Cabo Verde') & (df.Province_State==''), 'Lat'] = 16.5388

    df.loc[(df.Country_Region=='Cabo Verde') & (df.Province_State==''), 'Long']= -23.0418

    

    df.loc[(df.Country_Region=='United Kingdom') & (df.Province_State=='Bermuda'), 'Lat'] = 32.3078

    df.loc[(df.Country_Region=='United Kingdom') & (df.Province_State=='Bermuda'), 'Long']= -64.7505

    

    df.loc[(df.Country_Region=='Chad') & (df.Province_State==''), 'Lat'] = 15.4542

    df.loc[(df.Country_Region=='Chad') & (df.Province_State==''), 'Long']= 18.7322

    

    df.loc[(df.Country_Region=='Uganda') & (df.Province_State==''), 'Lat'] = 1.3733

    df.loc[(df.Country_Region=='Uganda') & (df.Province_State==''), 'Long']= 32.2903

    

    df.loc[(df.Country_Region=='Denmark') & (df.Province_State=='Greenland'), 'Lat'] = 71.7069

    df.loc[(df.Country_Region=='Denmark') & (df.Province_State=='Greenland'), 'Long']= -42.6043

    

    df.loc[(df.Country_Region=='Denmark') & (df.Province_State==''), 'Lat'] = 56.2639

    df.loc[(df.Country_Region=='Denmark') & (df.Province_State==''), 'Long']= 9.5018

    

    df.loc[(df.Country_Region=='Timor-Leste') & (df.Province_State==''), 'Lat'] = -8.8742

    df.loc[(df.Country_Region=='Timor-Leste') & (df.Province_State==''), 'Long']= 125.7275

    

    df.loc[(df.Country_Region=='Syria') & (df.Province_State==''), 'Lat'] = 34.8021

    df.loc[(df.Country_Region=='Syria') & (df.Province_State==''), 'Long']= 38.9968

    

    df.loc[(df.Country_Region=='Saint Kitts and Nevis') & (df.Province_State==''), 'Lat'] = 17.3578

    df.loc[(df.Country_Region=='Saint Kitts and Nevis') & (df.Province_State==''), 'Long']= -62.7830

    

    df.loc[(df.Country_Region=='Papua New Guinea') & (df.Province_State==''), 'Lat'] = -6.3150

    df.loc[(df.Country_Region=='Papua New Guinea') & (df.Province_State==''), 'Long']= 143.9555

    

    df.loc[(df.Country_Region=='Niger') & (df.Province_State==''), 'Lat'] = 17.6078

    df.loc[(df.Country_Region=='Niger') & (df.Province_State==''), 'Long']= 8.0817

    

    df.loc[(df.Country_Region=='El Salvador') & (df.Province_State==''), 'Lat'] = 13.7942

    df.loc[(df.Country_Region=='El Salvador') & (df.Province_State==''), 'Long']= -88.8965

    

    df.loc[(df.Country_Region=='Gambia') & (df.Province_State==''), 'Lat'] = 13.4432

    df.loc[(df.Country_Region=='Gambia') & (df.Province_State==''), 'Long']= -15.3101

    

    df.loc[(df.Country_Region=='Libya') & (df.Province_State==''), 'Lat'] = 26.3351

    df.loc[(df.Country_Region=='Libya') & (df.Province_State==''), 'Long']= 17.2283

    

    df.loc[(df.Country_Region=='Mali') & (df.Province_State==''), 'Lat'] = 17.5707

    df.loc[(df.Country_Region=='Mali') & (df.Province_State==''), 'Long']= -3.9962

    

    df.loc[(df.Country_Region=='Grenada') & (df.Province_State==''), 'Lat'] = 12.1165

    df.loc[(df.Country_Region=='Grenada') & (df.Province_State==''), 'Long']= -61.6790

    

    df.loc[(df.Country_Region=='Laos') & (df.Province_State==''), 'Lat'] = 19.8563

    df.loc[(df.Country_Region=='Laos') & (df.Province_State==''), 'Long']= 102.4955

    

    df.loc[(df.Country_Region=='Madagascar') & (df.Province_State==''), 'Lat'] = -18.7669

    df.loc[(df.Country_Region=='Madagascar') & (df.Province_State==''), 'Long']= 46.8691

    

    df.loc[(df.Country_Region=='Guinea-Bissau') & (df.Province_State==''), 'Lat'] = 11.8037

    df.loc[(df.Country_Region=='Guinea-Bissau') & (df.Province_State==''), 'Long']= -15.1804

    

    df.loc[(df.Country_Region=='Fiji') & (df.Province_State==''), 'Lat'] = -17.7134

    df.loc[(df.Country_Region=='Fiji') & (df.Province_State==''), 'Long']= 178.0650

    

    df.loc[(df.Country_Region=='Nicaragua') & (df.Province_State==''), 'Lat'] = 12.8654

    df.loc[(df.Country_Region=='Nicaragua') & (df.Province_State==''), 'Long']= -85.2072

    

    df.loc[(df.Country_Region=='Eritrea') & (df.Province_State==''), 'Lat'] = 15.1794

    df.loc[(df.Country_Region=='Eritrea') & (df.Province_State==''), 'Long']= 39.7823

    

    df.loc[(df.Country_Region=='Haiti') & (df.Province_State==''), 'Lat'] = 18.9712

    df.loc[(df.Country_Region=='Haiti') & (df.Province_State==''), 'Long']= -72.2852

    

    df.loc[(df.Country_Region=='Dominica') & (df.Province_State==''), 'Lat'] = 15.4150

    df.loc[(df.Country_Region=='Dominica') & (df.Province_State==''), 'Long']= -61.3710

    

    df.loc[(df.Country_Region=='Mozambique') & (df.Province_State==''), 'Lat'] = -18.6657

    df.loc[(df.Country_Region=='Mozambique') & (df.Province_State==''), 'Long']= 35.5296

    

    df.loc[(df.Country_Region=='Netherlands') & (df.Province_State==''), 'Lat'] = 52.1326

    df.loc[(df.Country_Region=='Netherlands') & (df.Province_State==''), 'Long']= 5.2913

    

    df.loc[(df.Country_Region=='Netherlands') & (df.Province_State=='Sint Maarten'), 'Lat'] = 18.0425

    df.loc[(df.Country_Region=='Netherlands') & (df.Province_State=='Sint Maarten'), 'Long']= -63.0548

    

    df.loc[(df.Country_Region=='France') & (df.Province_State==''), 'Lat'] = 46.2276

    df.loc[(df.Country_Region=='France') & (df.Province_State==''), 'Long']= 2.2137

    

    df.loc[(df.Country_Region=='France') & (df.Province_State=='New Caledonia'), 'Lat'] = -20.9043

    df.loc[(df.Country_Region=='France') & (df.Province_State=='New Caledonia'), 'Long']= 165.6180

    

    df.loc[(df.Country_Region=='France') & (df.Province_State=='Martinique'), 'Lat'] = 14.6415

    df.loc[(df.Country_Region=='France') & (df.Province_State=='Martinique'), 'Long']= -61.0242



print('done', datetime.now())









###############################################################################

## Read Data

###############################################################################

PATH = '/kaggle/input/covid19-global-forecasting-week-2/'

train  = pd.read_csv(PATH + 'train.csv')

test  = pd.read_csv(PATH + 'test.csv')



#df_geo  = pd.read_csv('/kaggle/input/df-geo/' + 'df_geo.csv')

df_geo  = pd.read_csv('../input/df-geo/df_geo2.csv')

df_geo.Province_State.fillna('', inplace=True)





dfp  = pd.read_csv('/kaggle/input/population3/' + 'population.csv')

dfp.columns = ['Province_State', 'Country_Region', 'pop']

print(train.shape, dfp.shape)





###############################################################################

## Clean Data

###############################################################################

print('Cleaning Data')

## Fill in those missing states

train.loc[train['Province_State'].isnull(), 'Province_State'] = ''

test.loc[test['Province_State'].isnull(), 'Province_State']   = ''





dfp.loc[dfp['Province_State'].isnull(), 'Province_State']     = ''





###############################################################################

## Joining Data

###############################################################################

print('Joining Data')



print(train.shape, test.shape)

n = train.shape[0]

train = pd.merge(train, dfp, on=['Country_Region','Province_State'], how='left')

assert train.shape[0] == n



n = test.shape[0]

test = pd.merge(test, dfp, on=['Country_Region','Province_State'], how='left')

assert test.shape[0] == n



test.reset_index(drop=True, inplace=True)

train.reset_index(drop=True, inplace=True)



n = train.shape[0]

train = pd.merge(train, df_geo, on=['Country_Region','Province_State'], how='left')

assert train.shape[0] == n



n = test.shape[0]

test = pd.merge(test, df_geo, on=['Country_Region','Province_State'], how='left')

assert test.shape[0] == n



test.reset_index(drop=True, inplace=True)

train.reset_index(drop=True, inplace=True)



train.loc[train['pop'].isnull(),'pop'] = 0

test.loc[test['pop'].isnull(),'pop']   = 0



print('\nNumber of countries with missing Lat/Lng: ',

      train[train.Lat.isnull()]['Country_Region'].value_counts().shape[0])



fill_missing_coords(train)

fill_missing_coords(test)



print('\nNumber of countries with missing Lat/Lng after fixing: ',

      train[train.Lat.isnull()]['Country_Region'].value_counts().shape[0])



print(train.shape, test.shape)





###############################################################################

## Enrich Data

###############################################################################

print('\nEnriching Data')



## Date Stuffs

mo = train['Date'].apply(lambda x: x[5:7])

da = train['Date'].apply(lambda x: x[8:10])

train['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )



mo = test['Date'].apply(lambda x: x[5:7])

da = test['Date'].apply(lambda x: x[8:10])

test['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )





#Create time features

create_time_features(train)

create_time_features(test)



print('done', datetime.now())
## Geographic stuffs



country1 = 'Luxembourg'

lat1     = df_geo[df_geo.Country_Region==country1]['Lat'].item()

lng1     = df_geo[df_geo.Country_Region==country1]['Long'].item()



country2 = 'Singapore'

lat2     = df_geo[df_geo.Country_Region==country2]['Lat'].item()

lng2     = df_geo[df_geo.Country_Region==country2]['Long'].item()



# This should be 10,436km

print('\nDistance between ' + country1, country2, calc_distance(lat1, lng1, lat2, lng2))

print('This should be 10,436km')



print(' Label Encoding the geographic features...')



df1 = train[['Country_Region','Province_State','Lat','Long']].copy()

df2 = test[['Country_Region','Province_State','Lat','Long']].copy()

geo = pd.concat([df1,df2], axis=0)

geo = geo.groupby(['Country_Region','Province_State'])[['Lat','Long']].max().reset_index()



le_country = LabelEncoder().fit( geo.Country_Region )

le_state   = LabelEncoder().fit( geo.Province_State )



train['country'] = le_country.transform( train.Country_Region)

train['state']   = le_state.transform( train.Province_State)



test['country'] = le_country.transform( test.Country_Region)

test['state']   = le_state.transform( test.Province_State)



print('done', train.shape)



# Probability Models

train['wg']     = 0

train['wga']    = 0

train['wgr']    = 0

train['wgc']    = 0

train['wghcnb'] = 0



test['wg']     = 0

test['wga']    = 0

test['wgr']    = 0

test['wgc']    = 0

test['wghcnb'] = 0



test['SARIMAX'] = 0

test['ARIMA']   = 0



countries = ['Afghanistan']



for country in train.Country_Region.unique():

    bool1 = train.Country_Region == country

    print(country)

    for state in train[bool1].Province_State.unique():

        bool2 = bool1 & (train.Province_State == state) & (train.ConfirmedCases>0)

        pop = np.max( train[bool2]['pop'] )

        

        data = train[ bool2 ].copy().reset_index()

        data['ts'] = data.index+1

        dfj   = data.iloc[0]['day_from_jan_first']

        

        cdata = data['ConfirmedCases'].values

        ts = data['ts'].values



        ## set up test stuffs

        boolt = (test.Country_Region==country) & (test.Province_State== state)

        datat = test[boolt].copy().reset_index()

        datat['ts'] = datat.day_from_jan_first - dfj

        datat.loc[datat.ts<=0,'ts'] = 1

        

        ## FIT WG

        x0 = [1343, 5.440110881178935, 2.188935325131958, 0.9897823619555628]

        sol  = minimize(wg_func, x0, constraints = cons)

        a,r,c,hcnb = sol.x[0], sol.x[1], sol.x[2], sol.x[3]

        if np.isnan(sol.x[0]):

            train.loc[bool2, 'wg']    = np.NaN

            train.loc[bool2,'wga']    = np.NaN

            train.loc[bool2,'wgr']    = np.NaN

            train.loc[bool2,'wgc']    = np.NaN

            train.loc[bool2,'wghcnb'] = np.NaN

            

            test.loc[boolt, 'wg']    = np.NaN

            test.loc[boolt,'wga']    = np.NaN

            test.loc[boolt,'wgr']    = np.NaN

            test.loc[boolt,'wgc']    = np.NaN

            test.loc[boolt,'wghcnb'] = np.NaN

        else:

            datat['wg'] = datat.ts.apply(lambda x: (1-(a/(a+x**c))**r)*pop*(1-hcnb)).values

            data['wg']  = data.ts.apply(lambda x: (1-(a/(a+x**c))**r)*pop*(1-hcnb)).values

            train.loc[bool2, 'wg'] = data['wg'].values

            train.loc[bool2,'wga']    = a

            train.loc[bool2,'wgr']    = r

            train.loc[bool2,'wgc']    = c

            train.loc[bool2,'wghcnb'] = hcnb

            

            test.loc[boolt, 'wg'] = datat.wg.values

            test.loc[boolt,'wga']    = a

            test.loc[boolt,'wgr']    = r

            test.loc[boolt,'wgc']    = c

            test.loc[boolt,'wghcnb'] = hcnb

        

        ## Calculate Arima and SARIMAX

        incr = [cdata[i] if i == 0 else cdata[i] - cdata[i-1] for i,item in enumerate(cdata)]

        

        ## get the last known cum count

        cc = np.max( data[data.day_from_jan_first <=np.min(datat.day_from_jan_first)]['ConfirmedCases'])

        try:

            model_arima = ARIMA( incr, order=(1,1,0)).fit()

            preds = [item if item >=0 else 0 for item in model_arima.predict(datat.ts[0].item() , datat.ts[-1:].item() )]

            cum_sum = cc

            preds_cc = []

            for item in preds:

                cum_sum = cum_sum + item

                preds_cc.append(cum_sum)

            test.loc[boolt, 'ARIMA'] = pd.Series(preds_cc).values

        except:

            test.loc[boolt, 'ARIMA']= np.NaN

        

        try:

            model_SARIMAX = SARIMAX(incr, order=(1,1,0), seasonal_order=(1,1,0,12),enforce_stationarity=False).fit()

            preds = [item if item >=0 else 0 for item in model_SARIMAX.predict(datat.ts[0].item() , datat.ts[-1:].item() )]

            cum_sum = cc

            preds_cc = []

            for item in preds:

                cum_sum = cum_sum + item

                preds_cc.append(cum_sum)

            

            test.loc[boolt, 'SARIMAX']= pd.Series(preds_cc).values

        except:

            test.loc[boolt, 'SARIMAX']= np.NaN



print('done', datetime.now())
train.loc[train.wg<0,'wg'] = 0

test.loc[test.wg<0,'wg'] = 0



train.loc[train.wg.isnull(),'wg'] = 0

test.loc[test.wg.isnull(),'wg'] = 0



train.loc[np.isinf(train.wg),'wg'] = 0

test.loc[np.isinf(test.wg),'wg'] = 0

print('done', datetime.now())
train['wg2'] = 0

test['wg2'] = 0

train['SARIMAX2'] = 0

train['SARIMAX2'] = 0



data = train.groupby(['day_from_jan_first'])['ConfirmedCases','pop'].sum().reset_index()



pop = np.max( data['pop'] )

data['ts'] = data.index+1

dfj   = data.iloc[0]['day_from_jan_first']

cdata = data['ConfirmedCases'].values

ts = data['ts'].values



## FIT WG

x0 = [1343, 5.440110881178935, 2.188935325131958]

sol  = minimize(wg_func2, x0, constraints = cons)

a,r,c = sol.x[0], sol.x[1], sol.x[2]



print(a,r,c)



for country in train.Country_Region.unique():

    bool1 = train.Country_Region == country

    for state in train[bool1].Province_State.unique():

        bool2 = bool1 & (train.Province_State == state)

        pop = np.max( train[bool2]['pop'] )

        

        data = train[ bool2 ].copy().reset_index()

        data['ts'] = data.index+1

        dfj   = data.iloc[0]['day_from_jan_first']

        

        cdata = data['ConfirmedCases'].values

        ts = data['ts'].values



        ## set up test stuffs

        boolt = (test.Country_Region==country) & (test.Province_State== state)

        datat = test[boolt].copy().reset_index()

        datat['ts'] = datat.day_from_jan_first - dfj

        datat.loc[datat.ts<=0,'ts'] = 1

        

        ## FIT WG

        x0 = [1343, 5.440110881178935, 2.188935325131958, 0.9897823619555628]

        sol  = minimize(wg_func, x0, constraints = cons)

        a,r,c,hcnb = sol.x[0], sol.x[1], sol.x[2], sol.x[3]

        if not np.isnan(sol.x[0]):

            datat['wg2'] = datat.ts.apply(lambda x: (1-(a/(a+x**c))**r)*pop*(1-hcnb)).values

            data[ 'wg2']  = data.ts.apply(lambda x: (1-(a/(a+x**c))**r)*pop*(1-hcnb)).values

            train.loc[bool2, 'wg2'] = data['wg2'].values

            test.loc[boolt, 'wg2']  = datat.wg2.values

            

        try:

            model_SARIMAX = SARIMAX(data.ConfirmedCases, order=(1,1,0), seasonal_order=(1,1,0,12),enforce_stationarity=False).fit()

            preds = [item if item >=0 else 0 for item in model_SARIMAX.predict(datat.ts[0].item() , datat.ts[-1:].item() )]

            preds2 = [item if item >=0 else 0 for item in model_SARIMAX.predict(data.ts[0].item() , data.ts[-1:].item() )]

            train.loc[bool2, 'SARIMAX2']= pd.Series(preds2).values

            test.loc[boolt, 'SARIMAX2']= pd.Series(preds).values

        except:

            test.loc[boolt, 'SARIMAX2']= np.NaN

print('done', datetime.now())
data = train.groupby(['day_from_jan_first'])['ConfirmedCases','pop'].sum().reset_index()



train['wg3'] = 0

test['wg3'] = 0



pop = np.max( data['pop'] )

data['ts'] = data.index+1

dfj   = data.iloc[0]['day_from_jan_first']

cdata = data['ConfirmedCases'].values

ts = data['ts'].values



## FIT WG

x0 = [1343, 5.440110881178935, 2.188935325131958]

sol  = minimize(wg_func2, x0, constraints = cons)

a,r,c = sol.x[0], sol.x[1], sol.x[2]





for country in train.Country_Region.unique():

    bool1 = train.Country_Region == country

    for state in train[bool1].Province_State.unique():

        bool2 = bool1 & (train.Province_State == state)

        pop = np.max( train[bool2]['pop'] )

        

        data = train[ bool2 ].copy().reset_index()

        data['ts'] = data.index+1

        dfj   = data.iloc[0]['day_from_jan_first']

        

        cdata = data['ConfirmedCases'].values

        ts = data['ts'].values



        ## set up test stuffs

        boolt = (test.Country_Region==country) & (test.Province_State== state)

        datat = test[boolt].copy().reset_index()

        datat['ts'] = datat.day_from_jan_first - dfj

        datat.loc[datat.ts<=0,'ts'] = 1

        

        datat['wg3'] = datat.ts.apply(lambda x: (1-(a/(a+x**c))**r)*pop*(1-hcnb)).values

        data[ 'wg3']  = data.ts.apply(lambda x: (1-(a/(a+x**c))**r)*pop*(1-hcnb)).values

        train.loc[bool2, 'wg3'] = data['wg3'].values

        test.loc[boolt, 'wg3']  = datat.wg3.values

print('done', datetime.now())

print(a,r,c)
train.loc[train.wg2<0,'wg2'] = 0

test.loc[test.wg2<0,'wg2'] = 0



train.loc[train.wg2.isnull(),'wg2'] = 0

test.loc[test.wg2.isnull(),'wg2']   = 0



train.loc[np.isinf(train.wg2),'wg2'] = 0

test.loc[ np.isinf(test.wg2),'wg2'] = 0



train.loc[train.wg3<0,'wg3'] = 0

test.loc[test.wg3<0,'wg3'] = 0



train.loc[train.wg3.isnull(),'wg3'] = 0

test.loc[test.wg3.isnull(),'wg3']   = 0



train.loc[np.isinf(train.wg3),'wg3'] = 0

test.loc[ np.isinf(test.wg3),'wg3'] = 0

print('done', datetime.now())
print(mean_squared_log_error(train.ConfirmedCases, train.wg))

print(mean_squared_log_error(train.ConfirmedCases, train.wg2))

print(mean_squared_log_error(train.ConfirmedCases, train.wg3))

train.columns
for country in train.Country_Region.unique():

    bool1 = train.Country_Region == country

    for state in train[bool1].Province_State.unique():

        bool2 = bool1 & (train.Province_State == state)

        data = train[ bool2 ].copy().reset_index()

        

        wg1 = np.round(mean_squared_log_error(data.ConfirmedCases, data.wg),2)

        wg2 = np.round(mean_squared_log_error(data.ConfirmedCases, data.wg2),2)

        wg3 = np.round(mean_squared_log_error(data.ConfirmedCases, data.wg3),2)

        

        ## set up test stuffs

        boolt = (test.Country_Region==country) & (test.Province_State== state)

        datat = test[boolt].copy().reset_index()

        

        if wg1<wg2 and wg1<wg3:

            data['best']  = data.wg

            datat['best'] = datat.wg

        elif wg2<wg1 and wg2 < wg3:

            data['best']  = data.wg2

            datat['best'] = datat.wg2

        else:

            data['best']  = data.wg3

            datat['best'] = datat.wg3

        

        train.loc[bool2, 'wg'] = data['best'].values

        test.loc[boolt, 'wg']  = datat['best'].values

print('done', datetime.now())
print(mean_squared_log_error(train.ConfirmedCases, train.wg))



train.drop('wg2', inplace=True, axis=1)

train.drop('wg3', inplace=True, axis=1)



test.drop('wg2', inplace=True, axis=1)

test.drop('wg3', inplace=True, axis=1)

print('done', datetime.now())
###############################################################################

## logistic

###############################################################################



from scipy.optimize import curve_fit





train_data = train.copy()

train_df = train_data

train_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(train_data['Country_Region'], train_data['Province_State'])]

train_df['Date'] = pd.to_datetime(train_df['Date'])

full_data = train_df



today = full_data['Date'].max()+timedelta(days=1) 



def get_country_data(train_df, area, metric):

    country_data = train_df[train_df['area']==area]

    country_data = country_data.drop(['Id','Province_State', 'Country_Region', 'Lat','Long'], axis=1)

    country_data = pd.pivot_table(country_data, values=['ConfirmedCases','Fatalities'], index=['Date'], aggfunc=np.sum) 

    country_data = country_data[country_data[metric]!=0]

    return country_data      



area_info = pd.DataFrame(columns=['area', 'cases_start_date', 'deaths_start_date', 'init_ConfirmedCases', 'init_Fatalities'])

for i in range(len(train_df['area'].unique())):

    area = train_df['area'].unique()[i]

    area_cases_data = get_country_data(train_df, area, 'ConfirmedCases')

    area_deaths_data = get_country_data(train_df, area, 'Fatalities')

    cases_start_date = area_cases_data.index.min()

    deaths_start_date = area_deaths_data.index.min()

    if len(area_cases_data) > 0:

        confirmed_cases = max(area_cases_data['ConfirmedCases'])

    else:

        confirmed_cases = 0

    if len(area_deaths_data) > 0:

        fatalities = max(area_deaths_data['Fatalities'])

    else:

        fatalities = 0

    area_info.loc[i] = [area, cases_start_date, deaths_start_date, confirmed_cases, fatalities]



area_info = area_info.fillna(pd.to_datetime(today))

area_info['init_cases_day_no'] = pd.to_datetime(today)-area_info['cases_start_date']

area_info['init_cases_day_no'] = area_info['init_cases_day_no'].dt.days.fillna(0).astype(int)

area_info['init_deaths_day_no'] = pd.to_datetime(today)-area_info['deaths_start_date']

area_info['init_deaths_day_no'] = area_info['init_deaths_day_no'].dt.days.fillna(0).astype(int)

area_info.head()





def log_curve(x, k, x_0, ymax):

    return ymax / (1 + np.exp(-k*(x-x_0)))

    

def log_fit(train_df, area, metric):

    area_data = get_country_data(train_df, area, metric)

    x_data = range(len(area_data.index))

    y_data = area_data[metric]

    if len(y_data) < 5:

        estimated_k = -1  

        estimated_x_0 = -1 

        ymax = -1

    elif max(y_data) == 0:

        estimated_k = -1  

        estimated_x_0 = -1 

        ymax = -1

    else:

        try:

            popt, pcov = curve_fit(log_curve, x_data, y_data, bounds=([0,0,0],np.inf), p0=[0.3,100,10000], maxfev=1000000)

            estimated_k, estimated_x_0, ymax = popt

        except RuntimeError:

            print(area)

            print("Error - curve_fit failed") 

            estimated_k = -1  

            estimated_x_0 = -1 

            ymax = -1

    estimated_parameters = pd.DataFrame(np.array([[area, estimated_k, estimated_x_0, ymax]]), columns=['area', 'k', 'x_0', 'ymax'])

    return estimated_parameters



def get_parameters(metric):

    parameters = pd.DataFrame(columns=['area', 'k', 'x_0', 'ymax'], dtype=np.float)

    for area in train_df['area'].unique():

        estimated_parameters = log_fit(train_df, area, metric)

        parameters = parameters.append(estimated_parameters)

    parameters['k'] = pd.to_numeric(parameters['k'], downcast="float")

    parameters['x_0'] = pd.to_numeric(parameters['x_0'], downcast="float")

    parameters['ymax'] = pd.to_numeric(parameters['ymax'], downcast="float")

    parameters = parameters.replace({'k': {-1: parameters[parameters['ymax']>0].median()[0]}, 

                                     'x_0': {-1: parameters[parameters['ymax']>0].median()[1]}, 

                                     'ymax': {-1: parameters[parameters['ymax']>0].median()[2]}})

    return parameters





cases_parameters = get_parameters('ConfirmedCases')

cases_parameters.head(20)



deaths_parameters = get_parameters('Fatalities')

deaths_parameters.head(20)



fit_df = area_info.merge(cases_parameters, on='area', how='left')

fit_df = fit_df.rename(columns={"k": "cases_k", "x_0": "cases_x_0", "ymax": "cases_ymax"})

fit_df = fit_df.merge(deaths_parameters, on='area', how='left')

fit_df = fit_df.rename(columns={"k": "deaths_k", "x_0": "deaths_x_0", "ymax": "deaths_ymax"})

fit_df['init_ConfirmedCases_fit'] = log_curve(fit_df['init_cases_day_no'], fit_df['cases_k'], fit_df['cases_x_0'], fit_df['cases_ymax'])

fit_df['init_Fatalities_fit'] = log_curve(fit_df['init_deaths_day_no'], fit_df['deaths_k'], fit_df['deaths_x_0'], fit_df['deaths_ymax'])

fit_df['ConfirmedCases_error'] = fit_df['init_ConfirmedCases']-fit_df['init_ConfirmedCases_fit']

fit_df['Fatalities_error'] = fit_df['init_Fatalities']-fit_df['init_Fatalities_fit']

fit_df.head()



test_data = test.copy()

test_df = test_data

test_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(test_data['Country_Region'], test_data['Province_State'])]

test_df = test_df.merge(fit_df, on='area', how='left')

test_df['Date'] = pd.to_datetime(test_df['Date'])



test_df['cases_start_date'] = pd.to_datetime(test_df['cases_start_date'])

test_df['deaths_start_date'] = pd.to_datetime(test_df['deaths_start_date'])

test_df['cases_day_no'] = test_df['Date']-test_df['cases_start_date']

test_df['cases_day_no'] = test_df['cases_day_no'].dt.days.fillna(0).astype(int)

test_df['deaths_day_no'] = test_df['Date']-test_df['deaths_start_date']

test_df['deaths_day_no'] = test_df['deaths_day_no'].dt.days.fillna(0).astype(int)

test_df['ConfirmedCases_fit'] = log_curve(test_df['cases_day_no'], test_df['cases_k'], test_df['cases_x_0'], test_df['cases_ymax'])

test_df['Fatalities_fit'] = log_curve(test_df['deaths_day_no'], test_df['deaths_k'], test_df['deaths_x_0'], test_df['deaths_ymax'])

test_df['ConfirmedCases_pred'] = round(test_df['ConfirmedCases_fit']+test_df['ConfirmedCases_error'])

test_df['Fatalities_pred'] = round(test_df['Fatalities_fit']+test_df['Fatalities_error'])

test_df.head()



train_df = train.copy()

train_df['area'] = [str(i)+str(' - ')+str(j) for i,j in zip(train_df['Country_Region'], train_df['Province_State'])]

train_df         = train_df.merge(fit_df, on='area', how='left')

train_df['Date'] = pd.to_datetime(train_df['Date'])

train_df['cases_start_date'] = pd.to_datetime(train_df['cases_start_date'])

train_df['deaths_start_date'] = pd.to_datetime(train_df['deaths_start_date'])

train_df['cases_day_no'] = train_df['Date']-train_df['cases_start_date']

train_df['cases_day_no'] = train_df['cases_day_no'].dt.days.fillna(0).astype(int)

train_df['deaths_day_no'] = train_df['Date']-train_df['deaths_start_date']

train_df['deaths_day_no'] = train_df['deaths_day_no'].dt.days.fillna(0).astype(int)

train_df['ConfirmedCases_fit'] = log_curve(train_df['cases_day_no'], train_df['cases_k'], train_df['cases_x_0'], train_df['cases_ymax'])

train_df['Fatalities_fit'] = log_curve(train_df['deaths_day_no'], train_df['deaths_k'], train_df['deaths_x_0'], train_df['deaths_ymax'])

train_df['ConfirmedCases_pred'] = round(train_df['ConfirmedCases_fit']+train_df['ConfirmedCases_error'])

train_df['Fatalities_pred'] = round(train_df['Fatalities_fit']+train_df['Fatalities_error'])



train_df.head()

print('done', datetime.now())
train['y_hat_log'  ]                = test_df.ConfirmedCases_fit

test['y_hat_log'  ]                = test_df.ConfirmedCases_pred
###############################################################################

## Modeling

###############################################################################

dropcols = ['Date', 'date', 'ConfirmedCases', 'Id', 'ForecastId', 'Fatalities']

dropcols = dropcols + ['Country_Region','Province_State']

dropcols = dropcols + ['eg', 'egr', 'mae_eg', 'wga', 'wgr', 'wgc','wghcnb', 'mae_wg', 'SARIMAX', 'ARIMA',

           'cc_es',]



print('\nModeling...')

features = [f for f in train.columns if f not in dropcols + 

            ['shift4w', 'shift6w', 'dist'

             , 'dayofyear','year','quarter','hour','month','dayofmonth','dayofweek','weekofyear', 'Lat', 'Long']]



print(features)



X_train = train[features].copy()

X_test  = test[features].copy()



X_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)



y_train    = train["Fatalities"]

y_train_cc = train["ConfirmedCases"]



print('done', datetime.now())
isTraining = False

params_xgb = {}

params_xgb['n_estimators']       = 1100

params_xgb['max_depth']          = 10

params_xgb['seed']               = 2020

params_xgb['colsample_bylevel']  = 1

params_xgb['colsample_bytree']   = 1

params_xgb['learning_rate']      = 0.3

params_xgb['reg_alpha']          = 0

params_xgb['reg_lambda']         = 1

params_xgb['subsample']          = 1





if isTraining:

    X_TRAIN = X_train[features].values



    kf      = KFold(n_splits = 5, shuffle = True, random_state=2020)

    acc     = []



    for tr_idx, val_idx in kf.split(X_TRAIN, y_train_cc):

        ## Set up XY train/validation

        X_tr, X_vl = X_TRAIN[tr_idx], X_TRAIN[val_idx, :]

        y_tr, y_vl = y_train_cc[tr_idx], y_train_cc[val_idx]

        print(X_tr.shape)



        model_xgb_cc = xgb.XGBRegressor(**params_xgb)

        model_xgb_cc.fit(X_tr, y_tr, verbose=True)

        y_hat = model_xgb_cc.predict(X_vl)



        print('xgb mae :', mean_absolute_error(  y_vl, y_hat) )

        acc.append(mean_absolute_error( y_vl, y_hat) )





    print('done', np.mean(acc))

print('done', datetime.now())
## Fit confirmed cases with out wg

dropcols = ['Date', 'date', 'ConfirmedCases', 'Id', 'ForecastId', 'Fatalities']

dropcols = dropcols + ['Country_Region','Province_State']

dropcols = dropcols + ['eg', 'egr', 'mae_eg', 'wg', 'wga', 'wgr', 'wg_xgb', 'wgc','wghcnb', 'mae_wg', 'SARIMAX', 'ARIMA',

           'cc_es',]



print('\nModeling...')

features = [f for f in train.columns if f not in dropcols + 

            ['shift4w', 'shift6w', 'dist'

             , 'dayofyear','year','quarter','hour','month','dayofmonth','dayofweek','weekofyear']]



print(features)



X_train = train[features].copy()

X_test  = test[features].copy()



X_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)



X_train.head()

print('done', datetime.now())
params_xgb = {}

params_xgb['n_estimators']       = 1100

params_xgb['max_depth']          = 10

params_xgb['seed']               = 2020

params_xgb['colsample_bylevel']  = 1

params_xgb['colsample_bytree']   = 1

params_xgb['learning_rate']      = 0.3

params_xgb['reg_alpha']          = 0

params_xgb['reg_lambda']         = 1                                          

params_xgb['subsample']          = 1





isTraining = True



X_train.reset_index(drop=True, inplace=True)



if isTraining:

    print('Evaluating model...')

    booll = X_train.day_from_jan_first < 86

    X_tr, X_vl = X_train[booll][features        ], X_train[~booll][features]

    y_tr, y_vl = train[booll]['ConfirmedCases'], train[~booll]['ConfirmedCases']



    model_xgb_cc = xgb.XGBRegressor(**params_xgb)

    model_xgb_cc.fit(X_tr, y_tr, verbose=True)

    y_hat = model_xgb_cc.predict(X_vl)

    y_hat[y_hat<0] = 0

    print('xgb mae :', mean_absolute_error(  y_vl, y_hat), mean_squared_log_error(y_vl, y_hat), X_tr.shape, X_vl.shape ) 



    

print('done', datetime.now())
y_train    = train["Fatalities"]

y_train_cc = train["ConfirmedCases"]



params_xgb = {}

params_xgb['n_estimators']       = 1100

params_xgb['max_depth']          = 10

params_xgb['seed']               = 2020

params_xgb['colsample_bylevel']  = 1

params_xgb['colsample_bytree']   = 1

params_xgb['learning_rate']      = 0.3

params_xgb['reg_alpha']          = 0

params_xgb['reg_lambda']         = 1

params_xgb['subsample']          = 1



model_xgb_cc = xgb.XGBRegressor(**params_xgb).fit(X_train[features], y_train_cc, verbose=True)

y_hat_xgb_c  = model_xgb_cc.predict(X_test[features])

print('done', datetime.now())
###############################################################################

## Feature Importantce

###############################################################################



plot = plot_importance(model_xgb_cc, height=0.9, max_num_features=20)
## Fit fatalities

params_xgb = {}

params_xgb['n_estimators']       = 1100

params_xgb['max_depth']          = 10

params_xgb['seed']               = 2020

params_xgb['colsample_bylevel']  = 1

params_xgb['colsample_bytree']   = 1

params_xgb['learning_rate']      = 0.300000012

params_xgb['reg_alpha']          = 0

params_xgb['reg_lambda']         = 1

params_xgb['subsample']          = 1



model_xgb_f = xgb.XGBRegressor(**params_xgb).fit(X_train[features], y_train, verbose=True)

y_hat_xgb_f = model_xgb_f.predict(X_test[features])

print(np.mean(y_hat_xgb_f))

print('done', datetime.now())
features2 = [f for f in features if f not in ['SARIMAX2', 'y_hat_log']]



print(features2)

X_train2 = train[features2].copy()

X_test2  = test[features2].copy()



X_train2.reset_index(drop=True, inplace=True)

X_test2.reset_index(drop=True, inplace=True)



model_xgb_cc2 = xgb.XGBRegressor(**params_xgb).fit(X_train2, y_train_cc, verbose=True)

y_hat_xgb_c2  = model_xgb_cc2.predict(X_test2)

print('done', datetime.now())



plot = plot_importance(model_xgb_cc2, height=0.9, max_num_features=20)

print('done', datetime.now())
test['y_hat_xgb_c']                = y_hat_xgb_c

test['y_hat_xgb_f']                = y_hat_xgb_f

test['y_hat_xgb_c2']               = y_hat_xgb_c2



## Fix negative numbers

print('Fixing Negative Predictions:'

     , np.sum(test.y_hat_xgb_c < 0)

      , np.sum(test.y_hat_xgb_c2 < 0)

     , np.sum(test.y_hat_xgb_f< 0)

     , np.sum(test.wg< 0)

     , np.sum(test.ARIMA< 0)

     , np.sum(test.SARIMAX< 0)

     , np.sum(test.y_hat_log< 0))

test.loc[test.y_hat_xgb_c < 0, 'y_hat_xgb_c'] = 0

test.loc[test.y_hat_xgb_c2 < 0, 'y_hat_xgb_c2'] = 0

test.loc[test.y_hat_xgb_f < 0, 'y_hat_xgb_f'] = 0

test.loc[test.wg < 0, 'wg']               = 0

test.loc[test.ARIMA < 0, 'ARIMA']         = 0

test.loc[test.SARIMAX < 0, 'SARIMAX']     = 0

test.loc[test.y_hat_log < 0, 'y_hat_log'] = 0



print('done', datetime.now())
print('Fixing Inf Predictions:'

     , np.sum(test.y_hat_xgb_c.isnull())

     , np.sum(test.y_hat_xgb_c2.isnull())

     , np.sum(test.y_hat_xgb_f.isnull())

     , np.sum(test.wg.isnull())

     , np.sum(test.ARIMA.isnull())

     , np.sum(test.SARIMAX.isnull())

     , np.sum(test.y_hat_log.isnull()))



booll   = (test['SARIMAX'].isnull())

test.loc[booll, 'SARIMAX'] = test[booll]['y_hat_log']



booll   = (test['ARIMA'].isnull())

test.loc[booll, 'ARIMA'] = test[booll]['y_hat_log']



print('Fixing Inf Predictions:'

     , np.sum(test.y_hat_xgb_c.isnull())

     , np.sum(test.y_hat_xgb_c2.isnull())

     , np.sum(test.y_hat_xgb_f.isnull())

     , np.sum(test.wg.isnull())

     , np.sum(test.ARIMA.isnull())

     , np.sum(test.SARIMAX.isnull())

     , np.sum(test.y_hat_log.isnull()))

print('done', datetime.now())
print('Fixing Unrealistic Predictions:',

      np.sum( test['pop'] * (700/3100) < test.y_hat_xgb_c),

      np.sum( test['pop'] * (700/3100) < test.y_hat_xgb_c2),

      np.sum( test['pop'] * (700/3100) < test.y_hat_xgb_f),

      np.sum( test['pop'] * (700/3100) < test.wg),

      np.sum( test['pop'] * (700/3100) < test.ARIMA),

      np.sum( test['pop'] * (700/3100) < test.SARIMAX),

      np.sum( test['pop'] * (700/3100) < test.y_hat_log),

           )



booll   = (test['pop'] * (700/3100) < test.y_hat_xgb_c)

test.loc[booll, 'y_hat_xgb_c'] = test[booll]['pop'] * (700/3100) # from cruise ships



booll   = (test['pop'] * (700/3100) < test.y_hat_xgb_c2)

test.loc[booll, 'y_hat_xgb_c2'] = test[booll]['pop'] * (700/3100)



booll   = (test['pop'] * (700/3100) < test.y_hat_xgb_f)

test.loc[booll, 'y_hat_xgb_f'] = test[booll]['pop'] * (700/3100)



booll   = (test['pop'] * (700/3100) < test.wg)

test.loc[booll, 'wg'] = test[booll]['pop'] * (700/3100)



booll   = (test['pop'] * (700/3100) < test.ARIMA)

test.loc[booll, 'ARIMA'] = test[booll]['pop'] * (700/3100)



booll   = (test['pop'] * (700/3100) < test.SARIMAX)

test.loc[booll, 'SARIMAX'] = test[booll]['pop'] * (700/3100) 



booll   = (test['pop'] * (700/3100) < test.y_hat_log)

test.loc[booll, 'y_hat_log'] = test[booll]['pop'] * (700/3100)



print('Fixed Unrealistic Predictions:',

      np.sum( test['pop'] * (700/3100) < test.y_hat_xgb_c),

      np.sum( test['pop'] * (700/3100) < test.y_hat_xgb_f),

      np.sum( test['pop'] * (700/3100) < test.wg),

      np.sum( test['pop'] * (700/3100) < test.ARIMA),

      np.sum( test['pop'] * (700/3100) < test.SARIMAX),

      np.sum( test['pop'] * (700/3100) < test.y_hat_log),

           )

print('done', datetime.now())
train[

    (train.Province_State=='')&

    (train.ConfirmedCases>0)&

    (train.Country_Region=='France')][['Date','ConfirmedCases', 'wg']].tail(25)
test['y_hat_ens']  = test.y_hat_xgb_c *.15   + test.wg *.05 + test.y_hat_log *.30  + test['SARIMAX'] * .03 + test.y_hat_xgb_c2 *.47

test[(test.Province_State=='')&(test.Country_Region=='France')][['Date','y_hat_xgb_c','y_hat_xgb_c2', 'wg','SARIMAX', 'SARIMAX2','y_hat_log', 'y_hat_ens']]
# In case I missed anything

print('Empty Predictions?', np.sum(test.y_hat_ens.isnull()))



train[

    (train.Province_State=='')&

    (train.ConfirmedCases>0)&

    (train.Country_Region=='Turkey')][['Date','ConfirmedCases', 'wg', 'day_from_jan_first']].tail(25)
# fixing 'bad' models

narf = train[train.day_from_jan_first== 79].copy()

narf = narf.merge(test[test.day_from_jan_first== 79], on=['Country_Region','Province_State','day_from_jan_first'])



narf['err'] = np.abs((narf.wg_y - narf.ConfirmedCases)/(1+narf.ConfirmedCases))



print(mean_squared_log_error(narf.ConfirmedCases,narf.y_hat_ens))

print('wg', mean_squared_log_error(narf.ConfirmedCases,narf.wg_y))

print('SARIMAX', mean_squared_log_error(narf.ConfirmedCases,narf.SARIMAX))

print('y_hat_log', mean_squared_log_error(narf.ConfirmedCases,narf.y_hat_log_y))

print('y_hat_xgb_c', mean_squared_log_error(narf.ConfirmedCases,narf.y_hat_xgb_c))



narf = narf[narf.err>100].copy()

if narf.shape[0]> 0:

    print(mean_squared_log_error(narf.ConfirmedCases,narf.y_hat_ens))

else:

    print('no fixes for wg_y')

    

for index, row in narf.iterrows():

    country = row['Country_Region']

    state   = row['Province_State']

    booll   = (test.Country_Region==country) & (test.Province_State==state)

    test.loc[booll, 'wg'] = (test[booll].y_hat_log) * .3 + (test[booll].y_hat_xgb_c) * .7



print('done', datetime.now())
train[

    (train.Province_State=='')&

    (train.ConfirmedCases>0)&

    (train.Country_Region=='France')][['Date','ConfirmedCases', 'wg', 'day_from_jan_first']].tail(25)
test['y_hat_ens']  = test.y_hat_xgb_c *.15   + test.wg *.05 + test.y_hat_log *.30  + test['SARIMAX'] * .03 + test.y_hat_xgb_c2 *.47

test[(test.Province_State=='')&(test.Country_Region=='France')][['Date','y_hat_xgb_c','y_hat_xgb_c2', 'wg', 'ARIMA','SARIMAX','y_hat_log', 'y_hat_ens']]
# fixing 'bad' models

narf = train[train.day_from_jan_first== 79].copy()

narf = narf.merge(test[test.day_from_jan_first== 79], on=['Country_Region','Province_State','day_from_jan_first'])



narf['err'] = np.abs((narf.y_hat_log_y - narf.ConfirmedCases)/(1+narf.ConfirmedCases))



print(mean_squared_log_error(narf.ConfirmedCases,narf.y_hat_ens))

print('wg', mean_squared_log_error(narf.ConfirmedCases,narf.wg_y))

print('SARIMAX', mean_squared_log_error(narf.ConfirmedCases,narf.SARIMAX))

print('y_hat_log', mean_squared_log_error(narf.ConfirmedCases,narf.y_hat_log_y))

print('y_hat_xgb_c', mean_squared_log_error(narf.ConfirmedCases,narf.y_hat_xgb_c))



narf = narf[narf.err>10].copy()

if narf.shape[0]> 0:

    print(mean_squared_log_error(narf.ConfirmedCases,narf.y_hat_ens))

else:

    print('no fixes for y_hat_log')

    

for index, row in narf.iterrows():

    country = row['Country_Region']

    state   = row['Province_State']

    booll   = (test.Country_Region==country) & (test.Province_State==state)

    #test.loc[booll, 'wg'] = (test[booll].y_hat_log) * .3 + (test[booll].y_hat_xgb_c) * .7

print('done', datetime.now())
train[

    (train.Province_State=='')&

    (train.ConfirmedCases>0)&

    (train.Country_Region=='Turkey')][['Date','ConfirmedCases', 'wg', 'day_from_jan_first']].tail(25)
test[test.y_hat_ens==1521225]

test[(test.Province_State=='')&(test.Country_Region=='Turkey')][['Date','y_hat_xgb_c', 'wg', 'ARIMA','SARIMAX','y_hat_log', 'y_hat_ens']]
test['y_hat_ens'] = test.y_hat_ens.astype(int)

print(np.max( test['y_hat_ens']  ))



###############################################################################

## Submision

###############################################################################



submissionOrig = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

submissionOrig["ConfirmedCases"]= pd.Series( test.y_hat_ens)

submissionOrig["Fatalities"]    = pd.Series( test.y_hat_xgb_f)

submissionOrig.to_csv('submission.csv',index=False)

submissionOrig.head(25)

print('done', datetime.now())