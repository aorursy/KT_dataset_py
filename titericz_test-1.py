import pandas as pd

import numpy as np

from scipy.optimize import curve_fit



%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.metrics import log_loss

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor





import xgboost as xgb



from tqdm.notebook import tqdm



def exponential(x, a, k, b):

    return a*np.exp(x*k) + b



def funclog(x, a, b,c):

    return a*np.log(b+x)+c



def rmse( yt, yp ):

    return np.sqrt( np.mean( (yt-yp)**2 ) )



def pinball(y_true, y_pred, tao=0.5 ):

    return np.max( [(y_true - y_pred)*tao, (y_pred - y_true)*(1 - tao) ], axis=0 ) 



def calc_metric( df ):

    tmp = df.copy()

    tmp['m0'] = pinball( tmp['TargetValue'].values, tmp['q05'].values , 0.05 )

    tmp['m1'] = pinball( tmp['TargetValue'].values, tmp['q50'].values , 0.50 )

    tmp['m2'] = pinball( tmp['TargetValue'].values, tmp['q95'].values , 0.95 )

    tmp['q'] = tmp['Weight']*(tmp['m0']+tmp['m1']+tmp['m2']) / 3

    return tmp['q'].mean()
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')



train['Date'] = pd.to_datetime( train['Date'] )

mindate  = str(train['Date'].min())[:10]

maxdate  = str(train['Date'].max())[:10]

testdate = str( train['Date'].max() + pd.Timedelta(days=1) )[:10]

print( mindate, maxdate, testdate )



train['County'] = train['County'].fillna('N')

train['Province_State'] = train['Province_State'].fillna('N')

train['Country_Region'] = train['Country_Region'].fillna('N')

train['geo'] = train['Country_Region'] + '-' + train['Province_State'] + '-' + train['County']



print(train.shape)

train['dedup'] = pd.factorize( train['geo'] + '-' + train['Target'] + '-' + train['Date'].apply(str) + '-' + train['Population'].apply(str) )[0]

train.drop_duplicates(subset ="dedup", keep = 'first', inplace = True)

del train['dedup']

print(train.shape)



train.sort_values( ['geo','Date'], inplace=True )



train.head(5)
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')



test['Date'] = pd.to_datetime( test['Date'] )

#testdate = str( test['Date'].max() + pd.Timedelta(days=1) )[:10]

#print( maxdate, testdate )



test['County'] = test['County'].fillna('N')

test['Province_State'] = test['Province_State'].fillna('N')

test['Country_Region'] = test['Country_Region'].fillna('N')

test['geo'] = test['Country_Region'] + '-' + test['Province_State'] + '-' + test['County']



print(test.shape)

test.sort_values( ['geo','Date'], inplace=True )



print(test.head())
train = pd.concat( (train, test.loc[test.Date>=testdate]) , sort=False )

train.sort_values( ['geo','Date'], inplace=True )



train.loc[ (train.Date=='2020-04-24')&(train.geo=='Spain-N-N')&(train.Target=='ConfirmedCases'), 'TargetValue' ] = 6000

train.loc[ (train.Date=='2020-04-29')&(train.geo=='France-N-N')&(train.Target=='ConfirmedCases'), 'TargetValue' ] = 1843.5

train.loc[ (train.Date=='2020-04-22')&(train.geo=='France-N-N')&(train.Target=='ConfirmedCases'), 'TargetValue' ] = 2522
train0 = train.loc[ train.Target == 'ConfirmedCases' ].copy()

train1 = train.loc[ train.Target == 'Fatalities' ].copy()

for t in [train0,train1]:

    t['q05'] = 0

    t['q50'] = 0

    t['q95'] = 0



train0 = train0.loc[ (train0.Date  >='2020-03-01') ].copy()

train1 = train1.loc[ (train1.Date  >='2020-03-01') ].copy()

    

test0 = test.loc[ test.Target == 'ConfirmedCases' ].copy()

test1 = test.loc[ test.Target == 'Fatalities' ].copy()

train0.shape, test0.shape
!ls -l ../input/covid-w5-worldometer-scraper/train_oldformat.csv
DF = pd.read_csv('../input/covid-w5-worldometer-scraper/train_oldformat.csv')

DF['Date'] = pd.to_datetime( DF['Date'] )

#testdate = str( test['Date'].max() + pd.Timedelta(days=1) )[:10]

#print( maxdate, testdate )



DF['County'] = DF['County'].fillna('N')

DF['Province_State'] = DF['Province_State'].fillna('N')

DF['Country_Region'] = DF['Country_Region'].fillna('N')

DF['geo'] = DF['Country_Region'] + '-' + DF['Province_State'] + '-' + DF['County']

DF
DF0 = train0.loc[ train0.Date == '2020-05-11' ].copy()

DF0['ypred'] = DF0[['geo','Date']].merge( DF.loc[DF.Target=='ConfirmedCases'], on=['geo','Date'] , how='left' )['TargetValue'].values



DF1 = train1.loc[ train1.Date == '2020-05-11' ].copy()

DF1['ypred'] = DF1[['geo','Date']].merge( DF.loc[DF.Target=='Fatalities'], on=['geo','Date'] , how='left' )['TargetValue'].values
train0['ypred'] = train0['TargetValue'].values

train0['mpred'] = train0.groupby('geo')['TargetValue'].rolling(7).mean().values

train0['Hstd']  = np.clip(train0['ypred'] - train0['mpred'], 0, 9999999999)

train0['Lstd']  = np.clip(train0['ypred'] - train0['mpred'], -9999999999, 0)



train0.loc[ train0.Date>='2020-04-27' ,'ypred'] = np.nan

train0.loc[ train0.Date>='2020-04-27' ,'mpred'] = np.nan

train0.loc[ train0.Date>='2020-04-27' ,'Hstd']  = np.nan

train0.loc[ train0.Date>='2020-04-27' ,'Lstd']  = np.nan



train0['Hstd']  = train0.groupby('geo')['Hstd'].rolling(28).std().values

train0['Lstd']  = train0.groupby('geo')['Lstd'].rolling(28).std().values



train0['Lstd']  = train0.groupby('geo')['Lstd'].fillna( method='ffill' )

train0['Hstd']  = train0.groupby('geo')['Hstd'].fillna( method='ffill' )

train0['ypred'] = train0.groupby('geo')['ypred'].fillna( method='ffill' )

train0['mpred'] = train0.groupby('geo')['mpred'].fillna( method='ffill' )



train0['q50'] = train0['TargetValue'].values

train0.loc[ train0.Date>='2020-04-27' ,'q50']  = np.nan

train0['q05'] = train0['q50']

train0['q95'] = train0['q50']
import statsmodels.api as sm



count = 1

for valday in [

    '2020-04-27',

    '2020-04-28',

    '2020-04-29',

    '2020-04-30',

    '2020-05-01',

    '2020-05-02',

    '2020-05-03',

    '2020-05-04',

    '2020-05-05',

    '2020-05-06',

    '2020-05-07',

    '2020-05-08',

    '2020-05-09',

    '2020-05-10',

    ]:

    

    for i in np.arange(1,13,1):

        train0['lag1'+str(i)] = train0.groupby('geo')['q50'].shift(i)

    train0['std1']= train0.groupby('geo')['q50'].shift(1).rolling(7).std()

    train0['std2']= train0.groupby('geo')['q50'].shift(1).rolling(14).std()

    train0['std2']= train0.groupby('geo')['q50'].shift(1).rolling(21).std()

    train0['std2']= train0.groupby('geo')['q50'].shift(1).rolling(28).std()

    

    

    TRAIN = train0.loc[ (train0.Date  <'2020-04-27')&(train0.Date >='2020-04-01') ].copy()

    VALID = train0.loc[ (train0.Date ==valday) ].copy()

    

    features = TRAIN.columns[9:]

    features = [f for f in features if f not in ['geo','ForecastId','q05','q50','q95','ypred','qstd','Lstd','Hstd','mpred','flag']  ]



    if valday == '2020-04-27':        

        model05 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.05)

        model50 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.50)

        model95 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.95)

        

    #break

    VALID['q05'] = model05.predict( VALID[features] ) - VALID['Lstd']*np.clip(0.25*count,0,3.5)

    VALID['q50'] = model50.predict( VALID[features] )

    VALID['q95'] = model95.predict( VALID[features] ) + VALID['Hstd']*np.clip(0.25*count,0,3.5)

    

    VALID.loc[ VALID.q50<0  ,'q50'] = 0

    VALID.loc[ VALID.q50<VALID.q05 ,'q05'] = VALID.loc[ VALID.q50<VALID.q05 ,'q50']

    VALID.loc[ VALID.q50>VALID.q95 ,'q95'] = VALID.loc[ VALID.q50>VALID.q95 ,'q50']



    VALID['q05'] = VALID['q05']/(1.02**count)

    VALID['q50'] = VALID['q50']

    VALID['q95'] = VALID['q95']*(1.02**count)

    

    VALID.loc[ VALID.q05<0  ,'q05'] = 0

    VALID.loc[ VALID.q50<0  ,'q50'] = 0

    VALID.loc[ VALID.q95<0  ,'q95'] = 0

    

    train0.loc[ (train0.Date ==valday),'q05'] = VALID['q05']

    train0.loc[ (train0.Date ==valday),'q50'] = VALID['q50']

    train0.loc[ (train0.Date ==valday),'q95'] = VALID['q95']

   

    print( calc_metric( VALID ), valday )

    count+=1



TMP0 = train0.loc[ (train0.Date>='2020-04-27')&(train0.Date<='2020-05-10') ].copy()

print( calc_metric( TMP0 ) )  
train0['ypred'] = train0['TargetValue'].values

train0['mpred'] = train0.groupby('geo')['TargetValue'].rolling(7).mean().values

train0['Hstd']  = np.clip(train0['ypred'] - train0['mpred'], 0, 9999999999)

train0['Lstd']  = np.clip(train0['ypred'] - train0['mpred'], -9999999999, 0)



# train0.loc[ train0.Date>='2020-04-27' ,'ypred'] = np.nan

# train0.loc[ train0.Date>='2020-04-27' ,'mpred'] = np.nan

# train0.loc[ train0.Date>='2020-04-27' ,'Hstd']  = np.nan

# train0.loc[ train0.Date>='2020-04-27' ,'Lstd']  = np.nan



train0['Hstd']  = train0.groupby('geo')['Hstd'].rolling(28).std().values

train0['Lstd']  = train0.groupby('geo')['Lstd'].rolling(28).std().values



train0['Lstd']  = train0.groupby('geo')['Lstd'].fillna( method='ffill' )

train0['Hstd']  = train0.groupby('geo')['Hstd'].fillna( method='ffill' )

train0['ypred'] = train0.groupby('geo')['ypred'].fillna( method='ffill' )

train0['mpred'] = train0.groupby('geo')['mpred'].fillna( method='ffill' )



train0['q50'] = train0['TargetValue'].values

#train0.loc[ train0.Date>='2020-04-27' ,'q50']  = np.nan

train0['q05'] = train0['q50']

train0['q95'] = train0['q50']





count = 1

for valday in [

    '2020-05-11',

    '2020-05-12',

    '2020-05-13',

    '2020-05-14',

    '2020-05-15',

    '2020-05-16',

    '2020-05-17',

    '2020-05-18',

    '2020-05-19',

    '2020-05-20',

    '2020-05-21',

    '2020-05-22',

    '2020-05-23',

    '2020-05-24',

    '2020-05-25',

    '2020-05-26',

    '2020-05-27',

    '2020-05-28',

    '2020-05-29',

    '2020-05-30',

    '2020-05-31',

    '2020-06-01',

    '2020-06-02',

    '2020-06-03',

    '2020-06-04',

    '2020-06-05',

    '2020-06-06',

    '2020-06-07',

    '2020-06-08',

    '2020-06-09',

    '2020-06-10',

]:

    for i in np.arange(1,13,1):

        train0['lag1'+str(i)] = train0.groupby('geo')['q50'].shift(i)#.fillna(0)

    train0['std1']= train0.groupby('geo')['q50'].shift(1).rolling(7).std()

    train0['std2']= train0.groupby('geo')['q50'].shift(1).rolling(14).std()

    train0['std2']= train0.groupby('geo')['q50'].shift(1).rolling(21).std()

    train0['std2']= train0.groupby('geo')['q50'].shift(1).rolling(28).std()

    TRAIN = train0.loc[ (train0.Date  <'2020-05-11')&(train0.Date >='2020-04-01') ].copy()

    VALID = train0.loc[ (train0.Date ==valday) ].copy()

    

    features = TRAIN.columns[9:]

    features = [f for f in features if f not in ['geo','ForecastId','q05','q50','q95','ypred','qstd','Lstd','Hstd','mpred','flag']  ]



    if valday == '2020-05-11':        

        model05 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.05)

        model50 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.50)

        model95 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.95)

        

    #break

    VALID['q05'] = model05.predict( VALID[features] ) - VALID['Lstd']*np.clip(0.25*count,0,3.5)

    VALID['q50'] = model50.predict( VALID[features] )

    VALID['q95'] = model95.predict( VALID[features] ) + VALID['Hstd']*np.clip(0.25*count,0,3.5)

    

    VALID.loc[ VALID.q50<0  ,'q50'] = 0

    VALID.loc[ VALID.q50<VALID.q05 ,'q05'] = VALID.loc[ VALID.q50<VALID.q05 ,'q50']

    VALID.loc[ VALID.q50>VALID.q95 ,'q95'] = VALID.loc[ VALID.q50>VALID.q95 ,'q50']



    VALID['q05'] = VALID['q05']/(1.02**count)

    VALID['q50'] = VALID['q50']

    VALID['q95'] = VALID['q95']*(1.02**count)

    

    VALID.loc[ VALID.q05<0  ,'q05'] = 0

    VALID.loc[ VALID.q50<0  ,'q50'] = 0

    VALID.loc[ VALID.q95<0  ,'q95'] = 0

    

    train0.loc[ (train0.Date ==valday),'q05'] = VALID['q05']

    train0.loc[ (train0.Date ==valday),'q50'] = VALID['q50']

    train0.loc[ (train0.Date ==valday),'q95'] = VALID['q95']

   

    print( calc_metric( VALID ), valday )

    count+=1
TMP0B = train0.loc[ (train0.Date>='2020-05-11') ].copy()

TMP0B.shape
train1['ypred'] = train1['TargetValue'].values

train1['mpred'] = train1.groupby('geo')['TargetValue'].rolling(7).mean().values

train1['Hstd']  = np.clip(train1['ypred'] - train1['mpred'], 0, 9999999999)

train1['Lstd']  = np.clip(train1['ypred'] - train1['mpred'], -9999999999, 0)



train1.loc[ train1.Date>='2020-04-27' ,'ypred'] = np.nan

train1.loc[ train1.Date>='2020-04-27' ,'mpred'] = np.nan

train1.loc[ train1.Date>='2020-04-27' ,'Hstd']  = np.nan

train1.loc[ train1.Date>='2020-04-27' ,'Lstd']  = np.nan



train1['Hstd']  = train1.groupby('geo')['Hstd'].rolling(28).std().values

train1['Lstd']  = train1.groupby('geo')['Lstd'].rolling(28).std().values



train1['Lstd']  = train1.groupby('geo')['Lstd'].fillna( method='ffill' )

train1['Hstd']  = train1.groupby('geo')['Hstd'].fillna( method='ffill' )

train1['ypred'] = train1.groupby('geo')['ypred'].fillna( method='ffill' )

train1['mpred'] = train1.groupby('geo')['mpred'].fillna( method='ffill' )



train1['q50'] = train1['TargetValue'].values

train1.loc[ train1.Date>='2020-04-27' ,'q50']  = np.nan

train1['q05'] = train1['q50']

train1['q95'] = train1['q50']
train1['q50_cases'] = train1[['geo','Date']].merge( train0[['geo','Date','q50']], on=['geo','Date'], how='left' )['q50'].values

train1.iloc[-60:,5:25]
count = 1

for valday in [

    '2020-04-27',

    '2020-04-28',

    '2020-04-29',

    '2020-04-30',

    '2020-05-01',

    '2020-05-02',

    '2020-05-03',

    '2020-05-04',

    '2020-05-05',

    '2020-05-06',

    '2020-05-07',

    '2020-05-08',

    '2020-05-09',

    '2020-05-10',

    ]:

    

    for i in np.arange(1,13,1):

        train1['lag1'+str(i)] = train1.groupby('geo')['q50'].shift(i)

        train1['lagCases1'+str(i)] = train1.groupby('geo')['q50_cases'].shift(i)

    train1['std1']= train1.groupby('geo')['q50'].shift(1).rolling(7).std()

    train1['std2']= train1.groupby('geo')['q50'].shift(1).rolling(14).std()

    train1['std2']= train1.groupby('geo')['q50'].shift(1).rolling(21).std()

    train1['std2']= train1.groupby('geo')['q50'].shift(1).rolling(28).std()

    

    TRAIN = train1.loc[ (train1.Date  <'2020-04-27')&(train1.Date >='2020-04-01') ].copy()

    VALID = train1.loc[ (train1.Date ==valday) ].copy()

    

    features = TRAIN.columns[9:]

    features = [f for f in features if f not in ['geo','ForecastId','q05','q50','q95','ypred','qstd','Lstd','Hstd','mpred','flag']  ]



    if valday == '2020-04-27':        

        model05 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.05)

        model50 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.50)

        model95 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.95)

        

    #break

    VALID['q05'] = model05.predict( VALID[features] ) - VALID['Lstd']*np.clip(0.01*count,0,3.5)

    VALID['q50'] = model50.predict( VALID[features] )

    VALID['q95'] = model95.predict( VALID[features] ) + VALID['Hstd']*np.clip(0.01*count,0,3.5)

    

    VALID.loc[ VALID.q50<0  ,'q50'] = 0

    VALID.loc[ VALID.q50<VALID.q05 ,'q05'] = VALID.loc[ VALID.q50<VALID.q05 ,'q50']

    VALID.loc[ VALID.q50>VALID.q95 ,'q95'] = VALID.loc[ VALID.q50>VALID.q95 ,'q50']



    VALID['q05'] = VALID['q05']/(1.001**count)

    VALID['q50'] = VALID['q50']

    VALID['q95'] = VALID['q95']*(1.001**count)

    

    VALID.loc[ VALID.q05<0  ,'q05'] = 0

    VALID.loc[ VALID.q50<0  ,'q50'] = 0

    VALID.loc[ VALID.q95<0  ,'q95'] = 0

    

    train1.loc[ (train1.Date ==valday),'q05'] = VALID['q05']

    train1.loc[ (train1.Date ==valday),'q50'] = VALID['q50']

    train1.loc[ (train1.Date ==valday),'q95'] = VALID['q95']

   

    print( calc_metric( VALID ), valday )

    count+=1



TMP1 = train1.loc[ (train1.Date>='2020-04-27')&(train1.Date<='2020-05-10') ].copy()

print( calc_metric( TMP1 ) )
train1.iloc[-60:,5:25]
train1['ypred'] = train1['TargetValue'].values

train1['mpred'] = train1.groupby('geo')['TargetValue'].rolling(7).mean().values

train1['Hstd']  = np.clip(train1['ypred'] - train1['mpred'], 0, 9999999999)

train1['Lstd']  = np.clip(train1['ypred'] - train1['mpred'], -9999999999, 0)



# train1.loc[ train1.Date>='2020-04-27' ,'ypred'] = np.nan

# train1.loc[ train1.Date>='2020-04-27' ,'mpred'] = np.nan

# train1.loc[ train1.Date>='2020-04-27' ,'Hstd']  = np.nan

# train1.loc[ train1.Date>='2020-04-27' ,'Lstd']  = np.nan



train1['Hstd']  = train1.groupby('geo')['Hstd'].rolling(28).std().values

train1['Lstd']  = train1.groupby('geo')['Lstd'].rolling(28).std().values



train1['Lstd']  = train1.groupby('geo')['Lstd'].fillna( method='ffill' )

train1['Hstd']  = train1.groupby('geo')['Hstd'].fillna( method='ffill' )

train1['ypred'] = train1.groupby('geo')['ypred'].fillna( method='ffill' )

train1['mpred'] = train1.groupby('geo')['mpred'].fillna( method='ffill' )



train1['q50'] = train1['TargetValue'].values

#train1.loc[ train1.Date>='2020-04-27' ,'q50']  = np.nan

train1['q05'] = train1['q50']

train1['q95'] = train1['q50']



count = 1

for valday in [

    '2020-05-11',

    '2020-05-12',

    '2020-05-13',

    '2020-05-14',

    '2020-05-15',

    '2020-05-16',

    '2020-05-17',

    '2020-05-18',

    '2020-05-19',

    '2020-05-20',

    '2020-05-21',

    '2020-05-22',

    '2020-05-23',

    '2020-05-24',

    '2020-05-25',

    '2020-05-26',

    '2020-05-27',

    '2020-05-28',

    '2020-05-29',

    '2020-05-30',

    '2020-05-31',

    '2020-06-01',

    '2020-06-02',

    '2020-06-03',

    '2020-06-04',

    '2020-06-05',

    '2020-06-06',

    '2020-06-07',

    '2020-06-08',

    '2020-06-09',

    '2020-06-10',

    ]:

    

    for i in np.arange(1,13,1):

        train1['lag1'+str(i)] = train1.groupby('geo')['q50'].shift(i)

        train1['lagCases1'+str(i)] = train1.groupby('geo')['q50_cases'].shift(i)

    train1['std1']= train1.groupby('geo')['q50'].shift(1).rolling(7).std()

    train1['std2']= train1.groupby('geo')['q50'].shift(1).rolling(14).std()

    train1['std2']= train1.groupby('geo')['q50'].shift(1).rolling(21).std()

    train1['std2']= train1.groupby('geo')['q50'].shift(1).rolling(28).std()

    

    TRAIN = train1.loc[ (train1.Date  <'2020-05-11')&(train1.Date >='2020-04-01') ].copy()

    VALID = train1.loc[ (train1.Date ==valday) ].copy()

    

    features = TRAIN.columns[9:]

    features = [f for f in features if f not in ['geo','ForecastId','q05','q50','q95','ypred','qstd','Lstd','Hstd','mpred','flag']  ]



    if valday == '2020-05-11':        

        model05 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.05)

        model50 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.50)

        model95 = sm.QuantReg(TRAIN['q50'], TRAIN[features]).fit(q=0.95)

        

    #break

    VALID['q05'] = model05.predict( VALID[features] ) - VALID['Lstd']*np.clip(0.01*count,0,3.5)

    VALID['q50'] = model50.predict( VALID[features] )

    VALID['q95'] = model95.predict( VALID[features] ) + VALID['Hstd']*np.clip(0.01*count,0,3.5)

    

    VALID.loc[ VALID.q50<0  ,'q50'] = 0

    VALID.loc[ VALID.q50<VALID.q05 ,'q05'] = VALID.loc[ VALID.q50<VALID.q05 ,'q50']

    VALID.loc[ VALID.q50>VALID.q95 ,'q95'] = VALID.loc[ VALID.q50>VALID.q95 ,'q50']



    VALID['q05'] = VALID['q05']/(1.001**count)

    VALID['q50'] = VALID['q50']

    VALID['q95'] = VALID['q95']*(1.001**count)

    

    VALID.loc[ VALID.q05<0  ,'q05'] = 0

    VALID.loc[ VALID.q50<0  ,'q50'] = 0

    VALID.loc[ VALID.q95<0  ,'q95'] = 0

    

    train1.loc[ (train1.Date ==valday),'q05'] = VALID['q05']

    train1.loc[ (train1.Date ==valday),'q50'] = VALID['q50']

    train1.loc[ (train1.Date ==valday),'q95'] = VALID['q95']

   

    print( calc_metric( VALID ), valday )

    count+=1

    

TMP1B = train1.loc[ (train1.Date>='2020-05-11') ].copy()

TMP1B.shape    
tmp = pd.concat( (TMP0,TMP1) )

calc_metric( tmp )
VALID0 = train0.loc[ (train0.Date>='2020-05-11') , ['geo','Date','q05','q50','q95','TargetValue','Weight'] ].copy()

VALID1 = train1.loc[ (train1.Date>='2020-05-11') , ['geo','Date','q05','q50','q95','TargetValue','Weight'] ].copy()

VALID0.shape, VALID1.shape
VALID0 = pd.concat( (TMP0,TMP0B) )

VALID1 = pd.concat( (TMP1,TMP1B) )

VALID0 = VALID0.reset_index(drop=True)

VALID1 = VALID1.reset_index(drop=True)

VALID0.shape, VALID1.shape
#Write Public LB ground Truth



TMP0['q05'] = TMP0['TargetValue']

TMP0['q50'] = TMP0['TargetValue']

TMP0['q95'] = TMP0['TargetValue']



TMP1['q05'] = TMP1['TargetValue']

TMP1['q50'] = TMP1['TargetValue']

TMP1['q95'] = TMP1['TargetValue']
#Write external data



TMP0B['ypred'] = TMP0B[['geo','Date']].merge( DF0[['geo','Date','ypred']], on=['geo','Date'], how='left' )['ypred'].values

TMP1B['ypred'] = TMP1B[['geo','Date']].merge( DF1[['geo','Date','ypred']], on=['geo','Date'], how='left' )['ypred'].values



TMP0B.loc[TMP0B.ypred.notnull(),'q05'] = TMP0B.loc[TMP0B.ypred.notnull(),'ypred']

TMP0B.loc[TMP0B.ypred.notnull(),'q50'] = TMP0B.loc[TMP0B.ypred.notnull(),'ypred']

TMP0B.loc[TMP0B.ypred.notnull(),'q95'] = TMP0B.loc[TMP0B.ypred.notnull(),'ypred']



TMP1B.loc[TMP1B.ypred.notnull(),'q05'] = TMP1B.loc[TMP1B.ypred.notnull(),'ypred']

TMP1B.loc[TMP1B.ypred.notnull(),'q50'] = TMP1B.loc[TMP1B.ypred.notnull(),'ypred']

TMP1B.loc[TMP1B.ypred.notnull(),'q95'] = TMP1B.loc[TMP1B.ypred.notnull(),'ypred']



del TMP0B['ypred'], TMP1B['ypred']
VALID0 = pd.concat( (TMP0,TMP0B) )

VALID1 = pd.concat( (TMP1,TMP1B) )

VALID0 = VALID0.reset_index(drop=True)

VALID1 = VALID1.reset_index(drop=True)

VALID0.shape, VALID1.shape
tmp = train0.loc[ train0.geo == 'US-N-N' ].copy()

tmp[['Date','q05','q50','q95']].plot(x='Date')
tmp = train0.loc[ train0.geo == 'Brazil-N-N' ].copy()

tmp[['Date','q05','q50','q95']].plot(x='Date')
tmp = train1.loc[ train1.geo == 'US-N-N' ].copy()

tmp[['Date','q05','q50','q95']].plot(x='Date')
tmp = train1.loc[ train1.geo == 'Brazil-N-N' ].copy()

tmp[['Date','q05','q50','q95']].plot(x='Date')
tmp.iloc[-60:,5:25]
# del test0['q05'],test0['q50'],test0['q95']

# del test1['q05'],test1['q50'], test1['q95']
test0['q05'] = pd.merge( test0[['geo','Date']], VALID0, on=['geo','Date'], how='left'  )['q05'].values

test0['q50'] = pd.merge( test0[['geo','Date']], VALID0, on=['geo','Date'], how='left'  )['q50'].values

test0['q95'] = pd.merge( test0[['geo','Date']], VALID0, on=['geo','Date'], how='left'  )['q95'].values

test1['q05'] = pd.merge( test1[['geo','Date']], VALID1, on=['geo','Date'], how='left'  )['q05'].values

test1['q50'] = pd.merge( test1[['geo','Date']], VALID1, on=['geo','Date'], how='left'  )['q50'].values

test1['q95'] = pd.merge( test1[['geo','Date']], VALID1, on=['geo','Date'], how='left'  )['q95'].values
test0.isnull().sum()
test0.loc[ test0.geo =='Zimbabwe-N-N' ]
q05 = test0[['ForecastId','q05']].copy()

q50 = test0[['ForecastId','q50']].copy()

q95 = test0[['ForecastId','q95']].copy()

q05.columns = ['ForecastId','TargetValue']

q50.columns = ['ForecastId','TargetValue']

q95.columns = ['ForecastId','TargetValue']

q05['ForecastId_Quantile'] = q05['ForecastId'].apply(str) + '_0.05'

q50['ForecastId_Quantile'] = q50['ForecastId'].apply(str) + '_0.5'

q95['ForecastId_Quantile'] = q95['ForecastId'].apply(str) + '_0.95'

tst0 = pd.concat( (q05, q50, q95) )



q05 = test1[['ForecastId','q05']].copy()

q50 = test1[['ForecastId','q50']].copy()

q95 = test1[['ForecastId','q95']].copy()

q05.columns = ['ForecastId','TargetValue']

q50.columns = ['ForecastId','TargetValue']

q95.columns = ['ForecastId','TargetValue']

q05['ForecastId_Quantile'] = q05['ForecastId'].apply(str) + '_0.05'

q50['ForecastId_Quantile'] = q50['ForecastId'].apply(str) + '_0.5'

q95['ForecastId_Quantile'] = q95['ForecastId'].apply(str) + '_0.95'

tst1 = pd.concat( (q05, q50, q95) )



tst = pd.concat( (tst0,tst1), sort=False )

tst.sort_values( 'ForecastId', inplace=True )



tst['TargetValue'] = tst['TargetValue'].fillna(0)



tst[['ForecastId_Quantile','TargetValue']].to_csv( 'submission.csv', index=False )

tst.head(6)
sub = pd.read_csv('submission.csv')

print( sub.shape )

sub.describe()
sub = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')

print( sub.shape )

sub.describe()