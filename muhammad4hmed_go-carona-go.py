import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR 

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.preprocessing import PolynomialFeatures
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
def fuckProvince(s):

    if pd.isnull(s['Province_State']):

        if pd.isnull(s['Country_Region']):

            return 'Null'

        else:

            return s['Country_Region']

    else:

        return s['Province_State']
train['Province_State'] = train.apply(fuckProvince,axis=1)

test['Province_State'] = test.apply(fuckProvince,axis=1)
tr = train.copy()

ts = test.copy()
X = train.drop(['ConfirmedCases','Fatalities'],axis=1)

y = train[['ConfirmedCases','Fatalities']]

y1 = train['ConfirmedCases']

y2 = train['Fatalities']
def fuckDate(s):

    return s.month * 100 + s.day
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
train['Fucking_Useful_Dates'] = train['Date'].apply(fuckDate)

test['Fucking_Useful_Dates'] = test['Date'].apply(fuckDate)
train['Month_People_Were_Fucked'] = train['Date'].dt.month

test['Month_People_Were_Fucked'] = test['Date'].dt.month
train.drop(['Date'],axis=1,inplace=True)

test.drop(['Date'],axis=1,inplace=True)
pred = pd.DataFrame(columns=['ForecastId','ConfirmedCases','Fatalities'])
pred['ForecastId'] = test['ForecastId'].copy()
LE = LabelEncoder()

train['Province_State'] = LE.fit_transform(train['Province_State'])

test['Province_State'] = LE.transform(test['Province_State'])



LE2 = LabelEncoder()

train['Country_Region'] = LE2.fit_transform(train['Country_Region'])

test['Country_Region'] = LE2.transform(test['Country_Region'])
for state in tr['Province_State'].unique():

    try:

        tr_in = tr[ tr['Province_State'] == state ].index

        ts_in = ts[ ts['Province_State'] == state ].index

        ids = ts[ ts['Province_State'] == state ].loc[:,'ForecastId']

        X_train = train.iloc[tr_in,:].drop(['ConfirmedCases','Fatalities','Id'],axis=1)

        y1_train = train.iloc[tr_in,:].loc[:,'ConfirmedCases']

        y2_train = train.iloc[tr_in,:].loc[:,'Fatalities']

        X_test = test.iloc[ts_in, :].drop(['ForecastId'],axis=1)

        pred.at[ts_in,'ForecastId'] = ids



        model1 = XGBRegressor(n_estimators = 3000 )

        model1.fit(X_train,y1_train)

        pred.at[ts_in,'ConfirmedCases'] = model1.predict(X_test)



        model2 = XGBRegressor(n_estimators = 3000 )

        model2.fit(X_train,y2_train)

        pred.at[ts_in,'Fatalities'] = model2.predict(X_test)

    except:

        print(state)
pred['ConfirmedCases'] = np.round(pred['ConfirmedCases'].astype(np.double),0)

pred['Fatalities'] = np.round(pred['Fatalities'].astype(np.double),0)
pred
pred.to_csv('submission.csv',index=False)