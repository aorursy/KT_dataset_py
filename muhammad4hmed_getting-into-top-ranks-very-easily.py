import pandas as pd

import numpy as np



from xgboost import XGBRegressor



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
df.head()
def fill_States(s):

    if pd.isna(s['Province_State']):

        if pd.isna(s['Country_Region']):

            return "NA"

        else:

            return s['Country_Region']

    else:

        return s['Province_State']
df['Province_State'] = df.apply(fill_States,axis=1)
df_test['Province_State'] = df_test.apply(fill_States,axis=1)
df['Date'] = pd.to_datetime(df['Date'])

df_test['Date'] = pd.to_datetime(df_test['Date'])
df['Date'] = df['Date'].dt.month * 100 + df['Date'].dt.day

df_test['Date'] = df_test['Date'].dt.month * 100 + df_test['Date'].dt.day
preds = pd.DataFrame()
preds['ForecastId'] = df_test.ForecastId
df_test.columns
le = LabelEncoder()

le2 = LabelEncoder()
for country in df['Country_Region'].unique():

    for state in df[df['Country_Region']==country].loc[:,'Province_State'].unique():

        train = df[ df['Province_State'] == state ].copy()

        X = train.drop(['Id','ConfirmedCases','Fatalities'],axis=1)

        X['Province_State'] = le.fit_transform(X['Province_State'])

        X['Country_Region'] = le2.fit_transform(X['Country_Region'])

        y1 = train['ConfirmedCases']

        y2 = train['Fatalities']

        

        test = df_test[ df_test['Province_State'] == state ].copy()

        X_test = test.drop(['ForecastId'],axis=1)

        X_test['Province_State'] = le.transform(X_test['Province_State'])

        X_test['Country_Region'] = le2.transform(X_test['Country_Region'])

        

        model1 = XGBRegressor(n_estimators=1000)

        model1.fit(X,y1)

        preds.at[test['ForecastId'] - 1,'ConfirmedCases'] = model1.predict(X_test)

        

        model2 = XGBRegressor(n_estimators=1000)

        model2.fit(X,y2)

        preds.at[test['ForecastId'] - 1,'Fatalities'] = model2.predict(X_test)

        
preds.to_csv('submission.csv',index=False)