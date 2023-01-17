import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
EMPTY_VAL = "EMPTY_VAL"

NAN_VAL = "NaN"



def get_state(state, country):

    if state == EMPTY_VAL: return country

    if state == NAN_VAL: return country

    return state
def calc_score(y_true, y_pred):

    y_true[y_true<0] = 0

    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5

    return score
PATH_WEEK2='/kaggle/input/covid19-global-forecasting-week-2'

df_train = pd.read_csv(f'{PATH_WEEK2}/train.csv')

df_test = pd.read_csv(f'{PATH_WEEK2}/test.csv')

#df_train.head()

#df_test.head()



df_train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_test.rename(columns={'Country_Region':'Country'}, inplace=True)



df_train.rename(columns={'Province_State':'State'}, inplace=True)

df_test.rename(columns={'Province_State':'State'}, inplace=True)



df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)



y1_Train = df_train.iloc[:, -2]

#y1_Train.head()

y2_Train = df_train.iloc[:, -1]

#y2_Train.head()
df_train.info()
df_test.info()
X_Train = df_train.copy()



X_Train['State'].fillna(EMPTY_VAL, inplace=True)

X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : get_state(x['State'], x['Country']), axis=1)



X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Train.head()
X_Pred = df_test.copy()



X_Pred['State'].fillna(EMPTY_VAL, inplace=True)

X_Pred['State'] = X_Pred.loc[:, ['State', 'Country']].apply(lambda x : get_state(x['State'], x['Country']), axis=1)



X_Pred.loc[:, 'Date'] = X_Pred.Date.dt.strftime("%m%d")

X_Pred["Date"]  = X_Pred["Date"].astype(int)



X_Pred.head()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



X_Train.Country = le.fit_transform(X_Train.Country)

X_Train['State'] = le.fit_transform(X_Train['State'])



X_Train.head()
X_Pred.Country = le.fit_transform(X_Pred.Country)

X_Pred['State'] = le.fit_transform(X_Pred['State'])



X_Pred.head()
df_train.head()

df_train.loc[df_train.Country == 'Afghanistan', :]

df_test.tail()

X_Train.tail()
X_Train.head()
from warnings import filterwarnings

filterwarnings('ignore')



from sklearn import preprocessing



le = preprocessing.LabelEncoder()



from xgboost import XGBRegressor

import lightgbm as lgb



# Regressor parameters

n_estimators = 5000



n_poly_degree = 5



#Countries fo loop

countries = X_Train.Country.unique()





df_out_xgb = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

df_out_lgb = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

df_out_ply = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_Train.loc[X_Train.Country == country, :].State.unique()

    #print(country, states)

    # check string is nan

    for state in states:

        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        

        y_Train_CS_Cases = X_Train_CS.loc[:, 'ConfirmedCases']

        y_Train_CS_Fatal = X_Train_CS.loc[:, 'Fatalities']

        

        #X_Train_CS_Cases = X_Train_CS.loc[:, ['State', 'Country', 'Date', 'Fatalities']]

        #X_Train_CS_Fatal = X_Train_CS.loc[:, ['State', 'Country', 'Date', 'ConfirmedCases']]

        X_Train_CS = X_Train_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)

        X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])

        

        X_Pred_CS = X_Pred.loc[(X_Pred.Country == country) & (X_Pred.State == state), ['State', 'Country', 'Date', 'ForecastId']]

        

        X_Pred_CS_Id = X_Pred_CS.loc[:, 'ForecastId']

        

        X_Pred_CS = X_Pred_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Pred_CS.Country = le.fit_transform(X_Pred_CS.Country)

        X_Pred_CS['State'] = le.fit_transform(X_Pred_CS['State'])

        

        

        # XGBoost

        model_xgb_cases = XGBRegressor(n_estimators = n_estimators)

        model_xgb_cases.fit(X_Train_CS, y_Train_CS_Cases)

        y_xgb_cases_pred = model_xgb_cases.predict(X_Pred_CS)

        

        model_xgb_fatal = XGBRegressor(n_estimators = n_estimators)

        model_xgb_fatal.fit(X_Train_CS, y_Train_CS_Fatal)

        y_xgb_fatal_pred = model_xgb_fatal.predict(X_Pred_CS)

        

        # LightGBM

        model_lgb_cases = XGBRegressor(n_estimators = n_estimators)

        model_lgb_cases.fit(X_Train_CS, y_Train_CS_Cases)

        y_lgb_cases_pred = model_lgb_cases.predict(X_Pred_CS)

        

        model_lgb_fatal = XGBRegressor(n_estimators = n_estimators)

        model_lgb_fatal.fit(X_Train_CS, y_Train_CS_Fatal)

        y_lgb_fatal_pred = model_lgb_fatal.predict(X_Pred_CS)

        

        #polyfit

        pl_cases_pred = np.poly1d(np.polyfit(X_Train_CS.Date, y_Train_CS_Cases, n_poly_degree))

        y_ply_cases_pred = pl_cases_pred(X_Pred_CS.Date)

        

        pl_fatal_pred = np.poly1d(np.polyfit(X_Train_CS.Date, y_Train_CS_Fatal, n_poly_degree))

        y_ply_fatal_pred = pl_fatal_pred(X_Pred_CS.Date)

        

        df_xgb = pd.DataFrame({'ForecastId': X_Pred_CS_Id, 'ConfirmedCases': y_xgb_cases_pred, 'Fatalities': y_xgb_fatal_pred})

        df_lgb = pd.DataFrame({'ForecastId': X_Pred_CS_Id, 'ConfirmedCases': y_lgb_cases_pred, 'Fatalities': y_lgb_fatal_pred})

        df_ply = pd.DataFrame({'ForecastId': X_Pred_CS_Id, 'ConfirmedCases': y_ply_cases_pred, 'Fatalities': y_ply_fatal_pred})

        

        df_out_xgb = pd.concat([df_out_xgb, df_xgb], axis=0)

        df_out_lgb = pd.concat([df_out_lgb, df_lgb], axis=0)

        df_out_ply = pd.concat([df_out_ply, df_ply], axis=0)

        

    #for state loop

#for country Loop
df_out_xgb.ForecastId = df_out_xgb.ForecastId.astype('int')

df_out_lgb.ForecastId = df_out_lgb.ForecastId.astype('int')

df_out_ply.ForecastId = df_out_ply.ForecastId.astype('int')
df_out = df_out_ply.copy()
#df_out['ConfirmedCases'] = (1/2)*(df_out_xgb['ConfirmedCases'] + df_out_lgb['ConfirmedCases'])

#df_out['Fatalities'] = (1/2)*(df_out_xgb['Fatalities'] + df_out_lgb['Fatalities'])



#df_out['ConfirmedCases'] = df_out_ply['ConfirmedCases']

#df_out['Fatalities'] = df_out_ply['Fatalities']



df_out['ConfirmedCases'] = (1/4)*(df_out_xgb['ConfirmedCases'] + df_out_lgb['ConfirmedCases']) + (1/2) * df_out_ply['ConfirmedCases'] 

df_out['Fatalities'] = (1/4)*(df_out_xgb['Fatalities'] + df_out_lgb['Fatalities']) + (1/2) * df_out_ply['Fatalities'] 
df_out['ConfirmedCases'] = df_out['ConfirmedCases'].round().astype(int)

df_out['Fatalities'] = df_out['Fatalities'].round().astype(int)
df_out.to_csv('submission.csv', index=False)