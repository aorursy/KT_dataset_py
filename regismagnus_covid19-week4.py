import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_log_error



print('Setup complete')
covid_train=pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv', index_col='Id', parse_dates=['Date'])

covid_test=pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv', index_col='ForecastId', parse_dates=['Date'])



last_register= pd.to_datetime(covid_train["Date"].iloc[covid_train.shape[0]-1])



print("Len train %d, Len test %d"% (covid_train.shape[0], covid_test.shape[0]))

print("Last train \"Date\": ", last_register)
covid_train.info()
def adjustState(row):

    if pd.isna(row['Province_State']):

        row['Province_State'] = row['Country_Region']    

    return row

covid_train = covid_train.apply(adjustState, axis=1)

covid_test = covid_test.apply(adjustState, axis=1)



covid_train.fillna('NA', inplace=True)

covid_test.fillna('NA', inplace=True)



#define day after N cases

n_cases_cc = 50

n_cases_ft = 50



data_mark_date = pd.DataFrame(columns=['Country_Region', 'Province_State', 'Date_cc', 'Date_ft'])

data_mark_date.set_index(['Country_Region', 'Province_State'])



for country in covid_train['Country_Region'].unique():

    for state in covid_train[covid_train['Country_Region']==country]['Province_State'].unique():

        data_df = covid_train[(covid_train['Country_Region']==country) & (covid_train['Province_State']==state)]

        

        #date_cc=np.nan

        if data_df[data_df['ConfirmedCases']>=n_cases_cc].shape[0]>0:

            date_cc=data_df[data_df['ConfirmedCases']>=n_cases_cc].iloc[0]['Date']

        else:

            date_cc=last_register

        #date_ft=np.nan

        if data_df[data_df['Fatalities']>=n_cases_ft].shape[0]>0:

            date_ft=data_df[data_df['Fatalities']>=n_cases_ft].iloc[0]['Date']

        else:

            date_ft=last_register

        

        data_state = pd.DataFrame({ 'Country_Region': [country], 'Province_State': [state],

                                                  'Date_cc': [date_cc], 'Date_ft': [date_ft]})

        data_state.set_index(['Country_Region', 'Province_State'])

        data_mark_date=data_mark_date.append(data_state.iloc[0])

        

def mark_date(row):    

    data_df=data_mark_date[(data_mark_date['Country_Region']==row['Country_Region']) & (data_mark_date['Province_State']==row['Province_State'])].iloc[0]

    if not pd.isna(data_df['Date_cc']):

        row['Date_cc']=(row['Date']-data_df['Date_cc']).days

    if not pd.isna(data_df['Date_ft']):

        row['Date_ft']=(row['Date']-data_df['Date_ft']).days

    return row



#first register

covid_train = covid_train[(covid_train['ConfirmedCases']>0) | (covid_train['Fatalities']>0)]



covid_train['Date_cc'] = [0 for i in range(covid_train.shape[0])]

covid_train['Date_ft'] = [0 for i in range(covid_train.shape[0])]



covid_train = covid_train.apply(mark_date, axis=1)



covid_test['Date_cc'] = [0 for i in range(covid_test.shape[0])]

covid_test['Date_ft'] = [0 for i in range(covid_test.shape[0])]



covid_test = covid_test.apply(mark_date, axis=1)



covid_train['Date_st'] = covid_train['Date'].map(lambda x: x.timestamp())

covid_test['Date_st'] = covid_test['Date'].map(lambda x: x.timestamp())



enc = LabelEncoder()

covid_train['Province_State_enc'] = enc.fit_transform(covid_train['Province_State'])

covid_test['Province_State_enc'] = enc.transform(covid_test['Province_State'])



enc = LabelEncoder()

covid_train['Country_Region_enc'] = enc.fit_transform(covid_train['Country_Region'])

covid_test['Country_Region_enc'] = enc.transform(covid_test['Country_Region'])



X_features = ['Province_State', 'Country_Region', 'Date_st', 'Date_cc', 'Date_ft']



X = covid_train[X_features]

y_cc = covid_train['ConfirmedCases']

y_ft = covid_train['Fatalities']



print("Adjust data complete")
X_features_cc = ['Province_State_enc', 'Country_Region_enc', 'Date_st', 'Date_cc']

X_features_ft = ['Province_State_enc', 'Country_Region_enc', 'Date_st', 'Date_ft']
X_train, X_valid, y_train_cc, y_valid_cc = train_test_split(X, y_cc, random_state=42)

y_train_ft = y_ft[y_train_cc.index]

y_valid_ft = y_ft[y_valid_cc.index]
def preds_country(X_fit, X_pred, y_fit, n_estimators=5, learning_rate=0.1, n_jobs=4, predict_type='XGBR'):

    '''predict each by country'''

    X_pred_copy=X_pred.copy()#.drop(['Province_State', 'Country_Region', 'Date_st', 'Date_cc', 'Date_ft'], axis=1)

    X_pred_copy['preds']=[np.nan for i in range(X_pred_copy.shape[0])]   

    for country in X_fit['Country_Region'].unique():

        if country in X_pred['Country_Region'].unique():

            for state in X_fit[X_fit['Country_Region']==country]['Province_State'].unique():

                if state in X_pred[X_pred['Country_Region']==country]['Province_State'].unique():

                    X_fit_country = X_fit[(X_fit['Country_Region']==country) & (X_fit['Province_State']==state)].copy().drop(['Country_Region', 'Province_State'], axis=1)

                    X_pred_country = X_pred[(X_pred['Country_Region']==country) & (X_pred['Province_State']==state)].copy().drop(['Country_Region', 'Province_State'], axis=1)

                    y_fit_country = y_fit[X_fit_country.index]



                    if predict_type=='XGBR':

                        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs, random_state=42)

                    elif predict_type=='RFR':

                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                    else:

                        raise NameError('Model not valid')

                    

                    model.fit(X_fit_country, y_fit_country)

                    X_pred_country['preds'] = model.predict(X_pred_country)



                    for col in X_pred_country.index:

                        if col in X_pred_copy.index.values:

                            X_pred_copy.loc[col]=X_pred_country.loc[col]['preds']



    return X_pred_copy
#preds_cc = preds_country(X_train[['Province_State', 'Country_Region', 'Date_st', 'Date_cc']], X_valid[['Province_State', 'Country_Region', 'Date_st', 'Date_cc']], y_train_cc, n_estimators=100, predict_type='RFR')

#preds_ft = preds_country(X_train[['Province_State', 'Country_Region', 'Date_st', 'Date_ft']], X_valid[['Province_State', 'Country_Region', 'Date_st', 'Date_ft']], y_train_ft, n_estimators=50, predict_type='RFR')
'''mae = mean_absolute_error(y_valid_cc, preds_cc['preds'])

msle = mean_squared_log_error(y_valid_cc, preds_cc['preds'])

print("CC MAE: %f MSLE %f" % (mae, msle))



mae = mean_absolute_error(y_valid_ft, preds_ft['preds'])

msle = mean_squared_log_error(y_valid_ft, preds_ft['preds'])

print("FT MAE: %f MSLE %f" % (mae, msle))'''



'''

CC MAE: 53.621829 MSLE 0.032919

FT MAE: 3.685815 MSLE 0.008674

'''
'''X_train_2, X_valid_2, y_train_cc_2, y_valid_cc_2 = train_test_split(covid_train[X_features_cc], y_cc, random_state=42)



model_cc_2 = RandomForestRegressor(n_estimators=100, random_state=42)

model_cc_2.fit(X_train_2, y_train_cc_2)



predic = model_cc_2.predict(X_valid_2)



mae = mean_absolute_error(y_valid_cc_2, predic)

msle = mean_squared_log_error(y_valid_cc_2, predic)

print("CC MAE: %f MSLE %f" % (mae, msle))



X_train_2, X_valid_2, y_train_ft_2, y_valid_ft_2 = train_test_split(covid_train[X_features_ft], y_ft, random_state=42)



model_ft_2 = RandomForestRegressor(n_estimators=50, random_state=42)

model_ft_2.fit(X_train_2, y_train_ft_2)



predic = model_ft_2.predict(X_valid_2)



mae = mean_absolute_error(y_valid_ft_2, predic)

msle = mean_squared_log_error(y_valid_ft_2, predic)

print("CC MAE: %f MSLE %f" % (mae, msle))'''



'''

CC MAE: 217.231728 MSLE 0.881407

CC MAE: 14.093100 MSLE 0.503043

'''
'''model_cc_2 = RandomForestRegressor(n_estimators=100, random_state=42)

model_cc_2.fit(covid_train[X_features_cc], covid_train['ConfirmedCases'])



test_preds_cc = model_cc_2.predict(covid_test[X_features_cc]).round()



model_ft_2 = RandomForestRegressor(n_estimators=50, random_state=42)

model_ft_2.fit(covid_train[X_features_ft], covid_train['Fatalities'])



test_preds_ft = model_ft_2.predict(covid_test[X_features_ft]).round()'''
test_preds_cc = preds_country(covid_train[['Province_State', 'Country_Region', 'Date_st', 'Date_cc']], covid_test[['Province_State', 'Country_Region', 'Date_st', 'Date_cc']], covid_train['ConfirmedCases'], n_estimators=100, predict_type='RFR')

test_preds_ft = preds_country(covid_train[['Province_State', 'Country_Region', 'Date_st', 'Date_ft']], covid_test[['Province_State', 'Country_Region', 'Date_st', 'Date_ft']], covid_train['Fatalities'], n_estimators=50, predict_type='RFR')
submission = pd.DataFrame({'ForecastId': test_preds_cc.index,'ConfirmedCases':test_preds_cc['preds'].round(),'Fatalities':test_preds_ft['preds'].round()})

#submission = pd.DataFrame({'ForecastId': covid_test.index,'ConfirmedCases':test_preds_cc,'Fatalities':test_preds_ft})

filename = 'submission.csv'



submission.to_csv(filename,index=False)