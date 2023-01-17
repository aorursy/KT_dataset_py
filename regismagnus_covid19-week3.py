import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_log_error



print('Setup complete')
covid_train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv', index_col='Id')

covid_test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv', index_col='ForecastId')



last_register= pd.to_datetime(covid_train["Date"].iloc[covid_train.shape[0]-1])



print("Len train %d, Len test %d"% (covid_train.shape[0], covid_test.shape[0]))

print("Last train \"Date\": ", last_register)
covid_train.columns
covid_train['Date'] = pd.to_datetime(covid_train['Date'])

covid_test['Date'] = pd.to_datetime(covid_test['Date'])

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

            

covid_train['Date_cc'] = [0 for i in range(covid_train.shape[0])]

covid_train['Date_ft'] = [0 for i in range(covid_train.shape[0])]



covid_train = covid_train.apply(mark_date, axis=1)



covid_test['Date_cc'] = [0 for i in range(covid_test.shape[0])]

covid_test['Date_ft'] = [0 for i in range(covid_test.shape[0])]



covid_test = covid_test.apply(mark_date, axis=1)



covid_train['Date'] = covid_train['Date'].astype(int)

covid_test['Date'] = covid_test['Date'].astype(int)



X_features = ['Province_State', 'Country_Region', 'Date', 'Date_cc', 'Date_ft']

X_features_2 = ['Date']

X = covid_train[X_features]

y_cc = covid_train['ConfirmedCases']

y_ft = covid_train['Fatalities']



X_train, X_valid, y_train_cc, y_valid_cc = train_test_split(X, y_cc, random_state=42)

y_train_ft = y_ft[y_train_cc.index]

y_valid_ft = y_ft[y_valid_cc.index]



print("Adjust data complete")
covid_train[covid_train['Fatalities']>0]
cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor_1 = ColumnTransformer(

    transformers=[

        ('cat', cat_transformer, ['Country_Region', 'Province_State'])

    ])



preprocessor_2 = ColumnTransformer(

    transformers=[

        ('cat', cat_transformer, ['Province_State', 'Date'])

    ])



def preds_RFR(X_fit, X_pred, y_fit, n_estimators=5, preprocessor=preprocessor_1):

    '''predict using RandomForestRegressor'''

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                    ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=42))])

    model_pipeline.fit(X_fit, y_fit)

    

    return model_pipeline.predict(X_pred)



def score_RFR(X_fit, X_pred, y_fit, y_valid, n_estimators=5):

    '''Score using RanomForestRegressor'''

    preds = preds_RFR(X_fit, X_pred, y_fit, n_estimators)

    

    mae = mean_absolute_error(y_valid, preds)

    msle = mean_squared_log_error(y_valid, preds)

    print("(%s, %d) MAE: %f MSLE %f" % (y_valid.name, n_estimators, mae, msle))

    

def preds_XGBR(X_fit, X_pred, y_fit, n_estimators=5, learning_rate=0.1, n_jobs=4, preprocessor=preprocessor_1):

    '''predict using XGBRegressor'''

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                    ('model', XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs, random_state=42))])

        

    model_pipeline.fit(X_fit, y_fit)

    

    return model_pipeline.predict(X_pred)



def preds_XGBR_country(X_fit, X_pred, y_fit, n_estimators=5, learning_rate=0.1, n_jobs=4, predict_type='XGBR'):

    '''predict each by country'''

    X_pred_copy=X_pred.copy().drop(['Province_State', 'Country_Region', 'Date', 'Date_cc', 'Date_ft'], axis=1)

    X_pred_copy['preds']=[np.nan for i in range(X_pred_copy.shape[0])]   

    for country in X_fit['Country_Region'].unique():

        if country in X_pred['Country_Region'].unique():

            X_fit_country = X_fit[X_fit['Country_Region']==country].copy().drop(['Country_Region'], axis=1)

            X_pred_country = X_pred[X_pred['Country_Region']==country].copy().drop(['Country_Region'], axis=1)

            y_fit_country = y_fit[X_fit_country.index]

                        

            if predict_type=='XGBR':

                X_pred_country['preds'] = (preds_XGBR(X_fit_country, X_pred_country, y_fit_country, n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs, preprocessor=preprocessor_2))

            elif predict_type=='RFR':

                X_pred_country['preds'] = (preds_RFR(X_fit_country, X_pred_country, y_fit_country, n_estimators=n_estimators, preprocessor=preprocessor_2))

            else:

                raise NameError('Model not valid')

            

            for col in X_pred_country.index:

                if col in X_pred_copy.index.values:

                    X_pred_copy.loc[col]=X_pred_country.loc[col]['preds']

            

    return X_pred_copy

                

def score_XGBR(X_fit, X_pred, y_fit, y_valid, n_estimators=5, learning_rate=0.1, n_jobs=4):

    '''Score using XGBRegressor'''

    preds = preds_XGBR(X_fit, X_pred, y_fit, n_estimators, learning_rate, n_jobs)

    

    mae = mean_absolute_error(y_valid, preds)

    msle = mean_squared_log_error(y_valid, preds)

    print("(%s, %d) MAE: %f MSLE %f" % (y_valid.name, n_estimators, mae, msle))

    

def preds_country(X_fit, X_pred, y_fit, n_estimators=5, learning_rate=0.1, n_jobs=4, predict_type='XGBR'):

    '''predict each by country'''

    X_pred_copy=X_pred.copy().drop(['Province_State', 'Country_Region', 'Date', 'Date_cc', 'Date_ft'], axis=1)

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

    

def compare_preds(y_valid, preds):

    plt.figure(figsize=(14,6))

    plt.title("Compare predict of " + y_valid.name)

    sns.lineplot(y=y_valid, x=X_valid['Date'], label='y')

    sns.lineplot(y=preds, x=X_valid['Date'], label='Predict')

    plt.legend()
preds_cc = preds_country(X_train, X_valid, y_train_cc, n_estimators=100, predict_type='RFR')

preds_ft = preds_country(X_train, X_valid, y_train_ft, n_estimators=50, predict_type='RFR')
'''mae = mean_absolute_error(y_valid_cc, preds_cc)

msle = mean_squared_log_error(y_valid_cc, preds_cc)

print("CC MAE: %f MSLE %f" % (mae, msle))



mae = mean_absolute_error(y_valid_ft, preds_ft)

msle = mean_squared_log_error(y_valid_ft, preds_ft)

print("FT MAE: %f MSLE %f" % (mae, msle))'''



'''

CC MAE: 38.041236 MSLE 0.036174

FT MAE: 2.325117 MSLE 0.009207



CC MAE: 38.079204 MSLE 0.035471

FT MAE: 2.324699 MSLE 0.009112

'''
#ConfirmedCases 100 is enought MAE: 698 MSLE 9

#preds_cc = preds_RFR(X_train, X_valid, y_train_cc, n_estimators=100)
#Fatalities 50 is enought MAE: 36 MSLE 1

#preds_ft = preds_RFR(X_train, X_valid, y_train_ft, n_estimators=50)
#compare_preds(y_valid_cc, preds_cc)
#compare_preds(y_valid_ft, preds_ft)
#ComfirmedCases n_estimators=1200 | MAE: 698 MSLE 9

#preds_cc = preds_XGBR(X_train, X_valid, y_train_cc, n_estimators=1200)
#Fatalities n_estimators=150 |MAE: 37 MSLE 1

#preds_ft = preds_XGBR(X_train, X_valid, y_train_ft, n_estimators=150)
#compare_preds(y_valid_cc, preds_cc)
#compare_preds(y_valid_ft, preds_ft)
#preds_cc = preds_XGBR_country(X_train, X_valid, y_train_cc, n_estimators=1200)
#preds_ft = preds_XGBR_country(X_train, X_valid, y_train_ft, n_estimators=150)
#compare_preds(y_valid_cc, preds_cc['preds'])
#compare_preds(y_valid_ft, preds_ft['preds'])
#predict tests Score 1.38

#test_preds_cc = preds_RFR(X_train, covid_test[X_features], y_train_cc, n_estimators=100)

#test_preds_ft = preds_RFR(X_train, covid_test[X_features], y_train_ft, n_estimators=50)
#predict tests Score 1.39

#test_preds_cc = preds_XGBR(X_train, covid_test[X_features], y_train_cc, n_estimators=1200)

#test_preds_ft = preds_XGBR(X_train, covid_test[X_features], y_train_ft, n_estimators=150)
test_preds_cc = preds_country(covid_train[X_features], covid_test[X_features], covid_train['ConfirmedCases'], n_estimators=100, predict_type='RFR')

test_preds_ft = preds_country(covid_train[X_features], covid_test[X_features], covid_train['Fatalities'], n_estimators=50, predict_type='RFR')
submission = pd.DataFrame({'ForecastId': test_preds_cc.index,'ConfirmedCases':test_preds_cc['preds'],'Fatalities':test_preds_ft['preds']})

filename = 'submission.csv'



submission.to_csv(filename,index=False)