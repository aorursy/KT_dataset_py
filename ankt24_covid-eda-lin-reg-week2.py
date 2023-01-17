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
import pandas as pd



import numpy as np



import matplotlib.pyplot as plt



import seaborn as sns



from datetime import datetime



from sklearn.preprocessing import LabelEncoder



from sklearn.linear_model import LinearRegression



%matplotlib inline
PATH_WEEK2='/kaggle/input/covid19-global-forecasting-week-2'



train_df = pd.read_csv(f'{PATH_WEEK2}/train.csv')

test_df = pd.read_csv(f'{PATH_WEEK2}/test.csv')
test_df.head()
train_df.head()
train_df.describe()
test_df.describe()
train_df.shape
test_df.shape
train_df.info()
test_df.info()
def add_lag_trend(df, lag_list, col_name):

    for lag in lag_list:

        column_lag = col_name+ "_lag" + str(lag)

        col_trend = col_name+ "_trend" + str(lag)

        df[column_lag] = df[col_name].shift(lag, fill_value=0)

        df[col_trend] = (df[col_name] - df[column_lag])/df[column_lag]

    

#     df.fillna(0).round(3)

    

    return df
def process_state_date(df):

    df.loc[df.Province_State.isnull(),'Province_State'] = df.loc[df.Province_State.isnull(), 'Country_Region']

    df.Date = df.Date.apply(pd.to_datetime)

    df['day_of_year'] = df.Date.apply(lambda x: x.dayofyear)

    df.rename({"Country_Region":"country", "Province_State":"state"}, axis=1, inplace=True)

    return df
train_df2 = process_state_date(train_df)



test_df2 = process_state_date(test_df)
tgrp_df = train_df2.groupby("country")



tsgrp_df = test_df2.groupby("country")
def apply_log_tform(df, col_name):

    df[col_name] = df[col_name].apply(np.log)



    lag_df = df.filter(like='lag').apply(np.log)



    df2 = pd.concat([df.drop(lag_df.columns, axis=1), lag_df], axis=1)

    

    return df2
country_lockdown_dict = {

                    'Argentina' : datetime(2020,3,19), # 2020-03-19

                    'Australia' : datetime(2020,3,23), # 2020-03-23

                    'Austria' : datetime(2020,3,16), # 2020-03-16

                    'Belgium' : datetime(2020,3,18), # 2020-03-18

                    'Colombia' : datetime(2020,3,25), # 2020-03-25

                    'Czechia' : datetime(2020,3,16), # 2020-03-16

                    'Denmark' : datetime(2020,3,11), # 2020-03-11

                    'El Salvador' : datetime(2020,3,12), # 2020-03-12

                    'Fiji' : datetime(2020,3,9), # 2020-03-20

                    'France' : datetime(2020,3,17), # 2020-03-17

                    'Greece' : datetime(2020,3,23), # 2020-03-23

                    'Honduras' : datetime(2020,3,17), # 2020-03-17

                    'India': datetime(2020,3,22), # 2020-03-22

                    'Ireland' : datetime(2020,3,12), # 2020-03-12

                    'Italy' : datetime(2020,3,9), # 2020-03-09

                    'Lebanon' : datetime(2020,3,15), # 2020-03-15

                    'Lithuania' : datetime(2020,3,16), # 2020-03-16

                    'Malaysia' : datetime(2020,3,18), # 2020-03-18

                    'Morocco' : datetime(2020,3,19), # 2020-03-19

                    'Philippines' : datetime(2020,3,15), # 2020-03-15

                    'Poland' : datetime(2020,3,13), # 2020-03-13

                    'Romania' : datetime(2020,3,25), # 2020-03-25

                    'South Africa' : datetime(2020,3,26), # 2020-03-26

                    'Spain' : datetime(2020,3,14), # 2020-03-14

                    'Tunisia' : datetime(2020,3,22), # 2020-03-22

                    'United Kingdom' : datetime(2020,3,23), # 2020-03-23

                    'Venezuela' : datetime(2020,3,17), # 2020-03-17} 

                    }
def country_wise_ops(grpd_df):

    all_tdata = pd.DataFrame()

    for cntry,df in grpd_df:

    #     df.loc[df.index == df.loc[df.ConfirmedCases == 0].index[-1]]

        lockdown_date = country_lockdown_dict.get(cntry)

        if lockdown_date is not None:

            df.loc[df.Date >= lockdown_date, 'lockdown'] = 1

        df = add_lag_trend(df, range(1,8), 'ConfirmedCases')

        df = apply_log_tform(df, 'ConfirmedCases')

        df = add_lag_trend(df, range(1,8), 'Fatalities')

        df = apply_log_tform(df, 'Fatalities')

        all_tdata = all_tdata.append(df)

    

    all_tdata.replace([np.inf, -np.inf], 0, inplace=True)



    all_tdata = all_tdata.fillna(0).round(3)

    

    return all_tdata

#     countrydf_dict[cntry] = df
all_tdata = country_wise_ops(tgrp_df)



all_tdata2 = all_tdata.copy()
le_cntry = LabelEncoder()



le_state = LabelEncoder()
all_tdata['country'] = le_cntry.fit_transform(all_tdata['country'])

number_c = all_tdata['country']

countries = le_cntry.inverse_transform(all_tdata['country'])

country_dict = dict(zip(countries, number_c))
all_tdata['state'] = le_state.fit_transform(all_tdata['state'])

number_p = all_tdata['state']

province = le_state.inverse_transform(all_tdata['state'])

province_dict = dict(zip(province, number_p)) 
all_tdata.head()
all_tsdf = pd.DataFrame()



for cntry, df in tsgrp_df:

    lockdown_date = country_lockdown_dict.get(cntry)

    if lockdown_date is not None:

        df.loc[df.Date >= lockdown_date, 'lockdown'] = 1

    all_tsdf= all_tsdf.append(df)



all_tsdf = all_tsdf.fillna(0).round(3)
test_df3 = all_tsdf.copy()



# all_tsdf = test_df3.copy()
all_tsdf['country'] = le_cntry.transform(all_tsdf['country'])



all_tsdf['state'] = le_state.transform(all_tsdf['state'])
def get_train_test_data(train, test):

    x_train = train[['country','day_of_year','lockdown','state']]



    y_train1 = train[['ConfirmedCases']]



    y_train2 = train[['Fatalities']]

    

    x_test = test.drop(['ForecastId', 'Date'], axis=1)

    

    return x_train, y_train1, y_train2, x_test
# Linear regression model

def lin_reg(X_train, Y_train, X_test):

    # Create linear regression object

    regr = LinearRegression()



    # Train the model using the training sets

    regr.fit(X_train, Y_train)



    # Make predictions using the testing set

    y_pred = regr.predict(X_test)

    

    return regr, y_pred
def get_submission(df, target1, target2):

    

    prediction_1 = df[target1]

    prediction_2 = df[target2]



    # Submit predictions

    prediction_1 = [int(item) for item in list(map(round, prediction_1))]

    prediction_2 = [int(item) for item in list(map(round, prediction_2))]

    

    submission = pd.DataFrame({

        "ForecastId": df['ForecastId'].astype('int32'), 

        "ConfirmedCases": prediction_1, 

        "Fatalities": prediction_2

    })

    

    return submission

#     submission.to_csv('submission.csv', index=False)
# Set the dataframe where we will update the predictions

data_pred = all_tsdf[['country', 'state', 'day_of_year', 'ForecastId', 'lockdown']].copy()



# data_pred = data_pred.loc[data_pred['day_of_year']>=92]

data_pred['Predicted_ConfirmedCases'] = [0]*len(data_pred)

data_pred['Predicted_Fatalities'] = [0]*len(data_pred)

    

print("Currently running Logistic Regression for all countries")



# Main loop for countries

for cntry in all_tdata.country.unique():

    

    # List of provinces

    provinces_list = all_tdata.loc[all_tdata.country==cntry, 'state'].unique()

        

    # If the country has several Province/State informed

    if len(provinces_list)>1:

        for p in provinces_list:

            tdata_cp = all_tdata[(all_tdata['country']==cntry) & (all_tdata['state']==p)]

            tsdata_cp = all_tsdf[(all_tsdf['country']==cntry) & (all_tsdf['state']==p)]

            X_train, Y_train_1, Y_train_2, X_test = get_train_test_data(tdata_cp, tsdata_cp)

            model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

            model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

            data_pred.loc[((data_pred['country']==cntry) & (data_pred['state']==p)), 'Predicted_ConfirmedCases'] = pred_1

            data_pred.loc[((data_pred['country']==cntry) & (data_pred['state']==p)), 'Predicted_Fatalities'] = pred_2

  # No Province/State informed

    else:

        tdata_c = all_tdata[(all_tdata['country']==cntry)]

        tsdata_c = all_tsdf[(all_tsdf['country']==cntry)]

        X_train, Y_train_1, Y_train_2, X_test = get_train_test_data(tdata_c, tsdata_c)

        model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

        model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

        data_pred.loc[(data_pred['country']==cntry), 'Predicted_ConfirmedCases'] = pred_1

        data_pred.loc[(data_pred['country']==cntry), 'Predicted_Fatalities'] = pred_2
# Aplly exponential transf. and clean potential infinites due to final numerical precision

data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.exp(x))



data_pred.replace([np.inf, -np.inf], 0, inplace=True) 
# data_pred2 = data_pred.loc[data_pred.day_of_year >= 92]
submission_df = get_submission(data_pred, 'Predicted_ConfirmedCases', 'Predicted_Fatalities')
submission_df.to_csv('submission.csv', index=False)