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
! pip install pmdarima
import  math

from pmdarima.arima import auto_arima

import seaborn as sns

from datetime import datetime,date,timedelta

train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

submission_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")

print(train_data.shape)

print(test_data.shape)

print(submission_data.shape)
print(train_data.shape)

print(test_data.shape)

print(train_data.info())

print(test_data.info())

print(train_data.isna().sum())

print(test_data.isna().sum())

print(train_data.describe())

print(train_data.head(5))

print(test_data.head(5))
#### exclude null values

train_data.count()
print(train_data.columns)

print(test_data.columns)
train_data.Date = pd.to_datetime(train_data.Date)

test_data.Date = pd.to_datetime(test_data.Date)
print(train_data.info())

print(test_data.info())
print("train data range from ",min(train_data.Date)," to",max(train_data.Date))

print("test data range from ",min(test_data.Date)," to",max(test_data.Date))
print(test_data.head(5))
### train and test data manipulation



### train data manipulation



train_data.Date = pd.to_datetime(train_data.Date)

train_data['Province_State'] = train_data['Province_State'].astype(str)

train_data['Province_State'] = train_data['Province_State'].str.replace(',','_').str.replace('nan','').str.lower()

train_data['Country_Region'] = train_data['Country_Region'].astype(str).str.lower().str.replace(' ','_')

train_data['province_state_country'] = train_data['Country_Region'].astype(str).str.lower().str.replace(' ','_') + '_' + train_data['Province_State']

### change '_' to empty space'

train_data['province_state_country'] = train_data['province_state_country'].str.replace('_$','')





### train data manipulation

test_data.Date = pd.to_datetime(test_data.Date)

test_data['Province_State'] = test_data['Province_State'].astype(str)

test_data['Province_State'] = test_data['Province_State'].str.replace(',','_').str.replace('nan','').str.lower()

test_data['Country_Region'] = test_data['Country_Region'].astype(str).str.lower().str.replace(' ','_')

test_data['province_state_country'] = test_data['Country_Region'].astype(str).str.lower().str.replace(' ','_') + '_' + test_data['Province_State']

### change '_' to empty space'

test_data['province_state_country'] = test_data['province_state_country'].str.replace('_$','')

### sort the data

train_data = train_data.sort_values(['province_state_country','Date'])

test_data = test_data.sort_values(['province_state_country','Date'])
train_data_start_date = min(train_data.Date)

train_data_end_date = max(train_data.Date)

test_data_start_date = min(test_data.Date)

test_data_end_Date = max(test_data.Date)

# public_test_data_start_date = pd.datetime(2020,4,1)



public_test_data_start_date = test_data_start_date

public_test_data_end_date = pd.datetime(2020,4,15)

private_test_data_start_date = pd.datetime(2020,4,16)

private_test_data_end_date = test_data_end_Date

#### dates for filtering the train and test datasets

print("train_data_start_date",train_data_start_date)

print("train_data_end_date",train_data_end_date)

print("test_data_start_date",test_data_start_date)

print("test_data_end_Date",test_data_end_Date)

print("public_test_data_start_date",public_test_data_start_date)

print("public_test_data_end_date",public_test_data_end_date)

print("private_test_data_start_date",private_test_data_start_date)

print("private_test_data_end_date",private_test_data_end_date)
print(test_data.columns)
### function for creating feature

# feature_day = [1,20,50,100,200,500,1000,2000,5000,10000]



feature_day = [1]

def create_feature(data):

    feature_list = []

    for day in feature_day:

        data.loc[:,'Number day from ' + str(day) + ' case'] = 0

        if(data[data['ConfirmedCases'] < day]['Date'].count() > 0):

            

            fromday = data[data['ConfirmedCases'] < day]['Date'].max()

            print(fromday)

            

        elif(data[data['ConfirmedCases'] >= day]['Date'].count() > 0):

            max_date = data[data['ConfirmedCases'] >= day]['Date'].min()

            fromday = max_date- timedelta(days =1)

            print("fromday",fromday)

        

        else:

            fromday = data[data['ConfirmedCases'] < day]['Date'].min()

            

        for i in range(0,len(data)):

            if(data['Date'].iloc[i] > fromday):

                day_number = data['Date'].iloc[i] - fromday

                data['Number day from ' + str(day) + ' case'].iloc[i] = day_number.days

        feature_list = feature_list + ['Number day from ' + str(day) + ' case']

    

    return data[feature_list]

            
province_state_country_list = pd.concat([train_data['province_state_country'], test_data['province_state_country']]).unique()

province_state_country_list
province_state_country_list = pd.concat([train_data['province_state_country'], test_data['province_state_country']]).unique()



# province_state_country_list = np.asarray(['france_saint pierre and miquelon'])



#        'france_new caledonia', 'france_reunion',

#        'france_saint barthelemy', 'france_saint pierre and miquelon',

#        'france_st martin','gabon','france_mayotte','france_martinique','france_guadeloupe','china_anhui','china_beijing','us_california','us_new york'])



submission_df = pd.DataFrame(columns= ['ForecastId', 'Province_State', 'Country_Region', 'Date', 'province_state_country','days','Confirmed_pred','Fatalities_pred'])



validation_df = pd.DataFrame(columns= ['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases',

       'Fatalities', 'province_state_country','days','Confirmed_pred','Fatalities_pred'])



for s in province_state_country_list:

    

    print("province_country_region",s)

    ### train and test data for that country

    train_df = train_data.loc[(train_data['province_state_country'] == s)]

    test_df =  test_data.loc[(test_data['province_state_country'] == s)]

    train_df_2 = train_data.loc[(train_data['province_state_country'] == s)]

    

    

    train_df = train_df.sort_values(['province_state_country','Date'])

    test_df = test_df.sort_values(['province_state_country','Date'])

#     

    print(train_df.head(5))

    print("train_df_2",train_df_2.head(5))

    print(test_df.head(5))

    

    train_feature = create_feature(train_df_2)

    print("train_df",train_df.head(5))

    print("train_feature",train_feature.head(5))

    

    adjusted_train_feature_min_index = min(train_df.index)

    

    for day in sorted(feature_day, reverse = True):

        print("day",day)

        feature_use = 'Number day from ' + str(day) + ' case'

        print("feature_use",feature_use)

        train_feature_zero_count = train_feature[train_feature[feature_use] == 0].shape[0]

        train_feature_nonzero_count = train_feature[train_feature[feature_use] > 0].shape[0]

        print("train_feature_zero_count",train_feature_zero_count)

        print("train_feature_nonzero_count",train_feature_nonzero_count)

        

        adjusted_train_feature_min_index = min(train_feature[train_feature[feature_use]>0].index)

        

#         if(train_feature_nonzero_count >= 30):

#             adjusted_train_feature_min_index = min(train_feature[train_feature[feature_use]>0].index)

#             adjusted_train_feature_max_index = max(train_feature[train_feature[feature_use]>0].index)

#             print("adjusted_train_feature_min_index",adjusted_train_feature_min_index)

#             print("adjusted_train_feature_max_index",adjusted_train_feature_max_index)

#             break

            

           

    adjusted_train_data_1 = train_df.loc[train_df.index >= adjusted_train_feature_min_index]

    adjusted_train_data_start_date = adjusted_train_data_1['Date'].min()

    

#     adjusted_train_data_2 = adjusted_train_data_1.loc[(adjusted_train_data_1['Date'] < public_test_data_start_date)]

    

    if((adjusted_train_data_start_date < public_test_data_start_date) & (adjusted_train_data_1.shape[0] >= 20)) :

        adjusted_train_data = adjusted_train_data_1

        adjusted_train_data_start_date = adjusted_train_data_1['Date'].min()     

    else:

        adjusted_train_data = train_df

        adjusted_train_data_start_date = train_df['Date'].min()

    

    if((adjusted_train_data_1.shape[0] >= 13)) :

        adjusted_private_train_data = adjusted_train_data_1

        adjusted_private_train_data_start_date = adjusted_train_data_1['Date'].min()     

    else:

        adjusted_private_train_data = train_df

        adjusted_private_train_data_start_date = train_df['Date'].min()

    

        

    

    print(adjusted_train_data)



    #     adjusted_train_feature = train_feature[train_feature_zero_count:][feature_use].values.reshape(-1,1)

#     print("adjusted_train_feature",adjusted_train_feature)



    ### training data for public board

    train_data_1 = adjusted_train_data.loc[(adjusted_train_data['Date'] < public_test_data_start_date)]

    train_data_1['days'] = (train_data_1.Date - adjusted_train_data_start_date).dt.days

    train_data_1.index = pd.to_datetime(train_data_1.Date)

    train_data_exog_1 = np.array(train_data_1[['days']])

    print("train_data_1",train_data_1.head(5))



    #### Validation data

    train_data_validation_1 = adjusted_train_data.loc[(adjusted_train_data['Date'] >= public_test_data_start_date)]

    train_data_validation_1['days'] = (train_data_validation_1.Date - adjusted_train_data_start_date).dt.days

    train_data_validation_exog_1 = np.array(train_data_validation_1[['days']])

    print("train_data_validation_1",train_data_validation_1.head(5))

    

    ###3 test data for public data

    ### test data starts from '2020 -04-02' and public data starts from '2020-04-01'

    test_data_1 = test_data.loc[(test_data['province_state_country'] == s) & (test_data['Date'] <= public_test_data_end_date)]

    test_data_1['days'] = (test_data_1.Date - adjusted_train_data_start_date).dt.days

    test_data_exog_1 = np.array(test_data_1[['days']])

    print("test_data_1", test_data_1.head(5))

    

    #### train data for private board

    

    train_data_2 = adjusted_private_train_data.loc[(adjusted_private_train_data['Date'] <= train_data_end_date )]

    train_data_2['days'] = (train_data_2.Date - adjusted_private_train_data_start_date).dt.days

    train_data_2.index = pd.to_datetime(train_data_2.Date)

    train_data_exog_2 = np.array(train_data_2[['days']])

    print("train_data_2",train_data_2.head(5))

    

    ###3 test data for private data

    ### Private board data from '2020-04-16 to '2020-05-14'

    test_data_2 = test_data.loc[(test_data['province_state_country'] == s) & (test_data['Date'] >= private_test_data_start_date)]

    

    test_data_2['days'] = (test_data_2.Date - adjusted_private_train_data_start_date).dt.days

    test_data_exog_2 = np.array(test_data_2[['days']])

    print("test_data_2", test_data_2.head(4))

    

    ### for public board

    ### for confirmed cases

    trained_Model_Confirmed_1 = auto_arima(train_data_1['ConfirmedCases'], exogenous = train_data_exog_1 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

    print("trained_Model_Confirmed_1",trained_Model_Confirmed_1)

    

    prediction_confirmed_1 = trained_Model_Confirmed_1.predict(exogenous = train_data_validation_exog_1 , n_periods =train_data_validation_1['days'].shape[0])

    

    print("prediction_confirmed_1",prediction_confirmed_1)

    

    prediction_confirmed_public_1 = trained_Model_Confirmed_1.predict(exogenous = test_data_exog_1 , n_periods =test_data_1['days'].shape[0])

    print("prediction_confirmed_public_1",prediction_confirmed_public_1)



    ### for Fatalities

    trained_Model_Fatalities_1 = auto_arima(train_data_1['Fatalities'], exogenous = train_data_exog_1 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

    print("trained_Model_Fatalities_1",trained_Model_Fatalities_1)

    

    prediction_Fatalities_1 = trained_Model_Fatalities_1.predict(exogenous = train_data_validation_exog_1 , n_periods =train_data_validation_1['days'].shape[0])

    print("prediction_Fatalities_1",prediction_Fatalities_1)

    

    prediction_Fatalities_public_1 = trained_Model_Fatalities_1.predict(exogenous = test_data_exog_1 , n_periods =test_data_1['days'].shape[0])

    

    print("prediction_Fatalities_public_1",prediction_Fatalities_public_1)



    ### for private board

    ### for confirmed cases

    trained_Model_Confirmed_2 = auto_arima(train_data_2['ConfirmedCases'], exogenous = train_data_exog_2 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

    print("trained_Model_Confirmed_2",trained_Model_Confirmed_2)

    

    prediction_confirmed_private_2 = trained_Model_Confirmed_2.predict(exogenous = test_data_exog_2 , n_periods =test_data_2['days'].shape[0])

    

    print("prediction_confirmed_public_2",prediction_confirmed_private_2)



    ### for Fatalities

    trained_Model_Fatalities_2 = auto_arima(train_data_2['Fatalities'], exogenous = train_data_exog_2 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

    print("trained_Model_Fatalities_2",trained_Model_Fatalities_2)

    

    prediction_Fatalities_private_2 = trained_Model_Fatalities_2.predict(exogenous = test_data_exog_2 , n_periods =test_data_2['days'].shape[0])

    

    print("prediction_Fatalities_private_2",prediction_Fatalities_private_2)

    

    ##### public data prediction from 2020-04-01 to 2020-04-15'

    ### test data starts from '2020-04-02'

    #### Validation data start from '2020-04-01  to maimum date of training dataset'

    ### public data prediction data frame

    public_data_pred = test_data_1

    public_data_pred['Confirmed_pred'] = prediction_confirmed_public_1

    public_data_pred['Fatalities_pred'] = prediction_Fatalities_public_1

    

    ### Validation data

    validation_data = train_data_validation_1

    validation_data['Confirmed_pred'] = prediction_confirmed_1

    validation_data['Fatalities_pred'] = prediction_Fatalities_1

    

    

    #### Private dataset prediction from 2020-04-15 to 2020-05-14

    ### test data have data from '2020-04-02' to '2020-05-14'

    private_data_pred = test_data_2

    private_data_pred['Confirmed_pred'] = prediction_confirmed_private_2

    private_data_pred['Fatalities_pred'] = prediction_Fatalities_private_2

    

    ### Private data prediction filter

    private_data_pred_till_date = private_data_pred.loc[private_data_pred.Date >= private_test_data_start_date]

    

    Evaluation_df = public_data_pred

    Evaluation_df = Evaluation_df.append(private_data_pred_till_date , ignore_index = True)

    

    submission_df = submission_df.append(Evaluation_df , ignore_index = True)

    

    validation_df = validation_df.append(validation_data , ignore_index = True)

    





# province_state_country_list = pd.concat([train_data['province_state_country'], test_data['province_state_country']]).unique()

# # province_state_country_list = np.asarray(['us_california','us_new york'])

# submission_df = pd.DataFrame(columns= ['ForecastId', 'Province_State', 'Country_Region', 'Date', 'province_state_country','days','Confirmed_pred','Fatalities_pred'])



# validation_df = pd.DataFrame(columns= ['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases',

#        'Fatalities', 'province_state_country','days','Confirmed_pred','Fatalities_pred'])



# for s in province_state_country_list:

    

#     print("province_country_region",s)

#     ### train and test data for that country

#     train_df = train_data.loc[(train_data['province_state_country'] == s)]

#     test_df =  test_data.loc[(test_data['province_state_country'] == s)]

#     train_df_2 = train_data.loc[(train_data['province_state_country'] == s)]

    

    

#     train_df = train_df.sort_values(['province_state_country','Date'])

#     test_df = test_df.sort_values(['province_state_country','Date'])

# #     

#     print(train_df.head(5))

#     print(test_df.head(5))

    

#     train_feature = create_feature(train_df_2)

#     print("train_df",train_df.head(5))

#     print("train_feature",train_feature.head(5))

    

#     adjusted_train_feature_min_index = min(train_df.index)

#     for day in sorted(feature_day, reverse = True):

#         print("day",day)

#         feature_use = 'Number day from ' + str(day) + ' case'

#         print("feature_use",feature_use)

#         train_feature_zero_count = train_feature[train_feature[feature_use] == 0].shape[0]

#         train_feature_nonzero_count = train_feature[train_feature[feature_use] > 0].shape[0]

#         print("train_feature_zero_count",train_feature_zero_count)

#         print("train_feature_nonzero_count",train_feature_nonzero_count)

#         if(train_feature_nonzero_count >= 30):

#             adjusted_train_feature_min_index = min(train_feature[train_feature[feature_use]>0].index)

#             adjusted_train_feature_max_index = max(train_feature[train_feature[feature_use]>0].index)

#             print("adjusted_train_feature_min_index",adjusted_train_feature_min_index)

#             print("adjusted_train_feature_max_index",adjusted_train_feature_max_index)

#             break

            

           

#     adjusted_train_data = train_df.loc[train_df.index >= adjusted_train_feature_min_index]

#     adjusted_train_data_start_date = adjusted_train_data['Date'].min()

#     print(adjusted_train_data)



#     #     adjusted_train_feature = train_feature[train_feature_zero_count:][feature_use].values.reshape(-1,1)

# #     print("adjusted_train_feature",adjusted_train_feature)



#     ### training data for public board

#     train_data_1 = adjusted_train_data.loc[(adjusted_train_data['Date'] < public_test_data_start_date)]

#     train_data_1['days'] = (train_data_1.Date - adjusted_train_data_start_date).dt.days

#     train_data_1.index = pd.to_datetime(train_data_1.Date)

#     train_data_exog_1 = np.array(train_data_1[['days']])

#     print("train_data_1",train_data_1.head(5))



#     #### Validation data

#     train_data_validation_1 = adjusted_train_data.loc[(adjusted_train_data['Date'] >= public_test_data_start_date)]

#     train_data_validation_1['days'] = (train_data_validation_1.Date - adjusted_train_data_start_date).dt.days

#     train_data_validation_exog_1 = np.array(train_data_validation_1[['days']])

#     print("train_data_validation_1",train_data_validation_1.head(5))

    

#     ###3 test data for public data

#     ### test data starts from '2020 -04-02' and public data starts from '2020-04-01'

#     test_data_1 = test_data.loc[(test_data['province_state_country'] == s) & (test_data['Date'] <= public_test_data_end_date)]

#     test_data_1['days'] = (test_data_1.Date - adjusted_train_data_start_date).dt.days

#     test_data_exog_1 = np.array(test_data_1[['days']])

#     print("test_data_1", test_data_1.head(5))

    

#     #### train data for private board

#     train_data_2 = adjusted_train_data.loc[(adjusted_train_data['Date'] <= train_data_end_date )]

#     train_data_2['days'] = (train_data_2.Date - adjusted_train_data_start_date).dt.days

#     train_data_2.index = pd.to_datetime(train_data_2.Date)

#     train_data_exog_2 = np.array(train_data_2[['days']])

#     print("train_data_2",train_data_2.head(5))

    

#     ###3 test data for private data

#     ### Private board data from '2020-04-16 to '2020-05-14'

#     test_data_2 = test_data.loc[(test_data['province_state_country'] == s) & (test_data['Date'] >= private_test_data_start_date)]

    

#     test_data_2['days'] = (test_data_2.Date - adjusted_train_data_start_date).dt.days

#     test_data_exog_2 = np.array(test_data_2[['days']])

#     print("test_data_2", test_data_2.head(4))

    

#     ### for public board

#     ### for confirmed cases

#     trained_Model_Confirmed_1 = auto_arima(train_data_1['ConfirmedCases'], exogenous = train_data_exog_1 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

#     print("trained_Model_Confirmed_1",trained_Model_Confirmed_1)

    

#     prediction_confirmed_1 = trained_Model_Confirmed_1.predict(exogenous = train_data_validation_exog_1 , n_periods =train_data_validation_1['days'].shape[0])

    

#     print("prediction_confirmed_1",prediction_confirmed_1)

    

#     prediction_confirmed_public_1 = trained_Model_Confirmed_1.predict(exogenous = test_data_exog_1 , n_periods =test_data_1['days'].shape[0])

#     print("prediction_confirmed_public_1",prediction_confirmed_public_1)



#     ### for Fatalities

#     trained_Model_Fatalities_1 = auto_arima(train_data_1['Fatalities'], exogenous = train_data_exog_1 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

#     print("trained_Model_Fatalities_1",trained_Model_Fatalities_1)

    

#     prediction_Fatalities_1 = trained_Model_Fatalities_1.predict(exogenous = train_data_validation_exog_1 , n_periods =train_data_validation_1['days'].shape[0])

#     print("prediction_Fatalities_1",prediction_Fatalities_1)

    

#     prediction_Fatalities_public_1 = trained_Model_Fatalities_1.predict(exogenous = test_data_exog_1 , n_periods =test_data_1['days'].shape[0])

    

#     print("prediction_Fatalities_public_1",prediction_Fatalities_public_1)



#     ### for private board

#     ### for confirmed cases

#     trained_Model_Confirmed_2 = auto_arima(train_data_2['ConfirmedCases'], exogenous = train_data_exog_2 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

#     print("trained_Model_Confirmed_2",trained_Model_Confirmed_2)

    

#     prediction_confirmed_private_2 = trained_Model_Confirmed_2.predict(exogenous = test_data_exog_2 , n_periods =test_data_2['days'].shape[0])

    

#     print("prediction_confirmed_public_2",prediction_confirmed_private_2)



#     ### for Fatalities

#     trained_Model_Fatalities_2 = auto_arima(train_data_2['Fatalities'], exogenous = train_data_exog_2 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

#     print("trained_Model_Fatalities_2",trained_Model_Fatalities_2)

    

#     prediction_Fatalities_private_2 = trained_Model_Fatalities_2.predict(exogenous = test_data_exog_2 , n_periods =test_data_2['days'].shape[0])

    

#     print("prediction_Fatalities_private_2",prediction_Fatalities_private_2)

    

#     ##### public data prediction from 2020-04-01 to 2020-04-15'

#     ### test data starts from '2020-04-02'

#     #### Validation data start from '2020-04-01  to maimum date of training dataset'

#     ### public data prediction data frame

#     public_data_pred = test_data_1

#     public_data_pred['Confirmed_pred'] = prediction_confirmed_public_1

#     public_data_pred['Fatalities_pred'] = prediction_Fatalities_public_1

    

#     ### Validation data

#     validation_data = train_data_validation_1

#     validation_data['Confirmed_pred'] = prediction_confirmed_1

#     validation_data['Fatalities_pred'] = prediction_Fatalities_1

    

    

#     #### Private dataset prediction from 2020-04-15 to 2020-05-14

#     ### test data have data from '2020-04-02' to '2020-05-14'

#     private_data_pred = test_data_2

#     private_data_pred['Confirmed_pred'] = prediction_confirmed_private_2

#     private_data_pred['Fatalities_pred'] = prediction_Fatalities_private_2

    

#     ### Private data prediction filter

#     private_data_pred_till_date = private_data_pred.loc[private_data_pred.Date >= private_test_data_start_date]

    

#     Evaluation_df = public_data_pred

#     Evaluation_df = Evaluation_df.append(private_data_pred_till_date , ignore_index = True)

    

#     submission_df = submission_df.append(Evaluation_df , ignore_index = True)

    

#     validation_df = validation_df.append(validation_data , ignore_index = True)
# province_state_country_list = pd.concat([train_data['province_state_country'], test_data['province_state_country']]).unique()



# # province_state_country_list = np.asarray(['us_california','us_new york'])



# # print(province_state_country_list)



# submission_df = pd.DataFrame(columns= ['ForecastId', 'Province_State', 'Country_Region', 'Date', 'province_state_country','days','Confirmed_pred','Fatalities_pred'])



# validation_df = pd.DataFrame(columns= ['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases',

#        'Fatalities', 'province_state_country','days','Confirmed_pred','Fatalities_pred'])



# for s in province_state_country_list:

#     print("province_country_region",s)

    

#     #### train data for public data

    

#     train_data_1 = train_data.loc[(train_data['province_state_country'] == s) & (train_data['Date']< public_test_data_start_date)]

    

#     train_data_1['days'] = (train_data_1.Date - train_data_start_date).dt.days

#     train_data_1.index = pd.to_datetime(train_data_1.Date)

#     train_data_exog_1 = np.array(train_data_1[['days']])

    

#     print("train_data_1",train_data_1.head(5))

    

#     #### Validation data

#     train_data_validation_1 = train_data.loc[(train_data['province_state_country'] == s) & (train_data['Date'] >= public_test_data_start_date)]

    

#     train_data_validation_1['days'] = (train_data_validation_1.Date - train_data_start_date).dt.days

#     train_data_validation_exog_1 = np.array(train_data_validation_1[['days']])

    

#     ###3 test data for public data

#     ### test data starts from '2020 -04-02' and public data starts from '2020-04-01'

#     test_data_1 = test_data.loc[(test_data['province_state_country'] == s) & (test_data['Date'] <= public_test_data_end_date)]

#     test_data_1['days'] = (test_data_1.Date - train_data_start_date).dt.days

#     test_data_exog_1 = np.array(test_data_1[['days']])

    

#     print("test_data_1", test_data_1.head(4))

    

#     #### train data for private board

    

#     train_data_2 = train_data.loc[(train_data['province_state_country'] == s) & (train_data['Date']<=train_data_end_date )]

#     train_data_2['days'] = (train_data_2.Date - train_data_start_date).dt.days

#     train_data_2.index = pd.to_datetime(train_data_2.Date)

#     train_data_exog_2 = np.array(train_data_2[['days']])

    

#     print("train_data_2",train_data_2.head(5))

    

#     ###3 test data for private data

#     ### Private board data from '2020-04-16 to '2020-05-14'

#     test_data_2 = test_data.loc[(test_data['province_state_country'] == s) & (test_data['Date'] >= private_test_data_start_date)]

    

#     test_data_2['days'] = (test_data_2.Date - train_data_start_date).dt.days

#     test_data_exog_2 = np.array(test_data_2[['days']])

#     print("test_data_2", test_data_2.head(4))

    

#     ### for public board

#     ### for confirmed cases

#     trained_Model_Confirmed_1 = auto_arima(train_data_1['ConfirmedCases'], exogenous = train_data_exog_1 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

#     print("trained_Model_Confirmed_1",trained_Model_Confirmed_1)

    

#     prediction_confirmed_1 = trained_Model_Confirmed_1.predict(exogenous = train_data_validation_exog_1 , n_periods =train_data_validation_1['days'].shape[0])

    

#     print("prediction_confirmed_1",prediction_confirmed_1)

    

#     prediction_confirmed_public_1 = trained_Model_Confirmed_1.predict(exogenous = test_data_exog_1 , n_periods =test_data_1['days'].shape[0])

#     print("prediction_confirmed_public_1",prediction_confirmed_public_1)



#     ### for Fatalities

#     trained_Model_Fatalities_1 = auto_arima(train_data_1['Fatalities'], exogenous = train_data_exog_1 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

#     print("trained_Model_Fatalities_1",trained_Model_Fatalities_1)

    

#     prediction_Fatalities_1 = trained_Model_Fatalities_1.predict(exogenous = train_data_validation_exog_1 , n_periods =train_data_validation_1['days'].shape[0])

#     print("prediction_Fatalities_1",prediction_Fatalities_1)

    

#     prediction_Fatalities_public_1 = trained_Model_Fatalities_1.predict(exogenous = test_data_exog_1 , n_periods =test_data_1['days'].shape[0])

    

#     print("prediction_Fatalities_public_1",prediction_Fatalities_public_1)



#     ### for private board

#     ### for confirmed cases

#     trained_Model_Confirmed_2 = auto_arima(train_data_2['ConfirmedCases'], exogenous = train_data_exog_2 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

#     print("trained_Model_Confirmed_2",trained_Model_Confirmed_2)

    

#     prediction_confirmed_private_2 = trained_Model_Confirmed_2.predict(exogenous = test_data_exog_2 , n_periods =test_data_2['days'].shape[0])

    

#     print("prediction_confirmed_public_2",prediction_confirmed_private_2)



#     ### for Fatalities

#     trained_Model_Fatalities_2 = auto_arima(train_data_2['Fatalities'], exogenous = train_data_exog_2 , supress_warnings =  True, m = 1, stepwise = True, error_action = 'ignore', stationary = False)

#     print("trained_Model_Fatalities_2",trained_Model_Fatalities_2)

    

#     prediction_Fatalities_private_2 = trained_Model_Fatalities_2.predict(exogenous = test_data_exog_2 , n_periods =test_data_2['days'].shape[0])

    

#     print("prediction_Fatalities_private_2",prediction_Fatalities_private_2)

    

#     ##### public data prediction from 2020-04-01 to 2020-04-15'

#     ### test data starts from '2020-04-02'

#     #### Validation data start from '2020-04-01  to maimum date of training dataset'

#     ### public data prediction data frame

#     public_data_pred = test_data_1

#     public_data_pred['Confirmed_pred'] = prediction_confirmed_public_1

#     public_data_pred['Fatalities_pred'] = prediction_Fatalities_public_1

    

#     ### Validation data

#     validation_data = train_data_validation_1

#     validation_data['Confirmed_pred'] = prediction_confirmed_1

#     validation_data['Fatalities_pred'] = prediction_Fatalities_1

    

    

#     #### Private dataset prediction from 2020-04-15 to 2020-05-14

#     ### test data have data from '2020-04-02' to '2020-05-14'

#     private_data_pred = test_data_2

#     private_data_pred['Confirmed_pred'] = prediction_confirmed_private_2

#     private_data_pred['Fatalities_pred'] = prediction_Fatalities_private_2

    

#     ### Private data prediction filter

#     private_data_pred_till_date = private_data_pred.loc[private_data_pred.Date >= private_test_data_start_date]

    

#     Evaluation_df = public_data_pred

#     Evaluation_df = Evaluation_df.append(private_data_pred_till_date , ignore_index = True)

    

#     submission_df = submission_df.append(Evaluation_df , ignore_index = True)

    

#     validation_df = validation_df.append(validation_data , ignore_index = True)
submission_df.loc[submission_df['Confirmed_pred'] <0 ,'Confirmed_pred'] =0

submission_df.loc[submission_df['Fatalities_pred'] <0 ,'Fatalities_pred'] =0

validation_df.loc[submission_df['Confirmed_pred'] <0 ,'Confirmed_pred'] =0

validation_df.loc[submission_df['Fatalities_pred'] <0 ,'Fatalities_pred'] =0
submission_df.head(25)
train_data.columns
submission_data = submission_df[['ForecastId','Confirmed_pred','Fatalities_pred']]

### renaming columns

submission_data.rename(columns = {"Confirmed_pred":"ConfirmedCases" ,"Fatalities_pred":"Fatalities"},inplace =True)



submission_data['ConfirmedCases'] =submission_data['ConfirmedCases'].apply(np.ceil)

submission_data['Fatalities'] = submission_data['Fatalities'].apply(np.ceil)



submission_data['ConfirmedCases'] =submission_data['ConfirmedCases'].astype(int)

submission_data['Fatalities'] = submission_data['Fatalities'].astype(int)
submission_data.head(25)
submission_data.to_csv("submission.csv",index = None)