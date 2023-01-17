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
!pip install pmdarima
import math

import seaborn as sns

from pmdarima.arima import auto_arima

from datetime import datetime,date
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

submission_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")
print(train_data.shape)

print(test_data.shape)

print(train_data.info())

print(test_data.info())

print(train_data.isna().sum())

print(test_data.isna().sum())

display(train_data.head(5))

display(train_data.describe())
display(train_data.head(5))

display(train_data.describe())
### Gives count of each columns for Not Null values

train_data.count()
train_data.columns
train_data.Date = pd.to_datetime(train_data.Date)

test_data.Date = pd.to_datetime(test_data.Date)
print("train data range from ", min(train_data.Date) ,"to", max(train_data.Date))

print("test data range from ", min(test_data.Date) ,"to", max(test_data.Date))
train_data.head(5)
## train and test data manipulation

train_data.Date = pd.to_datetime(train_data.Date)

train_data['Province_State'] = train_data['Province_State'].astype(str)

train_data['Province_State'] = train_data['Province_State'].str.replace(',','_').str.replace('nan','').str.lower()

train_data['Country_Region'] = train_data['Country_Region'].astype(str).str.lower().str.replace(' ','_')

train_data['province_state_country'] = train_data['Country_Region'].astype(str).str.lower().str.replace(' ','_') + ('_') + train_data['Province_State']

train_data['province_state_country'] = train_data['province_state_country'].str.replace('_$','')

train_data['DayWiseConfirmedCases'] = train_data['ConfirmedCases'] - train_data['ConfirmedCases'].shift(1)

train_data['DayWiseConfirmedCases'] = train_data['DayWiseConfirmedCases'].fillna(0)

train_data['DayWiseFatalities'] = train_data['Fatalities'] - train_data['Fatalities'].shift(1)

train_data['DayWiseFatalities'] = train_data['DayWiseFatalities'].fillna(0)





##3 test data manipulation

test_data.Date = pd.to_datetime(test_data.Date)

test_data['Province_State'] = test_data['Province_State'].astype(str)

test_data['Province_State'] = test_data['Province_State'].str.replace(',','_').str.replace('nan','').str.lower()

test_data['Country_Region'] = test_data['Country_Region'].astype(str).str.lower().str.replace(' ','_')

test_data['province_state_country'] = test_data['Country_Region'].astype(str).str.lower().str.replace(' ','_') + ('_') + test_data['Province_State']

test_data['province_state_country'] = test_data['province_state_country'].str.replace('_$','')

### sort the data

train_data = train_data.sort_values(['province_state_country','Date'])

test_data = test_data.sort_values(['province_state_country','Date'])
### Dates

train_data_start_date = pd.to_datetime(min(train_data.Date))

train_data_end_date = max(train_data.Date)

test_data_start_date = min(test_data.Date)

test_data_end_date = max(test_data.Date)

public_test_data_start_date = test_data_start_date

public_test_data_end_date = datetime(2020,4,8)

private_test_data_start_date = datetime(2020,4,9)

private_test_data_end_date = test_data_end_date

print(type(train_data_start_date))

# print(train_data_end_date)

# print(test_data_start_date)

# print(test_data_end_date)

# print(public_test_data_start_date)

# print(public_test_data_end_date)

# print(private_test_data_start_date)

# print(private_test_data_end_date)
train_data.info()
# province_state_country_list = pd.concat([train_data['province_state_country'], test_data['province_state_country']]).unique()



# # province_state_country_list = np.asarray(['china_xinjiang', 'china_yun'])



# submission_df = pd.DataFrame(columns = ['ForecastId', 'Province_State', 'Country_Region', 'Date',

#        'province_state_country','days','Confirmed_pred','Fatalities_Pred'])



# validation_df = pd.DataFrame(columns = ['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases',

#        'Fatalities', 'province_state_country','days','Confirmed_pred','Fatalities_Pred'])

# for s in province_state_country_list:

#     print(s)

    

#     ### train data for public data

# #     print(train_data.head(2))

#     train_data_1 = train_data.loc[(train_data['province_state_country'] == s ) & (train_data['Date'] <test_data_start_date)]

# #     print(train_data_1)

# #     print(train_data_start_date)

#     train_data_1['days'] =  (train_data.Date - train_data_start_date).dt.days

#     train_data_1.index = pd.to_datetime(train_data_1.Date)

#     train_data_exog_1 = np.array(train_data_1[['days']])

    

# #     print(train_data_1.head(1))

# #     print(train_data_1.tail(1))

    

#     ### Validation data

#     print(train_data)

#     train_data_validation_1 = train_data.loc[(train_data['province_state_country'] == s ) & (train_data['Date'] >= test_data_start_date)]

#     train_data_validation_1['days'] =  (train_data.Date - train_data_start_date).dt.days

#     train_data_validation_exog_1 = np.array(train_data_validation_1[['days']])

    

# #     print(train_data_validation_1.head(1))

# #     print(train_data_validation_1.tail(1))

    

#     ### test data for public board

#     test_data_1 = test_data.loc[(test_data['province_state_country'] == s ) & (test_data['Date'] <= public_test_data_end_date)]

#     test_data_1['days'] =  (test_data.Date - train_data_start_date).dt.days

#     test_data_exog_1 = np.array(test_data_1[['days']])

    

# #     print(test_data_1.head(1))

# #     print(test_data_1.tail(1))

    

#     ## train data for private board

#     train_data_2 = train_data.loc[(train_data['province_state_country'] == s ) & (train_data['Date'] <=train_data_end_date)]

#     train_data_2['days'] =  (train_data.Date - train_data_start_date).dt.days

#     train_data_2.index = pd.to_datetime(train_data_2.Date)

#     train_data_exog_2 = np.array(train_data_2[['days']])

    

# #     print(train_data_2.head(1))

# #     print(train_data_2.tail(1))

    

#     ### test data for private board

#     test_data_2 = test_data.loc[(test_data['province_state_country'] == s ) & (test_data['Date'] >= private_test_data_start_date)]

#     test_data_2['days'] =  (test_data.Date - train_data_start_date).dt.days

#     test_data_exog_2 = np.array(test_data_2[['days']])

# #     print(test_data_2.head(1))

# #     print(test_data_2.tail(1))

    

#     ### Set date as time index for train data before applying model

    

#     ##### For Public board

#     ### Model Building for confirmed cases

#     print(train_data_exog_1)

#     trained_Model_Confirmed_1 = auto_arima(train_data_1['ConfirmedCases'],exogenous = train_data_exog_1 ,supress_warnings = True,m= 1, stepwise = True,error_action = 'ignore',stationary = False)

#     print((trained_Model_Confirmed_1))

    

#     prediction_confirmed_1 = trained_Model_Confirmed_1.predict(exogenous = train_data_validation_exog_1 ,n_periods = train_data_validation_1['days'].shape[0])

    

#     print(prediction_confirmed_1)

#     prediction_confirmed_public_1 = trained_Model_Confirmed_1.predict(exogenous = test_data_exog_1 , n_periods = test_data_1['days'].shape[0])

#     print(prediction_confirmed_public_1)

    

#     ### Model Building for Fatalities cases

#     trained_Model_Fatalities_1 = auto_arima(train_data_1['Fatalities'],exogenous = train_data_exog_1,supress_warnings = True,m= 1, stepwise = True,error_action = 'ignore',stationary = False)

#     print((trained_Model_Fatalities_1))

    

#     prediction_Fatalities_1 = trained_Model_Fatalities_1.predict(exogenous = train_data_validation_exog_1,n_periods = train_data_validation_1['days'].shape[0])

    

#     print(prediction_Fatalities_1)

#     prediction_Fatalities_public_1 = trained_Model_Fatalities_1.predict(exogenous = test_data_exog_1,n_periods = test_data_1['days'].shape[0])

#     print(prediction_Fatalities_public_1)

    

#     ##### Model Building for Private board

#     ### Model Building for confirmed cases

    

#     trained_Model_Confirmed_2 = auto_arima(train_data_2['ConfirmedCases'],exogenous = train_data_exog_2,supress_warnings = True,m= 1, stepwise = True,error_action = 'ignore',stationary = False)

#     print((trained_Model_Confirmed_2))

    

#     print(test_data_exog_2)

#     prediction_confirmed_private_2 = trained_Model_Confirmed_2.predict(exogenous = test_data_exog_2 ,n_periods = test_data_2['days'].shape[0])

#     print(prediction_confirmed_private_2)

    

#     ### Model Building for Fatalities cases

#     trained_Model_Fatalities_2 = auto_arima(train_data_2['Fatalities'],exogenous = train_data_exog_2,supress_warnings = True,m= 1, stepwise = True,error_action = 'ignore',stationary = False)

#     print((trained_Model_Fatalities_2))

    

#     prediction_Fatalities_private_2 = trained_Model_Fatalities_2.predict(exogenous = test_data_exog_2,n_periods = test_data_2['days'].shape[0])

#     print(prediction_Fatalities_private_2)

    

#     ### Public data prediction dataframe

#     public_data_pred = test_data_1

#     public_data_pred['Confirmed_pred'] =  prediction_confirmed_public_1

#     public_data_pred['Fatalities_Pred'] =  prediction_Fatalities_public_1

    

#     ### Validation data

#     validation_data = train_data_validation_1

#     validation_data['Confirmed_pred'] =  prediction_confirmed_1

#     validation_data['Fatalities_Pred'] =  prediction_Fatalities_1

    

#     ### Private data prediction dataframe

#     private_data_pred = test_data_2

#     private_data_pred['Confirmed_pred'] =  prediction_confirmed_private_2

#     private_data_pred['Fatalities_Pred'] =  prediction_Fatalities_private_2

    

#     ##### Private data prediction filter(2020-04-09 to 2020-05-07)

#     private_data_pred_till_date = private_data_pred.loc[private_data_pred.Date>= private_test_data_start_date]

    

#     Evaluation_df = public_data_pred

    

#     Evaluation_df= Evaluation_df.append(private_data_pred_till_date, ignore_index = True)

    

#     submission_df= submission_df.append(Evaluation_df, ignore_index = True)

    

#     validation_df= validation_df.append(validation_data, ignore_index = True)
test_data.columns

# train_data.iloc[-1]["ConfirmedCases"]
province_state_country_list = pd.concat([train_data['province_state_country'], test_data['province_state_country']]).unique()

province_state_country_list[0]
province_state_country_list = pd.concat([train_data['province_state_country'], test_data['province_state_country']]).unique()



# province_state_country_list = np.asarray(['afghanistan', 'china_yun'])



submission_df = pd.DataFrame(columns = ['ForecastId', 'Province_State', 'Country_Region', 'Date',

       'province_state_country','days','Confirmed_pred','Fatalities_Pred','DayConfirmed_pred','DayFatalities_Pred'])



validation_df = pd.DataFrame(columns = ['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases',

       'Fatalities', 'province_state_country','days','Confirmed_pred','Fatalities_Pred','DayConfirmed_pred','DayFatalities_Pred'])

for s in province_state_country_list:

    print(s)

    

    ### train data for public data

#     print(train_data.head(2))

    train_data_1 = train_data.loc[(train_data['province_state_country'] == s ) & (train_data['Date'] <test_data_start_date)]

#     print(train_data_1)

#     print(train_data_start_date)

    train_data_1['days'] =  (train_data.Date - train_data_start_date).dt.days

    train_data_1.index = pd.to_datetime(train_data_1.Date)

    train_data_exog_1 = np.array(train_data_1[['days']])

    train_data_1_ConfirmedCases = train_data_1.iloc[-1]["ConfirmedCases"]

    train_data_1_Fatalities = train_data_1.iloc[-1]["Fatalities"]

    

#     print(train_data_1.head(1))

#     print(train_data_1.tail(1))

    

    ### Validation data

    print(train_data)

    train_data_validation_1 = train_data.loc[(train_data['province_state_country'] == s ) & (train_data['Date'] >= test_data_start_date)]

    train_data_validation_1['days'] =  (train_data.Date - train_data_start_date).dt.days

    train_data_validation_exog_1 = np.array(train_data_validation_1[['days']])

    

#     print(train_data_validation_1.head(1))

#     print(train_data_validation_1.tail(1))

    

    ### test data for public board

    test_data_1 = test_data.loc[(test_data['province_state_country'] == s ) & (test_data['Date'] <= public_test_data_end_date)]

    test_data_1['days'] =  (test_data.Date - train_data_start_date).dt.days

    test_data_exog_1 = np.array(test_data_1[['days']])

    

#     print(test_data_1.head(1))

#     print(test_data_1.tail(1))

    

    ## train data for private board

    train_data_2 = train_data.loc[(train_data['province_state_country'] == s ) & (train_data['Date'] <=train_data_end_date)]

    train_data_2['days'] =  (train_data.Date - train_data_start_date).dt.days

    train_data_2.index = pd.to_datetime(train_data_2.Date)

    train_data_exog_2 = np.array(train_data_2[['days']])

    train_data_2_ConfirmedCases = train_data_2.iloc[-1]["ConfirmedCases"]

    train_data_2_Fatalities = train_data_2.iloc[-1]["Fatalities"]

    

#     print(train_data_2.head(1))

#     print(train_data_2.tail(1))

    

    ### test data for private board

    test_data_2 = test_data.loc[(test_data['province_state_country'] == s ) & (test_data['Date'] >= private_test_data_start_date)]

    test_data_2['days'] =  (test_data.Date - train_data_start_date).dt.days

    test_data_exog_2 = np.array(test_data_2[['days']])

#     print(test_data_2.head(1))

#     print(test_data_2.tail(1))

    

    ### Set date as time index for train data before applying model

    

    ##### For Public board

    

#     'Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases',

#        'Fatalities', 'province_state_country', 'DayWiseConfirmedCases',

#        'DayWiseFatalities'],

#       dtype='object')

    ### Model Building for confirmed cases

    

    print(train_data_exog_1)

    trained_Model_Confirmed_1 = auto_arima(train_data_1['DayWiseConfirmedCases'],exogenous = train_data_exog_1 ,supress_warnings = True,m= 1, stepwise = True,error_action = 'ignore',stationary = False)

    print((trained_Model_Confirmed_1))

    

    prediction_confirmed_1 = trained_Model_Confirmed_1.predict(exogenous = train_data_validation_exog_1 ,n_periods = train_data_validation_1['days'].shape[0])

    

    print(prediction_confirmed_1)

    prediction_confirmed_public_1 = trained_Model_Confirmed_1.predict(exogenous = test_data_exog_1 , n_periods = test_data_1['days'].shape[0])

    print(prediction_confirmed_public_1)

    

    ### Model Building for Fatalities cases

    trained_Model_Fatalities_1 = auto_arima(train_data_1['DayWiseFatalities'],exogenous = train_data_exog_1,supress_warnings = True,m= 1, stepwise = True,error_action = 'ignore',stationary = False)

    print((trained_Model_Fatalities_1))

    

    prediction_Fatalities_1 = trained_Model_Fatalities_1.predict(exogenous = train_data_validation_exog_1,n_periods = train_data_validation_1['days'].shape[0])

    

    print(prediction_Fatalities_1)

    prediction_Fatalities_public_1 = trained_Model_Fatalities_1.predict(exogenous = test_data_exog_1,n_periods = test_data_1['days'].shape[0])

    print(prediction_Fatalities_public_1)

    

    ##### Model Building for Private board

    ### Model Building for confirmed cases

    

    trained_Model_Confirmed_2 = auto_arima(train_data_2['DayWiseConfirmedCases'],exogenous = train_data_exog_2,supress_warnings = True,m= 1, stepwise = True,error_action = 'ignore',stationary = False)

    print((trained_Model_Confirmed_2))

    

    print(test_data_exog_2)

    prediction_confirmed_private_2 = trained_Model_Confirmed_2.predict(exogenous = test_data_exog_2 ,n_periods = test_data_2['days'].shape[0])

    print(prediction_confirmed_private_2)

    

    ### Model Building for Fatalities cases

    trained_Model_Fatalities_2 = auto_arima(train_data_2['DayWiseFatalities'],exogenous = train_data_exog_2,supress_warnings = True,m= 1, stepwise = True,error_action = 'ignore',stationary = False)

    print((trained_Model_Fatalities_2))

    

    prediction_Fatalities_private_2 = trained_Model_Fatalities_2.predict(exogenous = test_data_exog_2,n_periods = test_data_2['days'].shape[0])

    print(prediction_Fatalities_private_2)

    

    ### Public data prediction dataframe

    

    public_data_pred = test_data_1

    public_data_pred['Confirmed_pred'] =  prediction_confirmed_public_1

    public_data_pred['Fatalities_Pred'] =  prediction_Fatalities_public_1

    public_data_pred.loc[public_data_pred['Confirmed_pred'] <0,'Confirmed_pred'] =0

    public_data_pred.loc[public_data_pred['Fatalities_Pred'] <0,'Fatalities_Pred'] =0

    

    public_data_pred['Confirmed_pred'].iloc[0]  = public_data_pred['Confirmed_pred'].iloc[0] + train_data_1_ConfirmedCases

    

    public_data_pred['Fatalities_Pred'].iloc[0]  = public_data_pred['Fatalities_Pred'].iloc[0] + train_data_1_Fatalities 

    

    print(public_data_pred.head(5))

    

#     DayConfirmed_pred','DayFatalities_Pred

#     public_data_pred['DayConfirmed_pred'].iloc[0]  = public_data_pred['Confirmed_pred'].iloc[0]

#     public_data_pred['DayConfirmed_pred'] =  public_data_pred['Confirmed_pred'] + public_data_pred['DayConfirmed_pred'].shift(1) 

#     public_data_pred['DayFatalities_Pred'] =  public_data_pred['Fatalities_Pred'] + public_data_pred['DayFatalities_Pred'].shift(1) 

    public_data_pred['DayConfirmed_pred'] =  public_data_pred['Confirmed_pred'].cumsum()

    public_data_pred['DayFatalities_Pred'] =  public_data_pred['Fatalities_Pred'].cumsum()

#     public_data_pred['DayConfirmed_pred'].iloc[0]  = public_data_pred['Confirmed_pred'].iloc[0]

    

    ### Validation data

    validation_data = train_data_validation_1

    validation_data['Confirmed_pred'] =  prediction_confirmed_1

    validation_data['Fatalities_Pred'] =  prediction_Fatalities_1

    validation_data.loc[validation_data['Confirmed_pred'] <0,'Confirmed_pred'] =0

    validation_data.loc[validation_data['Fatalities_Pred'] <0,'Fatalities_Pred'] =0

    

    validation_data['Confirmed_pred'].iloc[0]  = validation_data['Confirmed_pred'].iloc[0] + train_data_1_ConfirmedCases

    

    validation_data['Fatalities_Pred'].iloc[0]  = validation_data['Fatalities_Pred'].iloc[0] + train_data_1_Fatalities 

    

    

    print(validation_data.head(5))

#     DayConfirmed_pred','DayFatalities_Pred

#     validation_data['DayConfirmed_pred'].iloc[0]  = validation_data['Confirmed_pred'].iloc[0]

#     validation_data['DayConfirmed_pred'] =  validation_data['Confirmed_pred'] + validation_data['DayConfirmed_pred'].shift(1) 

#     validation_data['DayFatalities_Pred'] =  validation_data['Fatalities_Pred'] + validation_data['DayFatalities_Pred'].shift(1)  

    

    

    validation_data['DayConfirmed_pred'] =  validation_data['Confirmed_pred'].cumsum() 

    validation_data['DayFatalities_Pred'] =  validation_data['Fatalities_Pred'].cumsum()  

    

#     validation_data['DayConfirmed_pred'].iloc[0]  = validation_data['Confirmed_pred'].iloc[0]

    

    ### Private data prediction dataframe

#     train_data_2_ConfirmedCases = train_data_2.iloc[-1]["ConfirmedCases"]

#     train_data_2_Fatalities = train_data_2.iloc[-1]["Fatalities"]

    private_data_pred = test_data_2

    private_data_pred['Confirmed_pred'] =  prediction_confirmed_private_2

    private_data_pred['Fatalities_Pred'] =  prediction_Fatalities_private_2

    private_data_pred.loc[private_data_pred['Confirmed_pred'] <0,'Confirmed_pred'] =0

    private_data_pred.loc[private_data_pred['Fatalities_Pred'] <0,'Fatalities_Pred'] =0

    

    private_data_pred['Confirmed_pred'].iloc[0]  = private_data_pred['Confirmed_pred'].iloc[0] + train_data_2_ConfirmedCases

    

    private_data_pred['Fatalities_Pred'].iloc[0]  = private_data_pred['Fatalities_Pred'].iloc[0] + train_data_2_Fatalities 



    print(private_data_pred.head(6))

#     DayConfirmed_pred','DayFatalities_Pred

    

#     private_data_pred['DayConfirmed_pred'].iloc[0]  = private_data_pred['Confirmed_pred'].iloc[0]

#     private_data_pred['DayConfirmed_pred'] =  private_data_pred['Confirmed_pred'] + private_data_pred['DayConfirmed_pred'].shift(1) 

#     private_data_pred['DayFatalities_Pred'] =  private_data_pred['Fatalities_Pred'] + private_data_pred['DayFatalities_Pred'].shift(1)   

    

    private_data_pred['DayConfirmed_pred'] =  private_data_pred['Confirmed_pred'].cumsum()

    private_data_pred['DayFatalities_Pred'] =  private_data_pred['Fatalities_Pred'] .cumsum()   

    

    

#     private_data_pred['DayConfirmed_pred'].iloc[0]  = private_data_pred['Confirmed_pred'].iloc[0]

    ##### Private data prediction filter(2020-04-09 to 2020-05-07)

    private_data_pred_till_date = private_data_pred.loc[private_data_pred.Date>= private_test_data_start_date]

    

    Evaluation_df = public_data_pred

    

    Evaluation_df= Evaluation_df.append(private_data_pred_till_date, ignore_index = True)

    

    submission_df= submission_df.append(Evaluation_df, ignore_index = True)

    

    validation_df= validation_df.append(validation_data, ignore_index = True)
###Make -ve prediction as 0

submission_df.loc[submission_df['Confirmed_pred'] <0,'Confirmed_pred'] =0

submission_df.loc[submission_df['Fatalities_Pred'] <0,'Fatalities_Pred'] =0

validation_df.loc[validation_df['Confirmed_pred'] <0,'Confirmed_pred'] =0

validation_df.loc[validation_df['Fatalities_Pred'] <0,'Fatalities_Pred'] =0
validation_df[:43]
submission_df.head(5)
submission_data = submission_df[['ForecastId','DayConfirmed_pred','DayFatalities_Pred']]

print(submission_data.columns)



### renaming the columns

submission_data.rename(columns = {"DayConfirmed_pred" :"ConfirmedCases","DayFatalities_Pred": "Fatalities"},inplace = True)

print(submission_data.columns)

submission_data['ConfirmedCases'] = submission_data['ConfirmedCases'].astype(int)

submission_data['Fatalities'] = submission_data['Fatalities'].astype(int)
submission_data.head(3)
submission_data.to_csv('submission.csv', index = None)