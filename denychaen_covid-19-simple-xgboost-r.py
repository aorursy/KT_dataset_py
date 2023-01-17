import pandas as pd

import numpy as np

import gc

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train.rename(columns={'Country_Region':'Country','Province_State':'State','ConfirmedCases':'Confirmed'}, inplace=True)



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test.rename(columns={'Country_Region':'Country','Province_State':'State','ConfirmedCases':'Confirmed', 'ForecastId':'Id'},inplace=True)

train['Type']='train'

test['Type']='test'



test['Confirmed']=0

test['Fatalities']=0



print(train['Date'].min(),train['Date'].max())

print(test['Date'].min(),test['Date'].max())



train['id_x']=train['Date'].astype(str).values+'_'+train['State'].astype(str).values+'_'+train['Country'].astype(str).values+'_'+train['Type'].astype(str).values

test['id_x']=test['Date'].astype(str).values+'_'+test['State'].astype(str).values+'_'+test['Country'].astype(str).values+'_'+test['Type'].astype(str).values
raw = pd.concat([train,test], axis=0, sort=False)

raw.fillna('None_VAL', inplace=True)

raw['Date'] = pd.to_datetime(raw["Date"])

raw.sort_values('Date',inplace=True) # sort by date

Country_State = raw.Country+'_'+raw.State.astype(str)

raw['Country_State'] = Country_State

raw['Country_State_id'] = Country_State.astype('category').cat.codes

raw['Day'] = raw.Date.dt.strftime("%m%d")

raw['MonthOfYear'] = raw.Date.dt.strftime("%Y%m").astype('category').cat.codes.astype(int)

raw['Week'] = raw['Date'].dt.week

raw['DayOfWeek'] = raw['Date'].dt.dayofweek

raw['Month'] = raw['Date'].dt.month

raw['MonthWeek'] = raw.Month+raw.Week

raw.set_index('Country_State_id', inplace=True)

raw.Day=raw.Day.astype(int)

raw.reset_index(inplace=True)

raw['Country_']=raw.Country.astype('category').cat.codes

raw['State_']=raw.State.astype('category').cat.codes



features = ['id_x','Id','Day','Week','DayOfWeek','Month','MonthOfYear','MonthWeek','Country_State','Country_','State_','Country_State_id']



train = train.merge(raw[features], on=['id_x'], how='left')

test = test.merge(raw[features], on=['id_x'],  how='left')



print(train.shape, test.shape)   
train[['Day','Week','DayOfWeek','Month','MonthOfYear','MonthWeek','Country_','State_']].head()
test[['Day','Week','DayOfWeek','Month','MonthOfYear','MonthWeek','Country_','State_']].head()
from sklearn.preprocessing import MinMaxScaler



sc = MinMaxScaler()

sc.fit_transform(test[['Day','Week','DayOfWeek','Month','MonthOfYear','MonthWeek','Country_','State_']])
from xgboost import XGBRegressor



def fit_model(train,test, target, pre_train=False,  **params):

    

    model = XGBRegressor(**params)

        

    model.fit(train, target)

    

    if pre_train:

        return model.predict(train)

    else:

        return model.predict(test)



def sub_send():

    sub = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

    sub_new = sub[["ForecastId"]]

    result = pd.concat([test.reset_index().Confirmed,test.reset_index().Fatalities,sub_new],axis=1)

    result.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

    result = result[['ForecastId','ConfirmedCases', 'Fatalities']]

    result.to_csv("submission.csv",index=False)

    print(result.head())

    print('subfile was created')

    

def rmse(y_hat, y):

    return np.sqrt(((y_hat - y) ** 2).mean())



def main():



    features = ['Day','Week','DayOfWeek','Month','MonthOfYear','MonthWeek','Country_','State_']

    pred_c = fit_model(train[features],test[features], train.Confirmed, n_estimators=50000)

    test.Confirmed=pred_c

    pred_f = fit_model(train[features],test[features], train.Fatalities, n_estimators=30000)

    test.Fatalities=pred_f

    sub_send()

    

if __name__ == "__main__":

    main()
# import matplotlib.pyplot as plt

# from matplotlib import rcParams

# rcParams['figure.figsize'] = 18,4



# train.groupby(['Date'])['Confirmed'].sum().plot()

# train.groupby(['Date'])['Fatalities'].sum().plot()