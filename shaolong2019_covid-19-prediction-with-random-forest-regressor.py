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
#Import packages

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np

import plotly_express as px

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler
#Import train data 

df_train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

del df_train['Id']

df_test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

df_infor=pd.read_csv('../input/input/CountryInformation.csv')
def collectdata(df):

    df = pd.merge(df, df_infor, how='left', left_on='Country_Region',right_on='Country(or territory)')

    return df
columns1 = ['Province_State','Country_Region','Date','longitude', 'latitude','HealthCareIndex',

       'PollutionIndex', 'ClimateIndex', 'Population','ConfirmedCases','Fatalities']

columns2 = ['Province_State','Country_Region','Date','longitude', 'latitude','HealthCareIndex',

       'PollutionIndex', 'ClimateIndex', 'Population']
df_train=collectdata(df_train)[columns1]

ForecastId = df_test.ForecastId.values

df_test=collectdata(df_test)[columns2]
def StringToInteger(df):

    #convert NaN Province State values to a string

    df.Province_State.fillna('NaN', inplace=True)

    df.longitude.fillna(0, inplace=True)

    df.latitude.fillna(0, inplace=True)

    df.PollutionIndex.fillna(0, inplace=True)

    df.HealthCareIndex.fillna(0, inplace=True)

    df.ClimateIndex.fillna(0, inplace=True)

    df.Population.fillna(0, inplace=True)

    #Define Ordinal Encoder Model

    oe = OrdinalEncoder()

    df[['Province_State','Country_Region']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region']])

    return df
df_train_Integer = StringToInteger(df_train)

df_test_Integer = StringToInteger(df_test)
Year=pd.to_datetime(df_train_Integer['Date']).dt.year

Month=pd.to_datetime(df_train_Integer['Date']).dt.month

Day=pd.to_datetime(df_train_Integer['Date']).dt.day

df_train_Integer.insert(3,'Year',Year)

df_train_Integer.insert(3,'Month',Month)

df_train_Integer.insert(3,'Day',Day)

df_train_Integer.drop(columns=['Date'],inplace=True)
Year=pd.to_datetime(df_test_Integer['Date']).dt.year

Month=pd.to_datetime(df_test_Integer['Date']).dt.month

Day=pd.to_datetime(df_test_Integer['Date']).dt.day

df_test_Integer.insert(3,'Year',Year)

df_test_Integer.insert(3,'Month',Month)

df_test_Integer.insert(3,'Day',Day)

df_test_Integer.drop(columns=['Date'],inplace=True)
df_train_values = df_train_Integer.values

features, labels = df_train_values[:,:-2], df_train_values[:,-2:]
# Split it into training dataset and training dataset

X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,random_state=0)
#Random Forest

model1=RandomForestRegressor(n_estimators = 100)

model1.fit(X_train,y_train[:,0])# train model

y_pred1 = model1.predict(X_test)
errors = abs(y_pred1 - y_test[:,0])

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate rmse

mse=mean_squared_error( y_test[:,0],y_pred1)

rmse=np.sqrt(mse)

print('rmse:', round(rmse, 2))

r2score=r2_score(y_test[:,0], y_pred1, multioutput='variance_weighted')

print('r2_score:', round(r2score, 2))

model2=RandomForestRegressor(n_estimators = 100)

model2.fit(X_train,y_train[:,1])# train model

y_pred2 = model2.predict(X_test)
# Calculate the absolute errors

errors = abs(y_pred2 - y_test[:,1])

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate rmse

mse=mean_squared_error( y_test[:,1],y_pred2)

rmse=np.sqrt(mse)

print('rmse:', round(rmse, 2))

r2score=r2_score(y_test[:,1], y_pred2, multioutput='variance_weighted')

print('r2_score:', round(r2score, 2))
pred1 = model1.predict(df_test.values)

pred2 = model2.predict(df_test.values)
submission = []

for i in range(len(pred1)):

            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':round(pred1[i],0), 'Fatalities':round(pred2[i],0)}

            submission.append(d)
df_submit = pd.DataFrame(submission)

df_submit.to_csv(r'submission.csv',columns=['ForecastId','ConfirmedCases','Fatalities'], index=0)
def plot_prec_country(data,country_name,Type):

    data = data.loc[(data['Country_Region']==country_name)]

    if Type=='ConfirmedCases':

        cases = data.groupby(['Date','Type'],as_index=False)['ConfirmedCases'].sum()

        fig = px.line(cases, x="Date", y="ConfirmedCases",color="Type")

    else:

        cases = data.groupby(['Date','Type'],as_index=False)['Fatalities'].sum()

        fig = px.line(cases, x="Date", y="Fatalities",color="Type")

    fig.show()
df_train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

del df_train['Id']

df_train=df_train.loc[df_train['Date']<'2020-04-02',:]

df_test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

df_merge = pd.merge(df_test, df_submit, how='left', on='ForecastId')

del df_merge['ForecastId']
df_train.insert(3,'Type','Actual Cases')

df_merge.insert(3,'Type','Predicted Cases')

df_union=df_train.append(df_merge)
plot_prec_country(df_union,'China','ConfirmedCases')