from sklearn.metrics import mean_absolute_error

import pandas as pd

import numpy as np

from fbprophet import Prophet

import os

import matplotlib as plt

def actualvspredicted(validate_df,forecast_df):

    """

    actualvspredicted function is used to concatinate vadidation data and forecasted data into output.

    :param 1: validate_df, :type: class 'pandas.core.frame.DataFrame'

    :param 2: forecast_df, :type: class 'pandas.core.frame.DataFrame'

    :return: output

    :return type: class 'pandas.core.frame.DataFrame'

    """

    validate_df = validate_df.reset_index()

    forecast_df = forecast_df.reset_index()

    forecast_df.drop('index',axis = 1,inplace = True)

    validate_df.drop('index',axis = 1,inplace = True)

    print(forecast_df.shape)

    print(validate_df.shape)

    output = pd.concat([validate_df['sales'],forecast_df['yhat']],axis =1)

    return output



def accuracy_check(output):

    """

    This function is used to count Mean Absolute Error and Mean Absolute Percentage Error 

    :param: output, :type: class 'pandas.core.frame.DataFrame'

    :return 1: mape

    :return 2: mae

    :return 1 type: class 'float'

    :return 2 type: class 'numpy.float64'

    """

    mae = mean_absolute_error(output['sales'],output['yhat'])    

    mape = np.mean(np.abs(output['sales']-output['yhat'])/output['sales'])  

    print('mae',mae)

    print('mape',mape)

    return mape,mae



def fetch_data(filename):

    """

    This function is used to upload training file and convert into dataframe.

    :param: filename, :type: .csv

    :return: df

    :return type: class 'pandas.core.frame.DataFrame'

    """

    path = '../input/'+filename+'.csv'

    df = pd.read_csv(path)

    return df



def predict(model,future):

    """

    This function is used to forcast data for future values.

    :param 1: model, :type: class 'fbprophet.forecaster.Prophet'

    :param 2: future, :type: class 'pandas.core.frame.DataFrame'

    :return: forecast

    :return type: class 'pandas.core.frame.DataFrame'

    """

    forecast = model.predict(future)

    return forecast



def get_cutoff_date(len_df):

    """

    This function is used to calculate index of training data using its length and selecting cutoff date.

    :param: len_df, :type: int

    :return 1: date_cutoff

    :return 2: train_length

    :return 1 type: str

    :return 2 type: int 

    """

    train_length = int(np.round(len_df*0.7))

    v_length = len_df - len_df*0.7

    print(train_length)

    date_cutoff = train_df.iloc[train_length,:]['date']

    return date_cutoff,train_length
file = "train"

df = fetch_data(file)
df.head()

df
# config item no. 2 and store no. 2

item = 2

store = 3
train_df = df[(df['store'] == store) & (df['item'] == item)] 

# reseting the index value

train_df = train_df.reset_index()

train_df.drop('index',axis = 1,inplace = True)

# calculating lengths for dividing data as 70% train and 30% validation 

len_df = len(train_df)

print("train length",len_df)

date_cutoff,train_length = get_cutoff_date(len_df)

# setting cutoff date by index values of lenghts

train_df = train_df[(train_df['date'] <= date_cutoff)]

train_df = train_df.drop(['store','item'],axis =1)

# renaming 'date' as 'ds' and 'sales'as 'y' mandatory in fbprophet

train_df.rename(columns ={'date':'ds','sales':'y'},inplace = True)

# loading model

model = Prophet(changepoint_prior_scale=0.001)

model.fit(train_df)
period = (len_df - train_length)-1

print("period",period)

# creating future data frame for forecasting

future = model.make_future_dataframe(periods = period)

future

# predict data

forecast = predict(model,future)

forecast
forecast_df = forecast[(forecast['ds'] > date_cutoff)]

# validation df

validate_df = df[(df['store'] == item) & (df['item'] == store) & (df['date'] > date_cutoff)]

forecast_df
forecast_df = forecast_df[['ds','yhat']]

output  = actualvspredicted(validate_df,forecast_df)

mape, mae = accuracy_check(output)
print("mae",mae)

print("mape",mape)

print("accuracy:",1-mape)
model.plot(forecast)

model.plot_components(forecast)