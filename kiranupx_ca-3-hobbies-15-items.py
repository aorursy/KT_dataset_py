import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import datetime as dt

import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

from sklearn import metrics
os.chdir('/kaggle/input/wallmart-sales/')

total_value = pd.read_csv('total_value.csv')

total_value.head()
os.chdir('/kaggle/input/wallmart/')

cal = pd.read_csv('calendar.csv')
cal
# One-hot encoding months

month = pd.get_dummies(cal['month'],prefix='month',drop_first=True)



# Dropping unecessary cols

cal.drop(['wm_yr_wk', 'weekday','d', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_TX', 'snap_WI'],1,inplace=True)



# Handling events

cal['event_name_1'] = cal['event_name_1'].fillna(0)

cal['event_name_1'] = np.where(cal['event_name_1'] != 0,1,0)



# One-hot encoding day of months

cal['date'] = pd.to_datetime(cal['date'])

cal['dayofmonth'] = cal['date'].dt.day     

dom = pd.DataFrame(np.where(cal['dayofmonth']>=15,1,0),columns=['day_ge_15'])



# One-hot encoding years

year = pd.get_dummies(cal['year'],prefix='year_',drop_first=True)



# One-hot encoding weekdays

wday = pd.get_dummies(cal['wday'],prefix='wday_',drop_first=True)



# Combing and removing features

cal.drop(['month','year','dayofmonth','wday'],1,inplace=True)

cal = pd.concat([cal,month,year,dom,wday],axis=1)
# Function for filtering california store 3 items

def california_store_3(item):

    state = item.split('_')[3] 

    store_no = int(item.split('_')[4])

    if (state == 'CA' and store_no == 3): 

        return True

    else: 

        return False
item_list = list(total_value.columns[:-1])

california_store_3_item = filter(california_store_3, item_list)

california_store_3_item_list = [i for i in california_store_3_item]
california_store_3_df = total_value.loc[:,california_store_3_item_list]

california_store_3_df
# Concatinating categorical variables

california_store_3_df = pd.concat([cal.iloc[:1941,:],california_store_3_df],1)

california_store_3_df.head()
california_store_3_df['date'] = pd.to_datetime(california_store_3_df['date'])
# 3049 items

california_store_3_last_1_year_df = california_store_3_df[california_store_3_df['date'] >='2015-02-22']

california_store_3_last_1_year_df.head()
# 3021 items

california_store_3_last_1_year_df_without_Nas = california_store_3_last_1_year_df.dropna(axis=1)

california_store_3_last_1_year_df_without_Nas.head()
# Fuction for filtering hobbies

def california_store_3_hobbies(item):

    category = item.split('_')[0]

    if (category == 'HOBBIES'): 

        return True

    else: 

        return False
categorical_variables = list(california_store_3_last_1_year_df_without_Nas.columns[:26])

california_store_3_last_1_year_without_Nas_item_list = list(california_store_3_last_1_year_df_without_Nas.columns[26:])

california_store_3_last_1_year_without_Nas_item_hobbies = filter(california_store_3_hobbies, california_store_3_last_1_year_without_Nas_item_list)

california_store_3_last_1_year_without_Nas_item_hobbies_list = categorical_variables + [i for i in california_store_3_last_1_year_without_Nas_item_hobbies]
california_store_3_last_1_year_without_Nas_item_hobbies_df = california_store_3_last_1_year_df_without_Nas.loc[:,california_store_3_last_1_year_without_Nas_item_hobbies_list]

california_store_3_last_1_year_without_Nas_item_hobbies_df
master_df = california_store_3_last_1_year_without_Nas_item_hobbies_df.copy()
fig = go.Figure()



fig.add_trace(go.Scatter(x=master_df.date, 

                         y=master_df.iloc[26:].sum(axis=1),

                         mode='lines',

                         name='pred'))

   

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,

    margin=dict(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    paper_bgcolor="LightSteelBlue",

    title="Walmart California store 3 hobbies sales",

    xaxis_title="Date",

    yaxis_title="Sales",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#042a30"

    )

)





fig.update_xaxes(rangeslider_visible=True)

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=master_df.date, 

                         y=master_df.iloc[:,26:26+15].sum(axis=1),

                         mode='lines',

                         name='pred'))

   

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,

    margin=dict(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    paper_bgcolor="LightSteelBlue",

    title="Walmart California store 3 hobbies sales for 15 items",

    xaxis_title="Date",

    yaxis_title="Sales",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#042a30"

    )

)





fig.update_xaxes(rangeslider_visible=True)

fig.show()
master_df_15_items = master_df.iloc[:,:26+15]
# loop for lstm



categorical_varibales = master_df_15_items.iloc[:,:26] 

target_variables = master_df_15_items.iloc[:,26:].columns



predictions = pd.DataFrame(np.arange(1,29),columns=['index'])

actual = pd.DataFrame(np.arange(1,29),columns=['index'])



for target_variable in target_variables:

    

    # Making dataset

    dataset = pd.concat([categorical_varibales,master_df[target_variable]],1)



    # Splitting train and test data

    split_date = '2016-04-24'

    Train = dataset.loc[dataset['date'] <= split_date].copy()

    Test = dataset.loc[dataset['date'] > split_date].copy()

    

    # Dropping date

    Train.drop(['date'],axis=1,inplace=True)

    Test.drop(['date'],axis=1,inplace=True)

    

    x_train = Train.drop([target_variable],1)

    y_train = Train[target_variable]

    x_test = Test.drop([target_variable],1)

    y_test = Test[target_variable]



    #Create model layers

    model = tf.keras.Sequential([

        tf.keras.layers.LSTM(64,input_shape=(len(x_test.keys()),1)),

        tf.keras.layers.Dense(1)

    ])



    #Choose optimizer

    optimizer = tf.keras.optimizers.Adam()



    #Compile model with mean squared error as loss function

    model.compile(loss='mse',

                  optimizer=optimizer,

                  metrics=['mae', 'mse'])



    # Number of epochs

    EPOCHS = 10

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    history=model.fit(np.reshape(x_train.values, (x_train.shape[0], x_train.shape[1], 1))

    , y_train,epochs=EPOCHS, validation_split = 0.2,callbacks=[early_stop], verbose=1)

    

    # Predictions

    test_predictions = model.predict(np.reshape(x_test.values, (x_test.shape[0], x_test.shape[1], 1))).flatten()

    print(f'{target_variable}:{metrics.mean_squared_error(test_predictions,y_test)}')



    test_predictions = pd.DataFrame(test_predictions, columns=[target_variable])

    predictions = pd.concat([predictions, test_predictions], 1)



    y_test = pd.DataFrame(y_test.values, columns=[target_variable])

    actual = pd.concat([actual, y_test], 1, ignore_index = True)
fig = go.Figure()





fig.add_trace(go.Scatter(x=predictions.index, y=predictions.sum(axis=1),mode='lines',name='pred'))



fig.add_trace(go.Scatter(x=predictions.index, y=actual.sum(axis=1),mode='lines',name='actual'))

    

    

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,

    margin=dict(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    paper_bgcolor="LightSteelBlue",

    title="Walmart California store 3 hobbies sales for 15 items",

    xaxis_title="Date",

    yaxis_title="Sales",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#042a30"

    )

)





fig.update_xaxes(rangeslider_visible=True)

fig.show()
# Packages for arima

!python3.7 -m pip install --upgrade pip

!pip install pmdarima

from pmdarima.arima import auto_arima
# loop for arima



target_variables = master_df_15_items.iloc[:,26:].columns



predictions_ar = pd.DataFrame(np.arange(1,29),columns=['index'])

actual_ar = pd.DataFrame(np.arange(1,29),columns=['index'])



for target_variable in target_variables:

     

    # Making dataset

    dataset = master_df[target_variable]



    # Splitting train and test data

    df_arima_train = dataset[:-28]

    y_test = dataset[-28:]    

    

    # Defining model

    stepwise_model = auto_arima(df_arima_train,start_p=1,start_q=1,max_p=3,max_q=3,m=7,start_P=0,seasonal=True,d=1,D=1,trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)



    # Predictions

    test_predictions = stepwise_model.predict(n_periods=28)

    ar_day_rmse = np.sqrt(metrics.mean_squared_error(test_predictions, y_test))

    print(f'{target_variable}th rmse:{ar_day_rmse}')



    test_predictions = pd.DataFrame(test_predictions, columns=[target_variable])

    predictions_ar = pd.concat([predictions_ar, test_predictions], 1)



    y_test = pd.DataFrame(y_test.values, columns=[target_variable])

    actual_ar = pd.concat([actual_ar, y_test], 1, ignore_index = True)
fig = go.Figure()





fig.add_trace(go.Scatter(x=predictions.index, y=predictions_ar.sum(axis=1),mode='lines',name='pred_ar'))



fig.add_trace(go.Scatter(x=predictions.index, y=actual_ar.sum(axis=1),mode='lines',name='actual'))

    

    

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,

    margin=dict(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    paper_bgcolor="LightSteelBlue",

    title="Walmart California store 3 hobbies sales for 15 items",

    xaxis_title="Date",

    yaxis_title="Sales",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#042a30"

    )

)





fig.update_xaxes(rangeslider_visible=True)

fig.show()