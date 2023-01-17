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

california_store_3_df = pd.concat([cal,california_store_3_df],1)

california_store_3_df.head()
california_store_3_df['date'] = pd.to_datetime(california_store_3_df['date'])
# 3049 items

california_store_3_last_1_year_df = california_store_3_df[california_store_3_df['date'] >='2015-02-22']

california_store_3_last_1_year_df.head()
# Dropping Nas

california_store_3_last_1_year_df_without_Nas_list = list(california_store_3_last_1_year_df.iloc[0,:].dropna().index)

california_store_3_last_1_year_df_without_Nas = california_store_3_last_1_year_df.loc[:, california_store_3_last_1_year_df_without_Nas_list]

california_store_3_last_1_year_df_without_Nas.head() # 3021 items
# Fuction for filtering food

def california_store_3_food(item):

    category = item.split('_')[0]

    if (category == 'FOODS'): 

        return True

    else: 

        return False
categorical_variables = list(california_store_3_last_1_year_df_without_Nas.columns[:26])

california_store_3_last_1_year_without_Nas_item_list = list(california_store_3_last_1_year_df_without_Nas.columns[26:])

california_store_3_last_1_year_without_Nas_item_food = filter(california_store_3_food, california_store_3_last_1_year_without_Nas_item_list)

california_store_3_last_1_year_without_Nas_item_food_list = categorical_variables + [i for i in california_store_3_last_1_year_without_Nas_item_food]
california_store_3_last_1_year_without_Nas_item_food_df = california_store_3_last_1_year_df_without_Nas.loc[:,california_store_3_last_1_year_without_Nas_item_food_list]

california_store_3_last_1_year_without_Nas_item_food_df
master_df = california_store_3_last_1_year_without_Nas_item_food_df.copy()
master_df_all_items = master_df.iloc[:,:]
# loop for lstm



categorical_varibales = master_df_all_items.iloc[:,:26] 

target_variables = master_df_all_items.iloc[:,26:].columns



predictions = pd.DataFrame(np.arange(1,29),columns=['index'])



for target_variable in target_variables:

    

    # Making dataset

    dataset = pd.concat([categorical_varibales,master_df[target_variable]],1)



    # Splitting train and test data

    split_date = '2016-05-22'

    Train = dataset.loc[dataset['date'] <= split_date].copy()

    Test = dataset.loc[dataset['date'] > split_date].copy()

    

    # Dropping date

    Train.drop(['date'],axis=1,inplace=True)

    Test.drop(['date'],axis=1,inplace=True)

    

    x_train = Train.drop([target_variable],1)

    y_train = Train[target_variable]

    x_test = Test.drop([target_variable],1)



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

    

    test_predictions = pd.DataFrame(test_predictions, columns=[target_variable])

    predictions = pd.concat([predictions, test_predictions], 1)

fig = go.Figure()





fig.add_trace(go.Scatter(x=predictions.index, y=predictions.sum(axis=1),mode='lines',name='pred'))

    

    

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

    title="Walmart California store 3 food sales for 15 items",

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
predictions.to_csv('/kaggle/working/CA_3_food.csv', index=False)