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
# Packages for arima

!python3.7 -m pip install --upgrade pip

!pip install pmdarima

from pmdarima.arima import auto_arima
# loop for arima



target_variables = master_df_all_items.iloc[:,26:].columns



predictions_ar = pd.DataFrame(np.arange(1,29),columns=['index'])



for target_variable in target_variables:

     

    # Making dataset

    dataset = master_df[target_variable]



    # Splitting train and test data

    df_arima_train = dataset[:-28]

    

    # Defining model

    stepwise_model = auto_arima(df_arima_train,start_p=1,start_q=1,max_p=3,max_q=3,m=7,start_P=0,seasonal=True,d=1,D=1,trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)



    # Predictions

    test_predictions = stepwise_model.predict(n_periods=28)



    test_predictions = pd.DataFrame(test_predictions, columns=[target_variable])

    predictions_ar = pd.concat([predictions_ar, test_predictions], 1)
fig = go.Figure()





fig.add_trace(go.Scatter(x=predictions_ar.index, y=predictions_ar.sum(axis=1),mode='lines',name='pred_ar'))



    

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
predictions_ar.to_csv('/kaggle/working/CA_3_food_ar.csv', index=False)