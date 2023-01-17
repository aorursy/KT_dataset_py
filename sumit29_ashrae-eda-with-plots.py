"""reading training data"""

import pandas as pd

import plotly.express as px



train_data= pd.read_csv("../input/ashrae-energy-prediction/train.csv")

weather = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

## Function to reduce the DF size

import numpy as np

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df





train_data = reduce_mem_usage(train_data)



weather = reduce_mem_usage(weather)



metadata = reduce_mem_usage(metadata)

"""getting description"""

train_data.describe()
"""EDA for each variable"""

import plotly.express as px

out = train_data.meter.value_counts().reset_index()



fig = px.bar(out, x='index' ,  y='meter')

fig.show()
temp = train_data.groupby(['meter']).mean().reset_index()

fig = px.bar(temp, x="meter" ,  y='meter_reading')

fig.show()
"""histogram for skewed meter_reading"""

import matplotlib.pyplot as plt

train_data['meter_reading'].hist(bins = 100)

import numpy as np

train_data['log_meter_reading'] = np.log1p(train_data['meter_reading']) 

train_data['log_meter_reading'].hist(bins = 100)

"""conevrting time stamp into timeseries type"""

train_data['timestamp'] = pd.to_datetime(train_data['timestamp'] )



from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(rows=2, cols=2)



"""for meter type 0"""

temp = train_data.loc[train_data['meter'] == 0].groupby(['timestamp']).sum().reset_index()

fig.add_trace(

    go.Scatter(x=temp['timestamp'], y=temp['meter_reading']),

    row=1, col=1

)



"""for meter type 1"""

temp = train_data.loc[train_data['meter'] == 1].groupby(['timestamp']).sum().reset_index()

fig.add_trace(

    go.Scatter(x=temp['timestamp'], y=temp['meter_reading']),

    row=1, col=2

)





"""for meter type 2"""

temp = train_data.loc[train_data['meter'] == 2].groupby(['timestamp']).sum().reset_index()

fig.add_trace(

    go.Scatter(x=temp['timestamp'], y=temp['meter_reading']),

    row=2, col=1

)



"""for meter type 3"""

temp = train_data.loc[train_data['meter'] == 3].groupby(['timestamp']).sum().reset_index()

fig.add_trace(

    go.Scatter(x=temp['timestamp'], y=temp['meter_reading']),

    row=2, col=2

)



fig.show()
"""getting missing data"""

metadata.isnull().sum()
fig = px.box(metadata, y="year_built")

fig.show()
fig = px.box(metadata, y="floor_count")

fig.show()
merged_data = pd.merge(train_data, metadata, on="building_id")

from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(rows=4, cols=4)

usage = list(merged_data.primary_use.unique())

count = 0





for rows in [1,2,3,4]:

  for cols in [1,2,3,4]:



    temp = merged_data.loc[merged_data.primary_use == usage[count]].groupby(['timestamp']).sum().reset_index()

    count += 1

    fig.add_trace(

        go.Scatter(x=temp['timestamp'], y=temp['meter_reading']),

        row=rows, col=cols

    )





fig.update_layout(height=1200, width=1600, title_text="Subplots")

count = 0

for rows in [1,2,3,4]:

  for cols in [1,2,3,4]:

    fig.update_xaxes(title_text=usage[count], row=rows, col=cols)

    count += 1



fig.show()


temp = merged_data.groupby(["primary_use"]).mean()['meter_reading'].reset_index()

fig = px.bar(temp, x="primary_use" ,  y='meter_reading')

fig.show()

temp = merged_data.groupby(["primary_use"]).sum()['meter_reading'].reset_index()

fig = px.bar(temp, x="primary_use" ,  y='meter_reading')

fig.show()
weather.describe()
#weather.air_temperature.nunique()

fig = px.box(weather, y="air_temperature")

fig.show()
"""converting timestamp data to timestamp type"""

weather['timestamp'] = pd.to_datetime(weather['timestamp'] )
weather.head(5)
merged_data.head(4)





"""merging dataframes on site_id and timestamp"""



new_merges = pd.merge(merged_data, weather, on=['timestamp','site_id'])



"""verifying length """

print(len(new_merges))
"""counting the number of missing timestamp and site combination data from weather file"""

print('missing' ,  len(set(merged_data.timestamp.apply(str) + "_" + merged_data.site_id.apply(str)) - set(weather.timestamp.apply(str) + "_" + weather.site_id.apply(str)) ))
print("data lost due to missing of wether data" , len(merged_data)- len(new_merges))
"""checking missing values"""

new_merges.isnull().sum().plot(kind= "bar")

print(new_merges.isnull().sum())
import plotly.express as px

temp = new_merges.groupby(['air_temperature']).sum()['meter_reading'].reset_index()

fig = px.bar(temp, x="air_temperature" ,  y='meter_reading')

fig.show()

import plotly.express as px

temp = new_merges.groupby(['wind_speed']).sum()['meter_reading'].reset_index()

fig = px.bar(temp, x="wind_speed" ,  y='meter_reading')

fig.show()

import plotly.express as px

temp = new_merges.groupby(['dew_temperature']).sum()['meter_reading'].reset_index()

fig = px.bar(temp, x="dew_temperature" ,  y='meter_reading')

fig.show()
