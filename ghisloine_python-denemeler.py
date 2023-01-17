# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

!pip install plotly==4.4.1

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
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
metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv",dtype=metadata_dtype)

metadata.info(memory_usage='deep')
weather_dtype = {"site_id":"uint8"}

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=['timestamp'],dtype=weather_dtype)

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=['timestamp'],dtype=weather_dtype)

print (weather_train.info(memory_usage='deep'))

print ("-------------------------------------")

print (weather_test.info(memory_usage='deep'))
train_dtype = {'meter':"uint8",'building_id':'uint16','meter_reading':"float32"}

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv",parse_dates=['timestamp'],dtype=train_dtype)

train = train.sample(n=100000, random_state=1)
# Burda herseyi birlestirme zamani



new = pd.merge(left= train,right=metadata,how='left' ,left_on ='building_id', right_on = 'building_id')

data = pd.merge(left = new, right=weather_train, how='left', left_on=['site_id','timestamp'],right_on= ['site_id','timestamp'])

data = reduce_mem_usage(data)

# Burda building metadata ile train dosyasini birlestirdik



# Memory kaybetmemek icin gereksizleri siliyoruz.
del train

del weather_train

del weather_test

del metadata
data.describe()
q1 = data['meter_reading'].quantile(.25)

print('q1 is {}'.format(q1))

q3 = data['meter_reading'].quantile(.75)

print('q3 is {}'.format(q3))

IQR_main = q3- q1

#mask = data['meter_reading'].between(0, 1.5 * IQR_main, inclusive=True)

#iqr = data.loc[mask, 'meter_reading']



data.drop(data[(data['meter_reading'] < 0) | (data['meter_reading'] > (1.5 * IQR_main)) ].index , inplace=True)

print(data['meter_reading'].value_counts())
print(data.info())

#Scientific notation yapmasin diye float yapiyoruz

data['meter_reading'].describe().apply(lambda x: format(x, 'f'))
# Degiskeni incelemeden once biraz uzerinde oynama yapicaz.

# NaN olanlari elememiz gerekiyor



print(np.isnan(data['meter_reading']).value_counts())

print(np.isinf(data['meter_reading']).value_counts())



# NaN var ancak inf yok



#data['meter_reading'] = np.nan_to_num(data['meter_reading'])
# Null satirlar var mi ? Var meter_reading'i degistirdik. Degismeyenler NULL olarak geldi. Onlari eleyecegiz.

print(data.isnull().sum())
data.dropna(subset =['wind_speed','wind_direction'],inplace = True)
def funct(a):

    if a >= 0 and a<= 1:

        return '0-1'

    elif a > 1 and a<= 2:

        return '1-2'

    elif a > 2 and a<= 3:

        return '2-3'

    elif a > 3 and a<= 4:

        return '3-4'

    elif a > 4 and a<= 5:

        return '4-5'

    elif a > 5 and a<= 6:

        return '5-6'

    else :

        return '6+'

data['wind_speed'] = data['wind_speed'].apply(funct)

data['wind_speed'].value_counts()
wind_data = data.groupby(['wind_speed','wind_direction']).size()

wind_data = wind_data.reset_index()

wind_data.columns = ['wind_speed', 'wind_direction','frequency']
# Tekrar eden satirlar var mi ?

print(data.shape)

duplicate_rows_df = data[data.duplicated()]

print('number of duplicate rows: ', duplicate_rows_df.shape)

# Hic tekrar eden satirimiz yok !
# Birlestirmeden sonra tarihlerde ufak degisiklikler



for df in [data]:

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.weekday.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")
# Burdaki meter degiskeni kategorik bir degisken ondan descriptive stat anlamsiz.



data['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

data['DayOfWeek'].replace({0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'},inplace=True)

data.describe(include= 'all')
#small_data = data

#small_data['meter_reading'] = np.log1p(data['meter_reading'])
fig = px.histogram(data, x="meter_reading",nbins = 20)

fig.update_layout(

    title='<b>Distribution of Meter Reading</b><br><span> Without Outlier and Log Transform<br>',font=dict(

        family="Courier New, monospace",

        size=18,

        color="#000"

    ))

fig.show(renderer="kaggle")
c = data.corr()



mask = np.zeros_like(c, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(c, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
x1 = np.log1p(data[data['meter']=='Electricity']['meter_reading'])

x2 = np.log1p(data[data['meter']=='ChilledWater']['meter_reading'])

x3 = np.log1p(data[data['meter']=='Steam']['meter_reading'])

x4 = np.log1p(data[data['meter']=='HotWater']['meter_reading'])

              

hist_data = [x1, x2, x3, x4]

colors = ['#835AF1', '#7FA6EE', '#B8F7D4','#F66095']              

group_data = ['Electricity', 'ChilledWater', 'Steam', 'HotWater']

              

fig = ff.create_distplot(hist_data, group_data,colors=colors, bin_size = .1)

fig.update_layout(

    title='Distribution of Meter Reading by Meter Type')

fig.show(renderer='kaggle')
'''y = data['wind_direction']

y = y/57.324

x = data['wind_speed']

fig = plt.figure(figsize = (18,18))

ax = fig.add_subplot(111, projection='polar')

c = ax.scatter(y,x, s= x * 10,c = data['site_id'],cmap='twilight',alpha=0.5)'''
fig = px.bar_polar(wind_data, r =wind_data['frequency'], theta=wind_data["wind_direction"],

                   color=wind_data['wind_speed'], template="plotly_dark",

                   color_discrete_sequence= px.colors.sequential.Plasma[-2::-1])

fig.update_layout(title = 'Wind Polar Bar')                  

fig.show()
meter_data = data.loc[:,['meter_reading','primary_use','meter','site_id']]

meter_dist = meter_data.loc[:,['meter_reading','primary_use','meter']]

meter_dist['index'] = meter_dist.reset_index().index

meter_dist.reset_index()



# Burda yapmaya calistigim birazdan daha net belli olacak
fig = px.violin(meter_dist, y="meter_reading", color = 'primary_use')

fig.update_layout(

    title='Distribution of Meter Reading by Primary Use')

fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Violin(x=meter_dist['primary_use'][ meter_dist['meter'] == 'Electricity' ],

                        y=df['meter_reading'][ df['meter'] == 'Electricity' ],

                        legendgroup='Yes', scalegroup='No', name='Electricity',

                        side='positive',

                        line_color='blue')

             )

fig.add_trace(go.Violin(x=meter_dist['primary_use'][ meter_dist['meter'] == 'Steam' ],

                        y=df['meter_reading'][ df['meter'] == 'Steam' ],

                        legendgroup='No', scalegroup='No', name='Steam',

                        side='negative',

                        line_color='orange')

             )

fig.add_trace(go.Violin(x=meter_dist['primary_use'][ meter_dist['meter'] == 'ChilledWater' ],

                        y=df['meter_reading'][ df['meter'] == 'ChilledWater' ],

                        legendgroup='No', scalegroup='No', name='ChilledWater',

                        side='positive',

                        line_color='green')

             )

fig.add_trace(go.Violin(x=meter_dist['primary_use'][ meter_dist['meter'] == 'HotWater' ],

                        y=df['meter_reading'][ df['meter'] == 'HotWater' ],

                        legendgroup='No', scalegroup='No', name='HotWater',

                        side='negative',

                        line_color='red')

             )

fig.update_layout(

    title='Distribution of Meter Reading by Primary Use and Meter')

fig.show()
fig = px.box(data, x="site_id", y="air_temperature", color = 'site_id')

fig.update_layout(

    title='Distribution of Air Temperature by Site')

fig.show()
time_data = data.loc[:,['timestamp', 'meter_reading','meter','site_id','DayOfMonth','Month','DayOfWeek','Hour','primary_use']]

time_data['meter_reading'] = np.log1p(time_data['meter_reading'])

time_data.reset_index()
fig = px.box(time_data, x="primary_use", y="meter_reading")

fig.update_layout(

    title='Distribution of Meter Reading by Primary Use')

fig.show()
fig = px.box(time_data, x="DayOfWeek", y="meter_reading")

fig.update_layout(

    title='Distribution of Meter Reading by Day')

fig.show()
time_temp = time_data.groupby(['Hour']).median().reset_index()

fig = px.line(time_temp, x="Hour", y="meter_reading", title='<b>Meter Reading by Hour')

fig.show()
time_temp = time_data.groupby(['Hour','primary_use']).median().reset_index()

fig = px.line(time_temp, x="Hour", y="meter_reading", title='<b>Meter Reading by Hour with Primary Use', facet_col='primary_use', facet_col_wrap=4)

fig.show()
time_temp = time_data.groupby(['DayOfWeek','primary_use']).median().reset_index()



fig = px.box(time_data, x="meter_reading", y="DayOfWeek",facet_col='primary_use', facet_col_wrap=4)

fig.update_layout(

    title='Distribution of Meter Reading by Primary Use')

fig.update_traces(orientation='h')

fig.show()
time_temp = time_data.groupby(['timestamp']).median().reset_index()



fig = px.line(time_temp, x="timestamp", y="meter_reading", title='<b>Meter Reading by Hour')

fig.show()