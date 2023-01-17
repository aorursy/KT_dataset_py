# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import datetime



weather = pd.read_csv('../input/weather-dataset-in-antwerp-belgium/weather_in_Antwerp.csv', ';')

power_info = pd.read_csv('../input/solarpanelspower/PV_Elec_Gas2.csv')

display(weather.head())

power_info.head()
power_info = power_info[['Unnamed: 0','cum_power']]

power_info = power_info.rename(columns= {'Unnamed: 0': 'date'})

power_info.info()
power_info.date = pd.to_datetime(power_info.date)

power_info.set_index(['date'], inplace=True)       #change the index

power_info.head()
power_info = power_info.shift(periods=-1, freq='D', axis=0)   #Correcting the measure error

                                                        # (mentioned in Frank's data description)

    

#Calculating daily power, because we have the cumulative one

temp = power_info.shift(periods=1, freq='D', axis=0)

power_info['day_power'] = power_info.loc[:, 'cum_power'] - temp.loc[:, 'cum_power']

power_info.drop(['cum_power'], axis=1, inplace=True)

power_info.day_power.iloc[0] = 5

power_info.head()
import seaborn as sns

from matplotlib import pyplot as plt



sns.set()

power_index= power_info.reset_index()

power_index.plot(kind='line', x='date', y='day_power', figsize=(15,5))



plt.title('Daily Power Produced By Solar Panels')

plt.ylabel('Daily Power')

plt.show()
def clear_wind(obj):

    if isinstance(obj, str):

        if obj == 'No wind':

            obj = 0

        else:

            obj = obj.replace(' km/h', '')

    return obj

def trans_from_objects(weather):

    weather.drop(['Unnamed: 0'], axis =1, inplace=True)

    

    #try statement is here for the future weather, 

    #as it is without barometer on the site

    try:

        weather.barometer = weather.barometer.apply(lambda x: x.replace(' mbar', '') 

                                    if isinstance(x, str) else x).astype(float)

        weather.drop(['visibility'], axis =1, inplace=True)

    except AttributeError:

        pass

    

    weather.humidity = weather.humidity.apply(lambda x: x.replace('%', '') 

                                    if isinstance(x, str) else x).astype(float)

    weather.temp = weather.temp.apply(lambda x: x.replace('°C', '') 

                                    if isinstance(x, str) else x).astype(float)

    weather.wind = weather.wind.apply(clear_wind).astype(float)

    

    return weather



#transfer dataframe from objects dtype to numbers

weather_tran = trans_from_objects(weather)

weather_tran.head()
weather_tran.info()
#Form the date column 

def create_date(weather):    

    weather['date'] = weather.apply(lambda row:

                                    f'{row.year}-{row.month}-{row.day} {row.clock}', axis=1)

    weather.date = pd.to_datetime(weather.date)

    return weather.drop(['clock', 'year', 'month', 'day'], axis = 1)



weather_pretty = create_date(weather_tran)

weather_pretty.head()
#to take the average of each day, so we have daily weather. Because we have the daily cum_power not hourly

def take_average_weather(weather, future = False):

    if future == False:

        average_weather = pd.DataFrame(columns = ['temp', 'weather', 'wind', 'humidity', 'barometer',

                                              'date'])

    else:

        average_weather = pd.DataFrame(columns = ['temp', 'weather', 'wind', 'humidity','date'])

    

    temp, wind, humidity, barometer, counter= [0]*5

    for i in range(len(weather)):

        if future == False:

            if (weather.loc[i, 'date'].time() ==datetime.time(0, 20)) and (i!=0):

                average_weather = average_weather.append({

                    'temp':temp/counter,

                    'wind':wind/counter,

                    'humidity':humidity/counter,

                    'barometer':barometer/counter,

                    'date':pd.to_datetime(weather.loc[i-1, 'date'].date()),

                    'weather':weath

                }, ignore_index=True)

                temp, wind, humidity, barometer, counter= [0]*5



            #Here we'll take the weather status in the most powerful hour (15:20), because you can't take averge 

                                                                                                        #here.

            if (weather.loc[i, 'date'].time()==datetime.time(15,20)):

                weath = weather.loc[i, 'weather']

        else:

            # or i==len(weather)-1 , so the last day in the data been appended

            if ((weather.loc[i, 'date'].time() ==datetime.time(0, 0)) and (i!=0)) or (i==len(weather)-1):

                average_weather = average_weather.append({

                    'temp':temp/counter,

                    'wind':wind/counter,

                    'humidity':humidity/counter,

                    'date':pd.to_datetime(weather.loc[i-1, 'date'].date()),

                    'weather':weath

                }, ignore_index=True)

                temp, wind, humidity, barometer, counter= [0]*5



            #Here we'll take the weather status in the most powerful hour (15:20),

            #because you can't take averge with categories.

            if (weather.loc[i, 'date'].time()==datetime.time(15,0)):

                weath = weather.loc[i, 'weather']

        counter += 1

        temp += weather.loc[i, 'temp']

        wind += weather.loc[i, 'wind']

        humidity += weather.loc[i, 'humidity']

        if future == False:

            barometer += weather.loc[i, 'barometer']

        

    return average_weather

average_weather = take_average_weather(weather_pretty)
def merge_weatherANDpower():

    dataset = average_weather.merge(power_info, on=['date'])

    return dataset.set_index('date')

final_dataset = merge_weatherANDpower()

final_dataset.head()
import seaborn as sns

weather_counts = final_dataset.weather.value_counts()

plt.figure(figsize=(16,5))

sns.barplot(weather_counts.index, weather_counts.values, alpha=0.8)

plt.xticks(rotation=90)

plt.title('Weather Status')

plt.xlabel('Status')

plt.ylabel('Number Of Repetition')

plt.show() # WHAT THE HECK! Let's reduce this amount of redundant information
#I need this, so I can deal with "loc"

final_dataset = final_dataset.reset_index()



def reduce_categories(weather):

    #Delete all first parts of two-part status, and highligh only the necessary categories. 

    #why the first part? Because we don't care about the raining or snowing weather, we care more about 

    #status of clouds

    for i in range(len(weather)):

        weather_list = weather.loc[i, 'weather'].split('.')

        if len(weather_list) > 2:

            weather.loc[i,'weather'] = weather_list[1].strip()

        elif len(weather_list) ==2:

            weather.loc[i, 'weather'] = weather_list[0].strip()



    weather.weather = weather.weather.map({

        'Ice fog':'Fog',

        'Haze':'Fog',

        'Fog':'Fog',

        'Clear':'Sunny',

        'Sunny':'Sunny',

        'Broken clouds':'Scattered clouds',

        'Scattered clouds':'Scattered clouds',

        'Overcast':'Cloudy',

        'More clouds than sun':'Cloudy',

        'More sun than clouds':'Sunny',

        'Low clouds':'Cloudy',

        'Mostly cloudy':'Cloudy',

        'Cloudy':'Cloudy',

        'Passing clouds':'Passing clouds',

        'Partly sunny':'Partly sunny',

        'Mostly sunny':'Sunny'

    },na_action='ignore')

    return weather

final_dataset = reduce_categories(final_dataset)



#get the index back to "date"

final_dataset.set_index('date', inplace=True)

from matplotlib import pyplot as plt

final_dataset.weather.value_counts()

weather_counts = final_dataset.weather.value_counts()



plt.figure(figsize=(12,6))

sns.barplot(weather_counts.index, weather_counts.values, alpha=0.8)

plt.xticks(rotation=33)

plt.title('Weather Status')

plt.xlabel('Status')

plt.ylabel('Number Of Repetition')

plt.show()
final_dataset.info()
final_dataset.hist(figsize=(16,12))

plt.show()
from sklearn.model_selection import train_test_split 

train_set, test_set = train_test_split(final_dataset, test_size=0.2, 

                                                   random_state=42) 

df = train_set.copy() 

df.describe() 
df.corr() 
from pandas.plotting import scatter_matrix 

scatter_matrix(df, figsize=(16,18), alpha=0.4) 

plt.show()
df.plot(kind='scatter', x= 'humidity',y='day_power', figsize=(9,7), alpha=0.4) 

plt.show()
#To delete data anomalies

import random

df.day_power = df.day_power.apply(lambda x: x+random.randint(0,50)/100 if x==0 else x)

for i in range(1,34):

    df.day_power = df.day_power.apply(lambda x: x+random.randint(-50,50)/100 if x==i else x)

df.plot(kind='scatter', x= 'humidity',y='day_power', figsize=(9,7), alpha=0.4) 

plt.show()
#As we have data (which we want to predict), without barometer column

features = df.drop(['day_power', 'barometer'], axis=1) 

columns=features.columns 

labels = df['day_power'].copy() 

features.head() 
num_attr = list(features.drop(['weather'],axis=1)) 

cat_attr = ['weather'] 
from sklearn.pipeline import Pipeline 

from sklearn.compose import ColumnTransformer 

from sklearn.preprocessing import OneHotEncoder 

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import MinMaxScaler 





num_pipeline = Pipeline([ 

    ('imputer', SimpleImputer(strategy='median')), 

    ('scaler', MinMaxScaler()) 

]) 



cat_pipeline = Pipeline([ 

    ('encoder', OneHotEncoder()) 

]) 



full_pipeline = ColumnTransformer([ 

    ('num_pipeline', num_pipeline, num_attr),

    ('cat_pipeline', cat_pipeline, cat_attr) 

]) 
prepared_features = full_pipeline.fit_transform(features) 
import numpy as np 



from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error as mse 

from sklearn.model_selection import cross_val_score  



lin_reg = LinearRegression() 

lin_reg.fit(prepared_features, labels) 

y_predicted = lin_reg.predict(prepared_features) 

scores = cross_val_score(lin_reg, prepared_features, labels, 

                         scoring='neg_mean_squared_error', cv=10) 

scores=np.sqrt(-scores) 

display(scores.mean()) 

scores.std() 
test = test_set.copy()

test_features = test.drop(['day_power', 'barometer'], axis=1) 

test_labels = test['day_power'].copy() 
prepared_test = full_pipeline.transform(test_features) 

test_predicted = lin_reg.predict(prepared_test)
scores = cross_val_score(lin_reg, prepared_test, test_labels, 

                         scoring='neg_mean_squared_error', cv=10) 

scores=np.sqrt(-scores) 

display(scores.mean()) 

scores.std() 
#Visualizing the difference between predicted and real values of day power for the test set



avg=[]

labels_avg = []

for i in range(len(test_labels)):

    avg.append(test_labels[i])

    if i % 40 == 0:

        labels_avg.append(np.array(avg).mean())

        avg.clear()

avg=[]

pred_avg = []

for i in range(len(test_predicted)):

    avg.append(test_predicted[i])

    if i % 40 == 0:

        pred_avg.append(np.array(avg).mean())

        avg.clear()
plt.figure(figsize=(16,6))

plt.plot(range(len(labels_avg)), labels_avg)

plt.plot(range(len(pred_avg)), pred_avg, 'r')

plt.title('Comparison between average predicted values and real ones in the test set')

plt.ylabel('Day Power')

plt.xlabel('Average Of Test Samples')

plt.legend(['Real Power', 'Predicted Power'])

plt.show()
weather_future= pd.read_csv('../input/weather-dataset-in-antwerp-belgium/weather_in_Antwerp_future2.csv', ';')
def predict_future_data(data):

    tran_fut = trans_from_objects(data)

    tran_fut = create_date(tran_fut)

    avg_fut = take_average_weather(tran_fut, future=True)

    red_fut = reduce_categories(avg_fut)

    red_fut = red_fut.set_index('date')

    prepared_future = full_pipeline.transform(red_fut)

    return red_fut.index, lin_reg.predict(prepared_future)



date, predicted_data = predict_future_data(weather_future)
plt.figure(figsize=(16,6))

plt.plot(date,predicted_data)

plt.title('Next Days Prediction')

plt.ylabel('Day Power')

plt.xlabel('Date')

plt.rcParams['xtick.labelsize']=14

plt.rcParams['ytick.labelsize']=14

plt.xticks(rotation=15)

plt.show()