import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib as mpl

import pickle

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
PATH = '../input/ashrae-energy-prediction/'

train_df = pd.read_csv(PATH + 'train.csv')

building_meta_df=pd.read_csv(PATH + 'building_metadata.csv')

weather_train_df=pd.read_csv(PATH + 'weather_train.csv')

weather_test_df=pd.read_csv(PATH + 'weather_test.csv')

test_df = pd.read_csv(PATH + '/test.csv')
print('Size of train data', train_df.shape)

print('Size of weather_train_df data', weather_train_df.shape)

print('Size of building_meta_df data', building_meta_df.shape)



print('Size of test data', test_df.shape)

print('Size of weather test data', weather_test_df.shape)







print('Dataset completo: ', train_df.shape)

train_df= train_df.sample(frac=0.1, random_state=0)

print('Porzione ridotta per limiti computazionali:', train_df.shape)

train_df.head()
train_df.columns.values
weather_train_df.columns.values
building_meta_df.columns.values
total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)

missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing__train_data.head(5)
total = weather_train_df.isnull().sum().sort_values(ascending = False)

percent = (weather_train_df.isnull().sum()/weather_train_df.isnull().count()*100).sort_values(ascending = False)

missing_weather_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_weather_data.head(5)
total = building_meta_df.isnull().sum().sort_values(ascending = False)

percent = (building_meta_df.isnull().sum()/building_meta_df.isnull().count()*100).sort_values(ascending = False)

missing_building_meta_df  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_building_meta_df.head(5)
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])

train_df["hour"] = train_df["timestamp"].dt.hour

train_df["day"] = train_df["timestamp"].dt.day

train_df["weekend"] = train_df["timestamp"].dt.weekday

train_df["month"] = train_df["timestamp"].dt.month

train_df["year"] = train_df["timestamp"].dt.year



weather_train_df["timestamp"] = pd.to_datetime(weather_train_df["timestamp"])



test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

test_df["hour"] = test_df["timestamp"].dt.hour

test_df["day"] = test_df["timestamp"].dt.day

test_df["weekend"] = test_df["timestamp"].dt.weekday

test_df["month"] = test_df["timestamp"].dt.month

test_df["year"] = train_df["timestamp"].dt.year

weather_test_df["timestamp"] = pd.to_datetime(weather_test_df["timestamp"])

building_meta_df['year_built'] = building_meta_df['year_built'].astype('Int64')

train_df['meter_label']=train_df['meter'].apply(lambda x: {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}.get(x,x))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

building_meta_df["primary_use_label"] = le.fit_transform(building_meta_df["primary_use"])
train_df = train_df.merge(building_meta_df, on='building_id', how='left')

train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

test_df = test_df.merge(building_meta_df, on='building_id', how='left')

test_df = test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

print('Size of train_df data', train_df.shape)

print('Size of weather_train_df data', weather_train_df.shape)

print('Size of building_meta_df data', building_meta_df.shape)
## Using IQR



def outlier_treatment(datacolumn):

    

    

    Q1,Q3 = np.percentile(sorted(datacolumn) , [25,75])

    print ("Q1:",Q1)

    print ("Q3:",Q3)

    IQR = Q3 - Q1

    print ("IQR",IQR)

    lower_range = Q1 - (3 * IQR)

    upper_range = Q3 + (3 * IQR)

    

    return lower_range,upper_range

  

l,u = outlier_treatment(train_df.meter_reading)

print(train_df [~(((train_df.meter_reading)> u) | ((train_df.meter_reading) < l)) ].shape)

df=train_df [~(((train_df.meter_reading)> u) | ((train_df.meter_reading) < l)) ]
df[df["building_id"] == 5].plot("timestamp", "meter_reading",figsize=(5,5))
df['timestamp'].value_counts().sort_index().plot()
df['air_temperature'].value_counts().sort_index().plot()
df['year_built'].value_counts().sort_index().plot()
df['building_id'].value_counts().sort_index().plot()
plt.figure(figsize=(16, 6))

g=sns.countplot(x="month", hue="meter_label", data=df).set_title("Monthly metering by meter label")
plt.figure(figsize=(16, 6))

g=sns.barplot(x="month", y="meter_reading",hue="meter_label", data=df).set_title("Monthly consumption by meter label")
g = sns.FacetGrid(df, hue="meter_label", col="primary_use", col_wrap=4 )

g.map(sns.barplot, "month","meter_reading")

g.add_legend()
def countplot(x, hue, **kwargs):

    sns.countplot(x=x, hue=hue, **kwargs)



grid = sns.FacetGrid(data=df,col='primary_use',col_wrap=4,aspect=1)

fig = grid.map(countplot,'month','meter_label',palette='Set1')

fig.add_legend()
g = sns.FacetGrid(data=df, hue="meter_label", col="primary_use", col_wrap=4, )



g.map(sns.lineplot, "month","meter_reading")

g.add_legend()

g.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
g = sns.FacetGrid(data=df, hue="meter_label", col="primary_use", col_wrap=4 )

#g = g.map(plt.scatter, "month", "meter_reading", alpha=.7)

g.map(sns.lineplot, "hour","meter_reading")

g.add_legend()