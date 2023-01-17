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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
filename = "/kaggle/input/indian-metro-data/Train.csv"

data = pd.read_csv(filename)
# Checking the first 5 rows of the dataset to get an overview of the dataset

data.head()
#Checking the null values in each coulmn of the dataset

data.isnull().sum()
# Checking the types for each column 

data.dtypes
data.date_time = pd.to_datetime(data.date_time)

#By using to_datetime method we have converted the object type date_time into a proper date and time format which is going to help us further in data visualization
data['year'] = data['date_time'].dt.year

data['month'] = data['date_time'].dt.month

data['day'] = data['date_time'].dt.day

data['dayofweek'] = data['date_time'].dt.dayofweek.replace([0,1,2,3,4,5,6],['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])

data['hour'] = data['date_time'].dt.hour

data.head()


data.describe()
sns.scatterplot(x = data['hour'], y = data['traffic_volume'])

plt.xlabel("Hour")

plt.ylabel("Traffic Volume")
plt.figure(figsize=(10,7))

sns.lineplot(x=data['hour'],y=data['traffic_volume'],hue=data['dayofweek']);

plt.title("Traffic volume and Hour on different days of the week");

plt.xlabel("Hour");

plt.ylabel("Traffic volume");
plt.figure(figsize=(10,7))

sns.lineplot(x=data['month'],y=data['traffic_volume']);

plt.title("Traffic Volume vs Month");

plt.xlabel("Month");

plt.ylabel("Traffic Volume");
weather = data.groupby('weather_type')

data.boxplot('traffic_volume',by = 'month');

title_boxplot="Traffic volume Grouped by Month"

plt.title(title_boxplot)

plt.suptitle('')

plt.xlabel("Month");

plt.ylabel("Traffic Volume");
data.boxplot('humidity',by = 'month');

title_boxplot="Humidity for every Month"

plt.title(title_boxplot)

plt.suptitle('')

plt.xlabel('Month');

plt.ylabel('Humidity');
g=sns.FacetGrid(col='month',data=data,col_wrap=4,height=2)

g.map(plt.scatter,"weather_type","clouds_all")

g.set_xticklabels(rotation=90)

plt.tight_layout()

plt.xlabel("Weather Type");

plt.ylabel("Clouds Percentage")
data['weather_type'].value_counts()
def tempconvert(t):

    return t-273.15
data['temperature']=data['temperature'].apply(tempconvert)

#By using function of tempconvert we have converted the temperature from kelvin to Celcius for our better understanding.
sns.barplot(y='temperature',x='month',data=data);

plt.title("Temperature per month");

plt.xlabel("Month");

plt.ylabel("Temperature");
plt.figure(figsize=(6,6))

sns.distplot(data['humidity']);

plt.title("Humidity Distribution");
plt.figure(figsize=(8,8))

sns.heatmap(data.corr(),linewidth=1);
plt.figure(figsize=(10,7))

sns.lineplot(x=data['year'],y=data['traffic_volume'])

plt.title('Traffic volume per year');

plt.xlabel("Year");

plt.ylabel("Traffic Volume");
plt.figure(figsize=(10,7))

sns.lineplot(x=data['year'],y=data['air_pollution_index']);

plt.title('Air pollution per year');

plt.xlabel("Year");

plt.ylabel("Air Pollution Index");
plt.figure(figsize=(10,7))

sns.lineplot(x=data['month'],y=data['temperature']);

plt.title("Temeperature per month");

plt.xlabel("Month");

plt.ylabel("Temperature");
plt.figure(figsize=(10,7))

sns.lineplot(x = data['month'],y=data['air_pollution_index']);

plt.xlabel("Month");

plt.ylabel("Air Pollution Index");

plt.title("Air Pollution Index per month");
plt.figure(figsize=(10,6))

sns.lineplot(x=data['month'],y=data['rain_p_h']);

plt.title("Rain per hour per month");

plt.xlabel("Month");

plt.ylabel("Rain Per hour(mm)");
plt.figure(figsize=(10,6))

sns.lineplot(x=data['year'],y=data['rain_p_h']);

plt.title("Rain per hour per year");

plt.xlabel("Year");

plt.ylabel("Rain per hour(mm)");
plt.figure(figsize=(10,6))

sns.lineplot(x=data['month'],y=data['snow_p_h']);

plt.xlabel("Month");

plt.ylabel("Snow per hour(mm)");

plt.title("Snow per hour per month");
plt.figure(figsize=(10,8))

sns.lineplot(x=data['year'],y=data['snow_p_h']);

plt.title("Snow per hour per year");

plt.xlabel("Year");

plt.ylabel("Snow per hour");