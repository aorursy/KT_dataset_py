# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting 
%matplotlib inline
import missingno as msn #for missing value
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# reading data from csv file and saving into df dataframe
df = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')

df.describe()
#checking how many year are present in dataframe 
df['Year'].unique()
#checking type basic information about features and attribute
df.info()
#sience there is 201 and 200 are in ou year column and it is not an year so we have to remove it.
df.drop(df[df['Year'] <= 201].index,inplace=True)

#removing Average temperature outlier
df.drop(df[df['AvgTemperature'] <= -94].index,inplace=True)
#converting different date column in single datetime format
df['Date']=df['Day'].astype(str)+'-'+df['Month'].astype(str)+'-'+df['Year'].astype(str)
df.drop(['Day','Month','Year'],inplace=True,axis=1)
df['Date']=pd.to_datetime(df['Date'])
# setting  the datetime column into datetime index for further time series analysis
df = df.set_index('Date')
#checking datetime index
df.index
df.head()
#ploting average temprature of different region

chart = sns.lineplot(x='Region',y='AvgTemperature',data=df)
for i in chart.get_xticklabels():
    i.set_rotation(45)
plt.show()
plt.tight_layout()
# segmenting data on the basis of Year,Region,and Country
Region_year = df.groupby(['Region','Country',pd.Grouper(freq = 'Y')]).mean().reset_index()
Region_year.head()
#checking year wise temp trend of differnt Region
fig,ax = plt.subplots(figsize=(10,6))

ax.set_xticks(Region_year['Date'].unique())
Region_year.groupby(["Date", "Region"]).mean()['AvgTemperature'].unstack().plot(ax=ax)
ax.legend(bbox_to_anchor=(1.1, 1.05))
plt.show()
# segment on the basis of country and year
Africa = Region_year[Region_year['Region']=='Africa'].set_index('Date').groupby(['Country',pd.Grouper(freq='Y')]).mean().reset_index()
Africa.set_index('Date').groupby(['Country',pd.Grouper(freq="Y")]).mean().reset_index().head()
#plotting different country and temp
plt.figure(figsize=(10,10))

sns.set(style="darkgrid")
chart = sns.lineplot(x="Country", y="AvgTemperature",
             data=Africa,markers='o')

for i in chart.get_xticklabels():
    i.set_rotation(90)

Asia = Region_year[Region_year['Region']=='Asia'].set_index('Date').groupby(['Country',pd.Grouper(freq='Y')]).mean().reset_index()
Asia.set_index('Date').groupby(['Country',pd.Grouper(freq="Y")]).mean().reset_index().head()
Asia.max()
plt.figure(figsize=(10,10))

sns.set(style="darkgrid")
chart = sns.lineplot(x="Country", y="AvgTemperature",
             data=Asia,markers='o')

for i in chart.get_xticklabels():
    i.set_rotation(90)
South_pac = Region_year[Region_year['Region']=='Australia/South Pacific'].set_index('Date').groupby(['Country',pd.Grouper(freq='Y')]).mean().reset_index()
South_pac.set_index('Date').groupby(['Country',pd.Grouper(freq="Y")]).mean().reset_index().head()
plt.figure(figsize=(10,10))

sns.set(style="darkgrid")
chart = sns.lineplot(x="Country", y="AvgTemperature",
             data=South_pac,markers='o')

for i in chart.get_xticklabels():
    i.set_rotation(90)
Europe = Region_year[Region_year['Region']=='Europe'].set_index('Date').groupby(['Country',pd.Grouper(freq='Y')]).mean().reset_index()
Europe.set_index('Date').groupby(['Country',pd.Grouper(freq="Y")]).mean().reset_index().head()
plt.figure(figsize=(10,10))

sns.set(style="darkgrid")
chart = sns.lineplot(x="Country", y="AvgTemperature",
             data=Europe,markers='o')

for i in chart.get_xticklabels():
    i.set_rotation(90)
Middle_East = Region_year[Region_year['Region']=='Middle East'].set_index('Date').groupby(['Country',pd.Grouper(freq='Y')]).mean().reset_index()
Middle_East.set_index('Date').groupby(['Country',pd.Grouper(freq="Y")]).mean().reset_index().head()
plt.figure(figsize=(10,10))

sns.set(style="darkgrid")
chart = sns.lineplot(x="Country", y="AvgTemperature",
             data=Middle_East,markers='o')

for i in chart.get_xticklabels():
    i.set_rotation(90)
North_America = Region_year[Region_year['Region']=='North America'].set_index('Date').groupby(['Country',pd.Grouper(freq='Y')]).mean().reset_index()
North_America.set_index('Date').groupby(['Country',pd.Grouper(freq="Y")]).mean().reset_index().head()
plt.figure(figsize=(10,10))

sns.set(style="darkgrid")
chart = sns.lineplot(x="Country", y="AvgTemperature",
             data=North_America,markers='o')

for i in chart.get_xticklabels():
    i.set_rotation(90)
South_Central = Region_year[Region_year['Region']=='South/Central America & Carribean'].set_index('Date').groupby(['Country',pd.Grouper(freq='Y')]).mean().reset_index()
South_Central.set_index('Date').groupby(['Country',pd.Grouper(freq="Y")]).mean().reset_index().head()
plt.figure(figsize=(10,10))

sns.set(style="darkgrid")
chart = sns.lineplot(x="Country", y="AvgTemperature",
             data=South_Central,markers='o')

for i in chart.get_xticklabels():
    i.set_rotation(90)
