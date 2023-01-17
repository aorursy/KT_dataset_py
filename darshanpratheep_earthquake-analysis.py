import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

df = pd.read_csv('../input/earthquake-dataset-us19652016/earthquake.csv')



# to suppress warnings

import warnings

warnings.filterwarnings('ignore')
#%matplotlib qt

%matplotlib inline

import plotly

plotly.offline.init_notebook_mode()

import plotly.express as px

import plotly.graph_objects as go
df.info()
df.head(5)
df.tail(10)
df.describe()
print(len(df))

print(df.shape)
print(list(df.columns))
df['Type'].value_counts()
classification = pd.DataFrame({'Class': ['Great','Major','Strong','Moderate','Light','Minor'],

                              'Magnitude': ['8 or more','7 - 7.9','6 - 6.9','5 - 5.9','4 - 4.9','3 - 3.9']})

classification
print("The number of Earthquakes classified as \"Great\" :",len(df[(df['Magnitude']>=8.0) & (df['Type']=='Earthquake')]))

print("The number of Earthquakes classified as \"Major\" :",len(df[(df['Magnitude']>=7.0) & (df['Magnitude']<=7.9) & (df['Type']=='Earthquake')]))

print("The number of Earthquakes classified as \"Strong\" :",len(df[(df['Magnitude']>=6.0) & (df['Magnitude']<7.0) & (df['Type']=='Earthquake')]))

print("The number of Earthquakes classified as \"Moderate\" :",len(df[(df['Magnitude']>=5.0) & (df['Magnitude']<6.0) & (df['Type']=='Earthquake')]))

print("The number of Earthquakes classified as \"Light\" :",len(df[(df['Magnitude']>=4.0) & (df['Magnitude']<5.0) & (df['Type']=='Earthquake')]))

print("The number of Earthquakes classified as \"Minor\" :",len(df[(df['Magnitude']>=3.0) & (df['Magnitude']<4.0) & (df['Type']=='Earthquake')]))
print("Number of Earthquakes in the Northern Hemisphere:",len(df[(df['Latitude']>=0.0) & (df['Type']=='Earthquake')]))

print("Number of Earthquakes in the Southern Hemisphere:",len(df[(df['Latitude']<0.0) & (df['Type']=='Earthquake')]))

print()

print("Number of Nuclear Explosions in the Northern Hemisphere:",len(df[(df['Latitude']>=0.0) & (df['Type']=='Nuclear Explosion')]))

print("Number of Nuclear Explosions in the Southern Hemisphere:",len(df[(df['Latitude']<0.0) & (df['Type']=='Nuclear Explosion')]))

print()

print("Number of Explosions in the Northern Hemisphere:",len(df[(df['Latitude']>=0.0) & (df['Type']=='Explosion')]))

print("Number of Explosions in the Southern Hemisphere:",len(df[(df['Latitude']<0.0) & (df['Type']=='Explosion')]))

print()

print("Number of Rock Bursts in the Northern Hemisphere:",len(df[(df['Latitude']>=0.0) & (df['Type']=='Rock Burst')]))

print("Number of Rock Bursts in the Southern Hemisphere:",len(df[(df['Latitude']<0.0) & (df['Type']=='Rock Burst')]))
print("Number of Earthquakes in the Eastern Hemisphere:",len(df[(df['Longitude']>=0.0) & (df['Type']=='Earthquake')]))

print("Number of Earthquakes in the Western Hemisphere:",len(df[(df['Longitude']<0.0) & (df['Type']=='Earthquake')]))

print()

print("Number of Nuclear Explosions in the Eastern Hemisphere:",len(df[(df['Longitude']>=0.0) & (df['Type']=='Nuclear Explosion')]))

print("Number of Nuclear Explosions in the Western Hemisphere:",len(df[(df['Longitude']<0.0) & (df['Type']=='Nuclear Explosion')]))

print()

print("Number of Explosions in the Eastern Hemisphere:",len(df[(df['Longitude']>=0.0) & (df['Type']=='Explosion')]))

print("Number of Explosions in the Western Hemisphere:",len(df[(df['Longitude']<0.0) & (df['Type']=='Explosion')]))

print()

print("Number of Rock Bursts in the Eastern Hemisphere:",len(df[(df['Longitude']>=0.0) & (df['Type']=='Rock Burst')]))

print("Number of Rock Bursts in the Western Hemisphere:",len(df[(df['Longitude']<0.0) & (df['Type']=='Rock Burst')]))
top_ten_eq = df[['Type','Magnitude']].sort_values(by='Magnitude',ascending = False).head(10)

top_ten_eq[top_ten_eq['Type']=='Earthquake']
top_ten_ne = df[df['Type']=='Nuclear Explosion']

top_ten_ne[['Type','Magnitude']].sort_values(by='Magnitude',ascending=False).head(10)
top_e = df[df['Type']=='Explosion']

top_e[['Type','Magnitude']].sort_values(by='Magnitude',ascending=False).head(10)
df[df['Type']=='Rock Burst'][['Type','Magnitude']]
df['Date'] = df['Date'] +  " " + df['Time']

del df['Time']

df.head()
df['Date'] = pd.to_datetime(df['Date'],errors = 'coerce')

df.head()

selected_date = df[(df['Date']>='2016-06-01 15:00:00') & (df['Date']<='2016-06-01 23:00:00')]

selected_date
print(selected_date['Type'].value_counts())
for i in selected_date[['Latitude','Longitude']]:

    print(selected_date[i])
for i in selected_date['ID']:

    print(i)
#finding missing values

missing_values_count = df.isnull().sum()

missing_values_count
count_of_missing_values = missing_values_count.sum()

print('Total number of missing values : ',count_of_missing_values)
total_values = np.product(df.shape)

total_missing = missing_values_count.sum()

print('percentage of missing value : %d'%((total_missing/total_values)*100)+'%')
droping =df.dropna(axis =1)

droping.head()
print("Columns in original dataset: %d \n" %df.shape[1])

print("Columns with na's dropped: %d" % droping.shape[1])
# get a small subset from  the  dataset

subset_data = df.loc[:, 'Depth Error':'Root Mean Square']

subset_data
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

bfill = subset_data.fillna(method = 'bfill', axis=0).fillna(0)

bfill
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

ffill =subset_data.fillna(method = 'ffill', axis=0).fillna(0)

ffill
# replace all NAN's with 0

subset_data.fillna(0)
bfill.isnull().sum()
ffill.isnull().sum()
subset_data.isnull().sum()
# getting row number 0 with all its columns

df.loc[0,:]
# Obtaining the columns of first 3 rows

df.loc[[0,1,2],:]
# Using loc function in terms of range

df.loc[0:5,:]
# To retrieve values of specified single column

df.loc[0:10,'Magnitude']
# To retrieve the values of specified multiple columns

df.loc[0:10,['Latitude', 'Longitude', 'Magnitude']]
# To retrieve the columns using range function

df.loc[0:10,'Date':'Magnitude']
df.columns
# To obtain the required columns in range

df.iloc[:,0:3]
f1 = pd.read_csv('../input/earthquake-dataset-us19652016/earthquake.csv')
f1.pivot_table(index = "Type", values = "Magnitude", aggfunc = max)
f1.pivot_table(index = "Type", values = "Magnitude", aggfunc = min)
df.columns
df = df[['Date', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

df.head()
df['Date'].head()
df['Date'].tail()
df['date_parsed'] = pd.to_datetime(df['Date'], infer_datetime_format=True, utc=True)

df['date_parsed']
day_of_year_earthquakes = df['date_parsed'].dt.year

day_of_year_earthquakes
# remove na's

day_of_year_earthquakes = day_of_year_earthquakes.dropna()



# plot the day of the month (count x dates)

sns.distplot(day_of_year_earthquakes, kde=False, axlabel='Years', color='r', vertical=False)
day_of_year_earthquakes.value_counts()
day_of_month_earthquakes = df['date_parsed'].dt.month

day_of_month_earthquakes
# remove na's

day_of_month_earthquakes = day_of_month_earthquakes.dropna()



# plot the day of the month (count x dates)

sns.distplot(day_of_month_earthquakes, kde=False, axlabel='Months', color = 'g')
day_of_month_earthquakes.value_counts()
day_of_date_earthquakes = df['date_parsed'].dt.day

day_of_date_earthquakes
# remove na's

day_of_date_earthquakes = day_of_date_earthquakes.dropna()



# plot the day of the month (count x dates)

sns.distplot(day_of_date_earthquakes, kde=False, bins=31, axlabel='Dates', color = 'r')
day_of_date_earthquakes.value_counts()

#day_of_date_earthquakes.value_counts().sort_index()   #sorted by index
plt.xlabel("Depth of magnitude")

plt.ylabel("Count")

plt.hist(x=df.Magnitude)
df.dropna()
sns.boxplot(x=df.Depth)
fig=px.box(data_frame=df,y='Magnitude')

fig.show()
fig=px.histogram(data_frame=df,y='Depth',nbins=10)

fig.show()
fig=px.scatter(data_frame=df,x='Magnitude',y='Depth')

fig.show()
sns.regplot(df.Magnitude,df.Depth)
df2=pd.cut(df.Magnitude,[5,6,7,8,9])
vc=df2.value_counts()
vc.plot(kind='pie',explode=[0,0,1,0],autopct='%.f',radius=1.5)