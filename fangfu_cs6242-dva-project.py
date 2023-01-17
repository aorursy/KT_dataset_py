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
import seaborn as sns
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import glob
import re
import io
from scipy.stats import boxcox
df = pd.read_csv('../input/us-accidents/US_Accidents_June20.csv')
print("The shape of data is:",(df.shape))
display(df.head(5))
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df
dtype_df.groupby("Column Type").aggregate('count').reset_index()
df_source = df.groupby(['Severity','Source']).size().reset_index().pivot(\
    columns='Severity', index='Source', values=0)
df_source.plot(kind='bar', stacked=True, title='Count of Accidents by Severity by Sources')
df.groupby('Severity').size()
states = df.State.unique()

count_by_state=[]
for i in df.State.unique():
    count_by_state.append(df[df['State']==i].count()['ID'])

fig,ax = plt.subplots(figsize=(16,10))
sns.barplot(states,count_by_state)
plt.title("Count of Accidents by States", size=15, y=1.05)
plt.xlabel('States',fontsize=15)
plt.ylabel('Number of Accidents',fontsize=15)
plt.show()

fig, ax=plt.subplots(figsize=(16,7))
df['Weather_Condition'].value_counts().sort_values(ascending=False).head(10).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Weather Conditions',fontsize=20)
plt.ylabel('Number of Accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Top 10 Weather Condition for accidents',fontsize=25)
plt.grid()
plt.ioff()
start = pd.to_datetime(df.Start_Time, format='%Y-%m-%d %H:%M:%S')
end = pd.to_datetime(df.End_Time, format='%Y-%m-%d %H:%M:%S')
laps=end-start
top_15 = laps.astype('timedelta64[m]').value_counts().nlargest(15) 
#Return the first n rows ordered by columns in descending order.
print('Top 15 longest accidents correspond to {:.1f}% of the data'.format(top_15.sum()*100/len(laps)))
(top_15/top_15.sum()).plot.bar(figsize=(10,8), color = 'plum')
plt.title('Top 15 Accident Durations', fontsize = 24, color='indigo')
plt.xlabel('Duration in minutes')
plt.ylabel('% of Total Data')
plt.grid(linestyle=':', linewidth = '0.2', color ='salmon');
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])
df['Month'] = df['Start_Time'].dt.month
df['Year'] = df['Start_Time'].dt.year
df['Hour'] = df['Start_Time'].dt.hour
df['Weekday'] = df['Start_Time'].dt.weekday
df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds()/60

# clean the data based on the condition that the impact on traffic is between zero-one week,
# and drop duplicates
oneweek = 60*24*7
df_clean = df[(df['Duration']>0) & (df['Duration']< oneweek)].drop_duplicates(subset=['Start_Time','End_Time','City','Street','Number','Description'])
#time series analysis
df1 = df_clean[['Country','Start_Time','End_Time','Year','Month','Weekday','Hour','Duration','Severity']]

sns.set_style('whitegrid')
sns.set_context('talk')
sns.set_palette('GnBu_d')
a = sns.catplot(x='Year',data=df_clean[df_clean['Year'] < 2020],kind='count')
a.fig.suptitle('Yearly Accident Cases(2016-2019)',y=1.03)
a.set(ylabel='yearly cases',xlabel='year')
plt.show()
sns.set_context('talk')
m = sns.catplot(x='Month',data=df1[df1['Year'] < 2019],kind='count')
m.fig.suptitle('Monthly Accident Cases(2016-2019)',y=1.03)
m.set(ylabel='monthly cases')
plt.show()

sns.set_context('talk')
w = sns.catplot(x='Weekday',data=df1,kind='count')
w.fig.suptitle('Weekday Accident Cases',y=1.03)
w.set(ylabel='weekday cases')
plt.show()
sns.set_context('paper')
h = sns.catplot(x='Hour',data=df1,kind='count')
h.fig.suptitle('Hourly accidents cases',y=1.03)
h.set(ylabel='hourly cases',xlabel='hour')
plt.annotate('morning peak',xy=(6,330000))
plt.annotate('afternoon peak',xy=(15,270000))
plt.annotate('bottom',xy=(1,25000))
plt.annotate('go to work',xy=(7.5,0),xytext=(1,125000),arrowprops={'arrowstyle':'fancy'})
plt.annotate('get off work',xy=(17.5,0),xytext=(19,150000),arrowprops={'arrowstyle':'fancy'})
plt.show()
fig=plt.gcf()
fig.set_size_inches(20,20)
fig=sns.heatmap(df_clean.corr(),annot=True,linewidths=0.2,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
df_clean = df_clean.drop(['ID','Source','Description', 'End_Time', 'End_Lat', 'End_Lng'], axis=1)
cat_names = ['Side', 'Country', 'Timezone', 'Amenity', 'Bump', 'Crossing', 
             'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 
             'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset', 
             'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
print("Unique count of categorical features:")
for i in cat_names:
  print(i,df[i].unique().size)
df_clean = df_clean.drop(['Country','Turning_Loop'], axis=1)
df_clean.columns
print("Wind Direction: ", df_clean['Wind_Direction'].unique())
df = df_clean
df.loc[df['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
df.loc[(df['Wind_Direction']=='West')|(df['Wind_Direction']=='WSW')|(df['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
df.loc[(df['Wind_Direction']=='South')|(df['Wind_Direction']=='SSW')|(df['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
df.loc[(df['Wind_Direction']=='North')|(df['Wind_Direction']=='NNW')|(df['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
df.loc[(df['Wind_Direction']=='East')|(df['Wind_Direction']=='ESE')|(df['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
df.loc[df['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'
print("Wind Direction after simplification: ", df['Wind_Direction'].unique())
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['columns_name','missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] /df.shape[0]
missing_df.loc[missing_df['missing_ratio']>0]
missin = missing_df.loc[missing_df['missing_count']>500000]
removelist = missin['columns_name'].tolist()
removelist
df = df.drop(['TMC', 'Number','Wind_Chill(F)'], axis=1)
df['Precipitation_NA'] = 0
df.loc[df['Precipitation(in)'].isnull(),'Precipitation_NA'] = 1
df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].median())
df.loc[:5,['Precipitation(in)','Precipitation_NA']]
df = df.dropna(subset=['City','Zipcode','Airport_Code',
                       'Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'])
# group data by 'Airport_Code' and 'Start_Month' then fill NAs with median value
Weather_data=['Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)']
print("The number of remaining missing values: ")
for i in Weather_data:
  df[i] = df.groupby(['Airport_Code','Month'])[i].apply(lambda x: x.fillna(x.median()))
  print( i + " : " + df[i].isnull().sum().astype(str))
df = df.dropna(subset=Weather_data)
# group data by 'Airport_Code' and 'Start_Month' then fill NAs with majority value
from collections import Counter
weather_cat = ['Weather_Timestamp','Wind_Direction','Weather_Condition']
print("Count of missing values that will be dropped: ")
for i in weather_cat:
  df[i] = df.groupby(['Airport_Code','Month'])[i].apply(lambda x: x.fillna(Counter(x).most_common()[0][0]) if all(x.isnull())==False else x)
  print(i + " : " + df[i].isnull().sum().astype(str))

# drop na
df = df.dropna(subset=weather_cat)
df = df.drop(['Start_Time'], axis=1)
df['Pressure_bc']= boxcox(df['Pressure(in)'].apply(lambda x: x+1),lmbda=6)
df['Visibility_bc']= boxcox(df['Visibility(mi)'].apply(lambda x: x+1),lmbda = 0.1)
df['Wind_Speed_bc']= boxcox(df['Wind_Speed(mph)'].apply(lambda x: x+1),lmbda=-0.2)
df = df.drop(['Pressure(in)','Visibility(mi)','Wind_Speed(mph)'], axis=1)
df = df.drop(['Street','Zipcode','Weather_Timestamp',
              'Airport_Code','Civil_Twilight', 'Nautical_Twilight', 
              'Astronomical_Twilight', 'Month', 'Year'], axis=1)
df.columns
df.info()
df.to_csv('/kaggle/working/df.csv')
# Generate dummies for categorical data
#df_dummy = pd.get_dummies(df,drop_first=True)

# Export data

#df_dummy.info()
#X = df.drop('Severity',axis=1)
#y = df['Severity']

# Standardizing the features based on unit variance
#from sklearn.preprocessing import StandardScaler

# split train test
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(\
 # X, y, test_size=0.20, random_state=42, stratify=y)
# Logistic regression
#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression(random_state=0)
#lr.fit(X_train,y_train)
#y_pred=lr.predict(X_test)

# Get the accuracy score
#acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
#accuracy_lst.append(acc)

#print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))

