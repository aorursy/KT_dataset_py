# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline



import seaborn as sns



from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the dataset

acc_df = pd.read_csv('../input/us-accidents/US_Accidents_May19.csv')
# First Look into the data

print('Rows, Columns - ', acc_df.shape)

acc_df.head()
# Get Attribute and Datatype Information

acc_df.info()
# Data represents how many States ?

print(f"Data represents {acc_df['State'].unique().shape[0]} states")
# Converting timestamp fileds into datetime.

acc_df['Start_Time'] = pd.to_datetime(acc_df['Start_Time'], infer_datetime_format=True)

acc_df['End_Time'] = pd.to_datetime(acc_df['End_Time'], infer_datetime_format=True)

acc_df['Weather_Timestamp'] = pd.to_datetime(acc_df['Weather_Timestamp'], infer_datetime_format=True)
# Missing Data

missing_data = []

total = acc_df.isna().sum()

percentage = round((acc_df.isna().sum()/acc_df.isna().count())*100, 0).astype(int)

missing_data = pd.concat([total, percentage], axis=1, keys=['Sum', 'Percentage(%)'])

missing_data = missing_data.sort_values('Percentage(%)', ascending=False)

missing_data
# Replace missing values with 0s

acc_df = acc_df.fillna(0)
# Missing Data

missing_data = []

total = acc_df.isna().sum()

percentage = round((acc_df.isna().sum()/acc_df.isna().count())*100, 0).astype(int)

missing_data = pd.concat([total, percentage], axis=1, keys=['Sum', 'Percentage(%)'])

missing_data = missing_data.sort_values('Percentage(%)', ascending=False)

missing_data
acc_df['Counter']=1
# Determining the timeframe of the data

print( 'State Date - ', acc_df['Start_Time'].min(), '\nEnd Date   - ', acc_df['Start_Time'].max())
# Different sources providing accident details 

print('Source of information:')

print('----------------------')

print((round(acc_df['Source'].value_counts(normalize=True)*100, 0)).astype(int).astype(str) + '%')

ax = sns.countplot(x="Source", data=acc_df, order = acc_df['Source'].value_counts().index)
# Frequency Distribution of TMC (Note - 0 represents missing data. 0 is not a valid TMC Code)

print('Frequency Distribution of TMC')

print('Code       Frequency ')

print('-----------------------------')

plt.figure(figsize = (16, 6))

print(acc_df['TMC'].value_counts())

sns.countplot(x="TMC", data=acc_df, order = acc_df['TMC'].value_counts().index)
# TMC Labels refence - https://wiki.openstreetmap.org/wiki/TMC/Event_Code_List

tmc_code_labels = ['(Q) accident(s)','Missing TMC Code',

'(Q) accident(s). Right lane blocked',

'(Q) accident(s). Two lanes blocked',

'(Q) accident(s). Slow traffic',

'multi-vehicle accident (involving Q vehicles)',

'(Q) accident(s). Queuing traffic',

'(Q) accident(s). Hard shoulder blocked',

'(Q th) entry slip road closed',

'(Q) serious accident(s)',

'(Q) accident(s). Three lanes blocked',

'accident. Delays (Q)',

'(Q) earlier accident(s)',

'(Q) accident(s). Heavy traffic',

'accident. Delays (Q) expected',

'(Q) fuel spillage accident(s)',

'(Q) jackknifed trailer(s)',

'(Q) jackknifed articulated lorr(y/ies)',

'multi vehicle pile up. Delays (Q)',

'(Q) oil spillage accident(s)',

'(Q) accident(s). Traffic building up',

'(Q) accident(s) in roadworks area']
# Create a lookup of TMC Code and Description. This will give us an idea of whats these codes mean

values = acc_df['TMC'].value_counts().keys().astype(int).tolist()

counts = acc_df['TMC'].value_counts().tolist()

tmc_lookup_df = pd.DataFrame({'TMC_CODE':values, 'TMC_CODE_DESC': tmc_code_labels, 'COUNT':counts})

tmc_lookup_df
# Frequency Distribution for Severity

plt.figure(figsize = (5, 5))

print('Severity:')

print('---------')

print((acc_df['Severity'].value_counts(normalize=True)*100).astype(int).astype(str) + '%')

sns.countplot(x='Severity', data=acc_df)
# Creating a new column called Accident Duration

acc_df['Accident_Duration_Mins'] = (acc_df['End_Time'] - acc_df['Start_Time']).astype('timedelta64[m]').astype(int)
# Accident Duration by Severity

acc_duration = acc_df[['Accident_Duration_Mins']].groupby(acc_df['Severity']).agg(['count', 'mean', 'std', 'median', 'min', 'max']).astype(int).round()

acc_duration
# No of accidents by States

plt.figure(figsize = (16, 6))

sns.countplot(x='State', data=acc_df, order = acc_df['State'].value_counts().index)

plt.title('Count of accidents by State')
# Select top 5 states

acc_df_top_state = acc_df[acc_df['State'].isin(['CA', 'TX', 'FL', 'NC', 'NY'])][['Start_Time', 'State', 'Severity']]
# How many accidents happened in last 4 years in top 5 states

acc_df_top_state['State'].value_counts()
# Breakup of accidents by Severity

acc_df_top_state['COUNTER'] =1

acc_df_top_state.groupby(['State','Severity'])['COUNTER'].sum()
# Extract Year and Month of accident from Start_Time

acc_df_top_state['Year'] = acc_df_top_state['Start_Time'].map(lambda x: x.year)

acc_df_top_state['Month'] = acc_df_top_state['Start_Time'].map(lambda x: x.month)

acc_df_top_state[['Start_Time', 'Year', 'Month']].head()
# Breakup of Accident by Year

plt.figure(figsize = (10, 6))

print(acc_df_top_state.groupby(['Year'])['COUNTER'].sum())

ax = acc_df_top_state.groupby(['Year'])['COUNTER'].sum().plot(kind='line', linestyle='-', marker='o', use_index=False)

acc_df_top_state.groupby(['Year'])['COUNTER'].sum().plot(kind='bar', color=['C0', 'C1', 'C2', 'C3'], title='No of Accidents by Year', ax=ax)
# Breakup of Accident by Month

plt.figure(figsize = (10, 6))

print(acc_df_top_state.groupby(['Month'])['COUNTER'].sum())

ax = acc_df_top_state.groupby(['Month'])['COUNTER'].sum().plot(kind='line', linestyle='-', marker='o', use_index=False)

acc_df_top_state.groupby(['Month'])['COUNTER'].sum().plot(kind='bar', color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], title='No of Accidents by Month', ax=ax)
# Which months have most severe 2 accidents

plt.figure(figsize = (10, 6))

print(acc_df_top_state.loc[acc_df_top_state["Severity"] == 2].groupby(['Month'])['COUNTER'].sum())

ax = acc_df_top_state.loc[acc_df_top_state["Severity"] == 2].groupby(['Month'])['COUNTER'].sum().plot(kind='line', linestyle='-', marker='o', use_index=False)

acc_df_top_state.loc[acc_df_top_state["Severity"] == 2].groupby(['Month'])['COUNTER'].sum().plot(kind='bar', color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], title='Months have Severity 2 accidents', ax=ax)
# Which months have most severe 3 accidents

plt.figure(figsize = (10, 6))

# print(acc_df_top_state.loc[acc_df_top_state["Severity"] == 2].groupby(['Month'])['COUNTER'].sum())

ax = acc_df_top_state.loc[acc_df_top_state["Severity"] == 3].groupby(['Month'])['COUNTER'].sum().plot(kind='line', linestyle='-', marker='o', use_index=False)

acc_df_top_state.loc[acc_df_top_state["Severity"] == 3].groupby(['Month'])['COUNTER'].sum().plot(kind='bar', color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], title='Months have Severity 3 accidents', ax=ax)
# Which months have most severe 4 accidents

plt.figure(figsize = (10, 6))

# print(acc_df_top_state.loc[acc_df_top_state["Severity"] == 4].groupby(['Month'])['COUNTER'].sum())

ax = acc_df_top_state.loc[acc_df_top_state["Severity"] == 4].groupby(['Month'])['COUNTER'].sum().plot(kind='line', linestyle='-', marker='o', use_index=False)

acc_df_top_state.loc[acc_df_top_state["Severity"] == 4].groupby(['Month'])['COUNTER'].sum().plot(kind='bar', color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], title='Months have Severity 4 accidents', ax=ax)
# Breakup of accidents by Severity and Year

plt.figure(figsize = (16, 6))

sby_df = acc_df_top_state.loc[acc_df_top_state["State"] == "CA"].groupby(['Year', 'Severity', 'State'])['COUNTER'].sum().reset_index()

sns.factorplot("Severity", "COUNTER", col="Year", data=sby_df, kind="bar")
# Which state has highest severity 4 ?

plt.figure(figsize = (10, 5))

acc_sev = acc_df[acc_df.Severity == 4][['State', 'Severity']]

sns.countplot(x='State', data=acc_sev, order = acc_sev['State'].value_counts().iloc[:10].index)
# Which state has highest severity 3 ?

plt.figure(figsize = (10, 5))

acc_sev = acc_df[acc_df.Severity == 3][['State', 'Severity']]

sns.countplot(x='State', data=acc_sev, order = acc_sev['State'].value_counts().iloc[:10].index)
# Which state has highest severity 2 ?

plt.figure(figsize = (10, 5))

acc_sev = acc_df[acc_df.Severity == 2][['State', 'Severity']]

sns.countplot(x='State', data=acc_sev, order = acc_sev['State'].value_counts().iloc[:10].index)
# Top 10 County having highest accidents

plt.figure(figsize = (10, 5))

sns.countplot(x='County', data=acc_df, order = acc_df['County'].value_counts().iloc[:10].index)
# Top 10 Zip Codes having highest accidents

plt.figure(figsize = (10, 5))

sns.countplot(x='Zipcode', data=acc_df, order = acc_df['Zipcode'].value_counts().iloc[:10].index)
# DescriptionShows natural language description of the accident.

plt.figure(figsize=(10, 6))

desc = acc_df["Description"].str.split("(").str[0].value_counts().keys()

wc_desc = WordCloud(scale=5,max_words=100,colormap="rainbow",background_color="white").generate(" ".join(desc))

plt.imshow(wc_desc,interpolation="bilinear")

plt.axis("off")

plt.title("Top 100 Accident Description",color='b')

plt.show()
# Distance(mi) by Severity The length of the road extent affected by the accident.

acc_df[['Distance(mi)']].groupby(acc_df['Severity']).agg(['count', 'mean', 'median', 'min', 'max'])
# Which side of Lane has more accidents

print((acc_df['Side'].value_counts(normalize=True)*100).astype(int).astype(str) + '%')
#SideShows the relative side of the street (Right/Left) in address field.

side_df = acc_df.groupby(['Side', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Side", data=side_df, kind="bar")
# Severity Impact by Temperature

plt.figure(figsize = (16, 6))

ax = sns.violinplot(y="Temperature(F)", x="Severity", data=acc_df, palette="Set1")
# Severity Impact by Wind_Chill

plt.figure(figsize = (16, 6))

ax = sns.violinplot(y="Wind_Chill(F)", x="Severity", data=acc_df, palette="Set2")
# Severity Impact by Humidity

plt.figure(figsize = (16, 6))

ax = sns.violinplot(y="Humidity(%)", x="Severity", data=acc_df, palette="Set3")
# Severity Impact by Visibility

plt.figure(figsize = (16, 6))

ax = sns.violinplot(y="Visibility(mi)", x="Severity", data=acc_df, palette="Set3")
# Severity Impact by Windspeed

plt.figure(figsize = (16, 6))

ax = sns.violinplot(y="Wind_Speed(mph)", x="Severity", data=acc_df)
# Severity Impact by Precipitation

plt.figure(figsize = (16, 6))

ax = sns.boxplot(y="Precipitation(in)", x="Severity", data=acc_df)
# Co-relation between Precipitation and Sverity

acc_df[['Precipitation(in)', 'Severity']].corr()
# Weather_Condition

acc_df['Weather_Condition'].unique().tolist()
# Top 10 weather condition

plt.figure(figsize = (15, 6))

acc_df[acc_df['Weather_Condition'] != 0]['Weather_Condition'].value_counts().iloc[:10].plot(kind='bar', color=['C0', 'C1', 'C2', 'C3', 'C4'], title='Top 10 Weather Condition')
# Weather_Condition Word Cloud

from wordcloud import WordCloud

weather_cond = acc_df["Weather_Condition"].str.split("(").str[0].value_counts().keys()

wc = WordCloud(scale=5,max_words=15,colormap="rainbow",background_color="white").generate(" ".join(weather_cond))

plt.figure(figsize=(10,10))

plt.imshow(wc,interpolation="bilinear")

plt.axis("off")

plt.title("Top 15 Weather conditions",color='b')

plt.show()
# Start_LatShows latitude in GPS coordinate of the start point.

# Start_LngShows longitude in GPS coordinate of the start point.
# Severity by Bump

bump_df = acc_df.groupby(['Bump', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Bump", data=bump_df, kind="bar")
# Severity by Crossing

cross_df = acc_df.groupby(['Crossing', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Crossing", data=cross_df, kind="bar")
# Severity by Junction 

junc_df = acc_df.groupby(['Junction', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Junction", data=junc_df, kind="bar")
# Severity by Roundabouts 

rndabt_df = acc_df.groupby(['Roundabout', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Roundabout", data=rndabt_df, kind="bar")
# Severity by Stop Sign 

stop_df = acc_df.groupby(['Stop', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Stop", data=stop_df, kind="bar")
# Severity by Traffic Signal 

stop_df = acc_df.groupby(['Traffic_Signal', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Traffic_Signal", data=stop_df, kind="bar")
# Severity by Traffic Signal 

ssnt_df = acc_df[acc_df['Sunrise_Sunset'] != 0].groupby(['Sunrise_Sunset', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Sunrise_Sunset", data=ssnt_df, kind="bar")
# Severity by Civil Twilight

ssnt_df = acc_df[acc_df['Civil_Twilight'] != 0].groupby(['Civil_Twilight', 'Severity'])['Counter'].sum().reset_index()

print(ssnt_df)

sns.factorplot("Severity", "Counter", col="Civil_Twilight", data=ssnt_df, kind="bar")
# Severity by Nautical_Twilight

ssnt_df = acc_df[acc_df['Nautical_Twilight'] != 0].groupby(['Nautical_Twilight', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Severity", "Counter", col="Nautical_Twilight", data=ssnt_df, kind="bar")
# Let's zoom into start_time column to see what hours are more severe

acc_df_ts = acc_df[['Start_Time', 'Severity', 'Counter']]

acc_df_ts['Time'] = acc_df_ts['Start_Time'].map(lambda x: x.time().strftime('%H:%M'))

acc_df_ts.head()
# Extracting hrs. from Start_time

acc_df_ts['Hrs'] = acc_df_ts['Start_Time'].map(lambda x: x.time().strftime('%H'))

acc_df_ts.head()
# Plotting hrs against complete accidient list

plt.figure(figsize = (10, 6))

ax = acc_df_ts.groupby(['Hrs'])['Counter'].sum().plot(kind='line', linestyle='-', marker='o', use_index=False)

acc_df_ts.groupby(['Hrs'])['Counter'].sum().plot(kind='bar', color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], title='Accidents by Hr', ax = ax)
# Severity by Hrs.

plt.figure(figsize = (15, 10))

acc_time_df = acc_df_ts[acc_df_ts['Severity']>1].groupby(['Hrs', 'Severity'])['Counter'].sum().reset_index()

sns.factorplot("Hrs", "Counter", col="Severity", data=acc_time_df, kind="bar")