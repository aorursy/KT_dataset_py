# importing libraries



import pandas as pd

import seaborn as sns
# loading data



df = pd.read_csv('../input/new-york-city-shooting-dataset/NYPD_Shooting_Incident_Data__Historic_.csv')
# inspecting data



df.head()
# inspecting data types



df.dtypes
df.shape
# inspecting types of victims



df['VIC_RACE'].unique()
# inspecting number of victims



df['VIC_RACE'].value_counts()
# visualizing victims



df['VIC_RACE'].value_counts().plot(kind='barh')
# inspecting victim sex



df['VIC_SEX'].value_counts()
# visulizing victim sex



df['VIC_SEX'].value_counts().plot(kind='barh')
# converting OCCUR_DATE from object to datetime



df['OCCUR_DATE'] = pd.to_datetime(df['OCCUR_DATE'] + ' ' + df['OCCUR_TIME'])
# inspecting data types



df.dtypes
# inspecting number of year the data is from



(df['OCCUR_DATE'].dt.year).unique()
# grouping by year



df_yearly = df.groupby(df['OCCUR_DATE'].dt.year)['INCIDENT_KEY'].agg('count')
# visualizing data



df_yearly.plot(kind = 'bar')
df_daily = df_yearly / 365

df_daily.plot()
# grouping by area



df_area = df.groupby(df['BORO'])['INCIDENT_KEY'].agg('count')
# visualizing data



df_area.plot(kind = 'bar')
# inspecting age groups



df['VIC_AGE_GROUP'].unique()
# inspecting victim counts per age group



df['VIC_AGE_GROUP'].value_counts()
# visualizing victim counts per age group



sns.countplot(x = df['VIC_AGE_GROUP'], hue = df['STATISTICAL_MURDER_FLAG'])
# inspecting percentages of muder flag based on age group



df_murder = (df.groupby(['VIC_AGE_GROUP'])['STATISTICAL_MURDER_FLAG'].value_counts(normalize=True).rename('PERCENTAGE').mul(100).reset_index())

df_murder.head()
# visualizing those percentages



sns.barplot(x = df_murder['VIC_AGE_GROUP'], y = df_murder['PERCENTAGE'], hue = df_murder['STATISTICAL_MURDER_FLAG'])
#grouping data by hours



df_date = df.set_index('OCCUR_DATE')

df_date = df_date.groupby(df_date.index.hour)['INCIDENT_KEY'].count()
# visulaizing hourly data



df_date.plot()