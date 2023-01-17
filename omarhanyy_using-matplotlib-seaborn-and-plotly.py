#Data analysis and wrangling

import numpy as np

import pandas as pd



#visualization

import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected = True)

cf.go_offline()

%matplotlib inline
#Reading our csv file into pandas dataframe

df = pd.read_csv('../input/train (3).csv')

df.head()
#Viewing the shape of the data to know how many features we have and how many rows 

df.shape
#As we can see there are null values on the Province_State

df.info()
#How many null values do we have in each feature 

df.isnull().sum()
#Although, we won't care much about dealing with missing data for now

#But Plotting missing data can be very helpful in understanding our dataset

total = df.isnull().sum().sort_values(ascending=False)

percent = ((df.isnull().sum()/df.isnull().count()) * 100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()
# How many unique values in our Id column

# I know this sounds strange as when seeing id you know that it will be unique but some datasets have duplicated id values 

df.Id.nunique()
# Now we are trying to find how many unique values are presented in Province_state column 

# Finding unique values may help us have better understanding of our data

print(f"It has { df.Province_State.nunique() } unique values and it's top 5 values are:")

print('-'*50)

print(df.Province_State.value_counts(dropna=False).head())
# trying to find how many unique values are presented in Country_Region column 

print(f"It has { df.Country_Region.nunique() } unique values and it's top 5 values are:")

print('-'*50)

print(df.Country_Region.value_counts(dropna=False).head())
df.ConfirmedCases.nunique()
print(df.columns.values)
# Convert Date from object to datetime64 format

df['Date'] = pd.to_datetime(df['Date'])
# As we can see day has 31 unique values which is better than before 

df.Date.dt.day.nunique()
# Month is even better as it has only 4 values

df.Date.dt.month.nunique()
# Year won't help us.. 

df.Date.dt.year.nunique()
# Create two columns for the day and month 

df['Day'] = df.Date.dt.day

df['Month'] = df.Date.dt.month
df.head()
# We can drop the Date column as it we got what we wanted from it 

# we can also drop the id column as it doesn't give us any infromation 

df.drop(['Id', 'Date'], axis=1, inplace=True)
# Our dataframe now looks like this

df.head()
# Our features data types

df.dtypes
# Now we want to know how to use the month column in our advantage

df.Month.value_counts(dropna=False)
# Use describe function to know more about the month and if we can use it or not 

df.groupby('Month').describe()
# Create a frequency distribution table as our month is a categorical variable although it's integer 

month_freq = df.groupby('Month').size().reset_index(name='Count')

plt.figure(figsize=(15, 10))

sns.set_style('whitegrid')

sns.countplot(x='Month', data=df)

plt.xlabel('Month', fontsize=20)

plt.ylabel('Count', fontsize=20)

month_freq
# Pie chart

# values for the chart

val = [df['Month'][df['Month'] == 1].count(),df['Month'][df['Month'] == 2].count(), df['Month'][df['Month'] == 2].count()]  # number of values of Jan, Feb & March

fig1, ax1 = plt.subplots(figsize=(15, 7))

ax1.pie(val, explode=(0, 0.05, 0.05), labels=['January', 'February', 'March'], colors=['#c03d3e','#095b97', '#3a923a'], autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 15, 'fontWeight':'bold'})

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')

plt.tight_layout()

plt.show()
# Using histogram for Fatalities 

plt.figure(figsize=(18, 10))

plt.hist(df.Fatalities)
# Using the distribution plot from seaborn 

plt.figure(figsize=(18, 10))

sns.distplot(df.Fatalities.dropna(), bins=30)
# Creating a scatter plot for ConfirmedCases & Fatalities

plt.figure(figsize=(15, 10))

plt.scatter(df.ConfirmedCases, df.Fatalities, marker='D')
sns.lmplot(x='ConfirmedCases', y='Fatalities', data=df, hue='Month', markers=['o', 'D', 'D', '*'])
# stripplot showed that every month number of cases and Fatalities increases

plt.figure(figsize=(15, 7))

sns.stripplot(x='Month', y='Fatalities', data=df)
sns.catplot(x='Month', y='Fatalities', data=df, height=10)
sns.catplot(x='Month', y='ConfirmedCases', data=df, height=10)
sns.pairplot(df, hue='Month', diag_kind='hist')
# Creating a coorelation heatmap to how features are affected by each other (relationship between features)

# using correlation coefficient 

plt.figure(figsize=(20, 12))

sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
# Creating a barplot for month column

df.iplot(kind = 'bar', x = 'Month', xTitle='Month', yTitle='Count')
# Seeing the relation between Month and Fatalities

df.iplot(kind = 'bar', x = 'Month', y = 'Fatalities', xTitle='Month', yTitle='Fatalities')
# Using a Scatter plot for the confirmedCases and fatalities   

df.iplot(kind = 'scatter', x = 'ConfirmedCases', y = 'Fatalities', mode='markers',symbol='circle-dot',colors=['orange','teal'], size=20)
df[['ConfirmedCases', 'Fatalities']].iplot(kind='spread')
df.iplot(x = 'ConfirmedCases', y = 'Fatalities')
df['Fatalities'].iplot(kind = 'hist', bins = 25)
# Box plot for our dataset 

df.iplot(kind = 'box')
df.iplot(kind = 'bubble', x = 'Fatalities', y = 'ConfirmedCases', size = 'Day')
df.iplot()
df[['ConfirmedCases', 'Fatalities']].iplot(kind='area',fill=True,opacity=1)
df.Country_Region.iplot()