!pip freeze | grep pandas
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import pandas_profiling

from IPython.display import display, HTML, IFrame
# This is our main dataset

df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

df.head()
df.shape
df1 = df.copy()
df = df1.copy()
df.columns
#@title Changing columns names
df.columns = ['sr', 'date', 'time', 'state', 'is_indian', 'is_foreigner', 'cured', 'deaths', 'positive']

df.columns
df.state.value_counts()
# Above we see that there are few values which are not states and some are reapeted

# Unassigned &  Cases being reassigned to states are not states, hence lets drop them

# Lets merge 'Dadar Nagar Haveli & 'Daman & Diu' in 'Dadra and Nagar Haveli and Daman and Diu'



df['state'] = df.state.replace({'Dadar Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu'})

df['state'] = df.state.replace({'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu'})



df['state'] = df.state.replace({'Cases being reassigned to states': np.nan})

df['state'] = df.state.replace({'Unassigned': np.nan})



df['state'].value_counts()
#@title Missing values

df.isna().sum()
df.is_indian.unique()
# There is a '-' value in feature is_indian which means missing value
df.is_indian.value_counts()
# We see '-' missing value is more 90%
df.is_foreigner.unique()
# There is a '-' value in feature is_foreigner which needs to be dropped 
df.is_foreigner.value_counts()
# We see '-' missing value is more 90%
# We see that is_indian and is_foreigner which provided data on how many Indian or Foreigners are effected has 

# missing values more then 90%, hence lets drop the features



df = df.drop(['is_indian', 'is_foreigner'], axis=1)

df.shape
# Lest drop nan values

df = df.dropna()

df.shape
df2 = df.copy()
df = df2.copy()
# Lets work in the date feature

# The date feature has few values with 2 digit day and month (ex- 01 & 1) and some has 1 digit values

# The year has few values of 4 digits YYYY and few with 2 digits YY

# Hence dorectly applying the pd.datetime function will create different dates which are actually not there in the database (bug)

# so we will split the string date into day, month & year and then convert them to int



# Splitting the string data

new_date = df['date'].str.split('/', 2, expand=True)

df['day'] = new_date[0]

df['month']= new_date[1]

df['year'] = new_date[2]



# Comnverting data type

df['day'] = df['day'].astype(int)

df['month'] = df['month'].astype(int)

df['year'] = df['year'].astype(int)



df.head()
# Lets equilize the year column



df['year'] = df['year'].apply(lambda x: 2020 if x < 30 else 2020 )

df['year'].values
# Now all the values in the year column is 2020

# Now to get the month name , lets merge the day, month and year to make ot datetime



df['date'] = pd.to_datetime(df[['day', 'month', 'year']])

df['Month'] = df['date'].dt.month_name()



# Lets drop day, month and year, since we have corrected date colum now

df = df.drop(['day', 'month', 'year'], axis=1)



df.head()
# Lets start by converting the data type to its correct 



# state & time to be converted to category



df['state'] = df['state'].astype('category')

df['time'] = df['time'].astype('category')



df.head()
# Now lets convert Month to category  for data visulization pourpose



df['Month'] = df['Month'].astype('category')



df.info()
jan = df[df['date']=='2020-01-31']

feb = df[df['date']=='2020-02-29']

mar = df[df['date']=='2020-03-31']

apr = df[df['date']=='2020-04-30']

may = df[df['date']=='2020-05-31']

jun = df[df['date']=='2020-06-22']



frame = [jan, feb, mar, apr, may, jun]



df = pd.concat(frame)



df.date.unique()
# Lets check the number of cases each month



f, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x=df.Month.unique(), y=df.Month.value_counts()[:6].sort_values(ascending=True), data=df, color= 'blue')

ax.set_title('Number of Cases each month', fontsize= 20)

ax.set_xlabel('Months', fontsize=16)

ax.set_ylabel('Number of Positive cases', fontsize=16)

plt.show()
# Lets check the number of cases statewise in the month of April 2020



#Lets create a new dataframe only for the month of April2020

df_April = df[df['Month']=='April']



# Creting a new dataframe with groupby

df_April_State = df_April.groupby(['state'], as_index=False)['positive'].agg('sum').sort_values('positive', ascending=False)



# Lets plot the new dataframe

f, ax = plt.subplots(figsize=(10, 10))

sns.barplot(x='positive', y='state',

            order=df_April_State['state'], data=df_April_State, color= 'blue', ax=ax)

ax.set_title('Number of cases statewise in the month of April 2020', fontsize= 20)

ax.set_ylabel('States/UT', fontsize=16)

ax.set_xlabel('Number of Positive cases', fontsize=16)

plt.close(2)

plt.show()
df_cured = df.groupby(df['state'], as_index=False)['cured'].agg('sum').sort_values('cured', ascending=False)



plt.figure(figsize=(10, 10))



sns.barplot(x='cured', y='state', order=df_cured['state'], data=df_cured, color='blue')

plt.title('Number of cases statewise in the month of April 2020', fontsize= 20)

plt.ylabel('States/UT', fontsize=16)

plt.xlabel('Number of Positive cases', fontsize=16)

plt.show()
# Current Status

positive = df['positive'].sum()

cured = df['cured'].sum()

deaths =df['deaths'].sum()

data_pie = [positive, cured, deaths]

plt.figure(figsize=(6, 6))

fig = px.pie(data_frame= df, values=data_pie, names=['Positive Cases', 'Cured People', 'Deaths'],

             title= 'Currrent status of Covid19 India')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()

# Monthwise graph of Number of positive cases found in last 6 months

df_month = df.groupby(df['Month'], as_index=False)['positive', 'cured', 'deaths'].agg('sum').sort_values(by='positive', ascending=True)



px.bar(df_month, x='Month', y='positive', color='Month', 

          labels={'positive': 'Positive cases', 'cured': 'People cured', 'deaths': 'Deaths occured'},

          hover_data= ['positive', 'cured', 'deaths'], title='Most number of cases/cure/deaths on Monthly basis', log_y=True)