# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Import Dependencies

%matplotlib inline



import calendar

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

import missingno as msno

plt.style.use('seaborn-whitegrid')



# Let's ignore warnings for now

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing Data



df=pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

df.head()
# Check for missing values



df.isnull().sum()
## Converting the Date column





df['date']=pd.to_datetime(df['date'],yearfirst=True)

df.set_index('date',inplace=True)

df.head()
# Dropping the fips column

df.drop(['fips'],axis=1,inplace=True)

df.head()
# Create a new data sets to see total cases by county



total_county=df.groupby('county')['cases','deaths'].sum() 

total_county['County']=total_county.index

total_county.reset_index(drop=True,inplace=True)

total_county['Mortality Rate (%)']=(total_county['deaths']/total_county['cases'])*100

total_county.head()
# Average Mortality Rate



total_county['Mortality Rate (%)'].mean()
# Create a new data set to see total cases by state



total_state=df.groupby('state')['cases','deaths'].sum() 

total_state['State']=total_state.index

total_state.reset_index(drop=True,inplace=True)

total_state['Mortality Rate (%)']=(total_state['deaths']/total_state['cases'])*100

# Seeing the top 5 worst affected states

top_states=df.groupby('state')['cases','deaths'].sum().sort_values(by='cases',ascending=False).head(5)

top_states
total_state['Mortality Rate (%)'].mean()
# Seeing the trend of reported COVID-19 cases and deaths over time



pd.pivot_table(df,index=['date'],values=['cases','deaths'],aggfunc=np.sum).plot()

plt.title("Number of reported COVID-19 cases & deaths", loc='center', fontsize=12, fontweight=0, color='orange')

plt.xlabel("date")

plt.ylabel("Total reported COVID-19 cases and deaths")
state_date = df.groupby(['state','date'])['deaths','cases'].apply(lambda x: x.sum())



state_date = state_date.reset_index()



state_date.head()
# Seeing the curve of all states for cases



import plotly.express as px

fig = px.line(state_date,x='date',y='cases',color='state')



fig.show()
# Seeing the curve of all states for deaths



import plotly.express as px

fig = px.line(state_date,x='date',y='deaths',color='state')



fig.show()
# Visualizing the worst affected COVID-19 counties by number of cases, deaths and Mortality Rate (Top 10)



fig, ax =plt.subplots(1,3,figsize=(24, 6))

sns.barplot(x="cases", y="County",ax=ax[0], data=total_county.sort_values(by='cases',ascending=False).head(10)).set_title('Top 10 worst affected counties by Covid-19 (Cases)')

sns.barplot(x="deaths", y="County",ax=ax[1], data=total_county.sort_values(by='deaths',ascending=False).head(10)).set_title('Top 10 worst affected counties by Covid-19 (Deaths)')

sns.barplot(x="Mortality Rate (%)", y="County",ax=ax[2], data=total_county.sort_values(by='Mortality Rate (%)',ascending=False).head(10)).set_title('Top 10 worst affected counties by Covid-19 (Mortality Rate %)')
# Visualizing the worst affected COVID-19 states by number of cases, deaths and Mortality Rate (Top 10)



fig, ax =plt.subplots(1,3,figsize=(24, 6))

sns.barplot(x="cases", y="State",ax=ax[0], data=total_state.sort_values(by='cases',ascending=False).head(10)).set_title('Top 10 worst affected states by Covid-19 (Cases)')

sns.barplot(x="deaths", y="State",ax=ax[1], data=total_state.sort_values(by='deaths',ascending=False).head(10)).set_title('Top 10 worst affected states by Covid-19 (Deaths)')

sns.barplot(x="Mortality Rate (%)", y="State",ax=ax[2], data=total_state.sort_values(by='Mortality Rate (%)',ascending=False).head(10)).set_title('Top 10 worst affected states by Covid-19 (Mortality Rate %)')
# Cleaning the Date column a bit



df['Date']=df.index

# df.reset_index(drop=True,inplace=True)

df['month'] = df['Date'].dt.month

df['month']=df['month'].apply(lambda x: calendar.month_name[x] )

df.head()
# Seeing the trend of reported COVID-19 cases over months



pd.pivot_table(df,index=['month'],values=['cases'],aggfunc=np.sum).plot(kind='bar',color='orange')

plt.title("Number of reported COVID-19 cases", loc='center', fontsize=12, fontweight=0, color='orange')

plt.xlabel("month")

plt.ylabel("Total reported COVID-19 cases")
# Seeing the trend of reported COVID-19 deaths over months



pd.pivot_table(df,index=['month'],values=['deaths'],aggfunc=np.sum).plot(kind='bar',color='red')

plt.title("Number of reported COVID-19 deaths", loc='center', fontsize=12, fontweight=0, color='orange')

plt.xlabel("month")

plt.ylabel("Total reported COVID-19 deaths")