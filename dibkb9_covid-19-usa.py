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
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
#convert 'date' column to 'datetime' dtype

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
#get latest data from each county 

df_latestdate = df[df.groupby('county').date.transform('max') == df['date']]
#aggregate all county data to state

df_state = df_latestdate.groupby('state').agg({

    'cases':'sum',

    'deaths':'sum',

})
df_state = df_state.sort_values(by = 'cases', ascending = False)

df_date = df.set_index('date')
df_state_ten = df_state.head(15).iloc[::-1]

df_state_ten_copy = df_state_ten.copy(deep =True)
#calulate fatality rate

df_state_ten_copy['fatality rate'] = (df_state_ten_copy['deaths']/df_state_ten_copy['cases'])*100
my_range=list(range(len(df_state_ten.index)+1))

fig, ax = plt.subplots(figsize=(18,9))

df_state_ten['cases'].plot(kind ='barh')

#labels

ax.set_xlabel('Total Cases', fontsize=18,)

ax.set_ylabel('')

ax.tick_params(axis='both', which='major', labelsize=13)

ax.set_ylabel('State', fontsize=18)

plt.yticks(my_range, df_state_ten.index)

#spines

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('data', -.6))

#annotations    

for x,y in zip(my_range,df_state_ten["cases"]):

    label = "{:}".format(y)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(27,-6), # distance from text to points (x,y)

                 ha='center',va="bottom")    
ax = df_state_ten.plot.barh(figsize=(18,10))

#annotations

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('data', -.6))

#annotations    

for x,y in zip(my_range,df_state_ten["cases"]):

    label = "{:}".format(y)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(27,-12), # distance from text to points (x,y)

                 ha='center',va="bottom")

for x,y in zip(my_range,df_state_ten["deaths"]):

    label = "{:}".format(y)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(21,0), # distance from text to points (x,y)

                 ha='center',va="bottom")        

#label

ax.set_xlabel('Confirmed Cases and Deaths', fontsize=18)

ax.set_ylabel('State', fontsize=18)



df_state_ten_copy = df_state_ten_copy.sort_values(by = 'fatality rate',ascending = True)
my_range=list(range(len(df_state_ten_copy.index)+1))

fig, ax = plt.subplots(figsize=(18,9))

df_state_ten_copy['fatality rate'].plot(kind ='barh',color='#b81215')

#labels

ax.set_xlabel('Case Fatality Rate', fontsize=18,)

ax.set_ylabel('')

ax.tick_params(axis='both', which='major', labelsize=13)

ax.set_ylabel('State', fontsize=18)

plt.yticks(my_range, df_state_ten_copy.index)

#spines

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('data', -.6))

#annotations    

for x,y in zip(my_range,df_state_ten_copy["fatality rate"]):

    y1 = round(y,3)

    label = "{:}".format(y1)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(3,0), # distance from text to points (x,y)

                 ha='left',va="center")    
df_county = df_latestdate.sort_values(by ='cases',ascending = False)
df_county_ten = df_county.head(15).iloc[::-1]

my_range= list(range(len(df_county_ten.index)+1))

fig,ax = plt.subplots(figsize=(18,9))

df_county_ten['cases'].plot(kind ='barh')

#labels

ax.set_xlabel('Total Cases', fontsize=18,)

ax.set_ylabel('')

ax.tick_params(axis='both', which='major', labelsize=13)

ax.set_ylabel('County', fontsize=18)

plt.yticks(my_range, df_county_ten['county'])

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('data', -.6))

#annotations    

for x,y in zip(my_range,df_county_ten["cases"]):

    label = "{:}".format(y)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(27,-6), # distance from text to points (x,y)

                 ha='center',va="bottom") 

plt.tight_layout()
df_county_ten_copy = df_county_ten.copy(deep =True)

df_county_ten_copy.drop(columns = ['date','state','fips'],inplace = True)

df_county_ten_copy.set_index('county',inplace =True)
ax = df_county_ten_copy.plot.barh(figsize=(18,10))

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('data', -.6))

#annotations    

for x,y in zip(my_range,df_county_ten_copy["cases"]):

    label = "{:}".format(y)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(27,-12), # distance from text to points (x,y)

                 ha='center',va="bottom")

for x,y in zip(my_range,df_county_ten["deaths"]):

    label = "{:}".format(y)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(21,0), # distance from text to points (x,y)

                 ha='center',va="bottom")        

#label

ax.set_xlabel('Confirmed Cases and Deaths', fontsize=18)

ax.set_ylabel('County', fontsize=18)

plt.tight_layout()
df_county_ten_copy['fatality rate'] = (df_county_ten_copy['deaths']/df_county_ten_copy['cases'])*100
df_county_ten_copy.sort_values(by ='fatality rate',ascending = True,inplace =True)
my_range=list(range(len(df_county_ten_copy.index)+1))

fig, ax = plt.subplots(figsize=(18,9))

df_county_ten_copy['fatality rate'].plot(kind ='barh',color='#b81215')

#labels

ax.set_xlabel('Case Fatality Rate', fontsize=18,)

ax.set_ylabel('County',fontsize=15)

ax.tick_params(axis='both', which='major', labelsize=13)

plt.yticks(my_range, df_county_ten_copy.index)

#spines

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_position(('data', -.6))

#annotations    

for x,y in zip(my_range,df_county_ten_copy["fatality rate"]):

    y1 = round(y,3)

    label = "{:}".format(y1)

    plt.annotate(label, # this is the text

                 (y,x), # this is the point to label

                  textcoords="offset points",# how to position the text

                 xytext=(3,0), # distance from text to points (x,y)

                 ha='left',va="center")
#function to plot time series data by county

def time_series(county):

    df_county = df_date[df_date['county'] == county]

    fig,ax = plt.subplots(figsize=(18,9))

    ax.plot(df_county.index,df_county['cases'],label ='Confirmed cases')

    ax.plot(df_county.index,df_county['deaths'],label='Deaths')

    ax.legend(loc="upper left")

    ax.set_xlabel(county,fontsize=18)
time_series('New York City')
time_series('Los Angeles')
time_series('Miami-Dade')
time_series('Maricopa')