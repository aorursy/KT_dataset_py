# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

covid_india.head()
covid_india.info()



covid_india_new = covid_india.copy()
# We can see that there are no null values, since 0 is also treated as a Numnber



# Let's have a,look at the columns



print(covid_india_new.describe())



print(covid_india_new['State/UnionTerritory'].value_counts())
covid_india_new['State/UnionTerritory'].replace('Chattisgarh','Chhattisgarh',inplace=True)



covid_india_new.head(10)
# So we will plot the Ditribution of States based on the number of Cases Recieved/Occurence,as this will tell us the top states which should be in high alert.



# The thing to note is that we are not looking at the number of Cases, but at the occurence of each state. We will be displaying the number of Cases also in the same.



covid_india_new['Confirmed'] = covid_india_new.ConfirmedIndianNational + covid_india_new.ConfirmedForeignNational

covid_india_new_summed = covid_india_new.groupby('State/UnionTerritory').agg({'Deaths':max,'Cured':max,'Confirmed':max}).reset_index()



fig = px.pie(covid_india_new_summed, values='Confirmed', names='State/UnionTerritory'

             ,color_discrete_sequence=px.colors.sequential.RdBu,title='The Distribution of States with the increase in count of Confirmed Cases.!')

fig.update_traces(textposition='inside', textinfo='value+label')

fig.show()



# So we can see that when we reached 620 cases, the Kerala was the State to have the 620th case indeed!



# We will look at the Distribution of the States also.
print('Total Confirmed Cases: ', covid_india_new_summed.Confirmed.sum())



print('Total Deaths occured: ', covid_india_new_summed.Deaths.sum())



print('Total Recovered cases: ', covid_india_new_summed.Cured.sum())
# 

covid_india_new_sorted = covid_india_new_summed.sort_values(by='Confirmed')



fig = go.Figure()

fig.add_trace(go.Scatter(x=covid_india_new_sorted['Confirmed'], y=covid_india_new_sorted['State/UnionTerritory'],hoverinfo=['all'],

                         mode='lines+markers',

                    name='The Line of Increasing Cases'))

fig.add_trace(go.Scatter(x=covid_india_new_sorted['Deaths'], y=covid_india_new_sorted['State/UnionTerritory'],hoverinfo=['all'],

                         mode='lines+markers',

                    name='The Line of Deaths faced!'))

fig.add_trace(go.Scatter(x=covid_india_new_sorted['Cured'], y=covid_india_new_sorted['State/UnionTerritory'],hoverinfo=['all'],

                         mode='lines+markers',

                    name='The Line of Recovered Cases'))

fig.update_layout(

    title="Which State gets to see the highest number of Confirmed cases??",

    yaxis_title="States",

    xaxis_title="Count of Cases",

    autosize=True,

    height=800,

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="darkblue"

    )

)

fig.show()
# We cannot do any modelling because we donot have a day by day data as far, but we would like to make some analysis based on the Individual Data, that has been provided to us.

# Like, what age of people are getting affected most,is gender impacting the situation somewhere and other questions we would like to answer



individual_details = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')



individual_details.head(10)
individual_details = individual_details.rename(columns=lambda x: x.strip())



cols_to_drop = ['unique_id','id','government_id','detected_city_pt','notes','current_location','current_location_pt','contacts']



filter_data = individual_details.drop(cols_to_drop,axis=1)



filter_data.head()
# Convert dates in one format

import datetime as dt



filter_data['status_change_date'] = pd.to_datetime(filter_data['status_change_date'])

filter_data['diagnosed_date'] = pd.to_datetime(filter_data['diagnosed_date'])



filter_data['Duration of Any Status'] = filter_data['status_change_date'] - filter_data['diagnosed_date']

filter_data['Duration of Any Status'] = filter_data['Duration of Any Status'].dt.days



filter_data['status_change_date'] = filter_data['status_change_date'].dt.strftime('%Y-%m-%d')

filter_data['diagnosed_date'] = filter_data['diagnosed_date'].dt.strftime('%Y-%m-%d')
filter_data.info()
# Next we will drop Detetcted City and District as we have values for State, plus we will drop the Age or we can fill it with the mean value, then we will be dropping the Nationality in order to not discriminate in such a pandemic



drop_cols = ['detected_city','detected_district']



covid_india_df = filter_data.drop(drop_cols,axis=1)



covid_india_df.info()
covid_india_df.describe()
covid_india_df['age'] = covid_india_df['age'].fillna(covid_india_df.age.median())

covid_india_df['current_status'] = covid_india_df['current_status'].fillna(method='ffill')



covid_india_df.info()
covid_india_df.head()
# Now we can look at the broader scale by looking at the Duration. And we'll see if we can do any Clustering

plt.figure(figsize=(18,9))

sns.scatterplot(y = covid_india_df['Duration of Any Status'],x = covid_india_df['current_status']);

plt.xlabel('Status of the Patient');

plt.ylabel('Duration of Days from the time they were Admitted');

plt.title('Distribution of Duration of Days wioth the Status of patients!');
# Let's disect age into bins to see which age group is affected most with covid-19

# We'll take a broad age group to form bins 

age_bins = [0,20,40,60,80,100]

plt.figure(figsize=(12,6))

sns.countplot(x=pd.cut(covid_india_df.age, age_bins), hue=covid_india_df.current_status)

plt.xticks(rotation=90)

plt.xlabel("Age Groups")

plt.yscale('log')

plt.title("Age Groups affected with Covid-19")

plt.grid(True)

plt.show()

covid_nationality = covid_india_df.groupby('nationality').count()

fig = px.pie(covid_nationality, values='current_status', names=covid_nationality.index

             ,color_discrete_sequence=px.colors.sequential.RdBu,title='The Distribution of Confirmed cases Nationality.!')

fig.update_traces(textposition='outside', textinfo='value+label')

fig.show()

# Let's check the current_status of confirmed cases of different nationalities in India



covid_diff = covid_india_df[covid_india_df.nationality!="India"]



plt.figure(figsize=(12,6))

sns.countplot(x=covid_diff.nationality, hue=covid_diff.current_status)

plt.grid(True)

plt.xlabel("Cases of Other Nationalities in India")

plt.title("Current Status of Other Nationalities in India")

plt.show()
covid_india_df.head()
covid_gender  = covid_india_df.groupby(['detected_state', 'gender']).size().reset_index().pivot(columns='gender', index='detected_state', values=0)



covid_gender = covid_gender.fillna(0)

covid_gender['total'] = covid_gender.Female + covid_gender.Male + covid_gender.Unknown



covid_gender_1 =  covid_gender.sort_values(by='total', ascending=False)

covid_gender_1.drop('total', axis=1, inplace=True)





covid_gender_1.plot(kind='bar', stacked=True, figsize=(15,8))

plt.xlabel("States")

plt.ylabel("Count")

plt.title("Distribution of Gender in States")

plt.grid(True)

plt.show()
# Let's split the detected_city_point feature of our original dataset to get the latitude and longitude 

new = individual_details.detected_city_pt.str.split(" ",n=1, expand=True)
# We'll form a new feature Detected_cordinates and assign it to the dataset and slpit it further 

individual_details['Detected_cordinates'] = new[1]



individual_details.head()
# Again split the coridinates column 

lat_long = individual_details.Detected_cordinates.str.split(" ", n=1, expand=True)



lat_long
# let's assign the latittude and longitude to our dataset 

covid_india_df['Detected_latitude'] = lat_long[0]

covid_india_df['Detected_longitude'] = lat_long[1]



# Let's clean the dataset further 

covid_india_df['Detected_latitude'] = covid_india_df['Detected_latitude'].map(lambda x : x.replace("(", "")).astype('float64')

covid_india_df['Detected_longitude'] = covid_india_df['Detected_longitude'].map(lambda x: x.replace(")","")).astype('float64')



covid_india_df.head()
# Dropping the features we don't need for the clustering 



covid_cluster =  covid_india_df.copy()



covid_cluster.drop(['diagnosed_date', 'status_change_date','Detected_latitude','Detected_longitude'], axis=1, inplace=True)



# Let's see how the dataset look after dropping it 

covid_cluster.head()
# let's convert these some categorical features in to numerical features to ease our clustering 



covid_cluster.nationality = covid_cluster.nationality.map(lambda x: 1 if x=='India' else 0)



le = LabelEncoder()

covid_cluster.gender =  le.fit_transform(covid_cluster.gender)

# We'll do one-hot encoding for detected_state feature 

covid_cluster.detected_state.value_counts()