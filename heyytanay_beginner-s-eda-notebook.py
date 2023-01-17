# Install DABL (it's a secret tool that'll help us later 😉)

! pip install -q dabl

! pip install -q country_converter
# Import some basic libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import plotly

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff



import country_converter as coco



import dabl
# Import the data

data = pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")

data.head()
# Drop the PassengerId column, since it's not a usefull feature

data = data.drop(['PassengerId'], axis=1)
# Check Null values in the dataset

data.isna().sum()
data['Country'].value_counts()
# Pie Chart of the different countries

vals = list(data['Country'].unique())

values, labels = [], []

for val in vals:

    values.append(len(data[data['Country']==val]))

    labels.append(val)



fig = px.pie(

    names=labels,

    values=values,

    title="Passenger Country Distribution",

    color_discrete_sequence=px.colors.sequential.RdBu,

)

fig.show()
# Also a bar chart for the same

fig = px.bar(

    x=labels,

    y=values,

    title="Passengers by Country",

    labels={

        'x': 'Country',

        'y': 'Passenger Count'

    },

    color=values

)

fig.show()
# Convert country name to ISO3 country code

con_cod = [coco.convert(x, to='ISO3') for x in data['Country']]



# Append the country codes to the orignal dataframe

data['code'] = con_cod



# Get the country codes and corresponding number of 

country_codes = list(dict(data['code'].value_counts()).keys())

country_pass = list(data['code'].value_counts())



# Make a new dataframe based on number of passengers based on each country

country_df = pd.DataFrame()

country_df['code'] = country_codes

country_df['passengers'] = country_pass



# View the new dataframe

country_df.head()
# Now let's plot it!

fig = px.choropleth(country_df, locations="code",

                    color="passengers",

                    hover_name="code",

                    color_continuous_scale=px.colors.sequential.Jet,

                    title="Passengers Country Distribution"

                   )

fig.show()
data['Lastname'].value_counts()[:10]
# Bar Chart for top-10 last names

values = data['Lastname'].value_counts().tolist()[:10]

names = list(dict(data['Lastname'].value_counts()).keys())[:10]



fig = px.bar(

    x=names,

    y=values,

    title="10 Most Popular Last Names",

    color=values,  

)



fig.show()
target = [data[data['Sex']=='M'].count().max(), data[data['Sex']=='F'].count().max()]

names = ['Male', 'Female']



fig = px.pie(

    names=names,

    values=target,

    hole=0.3,

    title="Gender Distribution among Passengers",

    color_discrete_sequence=['Blue', 'Magenta']

)

fig.show()
# Get Male and Female Ages in a List

male_ages = data[data['Sex'] == 'M']['Age'].tolist()

female_ages = data[data['Sex'] == 'F']['Age'].tolist()



fig = ff.create_distplot(

    hist_data=[male_ages, female_ages],

    group_labels=['Male', 'Female'],

    colors=['#1500ff', '#ff00e1'],

    show_hist=False,

    show_rug=False,

)



fig.layout.update({'title':f'Age Distribution of both Genders<br>[Average Age: {np.mean(male_ages+female_ages):.2f} years]'})



fig.show()
fig = ff.create_2d_density(x=data['Age'], 

                           y=data['Sex'].apply(lambda x: 1 if x=='M' else 0),

                           title="Age-Sex Density Plot",

                           colorscale=['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)])

fig.show()
target = [data[data['Category']=='P'].count().max(), data[data['Category']=='C'].count().max()]

names = ['Passengers', 'Crew Members']



fig = px.pie(

    names=names,

    values=target,

    hole=0.5,

    title="Crew members vs Passengers",

    color_discrete_sequence=['Red', 'Blue']

)

fig.show()
target = [data[data['Survived']==0].count().max(), data[data['Survived']==1].count().max()]

names = ['Did Not Survive', 'Survived']



fig = px.pie(

    names=names,

    values=target,

    hole=0.5,

    title="How many Survived?",

    color_discrete_sequence=['Black', 'Green']

)

fig.show()
# See How many crew members survived

target_c = [data[(data['Survived'] == 0)&(data['Category']=='C')].count().max(), data[(data['Survived'] == 1)&(data['Category']=='C')].count().max()]

names_c = ['Did Not Survive - Crew', 'Survived - Crew']



target_p = [data[(data['Survived'] == 0)&(data['Category']=='P')].count().max(), data[(data['Survived'] == 1)&(data['Category']=='P')].count().max()]

names_p = ['Did Not Survive - Passenger', 'Survived - Passenger']



fig = plotly.subplots.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(go.Pie(

    labels=names_c,

    values=target_c,

    hole=0.6,

    title="Crew Members",

), 1,1)



fig.add_trace(go.Pie(

    labels=names_p,

    values=target_p,

    hole=0.6,

    title="Passengers",

), 1,2)



fig.update_layout(title_text="Crew v/s Passenger Survival")



fig.show()
fig = px.bar(

    data_frame=data,

    x='Country',

    y='Survived',

    color='Category',

    title="Populace: Country and survival [P: Passenger | C: Crew Member]",

    color_discrete_sequence=['Cyan', 'Blue']

)

fig.show()
# First Drop the Country Code Column as it's redundant since we have country names

data = data.drop(['code'], axis=1)



# Plot!

dabl.plot(data, target_col='Survived')