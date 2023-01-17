import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.offline as py

from plotly.graph_objs import Scatter, Layout

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

#set the backgroung style sheet

sns.set_style("whitegrid")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
survey_df = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv')

multiChoice_df = pd.read_csv('../input//kaggle-survey-2019/multiple_choice_responses.csv')
survey_df.head(2)
p = survey_df.iloc[0, 1:-1]

table = go.Table(

    columnwidth=[0.4]+[5.8],

    header=dict(

        values=['Q.N.', 'Description'],

        line = dict(color='#506784'),

        fill = dict(color='lightblue'),

    ),

    cells=dict(

        values=[p.index] + [p.values],

        line = dict(color='#506784'),

        fill = dict(color=['rgb(173, 216, 220)', '#f5f5fa'])

    )

)

py.iplot([table], filename='table-of-mining-data')
p = multiChoice_df.iloc[0, 1:-1]

table = go.Table(

    columnwidth=[1.4]+[5.8],

    header=dict(

        values=['Q.N.', 'Description'],

        line = dict(color='#506784'),

        fill = dict(color='lightblue'),

    ),

    cells=dict(

        values=[p.index] + [p.values],

        line = dict(color='#506784'),

        fill = dict(color=['rgb(173, 216, 220)', '#f5f5fa'])

    )

)

py.iplot([table], filename='table-of-mining-data')
ss = pd.DataFrame(survey_df.loc[1])

ss = ss.drop(['2019 Kaggle Machine Learning and Data Science Survey','Time from Start to Finish (seconds)'],axis=0)

ss[1] = pd.to_numeric(ss[1])

ss = ss.rename(columns={1:'Number of Responders'})

ss.plot(kind='bar',figsize = (15,6))
ss = ss.sort_values('Number of Responders')

print("Questions Which get Least Responce: ")

print(ss.head(3))



print("\nMost Answered Questions: ")

print(ss.tail(3))
gd = multiChoice_df['Q2'][1:].value_counts()



data = [

go.Bar(

    x = list(gd.index),

    y = list(gd.values),

    marker=dict(color=['rgba(55, 128, 191, 1.0)', 'rgba(219, 64, 82, 0.7)',

               'rgba(50, 171, 96, 0.7)', 'rgb(128,0,128)'])

),]

layout= go.Layout(

    title= 'Gender Distribution',

    yaxis=dict(title='Count', ticklen=5, gridwidth=2),

    xaxis=dict(title='Gender', ticklen=5, gridwidth=2)

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Q2')
age_df = multiChoice_df['Q1'][1:].dropna()

order= ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+']

plt.figure(figsize=(12,5))

sns.countplot(age_df,order=order)

plt.xlabel('Age Group')

plt.ylabel('Count')

plt.title('Age Group Distribution')

plt.show()
lan = []

for i in range(1,13):

    lan.extend(multiChoice_df['Q18_Part_'+str(i)][1:])



f,ax=plt.subplots(1,2,figsize=(20,8))

pd.Series(lan).value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0])

ax[0].set_title('Distribution of Programming Language (Pie Chart)')

ax[0].set_ylabel('')

sns.countplot(pd.Series(lan).values,ax=ax[1])

ax[1].set_title('Distribution of Programming Language (Bar Plot)')

plt.xticks(rotation=35)

plt.show()

country_df = multiChoice_df['Q3'][1:].value_counts()



data = [dict(

        type='choropleth',

        locations = list(country_df.index),

        locationmode='country names',

        z=(country_df.values),

        text=list(country_df.index),

        colorscale='Portland',

        reversescale=True,

)]

layout = dict(

    title = 'A Map About Population of Data Scientists in Each Country',

    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))

)

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='world-map')
countries_map = {"Africa": ["Algeria", "Angola", "Benin", "Botswana", "Burkina", "Burundi", "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo", "Congo, Democratic Republic of", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"], "Asia": ["Afghanistan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Burma (Myanmar)", "Cambodia", "China", "East Timor", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Korea, North", "Korea, South", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Nepal", "Oman", "Pakistan", "Philippines", "Qatar", "Russian Federation", "Saudi Arabia", "Singapore", "Sri Lanka", "Syria", "Tajikistan", "Thailand", "Turkey", "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen", ""], "Europe": ["Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Macedonia", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom", "Vatican City"], "North America": ["Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador", "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States"], "Oceania": ["Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"], "South America": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"]}



countries_map['North America'].extend(['United States of America'])

countries_map['Asia'].extend(['Russia'])

countries_map['Asia'].extend(['South Korea'])
def parse(data):

    for i in countries_map.keys():

        if data in countries_map[i]:

            return i

    return "other"

multiChoice_df['Continent'] = multiChoice_df['Q3'].apply(parse)
multiChoice_df['Continent'].value_counts()
trace = go.Pie(labels = multiChoice_df['Continent'].value_counts().index, 

               values = multiChoice_df['Continent'].value_counts().values, opacity = 0.8,

               textfont=dict(size=15))

layout = dict(title =  'Data Scientists Vs Continents')

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
DP_df = multiChoice_df[['Q2','Continent']][1:].dropna()

DP_df = DP_df.groupby(['Continent'])['Q2'].value_counts()



trace1 = go.Bar(

    x=sorted(multiChoice_df.Continent.unique()),

    y=DP_df.loc[:,'Male'].values,

    name='Male',

    marker = dict(color="rgb(113, 50, 141)")

)

trace2 = go.Bar(

    x=sorted(multiChoice_df.Continent.unique()),

    y=DP_df.loc[:,'Female'].values,

    name='Female',

    marker = dict(color="rgb(119, 74, 175)")

)





trace3 = go.Bar(

    x=sorted(multiChoice_df.Continent.unique()),

    y=DP_df.loc[:,'Prefer not to say'].values,

    name='Prefer not to say',

    marker = dict(color="rgb(120, 100, 202)")

)



trace4 = go.Bar(

    x=sorted(multiChoice_df.Continent.unique()),

    y=DP_df.loc[:,'Prefer to self-describe'].values,

    name='Prefer to self-describe',

    marker = dict(color="rgb(117, 127, 221)")

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    barmode='group',

    title =  'Gender By Continent'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')
DP_df = multiChoice_df[['Q1','Q2']][1:].dropna()

DP_df = DP_df.groupby(['Q1'])['Q2'].value_counts()



trace1 = go.Bar(

    x=sorted(multiChoice_df.Q1.unique()),

    y=DP_df.loc[:,'Male'].values,

    name='Male',

    marker = dict(color="rgb(113, 50, 141)")

)

trace2 = go.Bar(

    x=sorted(multiChoice_df.Q1.unique()),

    y=DP_df.loc[:,'Female'].values,

    name='Female',

    marker = dict(color="rgb(119, 74, 175)")

)





trace3 = go.Bar(

    x=sorted(multiChoice_df.Q1.unique()),

    y=DP_df.loc[:,'Prefer not to say'].values,

    name='Prefer not to say',

    marker = dict(color="rgb(120, 100, 202)")

)



trace4 = go.Bar(

    x=sorted(multiChoice_df.Q1.unique()),

    y=DP_df.loc[:,'Prefer to self-describe'].values,

    name='Prefer to self-describe',

    marker = dict(color="rgb(117, 127, 221)")

)



data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    barmode='group',

    title =  'Gender By Age'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')