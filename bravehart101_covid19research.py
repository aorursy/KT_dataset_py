# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the dataset

perMilTestDF = pd.read_csv('/kaggle/input/uncover/UNCOVER/our_world_in_data/per-million-people-tests-conducted-vs-total-confirmed-cases-of-covid-19.csv')



#Grouping the data by country

countryDF = perMilTestDF.groupby(perMilTestDF['entity'])



height = 10

width = 10

plt.figure(figsize = (height, width))

ax = plt.subplot()



#PLotting data for India

x_india = list(range(1,87))

y_india = countryDF.get_group('India')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

ax.plot(x_india, y_india, 'r', label = 'India')



#Plotting data for USA

x_us = list(range(1,88))

y_us = countryDF.get_group('United States')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

ax.plot(x_us, y_us, 'g', label = 'USA')



#Plotting data for Italy

x_italy = list(range(1,88))

y_italy = countryDF.get_group('Italy')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

ax.plot(x_italy, y_italy, 'y', label = 'Italy')



#Plotting data for China

x_china = list(range(1,88))

y_china = countryDF.get_group('China')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

ax.plot(x_china, y_china, 'b', label = 'China')



#Plotting data for Spain

x_spain = list(range(1,88))

y_spain = countryDF.get_group('Spain')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

ax.plot(x_spain, y_spain, 'm', label = 'Spain')



#Plotting data for Germany

x_ger = list(range(1,88))

y_ger = countryDF.get_group('Germany')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

ax.plot(x_ger, y_ger, 'k', label = 'Germany')



plt.title('SPREAD OF PANDEMIC IN DIFFERENT COUNTRIES')

plt.xlabel('NUMBER OF DAYS SINCE OUTBREAK')

plt.ylabel('NUMBER OF CASES')

#plt.legend([line1, line2, line3, line4], ['India', 'USA', 'Italy', 'China'])

ax.legend()

plt.show()
# Loading the dataset 

canada_age_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/howsmyflattening/canada-testing-data.csv')



# Clearing up the dataset

canada_age_df.replace('Not Reported', np.nan, inplace = True)

canada_age_df.dropna(subset = ['age'], axis=0, inplace = True)

canada_age_df['age'].replace({'<20':'10-19', '<18':'10-19', '<10':'0-9', '61':'60-69', '<1':'0-9', '2':'0-9', '50':'50-59'}, inplace = True)



# PLotting the data

ageVScovid = canada_age_df['age'].value_counts()

ageVScovid = ageVScovid[:]

plt.figure(figsize = (10,10))

sns.barplot(ageVScovid.index, ageVScovid.values)

plt.xlabel('AGE GROUPS')

plt.ylabel('NUMBER OF CASES')

plt.title('CASES OBSERVED IN DIFFERENT AGE GROUPS')

plt.show()

plt.close()
# Loading the dataset

canada_sex_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/howsmyflattening/canada-testing-data.csv')



# Cleaning up the dataset

canada_sex_df.replace('Not Reported', np.nan, inplace = True)

canada_sex_df.dropna(subset = ['sex'], axis=0, inplace = True)



# PLotting the data

sexVScovid = canada_sex_df['sex'].value_counts()

sexVScovid = sexVScovid[:]

plt.figure(figsize = (5,5))

sns.barplot(sexVScovid.index, sexVScovid.values)

plt.xlabel('SEX')

plt.ylabel('NUMBER OF CASES')

plt.title('CASES OBSERVED (MALES vs FEMALES)')

plt.show()

plt.close()
# Importing the dataset

df = pd.read_csv('/kaggle/input/uncover/UNCOVER/nextstrain/covid-19-genetic-phylogeny.csv')

df = df[['age','sex']]

df.replace({'?': np.nan, 'Unknown': np.nan, 'U': np.nan, 'unknwon': np.nan, 'FEmale': 'Female' }, inplace = True)

df.dropna(inplace = True)

df = df.astype({'age': 'float64', 'sex': 'category'})



# Binning the age data

bins = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])

df['age_grp'] = pd.cut(df['age'], bins = bins, labels = ['0-20','20-40','40-60','60-80','80+'])



# Getting the number of male and female patients

grp = df.groupby(['age_grp'])

df = []

for key, item in grp:

    cases = item['sex'].value_counts().to_list()

    df.append([key, cases[0], cases[1]])

df = pd.DataFrame(df)

df.columns = ['age_grp', 'males', 'females']



# Plotting on a population graph

import plotly.graph_objs as go

import plotly.io as pio

           

women_bins = np.array(df['females'])*-1

men_bins = np.array(df['males'])



y = [10, 30, 50, 70, 90]



fig = dict(layout = go.Layout(yaxis=go.layout.YAxis(title='AGE GROUPS'),

                   xaxis=go.layout.XAxis(

                       range=[-800, 800],

                       tickvals=[-1000, -700, -300, 0, 300, 700, 1000],

                       ticktext=[1000, 700, 300, 0, 300, 700, 1000],

                       title='NUMBER OF PATIENTS'),

                   barmode='overlay',

                   bargap=0.1),



        data = [go.Bar(y=y,

               x=men_bins,

               orientation='h',

               name='Males',

               hoverinfo='x',

               marker=dict(color='#3283FE')

               ),

        go.Bar(y=y,

               x=women_bins,

               orientation='h',

               name='Females',

               text=-1 * women_bins.astype('int'),

               hoverinfo='text',

               marker=dict(color='#FE00FA')

               )]

            )



pio.show(fig)
# Loading the dataset

df = pd.read_csv('/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-mortality.csv')



# Cleaning up the data

df.replace('Not Reported', np.nan, inplace = True)

df.dropna(subset = ['age'], axis=0, inplace = True)

df['age'].replace({'>70':'70-79','78':'70-79', '>50':'50-59', '>65':'60-69','61':'60-69', '82':'80-89','83':'80-89','>80':'80-89', '92':'90-99'}, inplace = True)



# Plotting the data

ageVSdeath = df['age'].value_counts()

ageVSdeath = ageVSdeath[:]



#Plotting bar graph 

plt.figure(figsize = (10,10))

sns.barplot(ageVSdeath.index, ageVSdeath.values)

plt.xlabel('AGE GROUPS')

plt.ylabel('NUMBER OF DEATHS')

plt.title('DEATHS OBSERVED IN DIFFERENT AGE GROUPS')

plt.show()

plt.close()



#PLotting pie chart

df['age'].replace({'70-79':'60+', '80-89':'60+', '90-99':'60+', '60-69':'60+', '50-59':'20-60', '40-49':'20-60', '100-109':'60+', '30-39':'20-60', '20-29':'20-60'}, inplace = True)

ageVSdeath = df['age'].value_counts()

ageVSdeath = ageVSdeath[:]

plt.figure(figsize = (10,10))

plt.pie(ageVSdeath.values, labels = ageVSdeath.index, startangle = 90)

plt.title('AGE vs DEATHS')

plt.show()

plt.close()
# Loading the dataset

RiskData = pd.read_csv('/kaggle/input/uncover/UNCOVER/HDE_update/inform-covid-indicators.csv')



# Plotting the graph on a geomap



fig = go.Figure(data=go.Choropleth(

    locations = RiskData['iso3'],

    z = RiskData['inform_risk'],

    text = RiskData['country'],

    colorscale='bluered_r',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'RISK</br></br>FACTOR'

))



fig.update_layout(

    title_text='MOST DANGEROUS COUNTRIES TO LIVE IN DURING A PANDEMIC- HUMAN DATA EXCHANGE',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    ),

    annotations = [dict(

        x=0.55,

        y=0.1,

        xref='paper',

        yref='paper',

        text='',

        showarrow = False

    )]

)



fig.show()
df = pd.read_csv('/kaggle/input/uncover/UNCOVER/HDE_update/acaps-covid-19-government-measures-dataset.csv')

measures = df['country'].value_counts().to_frame()

measures = measures.reset_index()

measures.columns = ['country', 'total_govt_measures']



country = list(measures['country'])

countryDF = list(df['country'])

isoDF = list(df['iso'])



measures['iso'] = [isoDF[countryDF.index(i)] for i in country]



# Plotting the graph on a geomap



fig = go.Figure(data=go.Choropleth(

    locations = measures['iso'],

    z = measures['total_govt_measures'],

    text = measures['country'],

    colorscale='reds_r',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'TOTAL</br></br>MEASURES</br>TAKEN'

))



fig.update_layout(

    title_text='TOTAL MEASURES TAKEN BY DIFFERENT COUNTRIES OF THE WORLD',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    ),

    annotations = [dict(

        x=0.55,

        y=0.1,

        xref='paper',

        yref='paper',

        text='',

        showarrow = False

    )]

)



fig.show()