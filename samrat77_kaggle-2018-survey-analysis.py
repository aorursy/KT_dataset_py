import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

df = pd.DataFrame(data)
from plotly import tools

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
pd.set_option('display.max_columns',20)

pd.set_option('display.max_rows',30)
df_country = df['Q3'].value_counts().reset_index()

df_country.columns = ['country','People']

print(df_country)
df_country = df_country.replace('United Kingdom of Great Britain and Northern Ireland','UK')

df_country = df_country.replace('Iran, Islamic Republic of...','Iran')

df_country = df_country.replace('I do not wish to disclose my location','NotDisclosed')

df_country = df_country.replace('United States of America','USA')

df_country = df_country.replace('Viet Nam','Vietnam')

df_country = df_country[df_country.country != 'In which country do you currently reside?']

print(df_country)
plt.figure(figsize = (16,9))

df_top_country = df_country.head(15)

sns.barplot(df_top_country.country, df_top_country.People, palette = 'RdYlGn')

plt.title('Top Countries with Maximum numbers of Kaggle Users', fontsize = 20)

plt.xlabel('Countries', fontsize = 12)

plt.ylabel('Numbers of Users', fontsize = 12)

plt.show()
country_code = pd.read_csv("../input/plotly-country-code-mapping/2014_world_gdp_with_codes.csv")

print(country_code)

country_code.columns = [i.lower() for i in country_code.columns]

country_code = country_code.replace({'United States':'USA', 'United Kingdom':'UK'})

df_country = pd.merge(df_country, country_code, on = 'country')

print(df_country.head())
data = [ dict(type = 'choropleth',locations = df_country['code'], z = df_country['People'], text = df_country['country'],

        colorscale = 'inferno',autocolorscale = False, reversescale = True, marker = dict(line = dict (color = 'rgb(180,180,180)',width = 0.5)),

        colorbar = dict(autotick = False, title = 'Responders'))]



layout = dict(title = 'Responders by country',geo = dict(showframe = True, showcoastlines = True, projection = dict(type = 'Mercator')))



fig = dict( data=data, layout=layout )

iplot( fig, validate=False, filename='WorldMap_kagglers' )
df_age = df['Q2'].value_counts().reset_index()

df_age.columns = ['Age_group','Respondents']

df_age = df_age[df_age['Age_group'] != 'What is your age (# years)?']

print(df_age)
plt.figure(figsize = (16,9))

sns.barplot('Age_group','Respondents', data = df_age, palette = 'bright')
df = df[df['Q2'] != 'What is your age (# years)?'] 

pd.crosstab(df.Q3, df.Q2).plot.bar(stacked = True, color = sns.color_palette('inferno',3))

fig = plt.gcf()

fig.set_size_inches(16,9)

plt.title('Age Groups by Country')

plt.show()
print(df['Q5'].unique())
df = df.replace({'Engineering (non-computer focused)':'Engineering(NonComp)', 'Computer science (software engineering, etc.)':'Comp. Sci.',

                'Social sciences (anthropology, psychology, sociology, etc.)':'Social Science','Mathematics or statistics': 'Math & Stats','Physics or astronomy': 'Phy & Astro',

                 'Information technology, networking, or system administration':'Info. Tech.', 'A business discipline (accounting, economics, finance, etc.)': 'Business Descipline',

                 'Environmental science or geology':'Env. Science', 'Medical or life sciences (biology, chemistry, medicine, etc.)':'Medical','I never declared a major': 'NotDeclared',

                 'Humanities (history, literature, philosophy, etc.)': 'Humanities','Fine arts or performing arts':'Fine arts'})

print(df['Q5'].unique())
df_top = df[df['Q3'].isin(df['Q3'].value_counts()[0:14].index)]

print(df_top['Q3'].unique())
df_top = df_top[df_top['Q3'] != 'I do not wish to disclose my location']
df_top = df_top.replace({'United Kingdom of Great Britain and Northern Ireland':'UK','United States of America':'USA'})

print(df_top['Q3'].unique())
pd.crosstab(df_top.Q2 , df_top.Q3).plot(color = sns.color_palette('bright',9))

fig = plt.gcf()

fig.set_size_inches(16,9)

plt.show()
df_top = df_top.replace({'Master’s degree':'Masters', 'Doctoral degree':'Phd','Bachelor’s degree':'Bachelor','Professional degree':'Professional','Some college/university study without earning a bachelor’s degree':'SomeColStudy','No formal education past high school':'High School','I prefer not to answer':'NoAnswer'})

#df_top = df_top[df_top['Q4'] != 'I prefer not to answer']

df_top['Q4'].unique()
pd.crosstab(df_top.Q4 , df_top.Q3).plot(color = sns.color_palette('bright',9))

fig = plt.gcf()

fig.set_size_inches(16,9)





pd.crosstab(df_top.Q3, df_top.Q4).plot.bar(stacked = True, color = sns.color_palette('RdYlGn',7))

fig = plt.gcf()

fig.set_size_inches(16,9)

plt.title('Age Groups by Country')

plt.show()
df_top = df_top[df_top['Q1'] != 'What is your gender? - Selected Choice']

df_top['Q1'].unique()
pd.crosstab(df_top.Q3, df_top.Q1).plot.bar(stacked = True, color = sns.color_palette('bright',7))

fig = plt.gcf()

fig.set_size_inches(16,9)

plt.title('Age Groups by Country')

plt.show()
pd.crosstab(df_top.Q3, df_top.Q2).plot.bar(stacked = True, color = sns.color_palette('bright',3))

fig = plt.gcf()

fig.set_size_inches(16,9)

plt.title('Age Groups by Top 15 Countries')

plt.show()
pd.crosstab(df_top.Q3, df_top.Q5).plot.bar(stacked = True, color = sns.color_palette('pastel',3))

fig = plt.gcf()

fig.set_size_inches(16,9)

plt.title('Different Streams by Country')

plt.show()