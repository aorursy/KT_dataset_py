#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/task_2-owid_covid_data-21_June_2020.csv')

df.head()
na = (df.isnull().sum() / len(df)) * 100

na = na.drop(na[na == 0].index).sort_values(ascending=False)



f, ax = plt.subplots(figsize=(12,8))

sns.barplot(x=na.index, y=na)

plt.xticks(rotation='90')

plt.xlabel('Features', fontsize=15)

plt.title('Percentage Missing', fontsize=15)
# filling missing values with NA

df[['new_tests', 'new_tests_per_thousand', 'total_tests_per_thousand', 'total_tests', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units', 'handwashing_facilities', 'extreme_poverty', 'male_smokers', 'female_smokers', 'stringency_index', 'hospital_beds_per_thousand', 'gdp_per_capita', 'aged_65_older', 'aged_70_older', 'median_age', 'cvd_death_rate', 'diabetes_prevalence', 'population_density', 'life_expectancy', 'new_deaths_per_million', 'total_deaths_per_million', 'new_deaths', 'total_deaths', 'new_cases_per_million', 'total_cases_per_million', 'new_cases', 'total_cases', 'continent']] = df[['new_tests', 'new_tests_per_thousand', 'total_tests_per_thousand', 'total_tests', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_units', 'handwashing_facilities', 'extreme_poverty', 'male_smokers', 'female_smokers', 'stringency_index', 'hospital_beds_per_thousand', 'gdp_per_capita', 'aged_65_older', 'aged_70_older', 'median_age', 'cvd_death_rate', 'diabetes_prevalence', 'population_density', 'life_expectancy', 'new_deaths_per_million', 'total_deaths_per_million', 'new_deaths', 'total_deaths', 'new_cases_per_million', 'total_cases_per_million', 'new_cases', 'total_cases', 'continent']].fillna('NA')
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)

plt.show()
plt.style.use('fivethirtyeight')

sns.countplot(df['continent'],linewidth=3,palette="Set2", edgecolor='black')

plt.show()
ax = df['continent'].value_counts().plot.barh(figsize=(14, 6))

ax.set_title('Continent Total Cases  Distribution', size=18)

ax.set_ylabel('Continent', size=14)

ax.set_xlabel('total_cases', size=14)
ax = df['diabetes_prevalence'].value_counts().plot.barh(figsize=(14, 6), color='r')

ax.set_title('Diabetes Prevalence in the 65-older Group', size=18)

ax.set_ylabel('Diabetes Prevalence', size=14)

ax.set_xlabel('aged_65_older', size=14)
import plotly.express as px

fig = px.scatter_ternary(df, a="population", b="diabetes_prevalence",c="aged_65_older")

fig.show()
plt.style.use('dark_background')

ax = df['extreme_poverty'].value_counts().plot.barh(figsize=(14, 6), color='b')

ax.set_title('GDP per Capita vs Extreme Poverty', size=18)

ax.set_ylabel('Extreme Poverty', size=14)

ax.set_xlabel('gdp_per_capita', size=14)
plt.style.use('dark_background')

ax = df['cvd_death_rate'].value_counts().plot.barh(figsize=(14, 6), color='g')

ax.set_title('Handwashing Facilities vs Covid19 Death Rate', size=18)

ax.set_ylabel('Covid-19 Death Rate', size=14)

ax.set_xlabel('handwashing_facilities', size=14)
plt.style.use('dark_background')

ax = df.plot(figsize=(15,8), title='Covid19 Death Rate linked to Diabetes Prevalence in older 65 and Handwashing Facilities ')

ax.set_xlabel('diabetes_prevalence, aged_65_older, cvd_death_rate , handwashing_facilities ')

ax.set_ylabel('continent')
plt.style.use('dark_background')

from pandas.plotting import scatter_matrix

scatter_matrix(df, figsize= (4,4), diagonal='kde', color = 'b')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
fig = px.line_3d(df, x="total_cases", y="total_deaths", z="total_tests",color = "continent")

fig.show()
fig = px.line_3d(df, x="diabetes_prevalence", y="cvd_death_rate", z="aged_65_older",color = "continent")

fig.show()
fig = go.Figure()





fig.add_trace(go.Scatter(x=df.index, y=df['total_cases'],

                    mode='lines',marker_color='yellow',

                    name='total_tests',line=dict( dash='dot')))



fig.update_layout(

    title='Total tests vs. Total Cases',

        template='plotly_dark'



)



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=df.index, y=df['diabetes_prevalence'],name='aged_65_older',

                                   marker_color='black',mode='lines',line=dict( dash='dot') ))



fig.update_layout(

    title='Diabetes Prevalence in People Older than 65',

        template='plotly_white'



)



fig.show()
fig = px.pie(df,

             values="total_cases",

             names="continent",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['life_expectancy']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['population']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
plt.style.use('fivethirtyeight')

sns.countplot(df['stringency_index'],linewidth=3,palette="Set2", edgecolor='black')

plt.show()
fig=sns.lmplot(x="male_smokers", y="life_expectancy",data=df)
from plotly.subplots import make_subplots



Continents = df["continent"].value_counts().nlargest(n=10)

Continents_all = df["tests_units"].value_counts()

fig = make_subplots(1,2, 

                    subplot_titles = ["Continents", 

                                      "Tests Units"])

fig.append_trace(go.Bar(y=Continents.index,

                          x=Continents, 

                          orientation='h',

                          marker=dict(color=Continents.values, coloraxis="coloraxis", showscale=False),

                          texttemplate = "%{value:,s}",

                          textposition = "inside",

                          name="Tests Units by Continents",

                          showlegend=False),

                

                 row=1,

                 col=1)

fig.update_traces(opacity=0.7)

fig.update_layout(coloraxis=dict(colorscale='tealrose'))

fig.append_trace(go.Scatter(x=Continents_all.index,

                          y=Continents_all, 

                          line=dict(color="#008B8B",

                                    width=2),

                          showlegend=False),

                 row=1,

                 col=2)

fig.update_layout(showlegend=False)

fig.show()
Tests = df['tests_units'].value_counts().nlargest(n=10)

fig = px.pie(Tests, 

       values = Tests.values, 

       names = Tests.index, 

       title="Tests Units", 

       color=Tests.values,

       color_discrete_sequence=px.colors.qualitative.Prism)

fig.update_traces(opacity=0.7,

                  marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5)

fig.update_layout(title_x=0.5)

fig.show()
Beds = df['hospital_beds_per_thousand'].value_counts().nlargest(n=10)

fig = px.bar(y=Beds.values,

       x=Beds.index,

       orientation='v',

       color=Beds.index,

       text=Beds.values,

       color_discrete_sequence= px.colors.qualitative.Bold)



fig.update_traces(texttemplate='%{text:.2s}', 

                  textposition='outside', 

                  marker_line_color='rgb(8,48,107)', 

                  marker_line_width=1.5, 

                  opacity=0.7)



fig.update_layout(width=800, 

                  showlegend=False, 

                  xaxis_title="Hospital Beds",

                  yaxis_title="Count",

                  title="Hospital Beds per Thousand")

fig.show()
from nltk.tokenize import RegexpTokenizer

from wordcloud import WordCloud



location_words = df['location'].dropna().to_list()

tokenizer = RegexpTokenizer(r'\w+')

tokenized_list = [tokenizer.tokenize(i) for i in location_words]

tokenized_list = [w for l in tokenized_list for w in l]



tokenized_list = [w.lower() for w in tokenized_list]

string = " ".join(w for w in tokenized_list)

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='black', colormap='Set2',

                min_font_size = 10).generate(string) 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
from IPython.display import IFrame

IFrame('https://app.powerbi.com/view?r=eyJrIjoiMjcxNDIyNjAtOGM0Yi00ZWJhLWJkNmEtNjFiOTI0MWVlYjNiIiwidCI6IjI1NmNiNTA1LTAzOWYtNGZiMi04NWE2LWEzZTgzMzI4NTU3OCIsImMiOjh9', width=800, height=500)