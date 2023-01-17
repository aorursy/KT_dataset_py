!pip install calmap
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

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
df = pd.read_csv('../input/hackathon/BCG_COVID-19_clinical_trials-2020_06_06.csv', encoding='ISO-8859-2')

df.head()
df = df.rename(columns={'Study Start Date':'date'})
df.date = pd.to_datetime(df.date)
import calmap
fig,ax = calmap.calendarplot(df.groupby(['date']).Country.count(), monthticks=1, daylabels='MTWTFSS',cmap='PuRd',

                    linewidth=0, fig_kws=dict(figsize=(20,20)))

fig.show()
cnt_srs = df['Strain'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='BCG strain that has been used',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Strain")
plt.style.use('fivethirtyeight')

sns.countplot(df['Strain'],linewidth=3,palette="Set2",edgecolor='black')

plt.title('BCG Strain used')

plt.xticks(rotation=45)

plt.show()
#Codes from Mario Filho https://www.kaggle.com/mariofilho/live26-https-youtu-be-zseefujo0zq

from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['Strain']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Target Sample Size']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()



plt.yticks(rotation=45)

plt.title('BCG Strain used')
df1 = pd.read_csv('../input/hackathon/task_2-BCG_world_atlas_data-bcg_strain-7July2020.csv', encoding='utf8')

df1.head()
plt.style.use('fivethirtyeight')

sns.countplot(df1['vaccination_timing'],linewidth=3,palette="Set2",edgecolor='black')

plt.title('Vaccination Timing')

plt.xticks(rotation=45)

figsize=(16, 10)

plt.show()
plt.style.use('fivethirtyeight')

sns.countplot(df1['were_revaccinations_recommended'],linewidth=3,palette="Set2",edgecolor='black')

plt.title('Were Revaccinations Recommended?')

plt.xticks(rotation=45)

figsize=(16, 10)

plt.show()
plt.style.use('fivethirtyeight')

sns.countplot(df1['timing_of_revaccination'],linewidth=3,palette="Set2",edgecolor='black')

plt.title('Revaccination Timing')

plt.xticks(rotation=45)

figsize=(16, 10)

plt.show()
ax = df1['vaccination_timing'].value_counts().plot.barh(figsize=(16, 8))

ax.set_title('BCG Vaccination timing', size=18)

ax.set_ylabel('vaccination_timing', size=10)

ax.set_xlabel('bcg_strain_id', size=10)
US = df1[(df1['country_name']=='United States of America')].reset_index(drop=True)

US.head()
df2 = pd.read_csv('../input/hackathon/BCG_country_data.csv', encoding='ISO-8859-2')

df2.head()
plt.style.use('fivethirtyeight')

sns.countplot(df2['lockdown_start'],linewidth=3,palette="Set2",edgecolor='black')

plt.title('Lockdown Measures')

plt.xticks(rotation=45)

figsize=(12, 8)

plt.show()
plt.style.use('fivethirtyeight')

sns.countplot(df2['lockdown_end'],linewidth=3,palette="Set2",edgecolor='black')

plt.title('Lockdown Measures')

plt.xticks(rotation=45)

figsize=(12, 8)

plt.show()
ax = df2['lockdown_start'].value_counts().plot.barh(figsize=(16, 8))

ax.set_title('Lockdown Start', size=18)

ax.set_ylabel('lockdown_start', size=10)

ax.set_xlabel('country_name', size=10)
fig = px.line(df2, x="lockdown_start", y="country_name", color_discrete_sequence=['darkseagreen'], 

              title="Lockdown Measures Start")

fig.show()
fig = px.bar(df2, 

             x='median_age', y='population_per_km2', color_discrete_sequence=['crimson'],

             title='Average Age of Population Tested for Covid19', text='covid_19_test_cumulative_total')

fig.show()
fig = px.bar(df2, 

             x='hospital_bed_per_1000_people', y='country_name', color_discrete_sequence=['#27F1E7'],

             title='Hospital Beds by Country and Covid19 Tests', text='covid_19_test_cumulative_total')

fig.show()