# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# import plotly.plotly as py

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import matplotlib as ml

import matplotlib.pyplot as plt

%matplotlib inline

ml.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cont = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')

print(cont.shape)

cont.head()
cont.info()
cont.describe()
type(cont['Infant mortality (per 1000 births)'].values[2])
# Creating copy

cnt = cont.copy()

cnt.head()
# Renaming columns for better usability

new_column_name = {'Area (sq. mi.)':'Area' , 'Pop. Density (per sq. mi.)':'Pop_density' , 

                  'Coastline (coast/area ratio)':'Coastline' , 

                  'Infant mortality (per 1000 births)':'Infant_mortality' , 'GDP ($ per capita)':'GDP_per_capita' ,

                  'Literacy (%)':'Literacy_percent' , 'Phones (per 1000)':'Phones_per_k' , 'Arable (%)':'Arable' ,

                   'Crops (%)':'Crops' ,'Other (%)':'Other'}

cnt = cnt.rename(columns = new_column_name )

cnt
cnt = cnt.fillna(0)
cnt.isnull().sum()
cnt.info()
'''plt.figure(figsize=(20,10))

sns.heatmap(cnt.corr(),annot=True)

plt.show()'''



# Can't test before tidiness issues are handled.
# Nothing to test, no changes made
def rectify(cols):

    for c in cols:

        cnt[c] = cnt[c].astype(str)

        new_data = []

        for val in cnt[c]:

            val = val.replace(',','.')

            val = float(val)

            new_data.append(val)



        cnt[c] = new_data



# Running on dataset

cols = cnt[['Pop_density' , 'Coastline' , 'Net migration' , 'Infant_mortality' , 

                   'Literacy_percent' , 'Phones_per_k' , 'Arable' , 'Crops' , 'Other' , 'Climate' , 'Birthrate' , 'Deathrate' , 'Agriculture' ,

                   'Industry' , 'Service']]

rectify(cols)
cnt.head()
cnt.Climate.unique()
cnt['Climate'] = cnt['Climate'].astype('int')
cnt.Climate.unique()
cnt.Climate.value_counts().sort_values(ascending=False)
cnt.Climate.replace(0,2,inplace=True)
cnt.Climate.unique()
plt.figure(figsize=(20,10))

sns.heatmap(cnt.corr(),annot=True)

plt.show()
cont = cnt

cont.head()
cont.info()
cont.describe()
cont.isnull().sum()
cont.head()
plt.figure(figsize=(20,10))

sns.barplot(data = cnt.nlargest(20, 'Population'), x = 'Country', y = 'Population')

plt.title("TOP 20 MOST POPULATED COUNTRIES")

plt.show()



# Region

plt.figure(figsize=(20,10))

sns.barplot(data = cnt.nlargest(20, 'Population'), x = 'Region', y = 'Population')

plt.title("MOST POPULATED REGIONS")

plt.show()
# Group data together

hist_data = [cont['Infant_mortality'], cnt['Birthrate'], cnt['Deathrate']]



group_labels = ['Infant_mortality', 'Birth Rate', 'Death Rate']



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.update_layout(

    margin=dict(l=10, b=10))

fig.show()
fig = go.Figure(data=[

    go.Bar(name='INFANT MORTALITY', x=cont.nlargest(10, 'Population')['Country'], y=cont['Infant_mortality']),

    go.Bar(name='BIRTH RATE', x=cont.nlargest(10, 'Population')['Country'], y=cont['Birthrate']),

    go.Bar(name='DEATH RATE', x=cont.nlargest(10, 'Population')['Country'], y=cont['Deathrate'])

])



fig.update_layout(barmode='group')

fig.show()
trace1 = go.Scatter(

    x = cont.index,

    y = cont.Deathrate,

    mode = 'lines+markers',

    name = 'Death Rate',

    marker = dict(color = 'rgba(255, 81, 51, 0.5)'),

    text = cont.Country)



trace2 = go.Scatter(

    x = cont.index,

    y = cont.Birthrate,

    mode = 'lines+markers',

    name = 'Birth Rate',

    marker = dict(color = 'rgba(105, 100, 255, 0.5)'),

    text = cont.Country)



layout = dict(title = 'Birth Rate v/s Death Rate of Countries',

             xaxis= dict(zeroline= False)

             )



data = [trace1, trace2]



fig = dict(data = data, layout = layout)



iplot(fig)
cont[cont.Deathrate > cont.Birthrate]
perc = cont[cont.Deathrate > cont.Birthrate].shape[0]

perc
# PERCENTAGE OF COUNTRIES WITH HIGHER DEATH RATE

fig = go.Figure(data=[go.Pie(labels=['Death Rate > Birth Rate','Birth Rate > Death Rate'], values=[perc,(cont.shape[0]-perc)])])

fig.show()
plt.figure(figsize=(20,10))

sns.barplot(data = cont.nlargest(10, 'GDP_per_capita'), x = 'Country', y = 'Agriculture')

plt.title("ANALYSIS OF AGRICULTURE IN TOP 10 COUNTRIES WITH THE HIGHEST GDP_per_capita")

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(data = cont.nlargest(10, 'GDP_per_capita'), x = 'Country', y = 'Industry')

plt.title("ANALYSIS OF INDUSTRY IN TOP 10 COUNTRIES WITH THE HIGHEST GDP_per_capita")

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(data = cont.nlargest(10, 'GDP_per_capita'), x = 'Country', y = 'Service')

plt.title("ANALYSIS OF SERVICE IN TOP 10 COUNTRIES WITH THE HIGHEST GDP_per_capita")

plt.show()
cnt[cnt['Country'] == "San Marino "]
cont_gdp_sorted = pd.DataFrame(cont.sort_values(ascending=False,by=['GDP_per_capita']))

cont_gdp = cont_gdp_sorted.nlargest(30,'GDP_per_capita')



trace0 = go.Bar(

    x = cont_gdp.Country,

    y = cont_gdp['Agriculture'],

    name = "Agriculture",

    marker = dict(color = 'rgba(255, 26, 26, 0.5)',

                    line=dict(color='rgb(100,100,100)',width=3)))



trace1 = go.Bar(

    x = cont_gdp.Country,

    y = cont_gdp['Industry'],

    name = "Industry",

    marker = dict(color = 'rgba(255, 255, 51, 0.5)',

                line=dict(color='rgb(100,100,100)',width=3)))



trace2 = go.Bar(

    x = cont_gdp.Country,

    y = cont_gdp['Service'],

    name = "Service",

    marker = dict(color = 'rgba(77, 77, 255, 0.5)',

                    line=dict(color='rgb(100,100,100)',width=3)))



data = [trace0, trace1, trace2]

layout = go.Layout(barmode = "stack")

fig = go.Figure(data = data,layout = layout)

iplot(fig)
sns.jointplot(x="GDP_per_capita", y="Agriculture", data=cont, height=10, ratio=3, color="g")

plt.show()
sns.jointplot(x="GDP_per_capita", y="Industry", data=cont, height=10, ratio=3, color="y")

plt.show()
sns.jointplot(x="GDP_per_capita", y="Service", data=cont, height=10, ratio=3, color="maroon")

plt.show()
cont_lit_sorted = pd.DataFrame(cont.sort_values(ascending=False,by=['Literacy_percent'])).head(20)



fig = go.Figure([go.Bar(x=cont_lit_sorted.Country, y=cont_lit_sorted.Literacy_percent)])

fig.update_traces(marker_color='rgb(225,140,160)', marker_line_color='rgb(110,48,10)',

                  marker_line_width=1.5)

fig.show()
cont_lit_sorted
plt.figure(figsize=(20,10))

sns.distplot(cont_lit_sorted['Literacy_percent'])

plt.show()
fig = go.Figure(data=[go.Bar(x=cont.nlargest(20,'Population')['Country'], y=cont.nlargest(20,'Population')['Literacy_percent'])])

fig.update_layout(title_text='LITERACY RATES OF 20 MOST POPULATED COUNTRIES')

fig.show()
sns.jointplot(x="Literacy_percent", y="Population", data=cont, height=10, ratio=3, color="g")

plt.show()
fig = go.Figure(data=[go.Bar(x=cont.nlargest(20,'GDP_per_capita')['Country'], y=cont.nlargest(20,'GDP_per_capita')['Literacy_percent'])])

fig.update_layout(title_text='LITERACY RATES OF 20 HIGH GDP COUNTIRES')

fig.show()
sns.jointplot(x="Literacy_percent", y="GDP_per_capita", data=cont, height=10, ratio=3, color="r")

plt.show()