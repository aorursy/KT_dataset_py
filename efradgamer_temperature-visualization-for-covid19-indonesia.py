'''Import basic modules.'''

import pandas as pd

import numpy as np





'''Customize visualization

Seaborn and matplotlib visualization.'''

import altair as alt

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline

import folium 

from IPython.core.display import HTML

import urllib.request

from PIL import Image

from wordcloud import WordCloud ,STOPWORDS





'''Plotly visualization .'''

import plotly.express as px

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))



import warnings

warnings.filterwarnings('ignore')
complete = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

complete.head()
indonesia = complete[complete['Country/Region'] == 'Indonesia']

display(indonesia.head(3))

display(indonesia.tail(3))

display(indonesia.shape)
# Defining COVID-19 cases as per classifications 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Defining Active Case: Active Case = confirmed - deaths - recovered

complete['Active'] = complete['Confirmed'] - complete['Deaths'] - complete['Recovered']



# latest

full_latest = complete[complete['Date'] == max(complete['Date'])].reset_index()



# latest condensed

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()



temp = complete.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp.head()
confirmed = full_latest_grouped.sum().Confirmed

deaths =  full_latest_grouped.sum().Deaths

recovered =  full_latest_grouped.sum().Recovered

active =  full_latest_grouped.sum().Active

row = [temp.iloc[-1,].Date,confirmed,deaths,recovered,active]

latest = pd.DataFrame([row], columns = full_latest_grouped.columns)

latest.style.background_gradient(cmap='Blues')
temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f = temp_f.iloc[:50]

temp_f.head().style.background_gradient(cmap='Blues')
print('The most cases are confirmed:',temp_f.Confirmed.max())

print('The fewest cases are confirmed:',temp_f.Confirmed.min())
ratio_r = list()

ratio_d = list()

for i in range(len(temp_f)):

    ratio_r.append(temp_f.iloc[i,3]/temp_f.iloc[i,1])

    ratio_d.append(temp_f.iloc[i,2]/temp_f.iloc[i,1])

temp_f['Ratio_Recovered'] = ratio_r

temp_f['Ratio_Deaths'] = ratio_d



temp_ff = temp_f.sort_values(by='Ratio_Deaths', ascending=False)

temp_bb = temp_f.sort_values(by='Ratio_Recovered', ascending=False)

display(temp_ff.iloc[:10,].style.background_gradient(cmap='Blues'))

display(temp_bb.iloc[:10,].style.background_gradient(cmap='Blues'))

fig = px.bar(temp_f.sort_values('Ratio_Deaths', ascending=False).head(15), 

             y="Ratio_Deaths", x="Country/Region", color= "Country/Region",

             title='Top 15 Ratio_Deaths', 

             orientation='v',

             color_discrete_sequence = px.colors.cyclical.IceFire,

             width=700, height=600)

fig.update_traces(opacity=0.8)



fig.update_layout(template = 'plotly_white')

fig.show()
fig = px.bar(temp_f.sort_values('Ratio_Recovered', ascending=False).head(15), 

             y="Ratio_Recovered", x="Country/Region", color= "Country/Region",

             title='Top 15 Ratio_Recovered', 

             orientation='v',

             color_discrete_sequence = px.colors.cyclical.IceFire,

             width=700, height=600)

fig.update_traces(opacity=0.8)



fig.update_layout(template = 'plotly_white')

fig.show()
temp_f['Best_Ratio'] = temp_f['Ratio_Recovered'] - temp_f['Ratio_Deaths']
fig,axes = plt.subplots(1,1, figsize=(12,8))



sns.barplot(x='Best_Ratio', y='Country/Region', data=temp_f.sort_values(by='Best_Ratio'), palette='winter_r',ax=axes)

for i,p in enumerate(axes.patches):

    if i == 8:

        height = p.get_height()

        axes.annotate('Indonesia has {:.2f}% Best_Ratio '.format(p.get_width()),xy = (p.get_width(),p.get_xy()[1]),xytext = (p.get_x()+ p.get_width()+0.05, height+4),

                      arrowprops = dict(facecolor='black',shrink=0.05))

        

        

plt.title('Top 50 most cases Confirmed Best_Ratio', fontsize=20)

plt.show()
world_temp_2020 = pd.read_csv('/kaggle/input/world-average-temperature/Avg_World_Temp_2020.csv')

Continent = world_temp_2020.Continent



world_temp_2020 = world_temp_2020.iloc[:,:-9].drop('Unnamed: 0', axis=1)

world_temp_2020.drop('City',axis=1,inplace=True)



world_temp_2020['Avg_temp'] = world_temp_2020.groupby('Country').transform(lambda x: x.mean()).mean(axis=1)

world_temp_2020['Continent'] = Continent



Country = full_latest_grouped[full_latest_grouped['Country/Region'].isin(world_temp_2020['Country'])]

Country['Country'] = Country['Country/Region']



full = pd.merge(Country, world_temp_2020, on='Country', how='left')

full.drop(columns = ['Country/Region','Jan','Feb','Mar','Apr','May'],inplace=True)
display(world_temp_2020.head())

display(full.head())
africa_temp = full[full['Continent'] == 'Africa']

asia_temp = full[full['Continent'] == 'Asia']

europe_temp = full[full['Continent'] == 'Europe']

na_temp = full[full['Continent'] == 'North America']

ocenia_temp = full[full['Continent'] == 'Oceania']

sa_temp = full[full['Continent'] == 'South America']
fig, ax = plt.subplots(figsize = (18,8))

ax.scatter(asia_temp['Avg_temp'], asia_temp['Confirmed'], marker='v')

plt.xlabel('Average Temperature')

plt.ylabel('Confirmed Cases')

plt.title('Asia Temperature vs Confirmed Cases', fontsize=15)

for i, txt in zip(asia_temp.index,asia_temp.Country):

    ax.annotate(txt, (asia_temp['Avg_temp'][i], asia_temp['Confirmed'][i]),fontsize=13)
fig, ax = plt.subplots(figsize = (18,8))

ax.scatter(africa_temp['Avg_temp'], africa_temp['Confirmed'], marker='v')

plt.xlabel('Average Temperature')

plt.ylabel('Confirmed Cases')

plt.title('Africa Temperature vs Confirmed Cases', fontsize=15)

for i, txt in zip(africa_temp.index,africa_temp.Country):

    ax.annotate(txt, (africa_temp['Avg_temp'][i], africa_temp['Confirmed'][i]),fontsize=13)
fig, ax = plt.subplots(figsize = (18,8))

ax.scatter(europe_temp['Avg_temp'], europe_temp['Confirmed'], marker='v')

plt.xlabel('Average Temperature')

plt.ylabel('Confirmed Cases')

plt.title('Europe Temperature vs Confirmed Cases', fontsize=15)

for i, txt in zip(europe_temp.index,europe_temp.Country):

    ax.annotate(txt, (europe_temp['Avg_temp'][i], europe_temp['Confirmed'][i]),fontsize=13)
fig, ax = plt.subplots(figsize = (18,8))

ax.scatter(na_temp['Avg_temp'], na_temp['Confirmed'], marker='v')

plt.xlabel('Average Temperature')

plt.ylabel('Confirmed Cases')

plt.title('North America Temperature vs Confirmed Cases', fontsize=15)

for i, txt in zip(na_temp.index,na_temp.Country):

    ax.annotate(txt, (na_temp['Avg_temp'][i], na_temp['Confirmed'][i]),fontsize=13)
fig, ax = plt.subplots(figsize = (18,8))

ax.scatter(ocenia_temp['Avg_temp'], ocenia_temp['Confirmed'], marker='v')

plt.xlabel('Average Temperature')

plt.ylabel('Confirmed Cases')

plt.title('Oceania Temperature vs Confirmed Cases', fontsize=15)

for i, txt in zip(ocenia_temp.index,ocenia_temp.Country):

    ax.annotate(txt, (ocenia_temp['Avg_temp'][i], ocenia_temp['Confirmed'][i]),fontsize=13)
fig, ax = plt.subplots(figsize = (18,8))

ax.scatter(sa_temp['Avg_temp'], sa_temp['Confirmed'], marker='v' )

plt.xlabel('Average Temperature')

plt.ylabel('Confirmed Cases')

plt.title('South America Temperature vs Confirmed Cases', fontsize=15)

for i, txt in zip(sa_temp.index,sa_temp.Country):

    ax.annotate(txt, (sa_temp['Avg_temp'][i], sa_temp['Confirmed'][i]),fontsize=13)
import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot



init_notebook_mode(connected = True)

import plotly.graph_objs as go
trace2 = [go.Choropleth(

            colorscale = 'Blues',

            locationmode = 'country names',

            locations = full['Country'],

            text = full['Country'],

            z = full['Avg_temp'],colorbar= dict(title='Temperature')

)]



layout = go.Layout(title = 'Country vs Temperature')



fig = go.Figure(data = trace2, layout = layout)

py.iplot(fig)





trace = [go.Choropleth(

            colorscale = 'Blues',

            locationmode = 'country names',

            locations = full['Country'],

            text = full['Country'],

            z = full['Confirmed'],colorbar = dict(title='Confirmed Cases')

)]



layout = go.Layout(title = 'Country vs Confirmed')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)
indonesia = complete[complete['Country/Region'] == 'Indonesia']

indonesia.head()
world_temp_2020[world_temp_2020['Country'] == 'Indonesia']
# plot daily cases

colors = ['#FFA500']*85

colors[-5] = 'crimson'

fig = px.bar(indonesia, 

             x="Date", y="Confirmed", 

             title='<b>New Confirm Cases Per Day In Indonesia</b>', 

             orientation='v', 

             width=700, height=600)

fig.update_traces(marker_color=colors, opacity=0.8)



fig.add_annotation( # add a text callout with arrow

    text="Social Distancing", x='2020-04-10', y=indonesia.Confirmed.max(), arrowhead=1, showarrow=True

)



fig.add_annotation( # add a text callout with arrow

    text="Extended Social Distancing", x='2020-04-29', y=indonesia.Confirmed.max()-1000, arrowhead=1, showarrow=True

)

fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2020-04-10',

            y0=0,

            x1='2020-04-10',

            y1=indonesia.Confirmed.max(),

            line=dict(

                color="RoyalBlue",

                width=1,

                dash="dashdot"

            )))



fig.add_shape(

        # Line Vertical

        dict(

            type="line",

            x0='2020-04-29',

            y0=0,

            x1='2020-04-29',

            y1=indonesia.Confirmed.max()-1000,

            line=dict(

                color="RoyalBlue",

                width=1,

                dash="dashdot"

            )))



fig.update_layout(template = 'plotly_white',font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
indonesia_00 = indonesia[indonesia['Date'] == '2020-03-27']

indonesia_0 = indonesia[indonesia['Date'] == '2020-04-10']

indonesia_1 = indonesia[indonesia['Date'] == '2020-04-24']

indonesia_2 = indonesia[indonesia['Date'] == '2020-05-7']

indonesia_3 = indonesia[indonesia['Date'] == '2020-05-21']

Confirmed_0 = indonesia_0.Confirmed.iloc[0] - indonesia_00.Confirmed.iloc[0]

Confirmed_1 = indonesia_1.Confirmed.iloc[0] - indonesia_0.Confirmed.iloc[0]

Confirmed_2 = indonesia_2.Confirmed.iloc[0] - indonesia_1.Confirmed.iloc[0]

Confirmed_3 = indonesia_3.Confirmed.iloc[0] - indonesia_2.Confirmed.iloc[0]



confirmed = [Confirmed_0,Confirmed_1, Confirmed_2, Confirmed_3]

date = ['27 Mar - 10 Apr','10 Apr - 24 Apr', '24 Apr - 7 Mei', '7 Mei - 21 Mei']

social_distancing = ['Before','Before','After','After']



df = pd.DataFrame()

df['confirmed'] = confirmed

df['date'] = date

df['Social_Distancing'] = social_distancing
plt.figure(figsize=(10,7))

ax = sns.barplot(x = 'date', y= 'confirmed', hue='Social_Distancing', data=df);



for p in ax.patches:

    

            try:

                height = p.get_height()

                ax.text(p.get_x() + p.get_width() / 2.,

                    height + 10,

                    '{} Cases'.format(int(height)),

                    ha="center", fontsize=10)

            except:

                pass

plt.xlabel('14 Days', fontsize= 15)

plt.ylabel('Confirmed Cases during 14 Days', fontsize=15)

plt.show()
gdp = pd.read_csv('/kaggle/input/covid19-different-mitigate-scenarios/GDP_Country.csv')

suppression = pd.read_csv('/kaggle/input/covid19-different-mitigate-scenarios/Suppression.csv')

mitigate = pd.read_csv("/kaggle/input/covid19-different-mitigate-scenarios/Mitigation_Type.csv")
mitigate[mitigate['Country'] == 'Indonesia']
train = mitigate.drop('Social_distance',axis=1)

y = mitigate.Social_distance



total_infected = indonesia.iloc[-1,:].Confirmed

total_deaths = indonesia.iloc[-1,:].Deaths

total_critical = indonesia.iloc[-1,:].Active # In here I tried to change the total critical values into total Active

total_hospital = 2831

total_pop = 273277935

strategy = 'Social distancing whole population'

R0 = 2.5

row = ['Indonesia',R0,strategy,total_pop,total_infected, total_deaths,total_hospital,total_critical]

test = pd.DataFrame([row], columns= train.columns)

test.head()
train.Strategy.unique()
merged = pd.concat([train,test])
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

merged['Strategy'] = merged.Strategy.map({'Enhanced social distancing of elderly': 1,'Social distancing whole population':2, 'Unmitigated':0})
merged.head()
train = merged.iloc[:len(train),]

test = merged.iloc[len(train):,]
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)



X = train.drop('Country',axis=1)



target = list()

for i in y.str.split('%'):

    target.append(int(i[0]))

    

y = pd.Series(target)
from sklearn.model_selection import KFold

kf = KFold(n_splits = 2, random_state = 1, shuffle = True)

np.random.seed(1)

from sklearn.metrics import mean_squared_error



mean = 0

for i,(train_index, test_index) in enumerate(kf.split(X)):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)

    mean += np.sqrt(mean_squared_error(y_train, y_pred))

    

print(mean/2)
test.drop('Country', axis=1,inplace=True)

regressor.fit(X,target)

regressor.predict(test)