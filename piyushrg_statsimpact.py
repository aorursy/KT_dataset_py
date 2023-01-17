# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def RMSLE(y_true, y_pred):

    y_true = np.log(np.abs(y_true) + 1)

    y_pred = np.log(np.abs(y_pred) + 1)

    y = (y_true-y_pred)**2

    return np.mean(y)
train = pd.read_csv('/kaggle/input/jagritistatsimpact/train.csv')

test = pd.read_csv('/kaggle/input/jagritistatsimpact/test.csv')
train.head()
train[train['Country/Region'] == 'US'].head()
train['dat'] = [x.replace("-", "") for x in train['Date']]

test['dat'] = [x.replace("-", "") for x in test['Date']]
train['Date'] = pd.to_datetime(train['Date'], format = '%Y-%m-%d')

test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d')
min_date = min(train['Date'])

train['days'] = (train['Date']-min_date)

train['days'] = [x.days for x in train['days']]

test['days'] = (test['Date']-min_date)

test['days'] = [x.days for x in test['days']]
print(min(train['Date']), max(train['Date']))

print(min(test['Date']), max(test['Date']))
train[train['Date']== min_date].head()
test[test['Date']== min_date].head()
train[train['Lat']== 27.6104].sort_values(by=['Date']).head()
test[test['Lat']== 27.6104].sort_values(by=['Date']).head()
train['Province/State'] = train['Province/State'].fillna('NoData')

test['Province/State'] = test['Province/State'].fillna('NoData')
train['days2'] = np.square(train['days'])

test['days2'] = np.square(test['days'])
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[['Lat', 'Long', 'days', 'days2', 'Country/Region', 'Id', 'Province/State']], train[['ConfirmedCases', 'Country/Region', 'Province/State']], test_size=0.2, random_state=42)
countries = train['Country/Region'].unique()

col = ['Lat', 'Long', 'days2']

sub_df = pd.DataFrame(None, columns =['Id', 'ConfirmedCases'])

test_df = pd.DataFrame(None, columns =['Id', 'ConfirmedCases'])

results = pd.DataFrame(None, columns = ['country', 'Province/State','RMSLE'])

for country in countries:

    for state in train[train['Country/Region']== country]['Province/State'].unique():

        X_tr_new = X_train[(X_train['Country/Region'] == country) & (X_train['Province/State']== state)]

        y_tr_new = y_train[(y_train['Country/Region'] == country) & (y_train['Province/State']== state)]

        subs = pd.DataFrame(X_test[(X_test['Country/Region']== country) & (X_test['Province/State']== state)]['Id'])

        testdf =  pd.DataFrame(test[(test['Country/Region']== country) & (test['Province/State']== state)]['Id'])



        regr = RandomForestRegressor(random_state=0)

        regr.fit(X_tr_new[col], y_tr_new['ConfirmedCases'])



        pred = regr.predict(X_test[(X_test['Country/Region']== country) & (X_test['Province/State']== state)][col])

        sub_pred = regr.predict(test[(test['Country/Region']== country) & (test['Province/State']== state)][col])



        subs['ConfirmedCases'] = pred

        testdf['ConfirmedCases'] = sub_pred



        sub_df = sub_df.append(subs)

        test_df = test_df.append(testdf)

        error = RMSLE(pred, y_test[(y_test['Country/Region']== country) & (y_test['Province/State']== state)]['ConfirmedCases'])

        a = pd.DataFrame.from_dict({'country': [country], 'Province/State': [state],  'RMSLE': [error]})

        results = results.append(a)

    print(f'fiting for {country}')

    #print(RMSLE(pred2, y_test))
np.mean(results.RMSLE)
countries = train['Country/Region'].unique()

col = ['Lat', 'Long', 'days', 'days2']

sub_df = pd.DataFrame(None, columns =['Id', 'ConfirmedCases'])

test_df = pd.DataFrame(None, columns =['Id', 'ConfirmedCases'])

for country in countries:

    for state in train[train['Country/Region']== country]['Province/State'].unique():

        X_tr_new = train[(train['Country/Region'] == country) & (train['Province/State']== state)]

        y_tr_new = train[(train['Country/Region'] == country) & (train['Province/State']== state)]

        testdf =  pd.DataFrame(test[(test['Country/Region']== country) & (test['Province/State']== state)]['Id'])



        regr = RandomForestRegressor(random_state=0)

        regr.fit(X_tr_new[col], y_tr_new['ConfirmedCases'])



        sub_pred = regr.predict(test[(test['Country/Region']== country) & (test['Province/State']== state)][col])

        testdf['ConfirmedCases'] = sub_pred



        test_df = test_df.append(testdf)

    print(f'fiting for {country}')

    #print(RMSLE(pred2, y_test))
test_df.to_csv('sub_country_state_RF_4_all.csv', index = False)
import plotly.express as px
full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

full_table.sample(6)
import plotly
!pip install orca
plotly.io.orca.config.executable = ' /opt/conda/lib/python3.6/site-packages/orca'
#import orca

temp = full_table.groupby(['Country/Region'])['Confirmed'].max().reset_index()

fig = px.choropleth(temp, locations="Country/Region", 

                    locationmode='country names', color=np.log(temp["Confirmed"]), 

                    hover_name="Country/Region", hover_data=['Confirmed'],

                    color_continuous_scale="Sunsetdark", 

                    title='Countries with Confirmed Cases',

                    labels={'Sunsetdark':temp['Confirmed']})

fig.update(layout_coloraxis_showscale=False)

fig.update_layout({'paper_bgcolor': '#f3f3f3', 'plot_bgcolor': 'rgba(0,0,0,0)'})

fig.show()

fig.write_html("fig1.html")
full_table['Active'] = full_table['Confirmed'] - full_table['Recovered'] - full_table['Deaths']

temp = full_table.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case', height=800,

             title='Cases over time', color_discrete_sequence = ['green', 'black', 'blue'])

#fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout({'paper_bgcolor': '#f3f3f3', 'plot_bgcolor': 'rgba(0,0,0,0)'})

fig.show()

fig.write_html("fig1.html")
temp = full_table.groupby(['Country/Region'])['Confirmed', 'Deaths'].max().reset_index()

fig = px.scatter(temp.sort_values('Deaths', ascending=False).iloc[:15, :], 

                 x='Confirmed', y='Deaths', color='Country/Region', size='Confirmed', height=800,

                 text='Country/Region', log_x=True, log_y=True, title='Deaths vs Confirmed')

fig.update_traces(textposition='top center')

fig.update_layout({'paper_bgcolor': '#f3f3f3', 'plot_bgcolor': 'rgba(0,0,0,0)'})

#fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()

fig.write_html("fig1.html")
temp = full_table.groupby(['Country/Region', 'Date'])['Confirmed', 'Deaths'].sum()

temp = temp.reset_index()

temp = temp[temp['Country/Region']!= 'China']

fig = px.bar(temp.sort_values('Confirmed', ascending=False).iloc[:80, :], x="Date", y="Confirmed", color='Country/Region', orientation='v', height=600,#text='Country/Region',

             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)

#fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout({'paper_bgcolor': '#f3f3f3', 'plot_bgcolor': 'rgba(0,0,0,0)'})

fig.show()

fig.write_html("fig1.html")
# =========================================

temp = full_table.groupby(['Country/Region', 'Date'])['Confirmed', 'Deaths'].sum()

temp = temp.reset_index()

fig = px.bar(temp.sort_values('Confirmed', ascending=False).iloc[:80, :], x="Date", y="Deaths", color='Country/Region', orientation='v', height=600,

             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)

#fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout({'paper_bgcolor': '#f3f3f3', 'plot_bgcolor': 'rgba(0,0,0,0)'})

fig.show()

fig.write_html("fig1.html")
# =========================================



temp = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan



fig = px.bar(temp.sort_values('Confirmed', ascending=False).iloc[:30, :30], x="Date", y="Confirmed", color='Country/Region',title='New cases')

#fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Confirmed', ascending=False)



fig = px.line(temp, x="Date", y="Confirmed", color='Country/Region', title='Cases Spread', height=600)

#fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout({'paper_bgcolor': '#f3f3f3', 'plot_bgcolor': 'rgba(0,0,0,0)'})

fig.show()

fig.write_html("fig1.html")
#================================



temp = full_table.groupby(['Date', 'Country/Region'])['Deaths'].sum().reset_index().sort_values('Deaths', ascending=False)



fig = px.line(temp, x="Date", y="Deaths", color='Country/Region', title='Deaths', height=600)

#fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout({'paper_bgcolor': '#f3f3f3', 'plot_bgcolor': 'rgba(0,0,0,0)'})

fig.show()

fig.write_html("fig1.html")
#https://app.flourish.studio/visualisation/1708527/edit

from IPython.core.display import HTML

HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1708527"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')