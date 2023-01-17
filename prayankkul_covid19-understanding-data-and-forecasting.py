

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

from warnings import filterwarnings

filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook

import xgboost as xgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv(r'../input/covid19-global-forecasting-week-1/test.csv')

train.head()

test.head()
print("shape of train data is {}".format(train.shape))

print("shape of test data is {}".format(test.shape))
# checking basic info for test and train

train.info()
# Data missing information for train

data_info=pd.DataFrame(train.dtypes).T.rename(index={0:'column type'})

data_info=data_info.append(pd.DataFrame(train.isnull().sum()).T.rename(index={0:'null values (nb)'}))

data_info=data_info.append(pd.DataFrame(train.isnull().sum()/train.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

display(data_info)
# Data missing information for test

data_info=pd.DataFrame(test.dtypes).T.rename(index={0:'column type'})

data_info=data_info.append(pd.DataFrame(test.isnull().sum()).T.rename(index={0:'null values (nb)'}))

data_info=data_info.append(pd.DataFrame(test.isnull().sum()/test.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

display(data_info)
train['Date'] = pd.to_datetime(train['Date'])

train['Date'] = train['Date'].dt.date

confirmed_case = train.groupby('Date')['ConfirmedCases'].agg('sum')

Fatalities = train.groupby('Date')['Fatalities'].agg('sum')



fig = go.Figure()

fig.add_trace(go.Scatter(

                x=train.Date,

                y=confirmed_case,

                name="Confirmed Case around the World",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=train.Date,

                y=Fatalities,

                name="Death around around the world",

                line_color='red',

                opacity=0.8))



fig.update_layout(title_text="Growth Number of cases of COVID 19 Around the world",template="plotly_dark")



fig.show()
confirmed_case_country =train.groupby('Country/Region')['ConfirmedCases'].agg('max').sort_values(ascending =False)[:15]

Fatalities_country = train.groupby('Country/Region')['Fatalities'].agg('max').sort_values(ascending =False)[:15]

frame = {'Country':confirmed_case_country.index,'Cases': confirmed_case_country }

CountryWise = pd.DataFrame(frame)

frame = {'Country':Fatalities_country.index,'Cases':Fatalities_country}

CountryWise_death = pd.DataFrame(frame)
# Confirmed Cases analysis



fig = px.bar(CountryWise, x= 'Country', y='Cases',color='Cases',labels={'Country':'Confirmed Cases'},height=400,template="plotly_dark")

fig.show()
# Death Cases analysis



fig = px.bar(CountryWise_death, x= 'Country', y='Cases',color='Cases',labels={'Country':'Death Cases'},height=400,template="plotly_dark")

fig.show()
china = train.loc[train['Country/Region']=='China'].groupby('Date').agg('sum')

Italy = train.loc[train['Country/Region']=='Italy'].groupby('Date').agg('sum')

Iran = train.loc[train['Country/Region']=='Iran'].groupby('Date').agg('sum')
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=china.index,

                y=china.ConfirmedCases,

                name="Confirmed Case in China",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=china.index,

                y=china.Fatalities,

                name="Death case in China",

                line_color='red',

                opacity=0.8))



fig.update_layout(title_text="View of china",template="plotly_dark")



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=Italy.index,

                y=Italy.ConfirmedCases,

                name="Confirmed Case in Italy",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=Italy.index,

                y=Italy.Fatalities,

                name="Death case in Italy",

                line_color='red',

                opacity=0.8))



fig.update_layout(title_text="View of Italy",template="plotly_dark")



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=Iran.index,

                y=Iran.ConfirmedCases,

                name="Confirmed Case in Iran",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=Iran.index,

                y=Iran.Fatalities,

                name="Death case in Iran",

                line_color='red',

                opacity=0.8))



fig.update_layout(title_text="View of Iran",template="plotly_dark")



fig.show()
#train.drop('Proviance/state',axis=1,inplace = True)

train_country = train.groupby('Country/Region').agg('max')
train_country
fig = px.choropleth(train_country, locations=train_country.index, 

                    locationmode='country names', color="ConfirmedCases", 

                    hover_name="ConfirmedCases", range_color=[1,10000], 

                    color_continuous_scale="hot", 

                    title='Confimed Case around the world')



fig.update_layout(template="plotly_dark")



fig.show()







fig = px.choropleth(train_country, locations=train_country.index, 

                    locationmode='country names', color="Fatalities", 

                    hover_name="Fatalities", range_color=[1,2000], 

                    color_continuous_scale="hot", 

                    title='Death Case around the world')



fig.update_layout(template="plotly_dark")



fig.show()
CountryWise['death_case'] = CountryWise_death['Cases'] 

print("Correlation between Confirmed cases and death cases in worst hit countires {:.2f}%".format(CountryWise.corr()['death_case'][0]*100))
# Mean Confirmed Cases and Death Cases in Mainland China

print("Mean Confirmed Cases in world  {:.1f}".format(train_country['ConfirmedCases'].mean()))

print("Mean Confirmed Cases in world  {:.1f}".format(train_country['Fatalities'].mean()))

print("Total Death Rate in world is {:.1f}%".format((train_country['Fatalities'].mean()/train_country['ConfirmedCases'].mean())*100))
train_country_exclude_3 = train_country.loc[(train_country.index!='China') & (train_country.index!='Italy') & (train_country.index!= 'Iran')]
# Mean Confirmed Cases and Death Cases in Mainland China

print("Mean Confirmed Cases in rest of world  {:.1f}".format(train_country_exclude_3['ConfirmedCases'].mean()))

print("Mean Confirmed Cases in rest of world  {:.1f}".format(train_country_exclude_3['Fatalities'].mean()))

print("Total Death Rate in rest of world is {:.1f}%".format((train_country_exclude_3['Fatalities'].mean()/train_country_exclude_3['ConfirmedCases'].mean())*100))
train_italy = train.loc[train['Country/Region']=='Italy']

train_italy = train_italy[['Date','ConfirmedCases']]

train_italy['Date'] = pd.to_datetime(train_italy['Date'])

train_italy.index = train_italy['Date']

train_italy.drop('Date',inplace=True,axis=1)
train_italy
def create_features(df):

    """

    Creates time series features from datetime index

    """

    df['Cases'] = df['ConfirmedCases']

    df['date'] = df.index

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['quarter','month','year','dayofyear','dayofmonth','weekofyear','Cases']]

    return X
train_italy = create_features(train_italy)

y = train_italy.pop('Cases')

x = train_italy

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print("shape of x train and y trains is {} {}".format(x_train.shape, y_train.shape))

print("shape of x test and y test is {} {}".format(x_test.shape, y_test.shape))
reg = xgb.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.005,

                 max_depth=10,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)



reg.fit(x_train, y_train,

        eval_set=[(x_train, y_train), (x_test, y_test)],

        early_stopping_rounds=1000, #stop if 50 consequent rounds without decrease of error

        verbose=False) # Change verbose to True if you want to see it train
xgb.plot_importance(reg, height=0.9)
## Continue to Work 