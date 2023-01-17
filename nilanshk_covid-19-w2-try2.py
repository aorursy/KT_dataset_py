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
train_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

sub_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
train_data.isnull().sum()
from plotly.offline import iplot

from plotly import tools

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import plotly.io as pio

data = train_data.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()

data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%d/%m/%Y')

data['size'] = data['ConfirmedCases'].pow(0.2) * 2



fig = px.scatter_geo(data, locations="Country_Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Confirmed Cases Around the Globe', color_continuous_scale="purples")

fig.show()
train_data.rename(columns={'Country_Region':'Country'}, inplace=True)

test_data.rename(columns={'Country_Region':'Country'}, inplace=True)

sub_data.rename(columns={'Country_Region':'Country'}, inplace=True)
confirmed = train_data.groupby('Date').sum()['ConfirmedCases'].reset_index()

deaths = train_data.groupby('Date').sum()['Fatalities'].reset_index()

fig = go.Figure()

fig.add_trace(go.Bar(x=confirmed['Date'],

                y=confirmed['ConfirmedCases'],

                name='Confirmed',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=deaths['Date'],

                y=deaths['Fatalities'],

                name='Deaths',

                marker_color='Red'

                ))



fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths (Bar Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed['Date'],

                y=confirmed['ConfirmedCases'],

                name='Confirmed',

                marker_color='blue'

                ))

fig.add_trace(go.Scatter(x=deaths['Date'],

                y=deaths['Fatalities'],

                name='Deaths',

                marker_color='Red'

                ))





fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths (Line Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
country_data=train_data.groupby(["Country"]).sum().reset_index()
asia_countries = list(['China','India','Indonesia','Pakistan','Bangladesh','Russia','Japan','Phillippines','Vietnam','Turkey','Iran','Thailand','Myanmar','South Korea',

               'Iraq', 'Afghanistan','Uzbekistan','Malayisa','Nepal','South Arabia','Yemen','North Korea','Taiwan','Sri Lanka','Kazakhstan','Syria',

               'Cambodia', 'Azerbaijan', 'Tajikistan', 'Iceland', 'United Arab Emirates', 'Israel', 'Kyrgyzstan', 'Hong Kong', 'Jordan',

               'Laos', 'Turkmenistan', 'Singapore', 'Palestine', 'Lebanon', 'Oman', 'Kuwait','Georgia', 'Mongolia', 'Armenia', 'Qatar', 'Timor-Leste', 'Bahrain', 'Bhutan', 'Macau', 'Maldives', 'Brunei'])

asia_countries_list = country_data[country_data['Country'].isin(asia_countries)]
temp = train_data[train_data['Country'].isin(asia_countries)]

temp = temp.groupby(['Date', 'Country'])['ConfirmedCases'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date']).dt.strftime('%m/%d/%Y')

temp['size'] = temp['ConfirmedCases'].pow(0.3) * 3.5



fig = px.scatter_geo(temp, locations="Country", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country", 

                     range_color=[1,100],scope='asia',

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Cases Over Time', color_continuous_scale='Cividis_r')

fig.show()
population=pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')
population.head()
pop_data=pd.DataFrame({'Countrys':[],'Population':[]})
pop_data["Countrys"]=population["Country (or dependency)"]

pop_data["Population"]=population["Population (2020)"]
pop_data=pop_data.replace({"Countrys": "United States"},"US")

pop_data=pop_data.replace({"Countrys": "DR Congo"},"Congo (Kinshasa)")

pop_data=pop_data.replace({"Countrys": "Congo"},"Congo (Brazzaville)")

pop_data=pop_data.replace({"Countrys": "Taiwan"},"Taiwan*")

pop_data=pop_data.replace({"Countrys": "Côte d\'Ivoire"},"Cote d\'Ivoire")

pop_data=pop_data.replace({"Countrys": "St. Vincent & Grenadines"},"Saint Vincent and the Grenadines")

pop_data=pop_data.replace({"Countrys": "Czech Republic (Czechia)"},"Czechia")

pop_data=pop_data.replace({"Countrys": "Saint Kitts & Nevis"},"Saint Kitts and Nevis")

pop_data=pop_data.replace({"Countrys": "South Korea"},"Korea, South")

train_data = pd.merge(train_data, pop_data, how='left', left_on = 'Country', right_on = 'Countrys')

train_data["Population"]=pd.DataFrame(train_data["Population"].fillna(value="3700"))

train_data.drop("Countrys",axis=1)
test_data = pd.merge(test_data, pop_data, how='left', left_on = 'Country', right_on = 'Countrys')

test_data["Population"]=pd.DataFrame(test_data["Population"].fillna(value="3700"))

test_data.drop("Countrys",axis=1)
column_names = ['Id','Province_State','Country','Date','Population','ConfirmedCases','Fatalities']

train_data = train_data.reindex(columns=column_names)
train_data.isnull().sum()
train_data['Date'] = pd.to_datetime(train_data['Date'], infer_datetime_format=True)
test_data['Date'] = pd.to_datetime(test_data['Date'], infer_datetime_format=True)
X=train_data

X["Province_State"]=X.apply(

    lambda row: str(row['Country']) if pd.isnull(row['Province_State']) else row['Province_State'] , axis=1

)
test_data.head()
X_test=test_data

X_test["Province_State"]=X_test.apply(

    lambda row: str(row['Country']) if pd.isnull(row['Province_State']) else row['Province_State'] , axis=1

)


X.loc[:, 'Date'] = X.Date.dt.strftime("%m%d")

X["Date"]  = X["Date"].astype(int)



X_test.loc[:, 'Date'] = X_test.Date.dt.strftime("%m%d")

X_test["Date"]  = X_test["Date"].astype(int)

from sklearn.preprocessing import LabelEncoder

myencoder=LabelEncoder()

X.iloc[:,2]=myencoder.fit_transform(X.iloc[:,2])

X.iloc[:,1]=myencoder.fit_transform(X.iloc[:,1])

X=train_data.drop(columns="Id")





X_test.iloc[:,2]=myencoder.fit_transform(X_test.iloc[:,2])

X_test.iloc[:,1]=myencoder.fit_transform(X_test.iloc[:,1])
X_test["Population"]=X_test["Population"].astype(int)

X_test.head()
X["Population"]=X["Population"].astype(int)

X.head()
import seaborn as sns

sns.set_style('darkgrid')

corr=X.corr()

sns.heatmap(corr, 

            xticklabels = corr.columns.values,

            yticklabels = corr.columns.values,

            annot = True)
features3=['Province_State','Country','Date','Population']

features2=['Province_State','Country']

features1=['Province_State']



features5=['Country','Date']



features4=['Province_State','Country','Date','ConfirmedCases']



target_1=['ConfirmedCases']

target_2=['Fatalities']
X_ver=X[features3]

Y=X[target_1]
from sklearn.model_selection import train_test_split

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X_ver,Y,test_size=0.3,random_state=0)
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score
lasso = Lasso(alpha =2)

lasso.fit(X_Train,Y_Train)

Y_predict=lasso.predict(X_Test)

print(r2_score(Y_Test, Y_predict))
from xgboost import XGBRegressor

new_xgb_clf_1=XGBRegressor(n_estimators=1000)

new_xgb_clf_2=XGBRegressor(n_estimators=1000)
df=pd.DataFrame({'ForecastId':[],'ConfirmedCases':[],'Fatalities':[]})

df2=pd.DataFrame({'Predicted':[]})



for i in range(0,173):

    new_train=X.loc[X['Country']==i]

    new_xgb_clf_1.fit(new_train[features3],new_train[target_1])

    new_xgb_clf_2.fit(new_train[features3],new_train[target_2])

    

    new_test=X_test.loc[X_test['Country']==i]

    new_test_FI=new_test.ForecastId

    new_pred_1=new_xgb_clf_1.predict(new_test[features3])

    new_pred_2=new_xgb_clf_2.predict(new_test[features3])

    

    new_df=pd.DataFrame({'ForecastId':new_test_FI,'ConfirmedCases':new_pred_1,'Fatalities':new_pred_2})





    df=pd.concat([df,new_df],axis=0)

    
df['ForecastId']=df['ForecastId'].astype(int)

df['ConfirmedCases']=df['ConfirmedCases'].astype(int)

df['Fatalities']=df['Fatalities'].astype(int)
df.to_csv('submission.csv',index=False)