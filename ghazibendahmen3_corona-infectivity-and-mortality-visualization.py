!pip install ez-ml
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
import pandas as pd 

import plotly.express as px

import numpy as np 

import pandas as pd 

from ezml import preprocessing , interpreting

import os

import warnings 

from fbprophet import Prophet

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
env_vars = pd.read_csv('/kaggle/input/environmental-variables-for-world-countries/World_countries_env_vars.csv').set_index('Country')

indexes = pd.read_csv('/kaggle/input/65-world-indexes-gathered/Kaggle.csv').set_index('Id')

happiness_alcool = pd.read_csv('/kaggle/input/happiness-and-alcohol-consumption/HappinessAlcoholConsumption.csv').set_index('Country')

result = pd.concat([env_vars, indexes,happiness_alcool], axis=1, join='inner')
_ = preprocessing.missing_values(result,plot=True)
result.drop(['slope','aspect','elevation'],axis=1,inplace=True)

_ = preprocessing.missing_values(result,plot=True)
cleaned_data = pd.read_csv("/kaggle/input/hemzacsv/covid_19_clean_complete.csv",parse_dates=['Date'])

df_original = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df_original.loc[df_original['Province/State'].isnull(),'Province/State'] = df_original['Country/Region']
coords = pd.read_csv("/kaggle/input/hemzacsv/covid_19_clean_complete.csv")
corona_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

corona_df['Country/Region'].replace(['Mainland China'], 'China',inplace=True)

corona_df['Country/Region'].replace(['US'], 'United States',inplace=True)

corona_df['Country'] = corona_df['Country/Region']
a = corona_df.groupby(['Country','ObservationDate']).agg({

    'Confirmed':sum,

    'Deaths':sum,

    'Recovered':sum

}).reset_index()
a['Moratality'] = ((a['Deaths'] / a['Confirmed']) *1000).fillna(0)

a['RecoveryRate'] = ((a['Recovered'] / a['Confirmed']) *1000).fillna(0)

latest_mars = a[a['ObservationDate'] == '03/23/2020']

jour_avant = a[a['ObservationDate'] == '03/22/2020']
df = pd.concat([result,latest_mars.set_index('Country') ], axis=1, join='inner')
df = df.reset_index().drop('index',axis=1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Region'] = le.fit_transform(df['Region'])

le1 = LabelEncoder()

df['Hemisphere'] = le1.fit_transform(df['Hemisphere'])

final_date = '3/20/20'

coords_final = coords[coords.Date == final_date]

df_deaths = pd.DataFrame(coords_final.groupby('Country/Region')['Deaths'].sum())

df_confirmed = pd.DataFrame(coords_final.groupby('Country/Region')['Confirmed'].sum())

df_recovered = pd.DataFrame(coords_final.groupby('Country/Region')['Recovered'].sum())

df_confirmed['Deaths'] = df_deaths['Deaths']

df_confirmed['Recovered'] = df_recovered['Recovered']

df_global = df_confirmed

df_global['Mortality Rate'] = np.round((df_global.Deaths.values/df_global.Confirmed.values)*100,2)

df_global['Ratio_Death_recovered'] = df_global.Deaths.values/df_global.Recovered.values

df_global.Ratio_Death_recovered = df_global.Ratio_Death_recovered.replace(np.inf,1)

df_global.Ratio_Death_recovered = df_global.Ratio_Death_recovered.fillna(0)

df_global = df_global.reset_index()

corr = df_global.corr()

corr.style.background_gradient(cmap='coolwarm')
plt.scatter(df_global.Deaths,df_global.Recovered)

plt.xlabel('Deaths')

plt.ylabel('Recovered')

plt.show()
grouped = cleaned_data.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()



fig = px.line(grouped, x="Date", y="Confirmed", 

              title="Total confirmed cases")

fig.show()
fig = px.line(grouped, x="Date", y="Deaths", title="Mortality evolution",

             color_discrete_sequence=['#F42272'])

fig.show()
temp = cleaned_data[[col for col in cleaned_data.columns if col != 'state']]

latest = temp[temp['Date'] == max(temp['Date'])].reset_index()

latest_grouped=latest.groupby('Country/Region')['Confirmed','Deaths'].sum().reset_index()
fig = px.choropleth(latest_grouped, locations="Country/Region", 

                    locationmode='country names', color="Deaths", 

                    hover_name="Deaths", range_color=[1,100], 

                    color_continuous_scale="peach", 

                    title='Deaths per countries')



fig.show()
fig = px.bar(latest_grouped.sort_values('Deaths', ascending=False)[:10][::-1], 

             x='Deaths', y='Country/Region',

             title='Nombre des déces', text='Deaths', orientation='h')

fig.show()
formated_gdf = cleaned_data.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Deaths'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="Deaths", size='size', hover_name="Country/Region", 

                     range_color= [0, 100], 

                     projection="natural earth", animation_frame="Date", 

                     title='Death Evolution since it all started', color_continuous_scale="thermal")

# fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.choropleth(latest_mars, 

                    locations="Country", 

                    color="Moratality", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,100],

                    title='Global COVID-19 Moratality rate for the 03/23/2020 Per 1000')

fig.show()
fig = px.choropleth(df_global, 

                    locations="Country/Region", 

                    color="Ratio_Death_recovered", 

                    locationmode = 'country names', 

                    hover_name="Country/Region",

                    range_color=[0,1],

                    title='Global COVID-19 Ratio_Death_recovered as of '+final_date)

fig.show()
y = df['Moratality']

X = df.drop(['ObservationDate', 'Confirmed', 'Deaths', 'Recovered',

       'Moratality', 'RecoveryRate'] ,axis=1)
from catboost import CatBoostRegressor

cbr = CatBoostRegressor()

cbr.fit(X, y)
interpreting.rf_feat_importance(cbr,X,plot=True)
fig = px.choropleth(latest_mars, 

                    locations="Country", 

                    color="Confirmed", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,40000],

                    title='Global COVID-19 confirmed infections for the 03/23/2020')

fig.show()
y = df['Confirmed']

X = df.drop(['ObservationDate', 'Confirmed', 'Deaths', 'Recovered',

       'Moratality', 'RecoveryRate'] ,axis=1)
from catboost import CatBoostRegressor

cbr = CatBoostRegressor()

cbr.fit(X, y)
interpreting.rf_feat_importance(cbr,X,plot=True)
import warnings

warnings.filterwarnings("ignore")

print("-- List of Countries --")

print(df_original["Country/Region"].unique())

print("----")

country = "Mainland China"#input("Choose a country :")

fig, ax = plt.subplots()

print(df_original[df_original["Country/Region"] == country]["Province/State"].unique())

list_ = df_original[df_original["Country/Region"] == country]["Province/State"].unique()



for el in list_:

    ax.plot(df_original[(df_original["Country/Region"] == country) & (df_original["Province/State"] == el)]["ObservationDate"], df_original[(df_original["Country/Region"] == country) & (df_original["Province/State"] == el)]["Confirmed"],label=el)



plt.xticks(rotation=90)

plt.rcParams["figure.figsize"] = (20,15)

ax.legend(loc='upper left', frameon=False)

plt.show()
model = Prophet()



print("-- List of Countries --")



print(df_original["Country/Region"].unique())



print("----")



country = "Mainland China"#input("Choose a country :")



df_tmp = df_original[df_original["Country/Region"] == country]

df_tmp['Evolution'] = df_tmp.apply(lambda row: row.Confirmed - row.Recovered, axis = 1)

df_tmp = pd.DataFrame({'count' : df_tmp.groupby('ObservationDate')['Evolution'].sum()}).reset_index()

df_tmp = df_tmp.rename(columns={'ObservationDate':'ds', 'count':'y'})



print(df_tmp)



model.fit(df_tmp)

future = model.make_future_dataframe(periods=14)

forecast = model.predict(future)



print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])



model.plot_components(forecast)
fig = px.choropleth(latest_mars, 

                    locations="Country", 

                    color="Recovered", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,10000],

                    title='Global COVID-19 confirmed infections for the 03/23/2020')

fig.show()
y = df['Recovered']

X = df.drop(['ObservationDate', 'Confirmed', 'Deaths', 'Recovered',

       'Moratality', 'RecoveryRate'] ,axis=1)
from catboost import CatBoostRegressor

cbr = CatBoostRegressor()

cbr.fit(X, y)
interpreting.rf_feat_importance(cbr,X,plot=True)
from sklearn.cluster import KMeans

X = df_global[['Mortality Rate','Deaths' , 'Recovered','Ratio_Death_recovered','Confirmed']]

X = X.to_numpy()

y = df_global['Country/Region']
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=1000, n_init=10, random_state=0)

pred_y = kmeans.fit_predict(X)

dicti = {}

for i in range(len(y)):

    dicti[y[i]] = pred_y[i]

df_global['pr'] = pd.DataFrame(pred_y)
fig = px.choropleth(df_global, 

                    locations="Country/Region", 

                    color="pr", 

                    locationmode = 'country names', 

                    hover_name="Country/Region",

                    range_color=[0,3],

                    title='clusting result for the '+final_date)

fig.show()