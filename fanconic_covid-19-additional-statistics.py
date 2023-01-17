import numpy as np

import pandas as pd



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from pathlib import Path

data_dir = Path('../input/covid19-global-forecasting-week-1')



import os

os.listdir(data_dir)



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
cleaned_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])



cleaned_data.rename(columns={'ObservationDate': 'date', 

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Last Update':'last_updated',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

                     'Recovered':'recovered'

                    }, inplace=True)



# cases 

cases = ['confirmed', 'deaths', 'recovered', 'active']



# Active Case = confirmed - deaths - recovered

cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']



# replacing Mainland china with just China

cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')



# filling missing values 

cleaned_data[['state']] = cleaned_data[['state']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.rename(columns={'Date':'date'}, inplace=True)



data = cleaned_data



display(data.head())

display(data.info())
# Check if the data is updated

print("External Data")

print(f"Earliest Entry: {data['date'].min()}")

print(f"Last Entry:     {data['date'].max()}")

print(f"Total Days:     {data['date'].max() - data['date'].min()}")
group = data.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



fig = px.line(group, x="date", y="confirmed", 

              title="Worldwide Confirmed Cases Over Time")



fig.show()



fig = px.line(group, x="date", y="deaths", 

              title="Worldwide Deaths Over Time")



fig.show()
def p2f(x):

    """

    Convert urban percentage to float

    """

    try:

        return float(x.strip('%'))/100

    except:

        return np.nan



def age2int(x):

    """

    Convert Age to integer

    """

    try:

        return int(x)

    except:

        return np.nan



def fert2float(x):

    """

    Convert Fertility Rate to float

    """

    try:

        return float(x)

    except:

        return np.nan





countries_df = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv", converters={'Urban Pop %':p2f,

                                                                                                             'Fert. Rate':fert2float,

                                                                                                             'Med. Age':age2int})

countries_df.rename(columns={'Country (or dependency)': 'country',

                             'Population (2020)' : 'population',

                             'Density (P/Km²)' : 'density',

                             'Fert. Rate' : 'fertility',

                             'Med. Age' : "age",

                             'Urban Pop %' : 'urban_percentage'}, inplace=True)







countries_df['country'] = countries_df['country'].replace('United States', 'US')

countries_df = countries_df[["country", "population", "density", "fertility", "age", "urban_percentage"]]



countries_df.head()
data = pd.merge(data, countries_df, on='country')
cleaned_latest = data[data['date'] == max(data['date'])]

flg = cleaned_latest.groupby('country')['confirmed', 'population'].agg({'confirmed':'sum', 'population':'mean'}).reset_index()



flg['infectionRate'] = round((flg['confirmed']/flg['population'])*100, 5)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('infectionRate', ascending=False)



fig = px.bar(temp.sort_values(by="infectionRate", ascending=False)[:10][::-1],

             x = 'infectionRate', y = 'country', 

             title='% of infected people by country', text='infectionRate', height=800, orientation='h',

             color_discrete_sequence=['red']

            )

fig.show()
formated_gdf = data.groupby(['date', 'country'])['confirmed', 'population'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['infectionRate'] = round((formated_gdf['confirmed']/formated_gdf['population'])*100, 8)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="infectionRate", size='infectionRate', hover_name="country", 

                     range_color= [0, 0.2], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Spread Over Time (Normalized by Country Population)', color_continuous_scale="portland")

# fig.update(layout_coloraxis_showscale=False)

fig.show()
cleaned_latest = data[data['date'] == max(data['date'])]

flg = cleaned_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



flg['mortalityRate'] = round((flg['deaths']/flg['confirmed'])*100, 2)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('mortalityRate', ascending=False)



fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],

             x = 'mortalityRate', y = 'country', 

             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',

             color_discrete_sequence=['darkred']

            )

fig.show()
formated_gdf = data.groupby(['date', 'country'])['confirmed', 'deaths'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['mortalityRate'] = round((formated_gdf['deaths']/formated_gdf['confirmed'])*100, 2)



fig = px.scatter_geo(formated_gdf.fillna(0), locations="country", locationmode='country names', 

                     color="mortalityRate", size='mortalityRate', hover_name="country", 

                     range_color= [0, 10], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Mortality Rate in % by country', color_continuous_scale="portland")

# fig.update(layout_coloraxis_showscale=False)

fig.show()
icu_df = pd.read_csv("../input/hospital-beds-by-country/API_SH.MED.BEDS.ZS_DS2_en_csv_v2_887506.csv")

icu_df['Country Name'] = icu_df['Country Name'].replace('United States', 'US')

icu_df['Country Name'] = icu_df['Country Name'].replace('Russian Federation', 'Russia')

icu_df['Country Name'] = icu_df['Country Name'].replace('Iran, Islamic Rep.', 'Iran')

icu_df['Country Name'] = icu_df['Country Name'].replace('Egypt, Arab Rep.', 'Egypt')

icu_df['Country Name'] = icu_df['Country Name'].replace('Venezuela, RB', 'Venezuela')

data['country'] = data['country'].replace('Czechia', 'Czech Republic')

# We wish to have the most recent values, thus we need to go through every year and extract the most recent one, if it exists.

icu_cleaned = pd.DataFrame()

icu_cleaned["country"] = icu_df["Country Name"]

icu_cleaned["icu"] = np.nan



for year in range(1960, 2020):

    year_df = icu_df[str(year)].dropna()

    icu_cleaned["icu"].loc[year_df.index] = year_df.values
data = pd.merge(data, icu_cleaned, on='country')
data['state'] = data['state'].fillna('')

temp = data[[col for col in data.columns if col != 'state']]



latest = temp[temp['date'] == max(temp['date'])].reset_index()

latest_grouped = latest.groupby('country')['icu'].mean().reset_index()





fig = px.bar(latest_grouped.sort_values('icu', ascending=False)[:10][::-1], 

             x='icu', y='country',

             title='Ratio of ICU Beds per 1000 People', text='icu', orientation='h',color_discrete_sequence=['green'] )

fig.show()

fig = px.choropleth(latest_grouped, locations="country", 

                    locationmode='country names', color="icu", 

                    hover_name="country", range_color=[1,15], 

                    color_continuous_scale="algae", 

                    title='Ratio of ICU beds per 1000 people')

# fig.update(layout_coloraxis_showscale=False)

fig.show()
df_temperature = pd.read_csv("../input/covid19-global-weather-data/temperature_dataframe.csv")

df_temperature['country'] = df_temperature['country'].replace('USA', 'US')

df_temperature['country'] = df_temperature['country'].replace('UK', 'United Kingdom')

df_temperature = df_temperature[["country", "province", "date", "humidity", "sunHour", "tempC", "windspeedKmph"]].reset_index()

df_temperature.rename(columns={'province': 'state'}, inplace=True)

df_temperature["date"] = pd.to_datetime(df_temperature['date'])

df_temperature['state'] = df_temperature['state'].fillna('')





df_temperature.info()
data = data.merge(df_temperature, on=['country','date', 'state'], how='inner')

data.to_csv("countries_icu_temp.csv")
data.head()
train_data = data

print(train_data.shape)

train_data.head()
threshold = 0

train_data['infectionRate'] = round((train_data['confirmed']/train_data['population'])*100, 5)

train_data = train_data[train_data['infectionRate'] >= threshold]

print(train_data.shape)
train_data = train_data.drop([

                     "country", 

                     "active", 

                     "recovered", 

                     "infectionRate",

                     "state",

                     "Lat",

                     "Long",

                     "date",

                     "index"

                    ], axis= 1).dropna()



y = train_data[["confirmed", "deaths"]]

X = train_data.drop(["confirmed", "deaths"],axis=1)



display(X.head())

print(X.shape, y.shape)
import matplotlib.pyplot as plt

import seaborn as sns

cm = train_data.corr()

plt.figure(figsize=(20,10))

sns.heatmap(cm, annot=True)
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()

#X_scaled = scaler.fit_transform(X)
# Split into training and evaluation data:

from sklearn.model_selection import train_test_split as tts

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor



from sklearn.metrics import mean_squared_log_error, make_scorer

def rmsle(y_true, y_pred):

    """

    Computes the Root Mean Squared Logarithmic Error of a prediction set.

    params:

        y_true: numpy array of ground truth

        y_pred: numpy array of predictions

    """

    return np.sqrt(mean_squared_log_error(y_true, y_pred))



rmsle_scorer = make_scorer(rmsle)



X_train, X_val, y_train, y_val = tts(X, y, test_size= 0.2, random_state=42, shuffle=True)
model_infected = DecisionTreeRegressor(random_state=42, criterion="mae")



scores = cross_val_score(model_infected, 

                      X_train,

                      y_train["confirmed"],

                      cv=5, scoring=rmsle_scorer)



print("Cross Validation of Confirmed Cases: Mean = {}, std = {}".format(scores.mean(), scores.std()))

model_infected.fit(X_train, y_train["confirmed"])

result_infected = rmsle(y_val["confirmed"], model_infected.predict(X_val))

print("Validation Infected set RMSLE: {}".format(result_infected))
model_deaths = DecisionTreeRegressor(random_state=42, criterion="mae")



scores = cross_val_score(model_deaths, 

                      X_train,

                      y_train["deaths"],

                      cv=5, scoring=rmsle_scorer)



print("Cross Validation of Fatal Cases: Mean = {}, std = {}".format(scores.mean(), scores.std()))

model_deaths.fit(X_train, y_train["deaths"])

result_deaths = rmsle(y_val["deaths"], model_deaths.predict(X_val))

print("Validation Death set RMSLE: {}".format(result_deaths))
# Final Evalutation

print("Final Validatio score: {}".format(np.mean([result_infected, result_deaths])))
model_infected = model_infected.fit(X, y["confirmed"])

model_deaths = model_deaths.fit(X, y["deaths"])
def show_feature_importance(forest):

    """

    Creates a sorted list of the feature importance of a decision tree algorithm.

    Furthermore it plots it.

    params:

        forest: Decision Tree algorithm

    """

    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]



    # Print the feature ranking

    print("Feature ranking:")



    for f in range(X.shape[1]):

        print("{}, Feature: {}, Importance: {}".format(f + 1, X.columns[indices[f]], importances[indices[f]]))



    # Plot the feature importances of the forest

    plt.figure(figsize=(20,10))

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")

    plt.xticks(range(X.shape[1]),  X.columns[indices], rotation='vertical')

    plt.xlim([-1, X.shape[1]])

    plt.show()
show_feature_importance(model_infected)
show_feature_importance(model_deaths)
test_df = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

test_df.rename(columns={'Date': 'date', 

                     'Province/State':'state',

                     'Country/Region':'country',

                    }, inplace=True)

test_df["date"] = pd.to_datetime(test_df['date'])

test_df['state'] = test_df['state'].fillna('')

test_df.info()
test_df = test_df.merge(df_temperature, on=['country','date', 'state'], how='left')

test_df = test_df.merge(countries_df, on=['country'], how='left')

test_df = test_df.merge(icu_cleaned, on=['country'], how='left')

test_df.shape
X_test = test_df.set_index("ForecastId").drop(["Lat", "Long", "date", "state", "country", "index"], axis=1).fillna(0)

#X_test = scaler.fit_transform(X_test)

y_pred_confirmed = model_infected.predict(X_test)

y_pred_deaths = model_deaths.predict(X_test)
submission = pd.DataFrame()

submission["ForecastId"] = test_df["ForecastId"]

submission = submission.set_index(['ForecastId'])

submission["ConfirmedCases"] = y_pred_confirmed.astype(int)

submission["Fatalities"] = y_pred_deaths.astype(int)

submission.to_csv("submission.csv")

submission.head()
from fbprophet import Prophet

m = Prophet()

italy_data = data[data['country']=='Italy']

ts_df = pd.concat([italy_data['date'], np.log(italy_data['confirmed']+1)], axis=1, keys=['ds', 'y'])

ts_df.head()

m.fit(ts_df)
future = m.make_future_dataframe(periods=14)

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
ts_df = data

ts_df.info()
ts_df['infectionRate'] = round((ts_df['confirmed']/ts_df['population'])*100, 5)

ts_df = ts_df[ts_df['infectionRate'] >= threshold]

ts_df.index = ts_df.date
ts_df = ts_df.drop([

                     "country", 

                     "active", 

                     "recovered", 

                     "infectionRate",

                     "state",

                     "date",

                     "Lat",

                     "Long",

                     "population",

                     "density",

                     "fertility",

                     "age",

                     "urban_percentage",

                     "icu",

                     "index"

                    ], axis= 1).dropna()



#y = train_data[["confirmed", "deaths"]]

#X = train_data.drop(["confirmed", "deaths"],axis=1)
ts_df = ts_df[:60]

ts_df.head()
train_percentage = 0.75
train = ts_df[:int(train_percentage*(len(ts_df)))]

valid = ts_df[int((1-train_percentage)*(len(ts_df))):]
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)

model_fit = model.fit()
prediction = model_fit.forecast(model_fit.y, steps=len(valid))
pred = pd.DataFrame(index=range(0,len(prediction)),columns=ts_df.columns)

for j in range(0,prediction.shape[1]):

    for i in range(0, len(prediction)):

        pred.iloc[i][j] = prediction[i][j]
for i in ts_df.columns:

    print('rmse value for', i, 'is : ', np.sqrt(mean_squared_error(pred[i], valid[i])))
days_to_predict = 14

future_dt = pd.date_range(ts_df.last_valid_index(), periods=days_to_predict)



model = VAR(endog=ts_df)

model_fit = model.fit()

yhat = model_fit.forecast(model_fit.y, steps=days_to_predict)



pred_df = pd.DataFrame(yhat, columns=ts_df.columns)

pred_df = pred_df.drop([

                     "humidity", 

                     "sunHour", 

                     "tempC", 

                     "windspeedKmph"

                    ], axis=1)

pred_df['confirmed'] = pred_df['confirmed'].astype(int)

pred_df['deaths'] = pred_df['deaths'].astype(int)

pred_df.index = future_dt

pred_df.head()