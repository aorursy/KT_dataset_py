import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

init_notebook_mode(connected=True)

plt.rcParams.update({'font.size': 14})
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
# Drop date columns if they are mostly NaN

na_columns = (confirmed_df.isna().sum() / confirmed_df.shape[0]) > 0.99

na_columns = na_columns[na_columns]



confirmed_df = confirmed_df.drop(na_columns.index, axis=1)

deaths_df = deaths_df.drop(na_columns.index, axis=1)

recoveries_df = recoveries_df.drop(na_columns.index, axis=1)
## Tidy up the data

confirmed_df = confirmed_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='confirmed')

deaths_df = deaths_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='deaths')

recoveries_df = recoveries_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='recoveries')
confirmed_df['date'] = pd.to_datetime(confirmed_df['date'])

deaths_df['date'] = pd.to_datetime(deaths_df['date'])

recoveries_df['date'] = pd.to_datetime(recoveries_df['date'])
full_df = confirmed_df.merge(recoveries_df).merge(deaths_df)

full_df = full_df.rename(columns={'Country/Region': 'Country', 'date': 'Date', 'confirmed': "Confirmed", "recoveries": "Recoveries", "deaths": "Deaths"})

# Check null values

full_df.isnull().sum()
world_df = full_df.groupby(['Date']).agg({'Confirmed': ['sum'], 'Recoveries': ['sum'], 'Deaths': ['sum']}).reset_index()

world_df.columns = world_df.columns.get_level_values(0)



def add_rates(df):

    df['Confirmed Change'] = df['Confirmed'].diff().shift(-1)

 

    df['Mortality Rate'] = df['Deaths'] / df['Confirmed']

    df['Recovery Rate'] = df['Recoveries'] / df['Confirmed']

    df['Growth Rate'] = df['Confirmed Change'] / df['Confirmed']

    df['Growth Rate Change'] = df['Growth Rate'].diff().shift(-1)

    df['Growth Rate Accel'] = df['Growth Rate Change'] / df['Growth Rate']

    return df



world_df = add_rates(world_df)
def plot_aggregate_metrics(df, fig=None):

    if fig is None:

        fig = go.Figure()

    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Confirmed'],

                             mode='lines+markers',

                             name='Confirmed',

                             line=dict(color='Yellow', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Deaths'],

                             mode='lines+markers',

                             name='Deaths',

                             line=dict(color='Red', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Recoveries'],

                             mode='lines+markers',

                             name='Recoveries',

                             line=dict(color='Green', width=2)

                            ))

    return fig
plot_aggregate_metrics(world_df).show()
def plot_diff_metrics(df, fig=None):

    if fig is None:

        fig = go.Figure()



    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Mortality Rate'],

                             mode='lines+markers',

                             name='Mortality rate',

                             line=dict(color='red', width=2)))



    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Recovery Rate'],

                             mode='lines+markers',

                             name='Recovery rate',

                             line=dict(color='Green', width=2)))



    fig.add_trace(go.Scatter(x=df['Date'], 

                             y=df['Growth Rate'],

                             mode='lines+markers',

                             name='Growth rate confirmed',

                             line=dict(color='Yellow', width=2)))

    fig.update_layout(yaxis=dict(tickformat=".2%"))

    

    return fig
plot_diff_metrics(world_df).show()
fig = go.Figure()

fig.update_layout(template='plotly_dark')



tmp_df = world_df.copy()

tmp_df = tmp_df[tmp_df['Growth Rate Accel'] < 10]



fig.add_trace(go.Scatter(x=tmp_df['Date'], 

                         y=tmp_df['Growth Rate Accel'],

                         mode='lines+markers',

                         name='Growth Acceleration',

                         line=dict(color='Green', width=3)))

fig.update_layout(yaxis=dict(tickformat=".2%"))



fig.show()
confirmed_by_country_df = full_df.groupby(['Date', 'Country']).sum().reset_index()
fig = px.line(confirmed_by_country_df, x='Date', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

fig.update_layout(template='plotly_dark')

fig.show()
# Log scale to allow for view

#  (1) of countries other than China, and

#  (2) identifying linear sections, which indicate exponential growth

fig = px.line(confirmed_by_country_df, x='Date', y='Confirmed', color='Country', line_group="Country", hover_name="Country")

fig.update_layout(

    template='plotly_dark',

    yaxis_type="log"

)

fig.show()
confirmed_by_country_df.groupby('Country').max().sort_values(by='Confirmed', ascending=False)[:10]
k_layout_kwargs = {

    'font': dict(size=12,),

    'legend': dict(x=0, y=-0.7),

}
us_df = confirmed_by_country_df[confirmed_by_country_df['Country'] == 'US'].copy()

us_df = add_rates(us_df)
tmp_df = us_df[us_df['Confirmed'] > 100]



plot_aggregate_metrics(tmp_df).show()
plot_diff_metrics(tmp_df).show()
from sklearn.linear_model import LinearRegression



us_growth_rates = {}



us_n_days_to_fit = 5

confirmed_us_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == 'US') & (confirmed_by_country_df['Date'] >= (np.datetime64('today') - np.timedelta64(us_n_days_to_fit,'D')))]



x = (confirmed_us_df['Date'] - confirmed_us_df['Date'].min()).dt.days.to_numpy().reshape(-1, 1)

y = np.log(confirmed_us_df['Confirmed'])

reg = LinearRegression().fit(x, y)

print(f"Model fit score: {reg.score(x, y):.2f}")

print(f"Growth rate: {reg.coef_[0]:.3f}")

us_growth_rates[us_n_days_to_fit] = reg.coef_[0]



fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=x[:,0],

        y=np.exp(y),

        name='U.S.'

    )

)



xx = np.linspace(0, len(x[:,0]) + 14, 100)  # Forecast 14 days out

yy = reg.predict(xx.reshape(-1,1))



fig.add_trace(

    go.Scatter(

        x=xx,

        y=np.exp(yy),

        name='U.S. - Exponential fit',

        mode='lines',

    )

)



fig.update_layout(

    title=f"Exponential Model of U.S. Confirmed Cases<br>(fit to last {us_n_days_to_fit} days) with 14-Day Extrapolation",

    xaxis_title=f"Days since {confirmed_us_df['Date'].min()}",

    yaxis_title="Number of Confirmed Cases",

    yaxis_type="log",

    **k_layout_kwargs,

)



fig.show()
n_days_to_fit = 4

confirmed_us_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == 'US') & (confirmed_by_country_df['Date'] >= (np.datetime64('today') - np.timedelta64(n_days_to_fit,'D')))]



x = (confirmed_us_df['Date'] - confirmed_us_df['Date'].min()).dt.days.to_numpy().reshape(-1, 1)

y = np.log(confirmed_us_df['Confirmed'])

reg = LinearRegression().fit(x, y)

print(f"Model fit score: {reg.score(x, y):.2f}")

print(f"Growth rate: {reg.coef_[0]:.3f}")

us_growth_rates[us_n_days_to_fit] = reg.coef_[0]



fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=confirmed_by_country_df[confirmed_by_country_df['Country'] == 'US']['Date'],

        y=confirmed_by_country_df[confirmed_by_country_df['Country'] == 'US']['Confirmed'],

        name='U.S.',

        line=dict(width=4)

    )

)



predict_days_out = 7*2



exponential_fit_date_range = pd.date_range(confirmed_us_df['Date'].min(), confirmed_us_df['Date'].max() + np.timedelta64(predict_days_out,'D'))



xx = np.linspace(0, len(x[:,0]) + predict_days_out, exponential_fit_date_range.shape[0])  # Forecast 14 days out

yy = reg.predict(xx.reshape(-1,1))



fig.add_trace(

    go.Scatter(

        x=exponential_fit_date_range,

        y=np.exp(yy),

        name='U.S. - Exponential fit',

        mode='lines'

    )

)



fig.update_layout(

    title=f"Exponential Model of U.S. Confirmed Cases<br>(fit to last {us_n_days_to_fit} days) with {predict_days_out}-Day Extrapolation",

    xaxis_title=f"Date",

    yaxis_title="Number of Confirmed Cases",

    yaxis_type="log",

    **k_layout_kwargs,

)



fig.show()
proxy_country = 'Italy'



confirmed_iran_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == proxy_country) & (confirmed_by_country_df['Confirmed'] >= 100)]



x = (confirmed_iran_df['Date'] - confirmed_iran_df['Date'].min()).dt.days.to_numpy().reshape(-1, 1)

y = np.log(confirmed_iran_df['Confirmed'])

reg = LinearRegression().fit(x, y)

print(f"Model fit score: {reg.score(x, y):.2f}")

print(f"Growth rate: {reg.coef_[0]:.3f}")

iran_growth_rate = reg.coef_[0]



fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=x[:,0],

        y=np.exp(y),

        name=proxy_country

    )

)



xx = np.linspace(0, len(x[:,0]) + 14, 100)  # Forecast 14 days out

yy = reg.predict(xx.reshape(-1,1))



fig.add_trace(

    go.Scatter(

        x=xx,

        y=np.exp(yy),

        name=f'{proxy_country} - Exponential fit'

    )

)



fig.update_layout(

    title=f"Exponential Model of {proxy_country} Confirmed Cases<br>with 14-Day Extrapolation",

    xaxis_title=f"Days since {confirmed_iran_df['Date'].min()}",

    yaxis_title="Number of Confirmed Cases",

    yaxis_type="log",

    **k_layout_kwargs,

)



fig.show()
def linear_model(x, a, b):

    return b * x + a





def linear_model_fixed_slope(slope):

    def func(x, intercept):

        return linear_model(x, a=intercept, b=slope)

    

    return func



test_model = linear_model_fixed_slope(2)

x = np.array([1, 2, 3])

test_model(x=x, intercept=2)
from scipy.optimize import curve_fit





def get_model(model, popt):

    def fitted_model(x):

        return model(x, *popt)

    return fitted_model





x = (confirmed_us_df['Date'] - confirmed_us_df['Date'].min()).dt.days.to_numpy()

y = np.log(confirmed_us_df['Confirmed'].to_numpy())



# Pull the slope from the Iran model and use for the U.S., allowing only the intercept to vary

model = linear_model_fixed_slope(iran_growth_rate)

popt, pcov = curve_fit(model, x, y)



fitted_model_iran_rate = get_model(model, popt)



# Now do the same using the slope from the US model

model = linear_model_fixed_slope(us_growth_rates[us_n_days_to_fit])

popt, pcov = curve_fit(model, x, y)



fitted_model_us_rate = get_model(model, popt)



# Plot results

for layout_kwargs in [{}, {"yaxis_type": "log"}]:

    fig = go.Figure()



    fig.add_trace(

        go.Scatter(

            x=confirmed_by_country_df[confirmed_by_country_df['Country'] == 'US']['Date'],

            y=confirmed_by_country_df[confirmed_by_country_df['Country'] == 'US']['Confirmed'],

            name='U.S.',

            line=dict(width=4)

        )

    )



    exponential_fit_date_range = pd.date_range(confirmed_us_df['Date'].min(), confirmed_us_df['Date'].max() + np.timedelta64(14,'D'))



    xx = np.linspace(0, len(x) + 14, exponential_fit_date_range.shape[0])  # Forecast 14 days out

    yy = fitted_model_iran_rate(xx)



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy),

            name=f'U.S. - Exponential fit based on {proxy_country} growth rate ({iran_growth_rate:.0%})',

            mode='lines'

        )

    )



    #########



    yy = fitted_model_us_rate(xx)



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy),

            name=f'U.S. - Exponential fit based on US growth rate ({us_growth_rates[us_n_days_to_fit]:.0%}) (fitted to {us_n_days_to_fit} days)',

            mode='lines'

        )

    )



    fig.update_layout(

        title="Exponential Model of U.S. Confirmed Cases<br>with 14-Day Extrapolation",

        xaxis_title="Date",

        yaxis_title="Number of Confirmed Cases",

        **k_layout_kwargs,

        **layout_kwargs

    )



    fig.show()
from fbprophet.plot import plot_plotly

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot
full_pop = 330e6



#floor_model = lambda x: max(x - 1, 0)  # Use the value itself because the function only increases

floor_model = lambda x: round(0.65 * x)

cap_model = lambda x: round(min(full_pop, 1.5 * x + 10000))  # 50% above plus one to ensure floor > cap at 0



# Modeling Iran confirmed cases 

confirmed_training_df = confirmed_by_country_df[(confirmed_by_country_df['Country'] == 'US') & (confirmed_by_country_df['Confirmed'] > 0)]

confirmed_training_df = confirmed_training_df.rename(columns={'Date': 'ds', 'Confirmed': 'y'}).reset_index(drop=True)



confirmed_training_df['floor'] = confirmed_training_df.y.apply(floor_model)

confirmed_training_df['cap'] = confirmed_training_df.y.apply(cap_model)
confirmed_training_df.y = confirmed_training_df.y.apply(np.log10)

confirmed_training_df.floor = confirmed_training_df.floor.apply(np.log10)

confirmed_training_df.cap = confirmed_training_df.cap.apply(np.log10)
# Total confirmed model 

m = Prophet(

    growth='linear',

    #interval_width=0.90,

    changepoint_prior_scale=0.05,

    changepoint_range=0.9,

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=False,

    #n_changepoints=2

)

m.fit(confirmed_training_df)

future = m.make_future_dataframe(periods=14)

future['floor'] = confirmed_training_df.floor

future['cap'] = confirmed_training_df.cap

confirmed_forecast = m.predict(future)
for kwargs in [{}, {"yaxis_type": "log"}]:

    fig = plot_plotly(m, confirmed_forecast, plot_cap=False, changepoints=True)

    annotations = []

    annotations.append(dict(

        xref='paper',

        yref='paper',

        x=0.0,

        y=1.15,

        xanchor='left',

        yanchor='bottom',

        text='Predictions for log10 Confirmed cases U.S.',

        font=dict(

            family='Arial',

            size=30,

            color='rgb(37,37,37)'),

        showarrow=False))

    fig.update_layout(

        annotations=annotations,

        **kwargs

    )

    fig.show()
for kwargs in [{}, {"yaxis_type": "log"}]:

    fig = plot_plotly(m, confirmed_forecast, plot_cap=False, changepoints=True)

    annotations = []

    annotations.append(dict(

        xref='paper',

        yref='paper',

        x=0.0,

        y=1.15,

        xanchor='left',

        yanchor='bottom',

        text='Predictions for Confirmed cases U.S.',

        font=dict(

            family='Arial',

            size=30,

            color='rgb(37,37,37)'),

        showarrow=False))

    fig.update_layout(

        annotations=annotations,

        **kwargs

    )

    for trace in fig.data:

        trace.y = np.power(trace.y, 10)

    fig.show()
k_washington_state_min_date = np.datetime64('2020-02-24')



washington_state_selector = lambda x: x.endswith(', WA') or x == 'Washington'



us_df = full_df[full_df['Country'] == 'US'].copy()

wa_df = us_df[us_df['Province/State'].apply(washington_state_selector)]

wa_df = wa_df.drop(['Lat', 'Long'], axis=1).groupby('Date').sum().reset_index()

wa_df = wa_df[wa_df['Date'] >= k_washington_state_min_date]

wa_df = add_rates(wa_df)



tmp_df = wa_df[wa_df['Confirmed'] > 100]



fig = plot_aggregate_metrics(tmp_df)



fig.update_layout(

    title="Washington State",

    template='plotly_dark',

    yaxis_type='log',

    font=dict(

        size=18,

    ),

)



fig.show()
fig = plot_diff_metrics(tmp_df)



fig.update_layout(

    title="Washington State",

    template='plotly_dark',

    font=dict(

        size=18,

    ),

)



fig.show()
# Find Washington growth rate

wa_growth_rates = {}



n_days_to_fit = 5



wa_window_df = wa_df[wa_df['Date'] >= (np.datetime64('today') - np.timedelta64(n_days_to_fit,'D'))]



x = (wa_window_df['Date'] - wa_window_df['Date'].min()).dt.days.to_numpy().reshape(-1, 1)

y = np.log(wa_window_df['Confirmed'])

reg = LinearRegression().fit(x, y)

print(f"Model fit score: {reg.score(x, y):.2f}")

print(f"Growth rate: {reg.coef_[0]:.3f}")

wa_growth_rates[n_days_to_fit] = reg.coef_[0]



fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=wa_df['Date'],

        y=wa_df['Confirmed'],

        name='Washington State',

        line=dict(width=4)

    )

)



predict_days_out = 7*2



exponential_fit_date_range = pd.date_range(wa_window_df['Date'].min(), wa_window_df['Date'].max() + np.timedelta64(predict_days_out,'D'))



xx = np.linspace(0, len(x[:,0]) + predict_days_out, exponential_fit_date_range.shape[0])  # Forecast 14 days out

yy = reg.predict(xx.reshape(-1,1))



fig.add_trace(

    go.Scatter(

        x=exponential_fit_date_range,

        y=np.exp(yy),

        name='Washington State - Exponential fit',

        mode='lines'

    )

)



fig.update_layout(

    title=f"Exponential Model of Washington State Confirmed Cases<br>(fit to last {n_days_to_fit} days) with {predict_days_out}-Day Extrapolation",

    xaxis_title="Date",

    yaxis_title="Number of Confirmed Cases",

    yaxis_type="log",

    **k_layout_kwargs,

)



fig.show()
import requests

import json

import time



count = 0

while True:

    try:

        r = requests.get(url='https://covidtracking.com/api/states/daily')

        us_testing_df = pd.read_json(json.dumps(r.json()))

    except:

        time.sleep(np.power(2, count)) # exponential backoff

        count += 1

        continue

    break

us_testing_df['date'] = pd.to_datetime(us_testing_df['date'], format='%Y%m%d')
wa_testing_df = us_testing_df[us_testing_df['state'] == 'WA'].copy()

wa_testing_df = wa_testing_df.sort_values('date').reset_index(drop=True)
wa_testing_diff_df = pd.DataFrame({

    'date': wa_testing_df['date'],

    'positive': wa_testing_df['positive'].diff().shift(-1),

    'negative': wa_testing_df['negative'].diff().shift(-1),

})
from plotly.subplots import make_subplots



df = wa_testing_diff_df



fig = make_subplots(specs=[[{"secondary_y": True}]])



fig.add_trace(

    go.Scatter(

        x=df['date'], 

        y=df['positive'],

        mode='lines+markers',

        name='Daily Positive Tests',

        line=dict(color='red', width=2),

    ),

    secondary_y=False

)



fig.add_trace(

    go.Scatter(

        x=df['date'], 

        y=df['negative'],

        mode='lines+markers',

        name='Daily Negative Tests',

        line=dict(color='green', width=2),

    ),

    secondary_y=False

)



# Positive Rate



df = df[df['date'] > np.datetime64('2020-03-09')]



fig.add_trace(

    go.Scatter(

        x=df['date'], 

        y=df['positive'] / (df['positive'] + df['negative']),

        mode='lines+markers',

        name='Percentage of Tests Positive',

        line=dict(color='purple', width=2),

    ),

    secondary_y=True

)



fig.update_yaxes(title_text="<b>Daily Number of Tests</b>", secondary_y=False)

fig.update_yaxes(title_text="<b>Percentage of Tests Positive</b>", tickformat=".2%", secondary_y=True)



fig.show()
# Now fit for King County assuming the growth rate for Washington State



#

# Clean up the data

#

king_county_df = us_df[us_df['Province/State'].apply(lambda x: x == 'King County, WA')].reset_index(drop=True)

king_county_df = king_county_df[king_county_df['Date'] >= k_washington_state_min_date]

# Fill in datapoint from https://sccinsight.com/2020/03/10/does-king-county-have-enough-hospital-beds-to-deal-with-the-coronavirus/

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-10'), 'Confirmed'] = 190

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-10'), 'Deaths'] = np.NaN

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-10'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/12-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-11'), 'Confirmed'] = 270

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-11'), 'Deaths'] = 27

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-11'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/13-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-12'), 'Confirmed'] = 328

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-12'), 'Deaths'] = 32

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-12'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/14-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-13'), 'Confirmed'] = 388

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-13'), 'Deaths'] = 35

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-13'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/15-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-14'), 'Confirmed'] = 420

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-14'), 'Deaths'] = 37

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-14'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/16-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-15'), 'Confirmed'] = 488

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-15'), 'Deaths'] = 43

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-15'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/17-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-16'), 'Confirmed'] = 518

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-16'), 'Deaths'] = 46

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-16'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/18-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-17'), 'Confirmed'] = 562

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-17'), 'Deaths'] = 56

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-17'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/19-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-18'), 'Confirmed'] = 693

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-18'), 'Deaths'] = 60

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-18'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/20-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-19'), 'Confirmed'] = 793

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-19'), 'Deaths'] = 67

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-19'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/21-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-20'), 'Confirmed'] = 934

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-20'), 'Deaths'] = 74

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-20'), 'Recoveries'] = np.NaN

# Fill in datapoint from https://www.kingcounty.gov/depts/health/news/2020/March/22-covid.aspx

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-21'), 'Confirmed'] = 1040

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-21'), 'Deaths'] = 75

king_county_df.loc[king_county_df['Date'] == np.datetime64('2020-03-21'), 'Recoveries'] = np.NaN



# Fill NaN values using last valid value

king_county_df = king_county_df.fillna(method='ffill')



# Drop rows with non-sense data

king_county_df = king_county_df.loc[~((king_county_df['Date'] > np.datetime64('2020-03-01')) & (king_county_df['Confirmed'] == 0)), :]



# Add rates for each metric

king_county_df = add_rates(king_county_df)



#

# Hospital bed data

# based on https://sccinsight.com/2020/03/10/does-king-county-have-enough-hospital-beds-to-deal-with-the-coronavirus/

bed_capacity = 3600

hosptialization_rate = 0.05

hospital_max_caseload = bed_capacity / hosptialization_rate



#

# Fit model

#



king_county_window_df = king_county_df[king_county_df['Date'] >= (np.datetime64('today') - np.timedelta64(n_days_to_fit,'D'))]



x = (king_county_window_df['Date'] - king_county_window_df['Date'].min()).dt.days.to_numpy().reshape(-1, 1)

y = np.log(king_county_window_df['Confirmed'])



# Fit model to King County

king_county_wa_growth_rates = {}

reg = LinearRegression().fit(x, y)

print(f"Model fit score: {reg.score(x, y):.2f}")

print(f"Growth rate: {reg.coef_[0]:.3f}")

king_county_wa_growth_rates[n_days_to_fit] = reg.coef_[0]



x = x.ravel()

y = y.to_numpy()



# Pull the slope from the Washington model and use for the King County, allowing only the intercept to vary

model = linear_model_fixed_slope(wa_growth_rates[n_days_to_fit])

popt, pcov = curve_fit(model, x, y)



fitted_model_washington_rate = get_model(model, popt)



# Now do the same using the slope from the US model

us_growth_rate_n_days_to_fit = n_days_to_fit



model = linear_model_fixed_slope(us_growth_rates[us_n_days_to_fit])

popt, pcov = curve_fit(model, x, y)



fitted_model_us_rate = get_model(model, popt)



# Plot results

for layout_kwargs in [{}, {"yaxis_type": "log"}]:

    fig = go.Figure()



    fig.add_trace(

        go.Scatter(

            x=king_county_df['Date'],

            y=king_county_df['Confirmed'],

            name='King County, WA',

            line=dict(width=4)

        )

    )

    

    predict_days_out = 7*4



    exponential_fit_date_range = pd.date_range(wa_window_df['Date'].min(), wa_window_df['Date'].max() + np.timedelta64(predict_days_out,'D'))



    xx = np.linspace(0, len(x) + predict_days_out, exponential_fit_date_range.shape[0])  # Forecast number of days out

    

    ##########

    

    yy = reg.predict(xx.reshape(-1, 1))



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy),

            name=f'King County, WA - Exponential fit over last {n_days_to_fit} days'

        )

    )

    

    ##########

    

    yy = fitted_model_washington_rate(xx)



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy),

            name=f'King County, WA - Exponential fit to King County, WA, growth rate={reg.coef_[0]:.0%} (fitted to last {n_days_to_fit} days)',

            mode='lines'

        )

    )



    #########



    yy = fitted_model_us_rate(xx)



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy),

            name=f'King County, WA - Exponential fit based on US growth rate={us_growth_rates[us_n_days_to_fit]:.0%} (fitted to {us_n_days_to_fit} days)',

            mode='lines'

        )

    )

    

    #########

    

    fig.add_shape(

        # Line Horizontal

            type="line",

            x0=exponential_fit_date_range.min(),

            y0=hospital_max_caseload,

            x1=exponential_fit_date_range.max(),

            y1=hospital_max_caseload,

            line=dict(

                color="Red",

                width=4,

                dash='dash'

            ),

    )

    

    fig.add_trace(

        go.Scatter(

            x=[exponential_fit_date_range.min() - np.timedelta64(3,'D')],

            y=[np.exp(np.log(hospital_max_caseload) * 0.87),],

            mode='text',

            text='Hospital Max Caseload',

            showlegend=False

        )

    )

    

    #########



    fig.update_layout(

        title=f"Exponential Model of King County, WA Confirmed Cases<br>with {predict_days_out}-Day Extrapolation",

        xaxis_title="Date",

        yaxis_title="Number of Confirmed Cases",

        **k_layout_kwargs,

        **layout_kwargs

    )



    fig.show()
king_county_population = 2.189e6



# Plot results

for layout_kwargs in [{}, {"yaxis_type": "log"}]:

    fig = go.Figure()



    fig.add_trace(

        go.Scatter(

            x=king_county_df['Date'],

            y=king_county_df['Confirmed'] * 1e6 / king_county_population,

            name='King County, WA',

            line=dict(width=4)

        )

    )

    

    predict_days_out = 7*4



    exponential_fit_date_range = pd.date_range(wa_window_df['Date'].min(), wa_window_df['Date'].max() + np.timedelta64(predict_days_out,'D'))



    xx = np.linspace(0, len(x) + predict_days_out, exponential_fit_date_range.shape[0])  # Forecast number of days out

    

    ##########

    

    yy = reg.predict(xx.reshape(-1, 1))



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy) * 1e6 / king_county_population,

            name=f'King County, WA - Exponential fit to King County, WA, growth rate={reg.coef_[0]:.0%} (fitted to last {n_days_to_fit} days)'

        )

    )

    

    ##########

    

    yy = fitted_model_washington_rate(xx)



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy) * 1e6 / king_county_population,

            name=f'King County, WA - Exponential fit based on Washington State growth rate={wa_growth_rates[n_days_to_fit]:.0%} (fitted to {n_days_to_fit} days)',

            mode='lines'

        )

    )



    #########



    yy = fitted_model_us_rate(xx)



    fig.add_trace(

        go.Scatter(

            x=exponential_fit_date_range,

            y=np.exp(yy) * 1e6 / king_county_population,

            name=f'King County, WA - Exponential fit based on US growth rate={us_growth_rates[us_growth_rate_n_days_to_fit]:.0%} (fitted to {us_growth_rate_n_days_to_fit} days)',

            mode='lines'

        )

    )

    

    #########

    

    fig.add_shape(

        # Line Horizontal

            type="line",

            x0=exponential_fit_date_range.min(),

            y0=hospital_max_caseload,

            x1=exponential_fit_date_range.max(),

            y1=hospital_max_caseload,

            line=dict(

                color="Red",

                width=4,

                dash='dash'

            ),

    )

    

    fig.add_trace(

        go.Scatter(

            x=[exponential_fit_date_range.min() - np.timedelta64(3,'D')],

            y=[np.exp(np.log(hospital_max_caseload) * 0.87),],

            mode='text',

            text='Hospital Max Caseload',

            showlegend=False

        )

    )

    

    #########



    fig.update_layout(

        title=f"Exponential Model of King County, WA Confirmed Cases<br>with {predict_days_out}-Day Extrapolation",

        xaxis_title="Date",

        yaxis_title="Number of Confirmed Cases per Million Inhabitants",

        **k_layout_kwargs,

        **layout_kwargs

    )



    fig.show()
population_northern_italy = 27801460

population_italy = 60.48e6



print(f"Italy population correction factor: {population_italy / population_northern_italy:.2f}")
tmp_df = king_county_df[king_county_df['Confirmed'] > 50].copy()



fig = plot_aggregate_metrics(tmp_df)



fig.update_layout(

    title="King County, WA",

    template='plotly_dark',

    yaxis_type='log',

    font=dict(

        size=18,

    ),

)



fig.show()
fig = plot_diff_metrics(tmp_df)



fig.update_layout(

    title="King County, WA",

    template='plotly_dark',

    font=dict(

        size=18,

    ),

)



fig.show()
fig = go.Figure()

fig.update_layout(template='plotly_dark')



tmp_df = tmp_df[tmp_df['Growth Rate Accel'] < 10]



fig.add_trace(go.Scatter(x=tmp_df['Date'], 

                         y=tmp_df['Growth Rate Accel'],

                         mode='lines+markers',

                         name='Growth Acceleration',

                         line=dict(color='Green', width=3)))

fig.update_layout(yaxis=dict(tickformat=".2%"))



fig.update_layout(

    title="King County, WA",

    template='plotly_dark',

    font=dict(

        size=18,

    ),

)



fig.show()