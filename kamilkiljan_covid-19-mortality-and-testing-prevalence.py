import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import plotly.express as px

import plotly.graph_objects as pgo

from plotly.subplots import make_subplots

import numpy as np

from sklearn.linear_model import LinearRegression as LR

from sklearn.metrics import mean_squared_error, r2_score

import math

import datetime



# for time series

from fbprophet import Prophet

from fbprophet.plot import plot_plotly

import plotly.offline as py



py.init_notebook_mode()

pd.plotting.register_matplotlib_converters()

sns.set_style('whitegrid')

pd.set_option('display.max_columns', 30)

outlier_countries = ['Poland', 'Romania', 'Czech Republic']

other_countries = [

    'Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Ireland', 'Italy', 'Netherlands', 'Norway', 

    'Portugal', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom']



selected_countries = outlier_countries + other_countries

countries_colormap = {country: 'Purple' for country in outlier_countries}

countries_colormap.update({country: 'DodgerBlue' for country in other_countries})



plotly_default_layout = {

    'colorway': [color for _, color in countries_colormap.items()]

}

plotly_default_layout = {

    'colorway': ['DodgerBlue']

}



trace_primary = 'DodgerBlue'

trace_secondary = 'Purple'

annotation_primary = 'Coral'

annotation_secondary = 'DarkGrey'



# Converting dates (for testing data)

def convert_date_d_mon_Y(date_string):

    date_string = date_string.replace('-', ' ')+' 2020'

    date = None

    try:

        # 1 Apr 2020

        date = datetime.datetime.strptime(date_string, '%d %b %Y')

    except ValueError:

        # 1 April 2020

        date = datetime.datetime.strptime(date_string, '%d %B %Y')

    return date



# Load data

# Main datasource - COVID-19 cases

df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

df = df.rename({'Province/State': 'province', 'Country/Region': 'country', 'Lat': 'lat', 'Long': 'long', 'Date': 'date', 

                            'Confirmed': 'confirmed', 'Deaths': 'deaths', 'Recovered': 'recovered'}, axis=1)

df.country = df.country.replace({'Czechia': 'Czech Republic'})

# COVID-19 testing data

tests_stats = pd.concat([

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_Conducted_31Mar2020.csv'), 

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_05April2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_07April2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_09April2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_15April2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_26April2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_01May2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_05May2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_08May2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_11May2020.csv'),

    pd.read_csv('../input/covid19-tests-conducted-by-country/Tests_conducted_23May2020.csv'),

], sort=True)

tests_stats['Date'].fillna(tests_stats['As of'], inplace=True)

tests_stats['Country'].fillna(tests_stats['Country or region'], inplace=True)

tests_stats['Tests'].fillna(tests_stats['Tested'], inplace=True)

tests_stats = tests_stats.rename({'Country': 'country', 'Tests': 'tests_total', 'Date': 'date'}, axis=1)

tests_stats = tests_stats[['country', 'tests_total', 'date']]

tests_stats.country = tests_stats.country.replace({'Czechia': 'Czech Republic'})

tests_stats = tests_stats.loc[tests_stats.date.isnull() == False]

tests_stats.date = tests_stats.date.apply(convert_date_d_mon_Y)

tests_stats.drop_duplicates(['country', 'date'], keep='last', inplace=True)

tests_stats.dropna(axis=0, inplace=True)

# Area and population

area_pop = pd.read_csv('../input/countries-dataset-2020/Pupulation density by countries.csv', thousands=',')

area_pop = area_pop.drop(['Rank', 'Area mi2', 'Density pop./mi2', 'Date', 'Population source'], axis=1)

area_pop = area_pop.rename({'Country (or dependent territory)': 'country', 'Area km2': 'area', 'Population': 'population', 'Density pop./km2': 'density'}, axis=1)

# WDI data

wdi_health_system = pd.read_csv('../input/world-development-indicators-by-countries/health_system.csv')

columns = {

    'Country': 'country',

    'Health workers Physicians per 1,000 people 2009-18': 'physicians_per_thousand',

    'Health workers Nurses and midwives per 1,000 people 2009-18': 'nurses_per_thousand',

}

wdi_health_system = wdi_health_system[[k for k in columns]].rename(columns, axis=1)

wdi_health_risks = pd.read_csv('../input/world-development-indicators-by-countries/Health_Risk_factors.csv')

columns = {

    'Country': 'country',

    'Prevalence of smoking Male % of adults 2016': 'smoking_male',

    'Prevalence of smoking female % of adults 2016': 'smoking_female',

    'Incidence of tuberculosis  per 100,000 people 2018': 'tb_per_100k',

    'Prevalence of diabetes  % of population ages 20 to 79 2019': 'diabetes',

}

wdi_health_risks = wdi_health_risks[[k for k in columns]].rename(columns, axis=1)

wdi_pollution = pd.read_csv('../input/world-development-indicators-by-countries/sustainability.csv')

columns = {

    'Country': 'country',

    'Ambient PM2.5 air pollution mean annual exposure micrograms per cubic meter 2016': 'pm25_exposure',

}

wdi_pollution = wdi_pollution[[k for k in columns]].rename(columns, axis=1)

wdi_data = wdi_health_system.copy()

wdi_data = pd.merge(wdi_data, wdi_health_risks, on='country')

wdi_data = pd.merge(wdi_data, wdi_pollution, on='country')

wdi_data = wdi_data[wdi_data.country.isin(selected_countries)]

# Healthcare index and quality of life index

# https://www.numbeo.com/health-care/indices_explained.jsp

healthcare_index = pd.read_csv('../input/countries-dataset-2020/Quality of life index by countries 2020.csv')

healthcare_index = healthcare_index[['Country', 'Quality of Life Index', 'Health Care Index', 'Pollution Index']]

healthcare_index = healthcare_index.rename({

    'Country': 'country', 

    'Quality of Life Index': 'quality_of_life_index', 

    'Health Care Index': 'healthcare_index',

    'Pollution Index': 'pollution_index'}, axis=1)

# Age structure

age_structure = pd.read_csv('../input/countries-dataset-2020/Coutries age structure.csv')

age_structure = age_structure[['Country', 'Age above 65 Years']]

age_structure = age_structure.rename({'Country': 'country', 'Age above 65 Years': 'above_65_years'}, axis=1)

age_structure.above_65_years = age_structure.above_65_years.apply(lambda x: int(x.strip('%'))/100)





# Countries selection

df = df.loc[df.country.isin(selected_countries)].loc[df.province.isna()].drop(['province'], axis=1)



# Data preprocessing

df.date = pd.to_datetime(df.date)

# tests_stats.tests_for_day = pd.to_datetime(tests_stats.tests_for_day.apply(lambda x: x+' 2020'), infer_datetime_format=True)



# Data enhancement - timeseries

df = pd.merge(df, area_pop, how='left', on='country')

df = pd.merge(df, wdi_data, how='left', on='country')

df = pd.merge(df, healthcare_index, how='left', on='country')

df = pd.merge(df, age_structure, how='left', on='country')

day_0 = df.date.min()

df['days_since_day_0'] = df.date.apply(lambda date: (date-day_0).days)

date_1cpm = df.loc[df.confirmed/df.population*10**6>=1].groupby('country').date.min()-datetime.timedelta(days=1)

date_10cpm = df.loc[df.confirmed/df.population*10**6>=10].groupby('country').date.min()-datetime.timedelta(days=1)

df['date_1cpm'] = df.apply(lambda row: date_1cpm[row.country], axis=1)

df['date_10cpm'] = df.apply(lambda row: date_10cpm[row.country], axis=1)

df['days_since_1cpm'] = df.apply(lambda row: max(0, (row.date-row.date_1cpm).days), axis=1)

df['days_since_10cpm'] = df.apply(lambda row: max(0, (row.date-row.date_10cpm).days), axis=1)

df['mortality'] = df.deaths/df.confirmed



# Data enhancement - tests data

df_tests = pd.merge(tests_stats, df, how='inner', left_on=['country', 'date'], right_on=['country', 'date'])

df_tests['tests_per_million'] = df_tests.tests_total/df_tests.population

df_tests['testing_prevalence'] = df_tests.confirmed/df_tests.tests_total

df_tests['pop_per_test'] = df_tests.population/df_tests.tests_total



print(f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d')}")

print(f"pandas version: {pd.__version__}")
df.tail(5)
df.describe()
df_tests.tail(5)
df_tests.describe()
chart_limit_x = df.days_since_1cpm.max()

chart_limit_y = max(df.confirmed/df.population*10**6)

fig = pgo.Figure()

fig.update_layout({

    'title': 'Cumulative cases since one case per million', 

    'width': 700,

    'height': 550,

    **plotly_default_layout

})



# Plot doubling times

# https://en.wikipedia.org/wiki/Doubling_time

number_at_time = lambda initial_number, time, doubling_period: initial_number*2**(time/doubling_period)

for Td in range(2, 4):

    fig.add_trace(pgo.Scatter(

        x=[x for x in range(1, chart_limit_x+1)], 

        y=[number_at_time(1, x, Td) for x in range(0, chart_limit_x)],

        mode='lines', 

        showlegend=False,

        hoverinfo='skip',

        line_shape='spline',

        line={'color': annotation_primary, 'dash': 'dash'}

    ))

fig.add_trace(pgo.Scatter(

    name='Doubling time',

    x=[22.5, 35.5],

    y=[7000, 7000],

    hoverinfo='skip',

    text=['𝑇𝑑=2', '𝑇𝑑=3'],

    textfont={'color': annotation_primary},

    mode="text",

    showlegend=False

))



# Plot traces for each country

for country_name in selected_countries:

    country_data = df[(df.country==country_name) & (df.days_since_1cpm>0)]

    fig.add_trace(pgo.Scatter(

        x=country_data.days_since_1cpm, 

        y=country_data.confirmed/country_data.population*10**6, 

        mode='lines', 

        name=country_name,

        text = country_data.date.apply(lambda x: x.strftime('%Y-%m-%d')),

        hovertemplate = '''%{y:.2f} cases per million in %{x:.0f} days (%{text})''',

        line={'color': countries_colormap[country_name]},

    ))

    

fig.update_yaxes({

    'title': 'Cases', 

    'type': 'log', 'range': (0, 4),  # log scale offers easier comparison of growth dynamics

#     'range': (0, chart_limit_y),

    'ticktext': ['1 per million', '10 per million', '100 per million', '1,000 per million', '10,000 per million'],

    'tickvals': [1, 10, 100, 1000, 10000],

})

fig.update_xaxes({

    'title': 'Days since 1 per million', 

})

fig.show()
# Plotly Express Scatter Animated

animation_df = df[df.date.gt(datetime.datetime(2020, 3, 14))]

fig = px.scatter(

    animation_df, 

    y=animation_df.confirmed/animation_df.population*10**6,

    x="mortality", 

    animation_frame=animation_df.date.apply(lambda x: x.isoformat()[:10]), 

    animation_group='country',

    size='confirmed', 

    color='country',

    color_discrete_map=countries_colormap,

    hover_name='country',

    range_y=[0,(animation_df.confirmed/animation_df.population*10**6).max()*1.05], 

    range_x=[0,animation_df.mortality.max()*1.05],

    labels={'y': 'Cases per million', 'mortality': 'Mortality', 'animation_frame': 'Date', 'country': 'Country', 'confirmed': 'Confirmed cases'},

    title='Cumulative cases and mortality since one per million', 

    width=700,

    height=650,

)

fig.show()
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.03)

fig.update_layout({

    'title': 'Mortality correlation with testing statistics',

    'width': 700,

    'height': 550,

    **plotly_default_layout

})

lr_train_X1, lr_train_X2, lr_train_y = [], [], []

lr_dataset = None



# # Plotting countries' paths

# # Paths show a pattern of mortality and testing prevalence changes in time

# for country in selected_countries:

#     country_data = df_tests[df_tests.country==country]

#     fig.add_trace(pgo.Scatter(

#         x=country_data.testing_prevalence,

#         y=country_data.mortality,

#         mode='lines',

#         line_shape='spline',

#         showlegend=False,

#         name='',

#         line={'color': annotation_secondary},

#     ), row=1, col=1)



# Plotting values



for country in selected_countries:

    country_data = df_tests[df_tests.country==country]

    fig.add_trace(pgo.Scatter(

        x=country_data.testing_prevalence,

        y=country_data.mortality,

        mode='markers',

#         showlegend=False,

        text = ['Positive tests: {ps}%<br>Mortality: {m}%<br>{d}'.format(

            ps=str(round(row.testing_prevalence*100, 2)), 

            m=str(round(row.mortality*100, 2)), 

            d=row.date.strftime('%Y-%m-%d')) for _, row in country_data.iterrows()],

        hovertemplate = "%{text}",

        name=country,

        marker={'color': countries_colormap[country]},

    ), row=1, col=1)

    fig.add_trace(pgo.Scatter(

        x=country_data.pop_per_test,

        y=country_data.mortality,

        mode='markers',

        showlegend=False,

        text = ['Population per test: {ps}<br>Mortality: {m}%<br>{d} - {c}'.format(

            ps=str(round(row.pop_per_test, 2)), 

            m=str(round(row.mortality*100, 2)), 

            c=row.country, 

            d=row.date.strftime('%Y-%m-%d')) for _, row in country_data.iterrows()],

        hovertemplate = "%{text}",

        name='',

        marker={'color': annotation_secondary}

    ), row=1, col=2)

    

# To show correlation a linear regression model is fitted to the data

lr_data = df_tests[['testing_prevalence', 'pop_per_test', 'mortality']].copy()

lr_testing_prevalence, lr_pop_per_test = LR(), LR()

lr_testing_prevalence.fit(df_tests.testing_prevalence.to_numpy().reshape(-1, 1), df_tests.mortality.to_numpy().reshape(-1, 1))

lr_pop_per_test.fit(df_tests.pop_per_test.to_numpy().reshape(-1, 1), df_tests.mortality.to_numpy().reshape(-1, 1))



# Generate predictions for the data

lr_data['mortality_pred_1'] = lr_testing_prevalence.predict(lr_data.testing_prevalence.to_numpy().reshape(-1, 1))

lr_data['mortality_pred_2'] = lr_pop_per_test.predict(lr_data.pop_per_test.to_numpy().reshape(-1, 1))



lr_data.sort_values('testing_prevalence', inplace=True)

fig.add_trace(pgo.Scatter( 

        x=lr_data.testing_prevalence, y=lr_data.mortality_pred_1,

        mode='lines', 

        showlegend=False,

        hoverinfo='skip',

        line_shape='spline',

        line={'color': annotation_primary}

    ), row=1, col=1)



lr_data.sort_values('pop_per_test', inplace=True)

fig.add_trace(pgo.Scatter( 

        x=lr_data.pop_per_test, y=lr_data.mortality_pred_2,

        mode='lines', 

        showlegend=False,

        hoverinfo='skip',

        line_shape='spline',

        line={'color': annotation_primary}

    ), row=1, col=2)



# Plot R2 Score

# https://en.wikipedia.org/wiki/Coefficient_of_determination

fig.add_trace(pgo.Scatter(

    name='R2 Score',

    x=[0.22],

    y=[0.07],

    hoverinfo='skip',

    text='𝑅²='+str(round(r2_score(lr_data.mortality, lr_data.mortality_pred_1), 3)),

        textfont={'color': annotation_primary},

    mode="text",

    showlegend=False

), row=1, col=1)



fig.add_trace(pgo.Scatter(

    name='R2 Score',

    x=[50],

    y=[0.075],

    hoverinfo='skip',

    text='𝑅²='+str(round(r2_score(lr_data.mortality, lr_data.mortality_pred_2), 3)),

        textfont={'color': annotation_primary},

    mode="text",

    showlegend=False

), row=1, col=2)



fig.update_yaxes({'type': 'log', 'tickvals': [.005, .01, .02, .05, .1], 'title': 'Mortality', 'ticktext': ['0.5%', '1%', '2%', '5%', '10%']}, row=1, col=1)

fig.update_yaxes({'type': 'log', 'tickvals': [.005, .01, .02, .05, .1]}, row=1, col=2)

fig.update_xaxes({'title': 'Testing prevalence', 'tickvals': [.05, .1, .15, .2, .25, .3], 'ticktext': ['5%', '10%', '15%', '20%', '25%', '30%']}, row=1, col=1)

fig.update_xaxes({'title': 'Population per test', 'type': 'log', 'tickvals': [50, 100, 200, 500, 1000, 2000]}, row=1, col=2)

fig.show()



df_tests_corr = df_tests[['testing_prevalence', 'pop_per_test', 'mortality']].corr()[['mortality']].drop('mortality')

df_tests_corr.style.background_gradient(

    cmap='bwr',

    vmin=-1,

    vmax=1,

).set_caption("Mortality correlation")
compared_dims = [

    'quality_of_life_index', 

    'healthcare_index', 

    'pollution_index',

    'above_65_years',

    'mortality',

    'physicians_per_thousand',

    'nurses_per_thousand',

    'smoking_male',

    'smoking_female',

    'tb_per_100k',

    'diabetes',

    'pm25_exposure',

    'density',

]

mortality_corr_df = df.groupby('country')[['country'] + compared_dims]

mortality_corr_df = mortality_corr_df.tail(1).reset_index(drop=True)

mortality_corr_df = mortality_corr_df.corr()[['mortality']].drop('mortality')

mortality_corr_df.style.background_gradient(

    cmap='bwr',

    vmin=-1,

    vmax=1,

).set_caption("Mortality correlation")