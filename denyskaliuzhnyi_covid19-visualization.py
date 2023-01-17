UPDATE = True
import os
import urllib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.plotting import register_matplotlib_converters
from itertools import cycle
register_matplotlib_converters()
# data from https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases
URL_confirmed = 'https://bit.ly/3c7gjoh'
URL_deaths = 'https://bit.ly/39VZqeu'
URL_recovered = 'https://bit.ly/2UR4pca'
LOAD_DIR = '/kaggle/working/loaded'

def fetch_covid_data(url, fname, load_dir=LOAD_DIR):
    load_to = os.path.join(LOAD_DIR, fname)
    urllib.request.urlretrieve(url, load_to)

if UPDATE:
    if not os.path.exists(LOAD_DIR):
        os.mkdir(LOAD_DIR)
    fetch_covid_data(url=URL_confirmed, fname='confirmed.csv')
    fetch_covid_data(url=URL_deaths, fname='deaths.csv')
    fetch_covid_data(url=URL_recovered, fname='recovered.csv')

CONFIRMED_PATH = '/kaggle/working/loaded/confirmed.csv'
RECOVERED_PATH = '/kaggle/working/loaded/recovered.csv'
DEATHS_PATH = '/kaggle/working/loaded/deaths.csv'
expected_no_nan = ['Country/Region', 'Lat', 'Long', 'Date', 'Value']
confirmed = pd.read_csv(CONFIRMED_PATH)
confirmed.head()
assert confirmed[expected_no_nan].isna().sum().sum() == 0
print('RAW:')
confirmed.info()
# All the cleaning stuff
confirmed.drop(0, axis=0, inplace=True)
confirmed.columns = ['Province', 'Country', 'Lat', 'Long', 'Date', 'Confirmed', 
                     'ISO', 'Region_Code', 'Sub_Region_Code', 'Intermediate_Region_Code']
fill_values = dict(Region_Code=-1,
                   Sub_Region_Code=-1,
                   Intermediate_Region_Code=-1,
                   ISO='unknown',
                   Province='unknown')
confirmed.fillna(fill_values, downcast=True, inplace=True)

dtype = dict(Province="object",
             Country="object",
             Lat="float",
             Long='float',
             Date='datetime64',
             Confirmed='int', 
             ISO='object',
             Region_Code='int',
             Sub_Region_Code='int',
             Intermediate_Region_Code='int')
confirmed = confirmed.astype(dtype)
assert confirmed.isna().sum().sum() == 0
print('AFTER CLEANING:')
confirmed.info()
confirmed.head()
deaths = pd.read_csv(DEATHS_PATH)
deaths.head()
assert deaths[expected_no_nan].isna().sum().sum() == 0
print('RAW:')
deaths.info()
# All the cleaning stuff
deaths.drop(0, axis=0, inplace=True)
deaths.columns = ['Province', 'Country', 'Lat', 'Long', 'Date', 'Deaths', 
                  'ISO', 'Region_Code', 'Sub_Region_Code', 'Intermediate_Region_Code']
fill_values = dict(Region_Code=-1,
                   Sub_Region_Code=-1,
                   Intermediate_Region_Code=-1,
                   ISO='unknown',
                   Province='unknown')
deaths.fillna(fill_values, downcast=True, inplace=True)

dtype = dict(Province="object",
             Country="object",
             Lat="float",
             Long='float',
             Date='datetime64',
             Deaths='int', 
             ISO='object',
             Region_Code='int',
             Sub_Region_Code='int',
             Intermediate_Region_Code='int')
deaths = deaths.astype(dtype)
assert deaths.isna().sum().sum() == 0
print('AFTER CLEANING:')
deaths.info()
deaths.head()
recovered = pd.read_csv(RECOVERED_PATH)
recovered.head()
assert recovered[expected_no_nan].isna().sum().sum() == 0
print('RAW:')
recovered.info()
# All the cleaning stuff
recovered.drop(0, axis=0, inplace=True)
recovered.columns = ['Province', 'Country', 'Lat', 'Long', 'Date', 'Recovered', 
                  'ISO', 'Region_Code', 'Sub_Region_Code', 'Intermediate_Region_Code']
fill_values = dict(Region_Code=-1,
                   Sub_Region_Code=-1,
                   Intermediate_Region_Code=-1,
                   ISO='unknown',
                   Province='unknown')
recovered.fillna(fill_values, downcast=True, inplace=True)

dtype = dict(Province="object",
             Country="object",
             Lat="float",
             Long='float',
             Date='datetime64',
             Recovered='int', 
             ISO='object',
             Region_Code='int',
             Sub_Region_Code='int',
             Intermediate_Region_Code='int')
recovered = recovered.astype(dtype)
assert recovered.isna().sum().sum() == 0
print('AFTER CLEANING:')
recovered.info()
recovered.head()
assert confirmed.shape == deaths.shape and confirmed.shape >= recovered.shape
print('confirmed dataset shape:', confirmed.shape)
print('deaths dataset shape:', deaths.shape)
print('recovered dataset shape:', recovered.shape)
on = ['Province', 'Country', 'Lat', 'Long', 'Date', 'ISO', 
      'Region_Code', 'Sub_Region_Code', 'Intermediate_Region_Code']
covid19 = pd.merge(confirmed, deaths, on=on, how='inner', validate='1:1')
covid19 = pd.merge(covid19, recovered, on=on, how='left', validate='1:1')
covid19['Recovered'].fillna(0, downcast='int', inplace=True)

covid19.head()
assert covid19.isna().sum().sum() == 0
print('THREE SETS MERGED:')
covid19.info()
LOAD_DIR = "preprocessed"
FILE_NAME = 'covid19.csv'
PATH = os.path.join(LOAD_DIR, FILE_NAME)


if UPDATE:
    if not os.path.exists(LOAD_DIR):
        os.mkdir(LOAD_DIR) 
    covid19.to_csv(PATH, index=False)
# ignore regions and aggregate by country
values = ['Confirmed', 'Deaths', 'Recovered', 'Lat', 'Long']
aggfunc=dict(Confirmed='sum',
             Deaths='sum',
             Recovered='sum',
             Lat='last',  # select most infected
             Long='last') # select most infected
# to get lat and long of most infected country regions 

covid19_sorted_confirmed = covid19.sort_values('Recovered') 

# this made equal number of records for each country within time range
covid19_agg = covid19_sorted_confirmed.pivot_table(values=values, 
                                                   index='Country', 
                                                   columns='Date', 
                                                   aggfunc=aggfunc).stack().reset_index()

covid19_agg_sorted = covid19_agg.sort_values(["Country", "Date"])


# extract new data
def max_num_or_zero(x):
    return np.max(np.c_[x, np.zeros(x.shape)], axis=1)

def growth(x):
    increase = x.values[1:] - x.values[:-1]
    only_nonnegative = max_num_or_zero(increase)
    return np.r_[0, only_nonnegative]

# for plotly's plots
covid19_agg_sorted['Date_animation'] = covid19_agg_sorted["Date"].astype('str')
# extract growth from cumulative data
covid19_agg_sorted['ConfirmedDelta'] = covid19_agg_sorted.groupby(['Country'])["Confirmed"].transform(growth)
covid19_agg_sorted['RecoveredDelta'] = covid19_agg_sorted.groupby(['Country'])["Recovered"].transform(growth)
covid19_agg_sorted['DeathsDelta'] = covid19_agg_sorted.groupby(['Country'])["Deaths"].transform(growth)

# select important rows for future research
covid19_final = covid19_agg_sorted[['Country', 'Lat', 'Long', 
                                    'Confirmed', 'ConfirmedDelta', 
                                    'Deaths', 'DeathsDelta', 
                                    'Recovered', 'RecoveredDelta', 
                                    'Date', 'Date_animation']].copy()


# check whether aggregation was correct
OK = True

n_records = covid19_final.shape[0]
n_unique_dates = pd.unique(covid19_final["Date"]).size
n_unique_countries = pd.unique(covid19_final["Country"]).size

try:
    assert n_records == n_unique_countries * n_unique_dates, "aggregation issue"
except AssertionError as e:
    print(e)
    print('records:', n_records)
    print('must be: ', n_unique_countries * n_unique_dates,)
    OK = False
else:
    print('records:', n_records)
    print('countries:', n_unique_countries)
    

# date of first records within countries
f = covid19_final.groupby('Country')['Date'].min().unique()
# date of last records within countries
t = covid19_final.groupby('Country')['Date'].max().unique()
# period length
c = covid19_final.groupby('Country')['Date'].count().unique()

try:
    assert len(f) == 1 and len(t) == 1 and c == (t[0] - f[0]) / np.timedelta64(1, 'D') + 1, "aggregation issue"
except AssertionError as e:
    print(e)
    print('unique min dates:', f)
    print('unique max dates:', t)
    print('period length:', c)
    OK = False
else:
    print('period length:', c[0])
    print('start', np.datetime64(f[0], 'D'))
    print('end:', np.datetime64(t[0], 'D'))


if not OK:
    exit()
    
print()
covid19_final.info()
print()
print("OK" if OK else "ERROR")
covid19_final.head()
covid = covid19_final.copy()
# declare useful functionality

cumsum_cols = ['Confirmed', 'Deaths', 'Recovered']
delta_cols = ['ConfirmedDelta', 'DeathsDelta', 'RecoveredDelta']
pretty_print = {
    'Confirmed': 'Total Confirmed',
    'Deaths': 'Total Deaths',
    'Recovered': 'Total Recovered',
    'ConfirmedDelta': 'New Confirmed',
    'DeathsDelta': 'New Deaths',
    'RecoveredDelta': 'New Recovered'}
rolling_kw = dict(window=7,
                  win_type='gaussian',
                  center=True,
                  min_periods=1)
std = 1
    

def get_countries(frame):
    """Return a list of (unique) countries in 'frame'"""
    return frame['Country'].unique()



def get_countries_atleast(frame, atleast=10000, col='Confirmed'):
    """Return list of countries that have at least 'atleast' number of confirmed cases"""
    lr = last_records(frame)
    return lr.loc[lr[col] >= atleast, 'Country'].values



def omit_not_important_countries(frame, n_important=5, col='Confirmed'):
    n_countries = len(get_countries(frame))
    cols = delta_cols + cumsum_cols + ["Country", 'Date']
    frame_important = countries_top(frame, 
                                    n_important, 
                                    col=col)[cols]
    frame_unimportant = countries_top(frame, 
                                      n_countries-n_important, 
                                      col=col, 
                                      top_is_largest=False)[cols]
    frame_unimportant = frame_unimportant.groupby('Date')[cumsum_cols+delta_cols].sum().reset_index()
    frame_unimportant['Country'] = 'Others'
    return pd.concat([frame_important, frame_unimportant], sort=True)
    
    
    
def describe_country(frame, country, cols=delta_cols):
    """Return a frame of the growth statistics of a given country"""
    frame = frame.query(f"Country == '{country}'")
    describe = frame.describe()
    return describe[cols]



def last_records(frame, return_date=False):
    """Return only last records ia a frame"""
    date = frame["Date"]
    last_date = max(date)
    if return_date:
        return frame[date == last_date], last_date
    return frame[date == last_date]



def countries_top(frame, n=5, col='Confirmed', top_is_largest=True):
    """Return all of the records of countries that are top infected"""
    if n == -1:
        n = len(get_countries(frame))
    lr = last_records(frame)
    lr_srt = lr.sort_values(col)
    if top_is_largest:
        countries_top = lr_srt[-n:]['Country']
    else:
        countries_top = lr_srt[:n]['Country']
    return frame.query("Country in @countries_top")



def world_wide(frame):
    """Aggregate statistics across countries by date"""
    cols = delta_cols + cumsum_cols
    return frame.groupby('Date')[cols].sum().reset_index()

    
    
def plot_growth(frame, country=None, rolling=True):
    """
    Plot two figures. Left one is time vs total cases.
    Right one is total vs growth. There are two options:
    1) supply the whole dataframe and pass a country to plot statistics about
    2) supply global records (e.g. frame provided by world_wide function) and omit country
    Return figure with two axes
    """
    def plot_time_vs_total(frame, rolling=True, ax=None):
        ax = ax or plt.gca()
           
        x = frame['Date']
        ycols = cumsum_cols
        ys = [frame[col] for col in ycols]
        
        for y, label in zip(ys, ycols):
            if rolling:
                y = y.rolling(3, win_type='gaussian', center=True).mean(std=1).dropna()
            sns.lineplot(x=x, y=y, label=label, ax=ax)
   
        ax.set_ylim(0)
        ax.set_xlabel('')
        ax.set_ylabel('Total number of cases')
        ax.tick_params(axis='x', labelrotation=-30) 
    
    def plot_total_vs_growth(frame, rolling=True, ax=None):
        ax = ax or plt.gca()
        
        frame = frame.set_index('Date')[cumsum_cols+delta_cols]
        frame = frame.resample('7D', label='right').aggregate({'Confirmed': 'last',
                                                               'Deaths': 'last',
                                                               'Recovered': 'last',
                                                               'ConfirmedDelta': 'sum',
                                                               'RecoveredDelta': 'sum',
                                                               'DeathsDelta': 'sum'})
        frame = frame[:-1]
        
        xcols = cumsum_cols
        ycols = delta_cols
        xs = [frame[col] for col in xcols]
        ys = [frame[col] for col in ycols]

        for x, y, label in zip(xs, ys, xcols):
#             if rolling:
#                 y = y.rolling(**rolling_kw).mean(std=std)
#                 x = x.rolling(**rolling_kw).mean(std=std)
            sns.lineplot(x=x+1, y=y+1, label=label, ax=ax)
                   
        ax.set_xscale('log')
        ax.set_yscale('log')
        m = max(ax.get_ylim()[1], ax.get_xlim()[1])
        l = min(ax.get_ylim()[0], ax.get_xlim()[0])
        ax.set_xlim(l, m)
        ax.set_ylim(l, m)
        ax.set_xlabel('Total number of cases')
        ax.set_ylabel('Number of new cases')
        ax.set_aspect('equal', adjustable='box')
        
    
    fig, [axl, axr] = plt.subplots(1, 2, 
                                   figsize=(18, 8))
    if country is not None:
        fig.suptitle(country, x=0.51, y=0.97, color='0.1')
        frame = frame.query(f"Country == '{country}'")
    plot_time_vs_total(frame, rolling, axl)
    plot_total_vs_growth(frame, rolling, axr)
    return fig



def plot_growth_country_wise(frame, thumb_frac=0.04):
    """
    Plot three figures total vs growth (figures related to Confirmed, Deaths and Recovered)
    Return a list containing 3 figures with 1 axes
    """
    figs = []
    for colx, coly in zip(cumsum_cols, delta_cols):
        fig, ax = plt.subplots(figsize=(16, 16))
        figs.append(fig)
        
        lr = last_records(frame)
        X_last = lr[[colx, coly]].values
        X_last_log = np.log(X_last + 1)
        
        min_dist_2 = (thumb_frac * max(X_last_log.max(0) - X_last_log.min(0)))**2
        shown_images = np.array([[0, 0]])

        for country in get_countries(frame):
            country_data = frame.loc[frame['Country'] == country, [colx, coly]]
            
            X = (country_data.rolling(7, center=True).mean().dropna() + 1).values
            last_X = X[-1]
            last_Xlog = np.log(last_X + 1)
            dist = np.sum((last_Xlog - shown_images)**2, 1)
        
            if np.min(dist) >= min_dist_2:
                shown_images = np.vstack([shown_images, last_Xlog])
                ax.text(last_X[[0]], last_X[[1]], country, fontsize=10)
            ax.plot(X[:, 0], X[:, 1], color='black', alpha=0.3, linewidth=0.5)                
            ax.scatter(last_X[[0]], last_X[[1]], color='red', alpha=0.3)

        ax.set_xlabel(pretty_print[colx], fontsize=18)
        ax.set_ylabel(pretty_print[coly], fontsize=18)
        ax.set_aspect('equal')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1)
        ax.set_ylim(1)
    return figs



def plot_pie(frame, title_date=None, n_countries=None, order='Confirmed'):
    """
    Plot three pie charts to show percentage of Confirmed, Deaths and 
    Recovered cases across countries (percentage relative to a global situation). 
    n_countries specifies number of most infected countries to show on a pie. 
    Countries that are not in top will be aggregated in 'Others' piece. 
    """
    countries = last_records(frame).sort_values(order, ascending=False)['Country']
    colors = cycle(px.colors.qualitative.Plotly)
    color_map = {}
    for country in countries:
        color_map.update({country: next(colors)})
    color_map.update({'Others': '#d9d9d9'})
    
    for col in cumsum_cols:
        frame = last_records(frame)
        if n_countries is not None:
            frame = omit_not_important_countries(frame, n_important=n_countries, col=col)
        
        fig = px.pie(frame,  
                     values=col, 
                     names='Country', 
                     color='Country',
                     color_discrete_map=color_map,
                     hole=.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(annotations=[dict(text=col,
                                            x=0.5, 
                                            y=0.5,
                                            font_size=18,
                                            showarrow=False)],
                          margin = dict(t=50, l=10, r=10, b=10))
        fig.update_layout(title={
            'text': title_date.strftime('%d %B %Y'),
            'y':0.96,
            'x':0.42,
            'xanchor': 'center',
            'yanchor': 'top'})
        fig.show()
#       fig.write_image(f"{col}.png")



def plot_bar(frame, n_countries=5, title_date=None, include_others=False):
    """
    Plot bars for top n_countries infected countries.
    For each country there are 3 bars relative to Confirmed, Deaths and Recovered.
    """
    frame = last_records(frame)
    if include_others:
        frame = omit_not_important_countries(frame, n_important=n_countries)
    else:
        frame = countries_top(frame, n_countries)    
    frame = frame.sort_values(['Confirmed'], ascending=False)
    fig = go.Figure(data=[
        go.Bar(name='Confirmed', x=frame['Country'], y=frame['Confirmed']),
        go.Bar(name='Deaths', x=frame['Country'], y=frame['Deaths']),
        go.Bar(name='Recovered', x=frame['Country'], y=frame['Recovered'])
    ])

    fig.update_layout(
        template='seaborn',
        yaxis_title="Number of people", 
        width=800, 
        height=500)
    
    if title_date:
         fig.update_layout(title={
             'text': title_date.strftime('%d %B %Y'),
             'y':0.88,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'})
    fig.show()
    

def plot_choropleth(frame, animation=False):
    if not animation:
        frame = last_records(frame)
        
    fig = px.choropleth(frame,
                        locations="Country",
                        locationmode='country names',
                        color="Confirmed",
                        color_continuous_scale="Sunsetdark", 
                        projection="natural earth", 
                        animation_frame="Date_animation" if animation else None,
                        width=800,
                        height=600)
    fig.update(layout_coloraxis_showscale=False)
    
    text = "COVID-19 Growth Animation" if animation else actual_date.strftime('%d %B %Y') 
    fig.update_layout(title={
                     'text': text,
                     'y':0.98 if animation else 0.9,
                     'x':0.5,
                     'xanchor': 'center',
                     'yanchor': 'top'},
                      margin=dict(t=0, l=0, r=0, b=0))
    fig.show()

    
def plot_scatter_geo(frame, animation=False):
    if not animation:
        frame = last_records(frame)
    fig = px.scatter_geo(frame, 
                         lat="Lat", 
                         lon='Long',
                         size=np.sqrt(frame['Confirmed']),
                         hover_name="Country",
                         hover_data=["Confirmed"],
                         animation_frame="Date_animation" if animation else None,
                         projection="natural earth",
                         width=800,
                         height=600)
    
    text = "COVID-19 Growth Animation" if animation else actual_date.strftime('%d %B %Y') 
    fig.update_layout(title={
                     'text': text,
                     'y':0.98 if animation else 0.9,
                     'x':0.5,
                     'xanchor': 'center',
                     'yanchor': 'top'},
                      margin=dict(t=10, l=0, r=0, b=0))
    fig.show()
    
    
def plot_growth_rate(frame):
    """frame: expect global data"""
    frame = frame.set_index('Date')[delta_cols]
    frame = frame.resample('7D', label='right').sum()
    frame = frame[:-1]
    frame = frame.pct_change(1).dropna()
    
    ax = frame.plot(figsize=(16, 10))
    
    ax.set_xlim(frame.index.min())
    ticks_format = mpl.ticker.PercentFormatter(xmax=1)
    ax.yaxis.set_major_formatter(ticks_format)
    ax.set_xlabel('')
    ax.set_ylabel('Weekly Growth Rate (Compared to previous week)')
    ax.legend()
    

def plot_global_bar(frame, title_date=None):
    """frame: expect summarize data"""
    fig = px.bar(x=frame, y=frame.index, orientation='h')
    fig.update_layout(
        template='seaborn',
        yaxis_title="", 
        xaxis_title="Total number of cases",
        title={
         'text': title_date.strftime('%d %B %Y'),
         'x':0.5,
         'xanchor': 'center',
         'yanchor': 'top'})
    fig.show()
    
    
def plot_ts_country_wise(frame):      
    for col in cumsum_cols:
        fig = px.line(frame, x="Date", y=col, line_group='Country')
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        fig.show()
        

def plot_confirmed_vs_deaths(frame, title_date=None):
    """frame: expect actual information"""
    fig = px.scatter(frame, 
                     x="Confirmed",
                     y="Deaths", 
                     hover_name='Country',
                     log_x=True,
                     log_y=True)
    fig.update_layout(title={
        'text': title_date.strftime('%d %B %Y'),
        'y':0.96,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
   

rc = {
    'grid.color': '.8',

    'axes.grid': True,
    'axes.edgecolor': '.9',
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'axes.labelcolor': '0.1',

    'figure.titlesize': 22,

    'legend.fontsize': 16,

    'lines.linewidth': 3,
    'lines.markersize': 10,
 
    'xtick.direction': 'in',
    'xtick.labelsize': 16,
    'xtick.color': '0.1',
    'xtick.major.size': 5,
    'xtick.major.width': 3,
    'xtick.minor.size': 4,
    'xtick.minor.width': 1,

    'ytick.direction': 'in',
    'ytick.labelsize': 16,
    'ytick.color': '0.1',
    'ytick.major.size': 5,
    'ytick.major.width': 3,
    'ytick.minor.size': 4,
    'ytick.minor.width': 1,
}
# plt.rcParams
# mpl.rcParams.update(rc)
# define useful data patterns
covid_global = world_wide(covid)
covid_actual, actual_date = last_records(covid, return_date=True)
covid_top5 = countries_top(covid, 5)
covid_actual_top5 = countries_top(covid_actual, 5)
covid_summarize = covid_actual[cumsum_cols].sum().sort_values()
covid.describe()[delta_cols]
plot_global_bar(covid_summarize, title_date=actual_date)
with mpl.rc_context(rc=rc):
    fig = plot_growth(covid_global)
    fig.suptitle('COVID-19 Global Growth', x=0.51, y=0.97, color='0.1')
with sns.plotting_context('talk'), sns.axes_style('darkgrid'):
    plot_growth_rate(covid_global)
plot_bar(covid, title_date=actual_date, include_others=True)
plot_choropleth(covid, animation=False)
plot_pie(covid, title_date=actual_date, n_countries=10)
plot_ts_country_wise(covid)
with sns.axes_style('whitegrid'):
    plot_growth_country_wise(covid)
plot_confirmed_vs_deaths(covid_actual, title_date=actual_date)
plot_scatter_geo(covid, animation=True)