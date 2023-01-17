!git clone https://github.com/ihmeuw-msca/CurveFit.git
!ls -la
!ls CurveFit
!cp -r CurveFit/src/curvefit ./
!pip install xspline
# # FIXME: try installing this module the right way instead of this tomfoolery

# import sys

# sys.path.append('/opt/conda/lib/python3.6/site-packages/curvefit-0.0.0-py3.6.egg')
from curvefit.core.model import CurveModel

from curvefit.core.functions import gaussian_cdf, gaussian_pdf
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os, requests

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
plt.rcParams['figure.figsize'] = (11, 7.5)

plt.rcParams['font.size'] = 15

plt.rcParams['figure.facecolor'] = 'w'
# Kaggle quirk: "input" data is read-only. So we'll stream data into the "output" data at

# /kaggle/working

!mkdir data
prepath = '../input/covid-cases-and-forecasts/'
# OLD: hand-saved data file

# modcollab = pd.read_csv(prepath + 'ModCollab/results-2020-05-01.csv', index_col=0)

# NEW: read latest available data from HowsMyFlattening dataset

modcollab = pd.read_csv('http://flatteningthecurve-staging.herokuapp.com/data/predictivemodel').sort_values(['region', 'date'])
modcollab.head()
modcollab = modcollab[modcollab['region'] == 'base_on'].copy().reset_index(drop=True)

# modcollab['date'] = pd.date_range(start='2020-03-08', periods=len(modcollab), freq='D')

modcollab['New Cases'] = modcollab['cumulative_incidence'].diff()
modcollab.head()
modcollab.set_index('date').drop(columns=['id', 'cumulative_incidence', 'available_hospW']).plot();

plt.grid(which='both', alpha=0.3);
# OLD: read hand-saved data file

# fisman = pd.read_csv(prepath + 'IDEA/Ontario_projections_2020-05-01.csv')

#

# NEW: read latest available data from HowsMyFlattening dataset

fisman = pd.read_csv('http://flatteningthecurve-staging.herokuapp.com/data/ideamodel')



fisman['date'] = pd.to_datetime(fisman['date']).dt.round('d')

fisman = fisman[fisman['source'] == 'on'].copy().sort_values(by='date').reset_index(drop=True)
berry_case = pd.read_csv('https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/cases_timeseries_prov.csv')
isha_case_counts = berry_case[berry_case['province']=='Ontario'].reset_index(drop=True)

isha_case_counts['date'] = pd.to_datetime(isha_case_counts['date_report'], dayfirst=True)
# isha_case_counts = berry_case[berry_case['province']=='Ontario']['date_report'].apply(lambda x: '-'.join(x.split('-')[::-1])).value_counts().to_frame()

# isha_case_counts = isha_case_counts.sort_index()

# isha_case_counts = isha_case_counts.reset_index()

# isha_case_counts['date'] = pd.to_datetime(isha_case_counts['index'], dayfirst=True).dt.round('d')
url = 'https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv'



# # Status of Ontario cases, by reported date

# url = 'https://flatteningthecurve.herokuapp.com/data/covidtests'



on = pd.read_csv(url)

on['Accurate_Episode_Date'] = pd.to_datetime(on['Accurate_Episode_Date'])

# on['date'] = pd.to_datetime(on['date'])



on = on.sort_values(by='Accurate_Episode_Date')

on.head()
# 2020-05-05: let's move away from fitting anything to data by report date

# # Get daily counts from cumulative counts

# for col in ['positive', 'resolved', 'deaths']:

#     on[f'new_{col}'] = on[col].diff()

#

# on.set_index('date').filter(like='new_').plot()
counts = on.set_index('Accurate_Episode_Date')
cases_per_day = on['Accurate_Episode_Date'].value_counts().to_frame()

cases_per_day = cases_per_day.sort_index()



counts = on.groupby(['Accurate_Episode_Date', 'Outcome1']).size().unstack()

# counts.rename(columns={'Outcome1': 'Outcome'})

counts.plot();
# # FIXME: ModCollab isn't updating regularly, and it's not clear to us when their forecast kicks in. 

# For now, hardcode the forecast beginning as the cusp in their daily cases graph

n = modcollab[modcollab['New Cases'].diff() > 1].index.max()
on.head()
fisman.head()
fig = plt.figure(tight_layout=True)

plt.plot(fisman['date'], fisman['model_incident_cases'], label='Fisman et al. IDEA forecast', linewidth=2, c='#ee00ee')

for tick in fig.axes[0].get_xticklabels():

    tick.set_rotation(90)

plt.fill_between(fisman['date'], 

                 fisman['model_incident_cases_lower_PI'], 

                 fisman['model_incident_cases_upper_PI'], 

                 alpha=0.2, color='#ee00ee')



plt.scatter(fisman['date'], fisman['reported_cases'], label='Fisman et al. IDEA reported cases', marker='s', color='#ee00ee')

plt.scatter(isha_case_counts['date'], isha_case_counts['cases'], color='k', marker='x', label='Berry et al. new cases')



plt.scatter(cases_per_day.index, cases_per_day['Accurate_Episode_Date'], color='darkgreen', label='Ontario.ca new cases')

# plt.scatter(on['date'], on['new_positive'], color='darkgreen', label='Ontario.ca new cases')



plt.scatter(modcollab.loc[:n, 'date'], modcollab.loc[:n, 'New Cases'], color='orange', label='ModCollab new cases')

plt.plot(modcollab.loc[n:, 'date'], modcollab.loc[n:, 'New Cases'], color='orange', linewidth=2, label='ModCollab forecast')



plt.grid()

plt.legend()

plt.ylabel('Daily new cases');
# Let's get a running history of cases by accurate episode date, as reported by Ontario.ca

cases_on = on['Accurate_Episode_Date'].value_counts().rename('New Cases').to_frame()

cases_on = cases_on.sort_index().reset_index().rename(columns={'index': 'Episode Date'})

cases_on.head()



df = cases_on.copy()
df.plot(x='Episode Date', y='New Cases', figsize=(9, 6), marker='.', linewidth=0.5);
# First, drop the last seven rows, then interpolate any missing rows



recent_rows_to_drop = 7



df = df.iloc[:-recent_rows_to_drop]



start = df.iloc[0]['Episode Date']

end = df.iloc[-1]['Episode Date']



df = (df.set_index('Episode Date')

        .reindex(pd.date_range(start=start, 

                               end=end, 

                               freq='d'))

        .fillna(0))
df.head()
df = (df.reindex(pd.date_range(start=start, 

                               end=end + pd.offsets.DateOffset(months=1), 

                               freq='d'))

        .reset_index()

        .rename(columns={'index': 'Episode Date'}))

df.head()
df.head()
df.tail()
df['days since first case'] = list(range(1, df.shape[0] + 1))

df['province'] = 'ON'

df['intercept'] = 1
df.tail()
model = CurveModel(

    df=df.dropna(subset=['New Cases']),

    col_t='days since first case',

    col_obs='New Cases',

    col_group='province',

    col_covs=[['intercept'], ['intercept'], ['intercept']],

    param_names=['alpha', 'beta', 'p'],

    link_fun=[lambda x: x, lambda x: x, lambda x: x],

    var_link_fun=[lambda x: x, lambda x: x, lambda x: x],

    fun=gaussian_pdf

)
# FIXME: optimization is very brittle with respect to the initial values and gaussian priors.

# Can we improve how we choose these magic numbers?

model.fit_params([3.6e-02, 9.6e+01, 2.5e+04],

                fe_gprior=[[0, np.inf],

                           [0, np.inf],

                           [0, np.inf]])
model.result
model.params
df['IHME fit'] = model.predict(df['days since first case'])
df.plot(x='Episode Date', y=['New Cases', 'IHME fit'])
import plotly

plotly.__version__
!pip install --upgrade plotly==4.6
import plotly.express as px

import plotly.graph_objects as go

import plotly.io as pio

pio.templates
# Get each model's cutoff dates

model2meta = {'IDEA': [fisman.dropna(subset=['reported_cases']).iloc[-1]['date'], 'rgb(200, 0, 200)'],

              'ModCollab': [modcollab.iloc[n-1]['date'], 'orange'],

              'IHME': [df.dropna(subset=['New Cases']).iloc[-recent_rows_to_drop]['Episode Date'], '#009900']}
model2meta = pd.DataFrame(model2meta, index=['cutoff_date', 'color']).T

model2meta
# TODO: reduce repetitiveness of code in this cell



f = go.Figure()



# Ontario data



# All Ontario datapoints

f.add_trace(go.Scatter(

    x=df['Episode Date'],

    y=df['New Cases'],

    name='Cases by Episode Date (ON confirmed positive)',

    mode='markers',

    marker_symbol='square',

    marker_color='#80b0ff', # '#009900',

    marker_size=8,

    legendgroup='Ontario-data'

))





# Fisman et al. data, aggregated by Berry et al.

# https://github.com/ishaberry/Covid19Canada/) is:

# Berry I, Soucy J-PR, Tuite A, Fisman D. Open access epidemiologic data and 

# an interactive dashboard to monitor the COVID-19 outbreak in Canada. 

# CMAJ. 2020 Apr 14;192(15):E420. doi: https://doi.org/10.1503/cmaj.75262



f.add_trace(go.Scatter(

        x=fisman['date'],

        y=fisman['reported_cases'],

        name='Cases by Report Date (Berry et al.)',

        mode='markers',

        marker_color='#3070ff',# 'rgb(200, 0, 200)',

        showlegend=True,

        marker_size=7,

        legendgroup='Fisman-data'

))





# IHME-style forecast lines



df_before = df[df['Episode Date'] <= model2meta.loc['IHME', 'cutoff_date']]

df_after = df[df['Episode Date'] >= model2meta.loc['IHME', 'cutoff_date']]



for data, opacity, showlegend in zip([df_before, df_after], [0.2, 1], [False, True]):

    f.add_trace(go.Scatter(

        x=data['Episode Date'],

        y=data['IHME fit'],

        name='Forecast (IHME-style)',

        mode='lines',

        marker_color='#009900',

        marker_size=5,

        showlegend=showlegend,

        opacity=opacity,

        legendgroup='Ontario-forecast'

    ))





# IDEA forecast lines



fisman_before = fisman[fisman['date'] <= model2meta.loc['IDEA', 'cutoff_date']]

fisman_after = fisman[fisman['date'] >= model2meta.loc['IDEA', 'cutoff_date']]



for data, opacity, showlegend in zip([fisman_before, fisman_after], [0.2, 1], [False, True]):

    f.add_trace(go.Scatter(

            x=data['date'],

            y=data['model_incident_cases'],

            name='Forecast (IDEA)',

            mode="lines",

            line=go.scatter.Line(color=("rgb(200, 0, 200)")),

            showlegend=showlegend,

            legendgroup='Fisman-forecast',

            opacity=opacity

    ))



# ModCollab data and forecast



# # 2020-05-06: for now, let's hide all case data except Ontario's

# f.add_trace(go.Scatter(

#     x=modcollab.loc[:n, 'date'],

#     y=modcollab.loc[:n, 'New Cases'],

#     name='Actual Cases (COVID-19 ModCollab)',

#     mode='markers',

#     marker_color='orange',

#     marker_size=5,

#     legendgroup='ModCollab-data'

# ))



f.add_trace(go.Scatter(

    x=modcollab.loc[n:, 'date'],

    y=modcollab.loc[n:, 'New Cases'],

    name='Forecast (ModCollab - Expected Scenario)',

    mode='lines',

    marker_color='orange',

    marker_size=5,

    legendgroup='ModCollab-forecast'

))
# # Rest of ON datapoints not used in fit

# f.add_trace(go.Scatter(

#     x=df.dropna().iloc[-7:]['Episode Date'],

#     y=df.dropna().iloc[-7:]['New Cases'],

#     name='Actual Cases (Ontario Confirmed Positive) not fitted',

#     mode='markers',

#     marker_color='#009900',

#     marker_size=5,

#     marker_symbol='circle-open',

#     legendgroup='Ontario'

# ))



    

# # Berry et al.: same data as Fisman et al. use, and no forecasts



# f.add_trace(go.Scatter(

#     x=isha_case_counts['date'],

#     y=isha_case_counts['date_report'],

#     name='Cases (Berry et al.)',

#     mode='markers',

#     marker_color='black',

#     marker_symbol='circle-open',

#     marker_size=9,

#     legendgroup='Berry'

# ))
f.update_layout(template='plotly_white', 

                showlegend=True, 

                legend_x=0,

                legend_y=1.1,

                legend_orientation='v',

                legend_font_size = 13, # magic underscore notation ftw

                xaxis_title='Report date (Fisman, Berry); accurate episode date (ModCollab, ON)',

                yaxis_title='Daily cases',

               # hovermode="x unified",

                font=dict(

                    family='Helvetica',

                    size=15,

                    color="#303030")

)



# 2020-05-05: on second thought, don't bother with vertical lines

# for _, d, c in model_date_color[:2]:

#     f.add_shape(dict(type="line",

#                 x0=d,

#                 y0=0,

#                 x1=d,

#                 y1=1000,

#                 line=dict(color=c, width=2))

#                 )



f.show()
f.update_layout(legend_font_size = 15, # magic underscore notation ftw

                xaxis_title='Report date (IDEA); accurate episode date (ModCollab, ON)',

                yaxis_title='Daily Cases',

                font=dict(

                    family='Helvetica',

                    size=18,

                    color="#303030")

)
today = pd.to_datetime('today').strftime('%Y-%m-%d')

print(today)
# ## Uncomment to write to file

# f.write_html(f'COVID Cases and Forecasts - IDEA, ModCollab, IHME - {today}.html')