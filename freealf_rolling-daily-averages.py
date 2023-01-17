%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import datetime

import statsmodels.api as sm

import plotly.graph_objects as go

import plotly.express as px
canada_cases_df = pd.read_csv('/kaggle/input/covid19-challenges/test_data_canada.csv')

provincial_cases = canada_cases_df.groupby(['province', 'date'])['case_id'].count().unstack().T.fillna(0)

all_regional_cases = canada_cases_df.groupby(['region', 'date'])['case_id'].count().unstack().T.fillna(0)

on_regional_cases = canada_cases_df[canada_cases_df['province']=='Ontario'].groupby(['region', 'date'])['case_id'].count().unstack().T.fillna(0)

can_summ = pd.concat((provincial_cases, all_regional_cases), axis=1)
window_in_days = 7

rolling_can_summ = can_summ.rolling(window_in_days).mean()

rolling_can_days_since_30_cases_df = pd.DataFrame(index=range(len(rolling_can_summ.dropna())), columns=rolling_can_summ.columns)

for sr in rolling_can_days_since_30_cases_df.columns:

    if sr in ['NWT', 'Yukon', 'Nunavut']:

        # Territories are listed twice, once as a province and once as a region.  Neither hits the threshold

        continue

    srser = rolling_can_summ[sr]

    idx_first_30_cases = np.argmax(srser.values >= 30)

    # argmax returns 0 when there is no match, so check that this is real

    if srser.iloc[idx_first_30_cases] >= 30:

        diff_length = len(rolling_can_days_since_30_cases_df.index) - len(srser.iloc[idx_first_30_cases:])

        rolling_can_days_since_30_cases_df[sr] = list(srser.iloc[idx_first_30_cases:]) + [np.nan]*diff_length
# identify regions we want to colour on the chart... provinces and regions of Ontario, 

# plus Vancouver Coastal, which really stands out as seriously flattened

# and Calgary, because Alf's family lives there :-P

interesting_regions = ['Vancouver Coastal', 'Calgary']

pmax = provincial_cases.rolling(7).mean().max()

interesting_regions = interesting_regions + pmax[pmax>=30].index.to_list()

rmax = on_regional_cases.rolling(7).mean().max()

interesting_regions = interesting_regions + rmax[rmax>=30].index.to_list()

interesting_regions
colour_palette = px.colors.qualitative.G10

traces = []

colour_i = 0

for c in rolling_can_days_since_30_cases_df.columns:

    srser = rolling_can_days_since_30_cases_df[c].dropna()

    if len(srser) > 0:

        if c in interesting_regions:

            colour = colour_palette[colour_i]

        else:

            colour = '#B3B3B3'

        traces.append(go.Scatter(x=srser.index, y=srser.values, name=c,

                                 text=[c]*len(srser.index), mode='lines',

                                 hovertemplate = "<b>%{text}</b> Days Since 30 Cases: %{x}; Rolling Daily Cases: %{y:.1f}",

                                 line=dict(color=colour),

        ))

        if c in interesting_regions:

            traces.append(go.Scatter(

                x=[srser.index[-1]],

                y=[srser.values[-1]],

                text=[c],

                mode="text",

                textposition="top right",

                textfont=dict(color=colour),

                showlegend=False,

            ))

            colour_i += 1

            if colour_i >= len(colour_palette):

                colour_i = 0

layout = go.Layout(title='Daily COVID-19 Cases: %d-Day Rolling Average' % window_in_days,

                   xaxis_title="Days Since 30 Cases", 

                   yaxis_title="Average Daily Cases (%d-day rolling)" % window_in_days,

                   yaxis_type='log',

                  )

fig = go.Figure(data=traces, layout=layout)

#fig.update_yaxes(range=[0, 12])

fig.show()
canada_mortality_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_mortality.csv')

provincial_deaths = canada_mortality_df.groupby(['province', 'date'])['death_id'].count().unstack().T.fillna(0)

all_regional_deaths = canada_mortality_df.groupby(['region', 'date'])['death_id'].count().unstack().T.fillna(0)

on_regional_deaths = canada_mortality_df[canada_mortality_df['province']=='Ontario'].groupby(['region', 'date'])['death_id'].count().unstack().T.fillna(0)

can_death_summ = pd.concat((provincial_deaths, all_regional_deaths), axis=1)
window_in_days = 7

rolling_can_deaths = can_death_summ.rolling(window_in_days).mean()

rolling_can_days_since_3_deaths_df = pd.DataFrame(index=range(len(rolling_can_deaths.dropna())), columns=rolling_can_deaths.columns)

for sr in rolling_can_days_since_3_deaths_df.columns:

    if sr in ['NWT', 'Yukon', 'Nunavut']:

        # Territories are listed twice, once as a province and once as a region.  Neither hits the threshold

        continue

    srser = rolling_can_deaths[sr]

    idx_first_3_deaths = np.argmax(srser.values >= 3)

    # argmax returns 0 when there is no match, so check that this is real

    if srser.iloc[idx_first_3_deaths] >= 3:

        diff_length = len(rolling_can_days_since_3_deaths_df.index) - len(srser.iloc[idx_first_3_deaths:])

        rolling_can_days_since_3_deaths_df[sr] = list(srser.iloc[idx_first_3_deaths:]) + [np.nan]*diff_length
colour_palette = px.colors.qualitative.G10

traces = []

colour_i = 0

for c in rolling_can_days_since_3_deaths_df.columns:

    srser = rolling_can_days_since_3_deaths_df[c].dropna()

    if len(srser) > 0:

        if c in interesting_regions:

            colour = colour_palette[colour_i]

        else:

            colour = '#B3B3B3'

        traces.append(go.Scatter(x=srser.index, y=srser.values, name=c,

                                 text=[c]*len(srser.index), mode='lines',

                                 hovertemplate = "<b>%{text}</b> Days Since 3 Deaths: %{x}; Rolling Daily Deaths: %{y:.1f}",

                                 line=dict(color=colour),

        ))

        if c in interesting_regions:

            traces.append(go.Scatter(

                x=[srser.index[-1]],

                y=[srser.values[-1]],

                text=[c],

                mode="text",

                textposition="top right",

                textfont=dict(color=colour),

                showlegend=False,

            ))

            colour_i += 1

            if colour_i >= len(colour_palette):

                colour_i = 0

layout = go.Layout(title='Daily COVID-19 Deaths: %d-Day Rolling Average' % window_in_days,

                   xaxis_title="Days Since 3 Deaths", 

                   yaxis_title="Average Daily Deaths (%d-day rolling)" % window_in_days,

                   yaxis_type='log',

                  )

fig = go.Figure(data=traces, layout=layout)

#fig.update_yaxes(range=[0, 12])

fig.update_xaxes(range=[0, 18])

fig.show()