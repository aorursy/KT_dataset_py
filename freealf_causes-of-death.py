import pandas as pd

import numpy as np

import datetime

import plotly.graph_objects as go

import plotly.express as px
stats_df = pd.read_csv('/kaggle/input/causes-of-death-data/statscan_on_causes_death.csv')

stats_df.head()
causes_df = stats_df.loc[stats_df['Characteristics']=='Number of deaths', ['Leading causes of death (ICD-10)', 'VALUE']]

causes_df.rename(columns={

    'Leading causes of death (ICD-10)' : 'Cause',

    'VALUE' : 'Deaths',

}, inplace=True)

causes_df = causes_df[causes_df['Cause'] != 'Other causes of death']

causes_df.sort_values(by='Deaths', ascending=False, inplace=True)

causes_df
ONTARIO_POPULATION_2020Q1 = 14711827

ONTARIO_POPULATION_2018 = (14188919+14241379+14318545+14405726)/4
mortality_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_mortality.csv')

mortality_df['date'] = pd.to_datetime(mortality_df['date'], dayfirst=True)

mortality_df.head()
daily_death_ser = mortality_df[mortality_df['province']=='Ontario'].groupby('date')['death_id'].count()

daily_death_ser.name = 'daily_deaths_ontario'

daily_death_ser
causes_df['Daily_Deaths'] = causes_df['Deaths']/365*ONTARIO_POPULATION_2020Q1/ONTARIO_POPULATION_2018

causes_df.head(10)
causes_df['Cause'].head(10).to_list()
# Make some friendly names

def cause_to_friendly(cause):

  c2f = {

    'Malignant neoplasms [C00-C97]' : 'All Cancers',

    'Diseases of heart [I00-I09, I11, I13, I20-I51]' : 'Heart Disease',

    'Accidents (unintentional injuries) [V01-X59, Y85-Y86]' : 'Accidents / Injuries',

    'Cerebrovascular diseases [I60-I69]' : 'Stroke and Related Diseases',

    'Chronic lower respiratory diseases [J40-J47]' : 'Chronic Lower Respiratory Diseases',

    'Influenza and pneumonia [J09-J18]' : 'Influenza and Pneumonia',

    'Diabetes mellitus [E10-E14]' : 'Diabetes',

    "Alzheimer's disease [G30]" : "Alzheimer's Disease",

    'Intentional self-harm (suicide) [X60-X84, Y87.0]' : 'Suicide',

    'Chronic liver disease and cirrhosis [K70, K73-K74]' : 'Cirrhosis and Other Chronic Liver Diseases'

  }

  try:

    return c2f[cause]

  except KeyError:

    return np.nan
causes_df['Friendly_Name'] = causes_df['Cause'].apply(cause_to_friendly)
fig = px.bar(x=daily_death_ser.index, y=daily_death_ser.values, 

             text=daily_death_ser.values,

             )

for _, row in causes_df.head(8).iterrows():

  if row['Cause'] != 'Influenza and pneumonia [J09-J18]':

    fig.add_shape(

            # Line Horizontal

                type="line",

                x0=daily_death_ser.index.min(),

                y0=row['Daily_Deaths'],

                x1=daily_death_ser.index.max(),

                y1=row['Daily_Deaths'],

                line=dict(

                    color="Black",

                    width=1,

                    dash="dot",

                ),

                layer="below",

        )

    y_text = row['Daily_Deaths']+0.1

    # fix overlap

    if row['Cause'] == 'Chronic lower respiratory diseases [J40-J47]':

      y_text = row['Daily_Deaths']-1.7

    fig.add_trace(go.Scatter(

      x=[daily_death_ser.index.min()],

      y=[y_text],

      text=[row['Friendly_Name']],

      mode="text",

      textposition="top right",

      showlegend=False,

    ))

fig.update_layout(

    autosize=False,

    width=800,

    height=900,

    title='Daily Deaths in Ontario: Comparison of COVID-19 and Estimates of non-COVID-19 Leading Causes',

    xaxis_title="Date (since first Ontario death)", 

    yaxis_title="Estimated Deaths per Day",    

)

number_of_days = (daily_death_ser.index.max() - daily_death_ser.index.min()).days + 1

fig.update_xaxes(tickangle=-90, nticks=number_of_days)

fig.show()