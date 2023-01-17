def state_fix(state):

    #Is Abrv

    if pd.isnull(state): return

    if len(state) == 2:

        return abbrev_us_state[state] if state in abbrev_us_state.keys() else state

    #Is Fullname

    else:

        return us_state_abbrev[state] if state in us_state_abbrev.keys() else state

        

us_state_abbrev = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

}



abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))
#Load needed libraries

import pandas as pd



#Settings to display pandas

pd.set_option('display.max_columns', None)



#Read in dataframe and format datatime

base_path = "/kaggle/input/uncover/"

ihme_path = base_path + "ihme/2020_03_30/" + "Hospitalization_all_locs.csv"

ihme_df = pd.read_csv(ihme_path, usecols = ["location","date","allbed_mean", "ICUbed_mean", "InvVen_mean"])

ihme_df.rename(columns={"location":"state"}, inplace=True)

ihme_df["state_code"] = ihme_df["state"].apply(state_fix)

ihme_df.date = pd.to_datetime(ihme_df.date)

ihme_df.head(1)
hosp_cap_path = base_path + "harvard_global_health_institute/" + "hospital-capacity-by-state-20-population-contracted.csv"

hosp_cap_df = pd.read_csv(hosp_cap_path, 

                            usecols=["state", "total_hospital_beds", "total_icu_beds"])

hosp_cap_df.rename(columns={"state":"state_code"}, inplace=True)

hosp_cap_df["state"] = hosp_cap_df["state_code"].apply(state_fix)

hosp_cap_df.head(1)
resources_df = ihme_df.merge(hosp_cap_df, on=["state","state_code"])



#Make new col for total_vents

resources_df = resources_df.assign(total_vents=

                                   resources_df.total_icu_beds/4.0)



#Subtract from total_icu_beds

resources_df = resources_df.assign(total_icu_beds=

                                   resources_df.total_icu_beds-resources_df.total_vents)



#Calc net resources

resources_df[["avail_beds", "avail_icu", "avail_vents"]] = resources_df[["total_hospital_beds", "total_icu_beds", "total_vents"]] - resources_df[["allbed_mean", "ICUbed_mean", "InvVen_mean"]].values



#Calc avail score

def calc_score(row):

    return 3 * row["avail_vents"] + 2* row["avail_icu"] + 1* row["avail_beds"]

resources_df["avail_score"] = resources_df.apply(calc_score, axis=1)



#Make text column for display

resources_df["text"] = resources_df["avail_score"].astype(str)

resources_df.head(1)
import requests

import numpy as np



r= requests.get("https://api.census.gov/data/2019/pep/population?get=POP,NAME,DENSITY&for=state:*&key=c2049966d9c7bf9e0b31496d60c598ffdd999ad9")

results = r.json()

columns = results.pop(0)

pop_info = pd.DataFrame(results, columns=columns, dtype='float')

pop_info.rename(columns={"NAME":"state", "state":"state_ID"}, inplace=True)

pop_info.head(1)
resources_df = resources_df.merge(pop_info, on="state")

resources_df = resources_df.assign(avail_dens_scaled = resources_df["avail_score"] / resources_df["DENSITY"])

resources_df = resources_df.assign(avail_pop_scaled = resources_df["avail_score"] / resources_df["POP"])

resources_df.head(1)
import plotly

import plotly.graph_objs as go

import plotly.offline as offline

from plotly.graph_objs import *

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



### colorscale:    



scl = [[0.0, '#b30000'],[0.1, '#e34a33'],[0.2, '#fc8d59'],[0.3, '#fdcc8a'],[0.4, '#fef0d9'],[0.5, '#fef0d9'],

       [0.6, '#edf8fb'],[0.7, '#b2e2e2'],[0.8, '#66c2a4'],[0.9, '#2ca25f'],[1.0, '#006d2c']] # reds

### create empty list for data object:    



data_slider = []

data_slider = []

#### I populate the data object



all_dates = list(resources_df.date.unique())[::2]

zmax,zmin = resources_df.avail_dens_scaled.max(), resources_df.avail_dens_scaled.min()

for date in all_dates:



    # I select the year (and remove DC for now)

    resources_date_df = resources_df[(resources_df['state']!= 'District of Columbia' ) &  (resources_df['date']== date )]



    for col in resources_date_df.columns:  # I transform the columns into string type so I can:

        resources_date_df[col] = resources_date_df[col].astype(str)



    ### create the dictionary with the data for the current year

    data_one_year = dict(

                        type='choropleth',

                        locations = resources_date_df['state_code'],

                        z=resources_date_df['avail_dens_scaled'].astype(float),

                        locationmode='USA-states',

                        zmax = zmax,

                        zmin=zmin,

                        colorscale = scl,

                        text = resources_date_df['text'],

                        )



    data_slider.append(data_one_year)  # I add the dictionary to the list of dictionaries for the slider

steps = []



for i in range(len(data_slider)):

    step = dict(method='restyle',

                args=['visible', [False] * len(data_slider)],

                label='Year {}'.format(all_dates[i])) # label to be displayed for each step (year)

    step['args'][1][i] = True

    steps.append(step)





##  I create the 'sliders' object from the 'steps' 



sliders = [dict(active=0, pad={"t": 1}, steps=steps)]  
# I set up the layout (including slider option)



layout = dict(geo=dict(scope='usa',

                       projection={'type': 'albers usa'}),

              sliders=sliders)



# I create the figure object:



fig = dict(data=data_slider, layout=layout) 







# to plot in the notebook



plotly.offline.iplot(fig)