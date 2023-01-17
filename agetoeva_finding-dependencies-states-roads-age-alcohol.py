import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from plotly import tools

from plotly.offline import init_notebook_mode, iplot

import plotly.plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True)

accid_raw = pd.read_csv("../input/ACC_AUX.CSV")

accid_raw.head()
states = accid_raw.groupby('STATE')

count_by_state = np.asarray(states.YEAR.count())

fatals_by_state = np.asarray(states.FATALS.sum())



stcodes = np.asarray(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', \

                     'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', \

                     'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', \

                     'VA', 'WA', 'WV', 'WI', 'WY'])



stpopulation = np.asarray([4858979, 738432, 6828065, 2978204, 39144818, 5456574, 3590886, 945934, 646449, 20271272, 10214860, 1431603, 1654930, 12859995, 6619680, \

                           3123899, 2911641, 4425092, 4670724, 1329328, 6006401, 6794422, 9922576, 5489594, 2992333, 6083672, 1032949, 1896190, 2890845, \

                           1330608, 8958013, 2085109, 19795791, 10042802, 756927, 11613423, 3911338, 4028977, 12802503, 1056298, 4896146, 858469, 6600299, \

                           27469114, 2995919, 626042, 8382993, 7170351, 1844128, 5771337, 586107])



stpop_density = np.asarray([37, 0.5, 23.2, 22.1, 97, 20.3, 286.3, 187.4, 4088.4, 145.9, 68.6, 86.1, 7.7, 89.4, 71.4, \

                          21.6, 13.7, 43.2, 41.7, 16.6, 238.9, 336.3, 67.8, 26.6, 24.6, 34.2, 2.7, 9.5, 10.2, \

                          57.4, 467.2, 6.6, 162.2, 79.8, 4.2, 109.7, 22, 16.2, 110.5, 394.4, 62.9, 4.4, 61.8, \

                          40.6, 14.1, 26.2, 82.0, 41.7, 29.6, 41.2, 2.3])



stvehicles = np.asarray([1030, 960, 660, 700, 840, 340, 860, 950, 350, 710, 820, 760, 790, 750, 610, 1050, 830, 840, \

                        910, 780, 790, 820, 870, 870, 680, 830, 1120, 1000, 500, 830, 690, 770, 570, 670, 1080, \

                        910, 860, 770, 760, 730, 770, 950, 840, 720, 870, 910, 840, 870, 750, 860, 1140])



sturban = np.asarray([59.0, 66.0, 89.8, 56.2, 95.2, 86.2, 88.0, 83.3, 100.0, 91.2, 75.1, 91.9, 70.6, 88.5, 72.4, 64.0, 74.2, \

                      58.4, 73.2, 38.7, 87.2, 92.0, 74.6, 73.3, 49.3, 70.4, 55.9, 73.1, 94.2, 60.3, 94.7, 77.4, 87.9, 66.1, \

                      59.9, 77.9, 66.2, 81.0, 78.7, 90.7, 66.3, 56.7, 66.4, 84.7, 90.6, 38.9, 75.5, 84.0, 48.7, 70.2, 64.8])
# accidents per capita (10^5 )population

acc_per_capita = count_by_state / stpopulation * 1e+5

# accidents per population density

acc_per_density = count_by_state / stpop_density

# fatals per accidents

fatals_per_acc = fatals_by_state / count_by_state
# Accidents per capita (10^5 population)

data = [ dict(

        type='choropleth',

        autocolorscale = True,

        locations = stcodes,

        z = acc_per_capita,

        locationmode = 'USA-states',

        marker = dict(

                line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = 'Accidents per capita')

        ) ]



layout = dict(

        title = '2015 US Traffic Fatalities<br>(Accidents)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )



fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )
idx = []

for i in range(51):

    idx.append(False if i == 8 else True)

idx = np.asarray(idx)

  

# this plot shows dependency of the accidents on the population density    

acc_dot = go.Scatter(

    x = stpop_density[idx],

    y = count_by_state[idx],

    mode='markers',

    marker=dict(

        size='16',

        color = fatals_per_acc[idx], #np.random.randn(500), #set color equal to a variable

        colorscale='Hot',

        showscale=True

    )

)

data = [acc_dot]



iplot(data, filename='scatter-plot-with-colorscale')
acc_dot_veh = go.Scatter(

    x = stvehicles,

    y = count_by_state,

    mode='markers',

    marker=dict(

        size='16',

        color = fatals_per_acc, #np.random.randn(500), #set color equal to a variable

        colorscale='Hot',

        showscale=True

    )

)

data = [acc_dot_veh]



# this plot shows dependency of the accidents on the vehicles per capita

iplot(data, filename='scatter-plot-with-colorscale')
acc_vs_urban = go.Scatter(

    x = sturban,

    y = count_by_state,

    mode='markers',

    marker=dict(

        size='16',

        color = fatals_per_acc, #np.random.randn(500), #set color equal to a variable

        colorscale='Hot',

        showscale=True

    )

)

data = [acc_vs_urban]

# this plot shows dependency of the accidents on the urbanization ratio

iplot(data, filename='scatter-plot-with-colorscale')
hist_counts = [0 for i in range(14)]

hist_fat = [0 for i in range(14)]

for i in range(14):

    for j in range(51):

        if (i - 1) * 5 + 30 < sturban[j] <= i * 5 + 30:

            hist_counts[i] += 1

            hist_fat[i] += fatals_per_acc[j]

            

data = [go.Bar(

            x=list(range(14)),

            y=[(hist_fat[i] / hist_counts[i] - 1.05 if hist_counts[i] else 0) for i in range(14)]

    )]

layout = go.Layout(

    title='Avg fatals by urbanization ratio',

)



fig = go.Figure(data=data, layout=layout)



# Avg fatals by urbanization ratio

iplot(fig, filename='basic-bar')
rural_by_roads = accid_raw[accid_raw.A_RU == 1].groupby('A_ROADFC')

rural_count_by_roads = rural_by_roads.YEAR.count()

urban_by_roads = accid_raw[accid_raw.A_RU == 2].groupby('A_ROADFC')

urban_count_by_roads = urban_by_roads.YEAR.count()

unk_by_roads = accid_raw[accid_raw.A_RU == 3].groupby('A_ROADFC')

unk_count_by_roads = unk_by_roads.YEAR.count()

print(rural_count_by_roads)
rural = go.Bar(

            x=['Interstate', 'Freeway/expressway', 'Principal arterial (other)', 'Minor arterial', \

               'Collector', 'Local', 'Unknown'],

            y=rural_count_by_roads,

            name='Rural'

    )

urban = go.Bar(

            x=['Interstate', 'Freeway/expressway', 'Principal arterial (other)', 'Minor arterial', \

               'Collector', 'Local', 'Unknown'],

            y=urban_count_by_roads,

            name='Urban'

    )

unknown = go.Bar(

            x=['Interstate', 'Freeway/expressway', 'Principal arterial (other)', 'Minor arterial', \

               'Collector', 'Local', 'Unknown'],

            y=unk_count_by_roads,

            name='Unknown'

    )



data = [rural, urban, unknown]

layout = go.Layout(

    title='Types of roads',

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

# This plot shows different types of roads

iplot(fig, filename='stacked-bar')
speed_by_roads = accid_raw[accid_raw.A_SPCRA == 1].groupby('A_ROADFC')

speed_count_by_roads = speed_by_roads.YEAR.count()

not_by_roads = accid_raw[accid_raw.A_SPCRA == 2].groupby('A_ROADFC')

not_count_by_roads = not_by_roads.YEAR.count()

unk_by_roads = accid_raw[accid_raw.A_SPCRA == 3].groupby('A_ROADFC')

unk_count_by_roads = unk_by_roads.YEAR.count()



data = [go.Bar(

            x=['Interstate', 'Freeway/expressway', 'Principal arterial (other)', 'Minor arterial', \

               'Collector', 'Local', 'Unknown'],

            y=speed_count_by_roads/ (speed_count_by_roads + not_count_by_roads) - 0.15,

            name='Unknown'

    )]



layout = go.Layout(

    title='Speeding'

)



fig = go.Figure(data=data, layout=layout)



# Shows the distribution of accidents with speeding ratio

iplot(fig, filename='stacked-bar')
speed_by_roads = accid_raw[accid_raw.A_POSBAC == 1].groupby('A_ROADFC')

speed_count_by_roads = speed_by_roads.YEAR.count()

not_by_roads = accid_raw[accid_raw.A_POSBAC == 2].groupby('A_ROADFC')

not_count_by_roads = not_by_roads.YEAR.count()

unk_by_roads = accid_raw[accid_raw.A_POSBAC == 3].groupby('A_ROADFC')

unk_count_by_roads = unk_by_roads.YEAR.count()



speed = go.Bar(

            x=['Interstate', 'Freeway/expressway', 'Principal arterial (other)', 'Minor arterial', \

               'Collector', 'Local', 'Unknown'],

            y=speed_count_by_roads,

            name='Positive BAC'

    )

other = go.Bar(

            x=['Interstate', 'Freeway/expressway', 'Principal arterial (other)', 'Minor arterial', \

               'Collector', 'Local', 'Unknown'],

            y=not_count_by_roads,

            name='Negative BAC'

    )

unknown = go.Bar(

            x=['Interstate', 'Freeway/expressway', 'Principal arterial (other)', 'Minor arterial', \

               'Collector', 'Local', 'Unknown'],

            y=unk_count_by_roads,

            name='Unknown'

    )



data = [speed, other, unknown]

layout = go.Layout(

    title='BAC Test',

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)



# Show

iplot(fig, filename='stacked-bar')
np.mean(speed_count_by_roads / not_count_by_roads)
per_raw = pd.read_csv("../input/PER_AUX.CSV")

drivers = per_raw[per_raw.A_PTYPE == 1]
age3_drivers = np.asarray(drivers.groupby('A_AGE3').A_PTYPE.count())



data = [go.Bar(

            x=['0-3', '4-7', '8-12', '13-15', '16-20', '21-24', '25-34', '35-44', '45-54', \

               '55-64', '65-74', '75+', 'Unknown'],

            y=age3_drivers)]

layout = go.Layout(

    title='All drivers by age groups',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')
posalc = drivers[drivers.A_ALCTES == 2].groupby('A_AGE3')

pos = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0])

for a, st in posalc:

    pos[a - 1] = st.A_AGE3.count()



negalc = drivers[drivers.A_ALCTES == 1].groupby('A_AGE3')

neg = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0])

for a, st in negalc:

    neg[a - 1] = st.A_AGE3.count()



unkalc = drivers[drivers.A_ALCTES > 2].groupby('A_AGE3')

unk = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0])

for a, st in unkalc:

    unk[a - 1] = st.A_AGE3.count()



data = [go.Bar(

            x=['0-3', '4-7', '8-12', '13-15', '16-20', '21-24', '25-34', '35-44', '45-54', \

               '55-64', '65-74', '75+', 'Unknown'],

            y=pos,

            name='Alcohol positive'),

       go.Bar(

            x=['0-3', '4-7', '8-12', '13-15', '16-20', '21-24', '25-34', '35-44', '45-54', \

               '55-64', '65-74', '75+', 'Unknown'],

            y=neg, 

            name='Negative'),

       go.Bar(

            x=['0-3', '4-7', '8-12', '13-15', '16-20', '21-24', '25-34', '35-44', '45-54', \

               '55-64', '65-74', '75+', 'Unknown'],

            y=unk,

            name='Unknown')]

layout = go.Layout(

    title='All drivers by age groups',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')
data = [go.Bar(

            x=['0-3', '4-7', '8-12', '13-15', '16-20', '21-24', '25-34', '35-44', '45-54', \

               '55-64', '65-74', '75+', 'Unknown'],

            y=pos / (pos + neg + unk))]

layout = go.Layout(

    title='All drivers by age groups. Alcohol positive test distribution.',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')