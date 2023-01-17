import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from plotly import tools

from plotly.offline import init_notebook_mode, iplot

import plotly.plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True)



accid_raw = pd.read_csv("../input/ACC_AUX.CSV")

per_raw = pd.read_csv("../input/PER_AUX.CSV")

drivers = per_raw[per_raw.A_PTYPE == 1]



stcodes = np.asarray(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', \

                     'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', \

                     'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', \

                     'VA', 'WA', 'WV', 'WI', 'WY'])
acc_drivers = pd.merge(accid_raw, drivers, on=['ST_CASE', 'YEAR'])



alc_accdr = np.asarray(acc_drivers[acc_drivers.A_ALCTES == 2].groupby('STATE').STATE.count())

neg_accdr = np.asarray(acc_drivers[acc_drivers.A_ALCTES == 1].groupby('STATE').STATE.count())

unk_accdr = np.asarray(acc_drivers[acc_drivers.A_ALCTES > 2].groupby('STATE').STATE.count())

print(unk_accdr)

print(alc_accdr + neg_accdr)



data = [ dict(

        type='choropleth',

        locations = stcodes,

        z = alc_accdr / (neg_accdr + alc_accdr),

        locationmode = 'USA-states',

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = 'Ratio')

        ) ]



layout = dict(

        title = '2015 US Alcohol Positive Drivers<br>(only tested and known result drivers)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )



fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )
rest_hrace = np.asarray(drivers[drivers.A_HRACE != 2].groupby('A_HRACE').A_HRACE.count())[1:]

unrest_hrace = np.asarray(drivers[drivers.A_REST == 2].groupby('A_HRACE').A_HRACE.count())[1:]

unk_hrace = np.asarray(drivers[drivers.A_REST == 3].groupby('A_HRACE').A_HRACE.count())[1:]

races = ['Hispanic', 'White', 'Black', 'American Indian', 'Asian', 'Pacific Islander', 'Multiple Races', 'Other']
rest = go.Bar(

            x=['Hispanic', 'Black', 'American Indian', 'Asian', 'Pacific Islander', 'Multiple Races', \

              'Other'],

            y=rest_hrace,

            name='Restraint Used'

    )

unrest = go.Bar(

            x=['Hispanic', 'Black', 'American Indian', 'Asian', 'Pacific Islander', 'Multiple Races', \

              'Other'],

            y=unrest_hrace,

            name='Restraint Not Used'

    )

unknown = go.Bar(

            x=['Hispanic', 'Black', 'American Indian', 'Asian', 'Pacific Islander', 'Multiple Races', \

              'Other'],

            y=unk_hrace,

            name='Unknown'

    )



data = [rest]#, unrest, unknown]

layout = go.Layout(

    title='Different Races',

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')
stacdr = acc_drivers[0 < acc_drivers.A_HRACE]

stacdr = stacdr[stacdr.A_HRACE < 9].groupby('STATE')

st_races = []

for st, data in stacdr:

    c = data[data.A_HRACE != 2].groupby('A_HRACE').A_HRACE.count()

    st_races.append(races[c.idxmax() - 1] if c.any() else 'White')

st_races = st_races[:38] + ['White'] + st_races[38:]

print(len(st_races), st_races)



colours_race = {'Black': 'rgb(187, 170, 144)', 'Hispanic': 'rgb(200,100,120)', 'American Indian': 'rgb(68,94,150)', \

               'Other': 'rgb(87, 170, 44)', 'White': 'rgb(200, 200, 200)', 'Asian':'rgb(123, 43, 34)', \

                'Pacific Islander': 'rgb(21, 31, 43)', 'Multiple Races': 'rgb(213, 243, 123)'}



data = []

for i in races:

    locs = [stcodes[idx] for idx in range(51) if st_races[idx] == i]

    print(i,len(locs), locs)

    irace = go.Choropleth(

                        z=['1'] * len(locs),

                        autocolorscale=False,

                        colorscale=[[0, 'rgb(255, 255, 255)'], [1, colours_race[i]]],

                        hoverinfo='text',

                        locationmode='USA-states',

                        locations=locs,

                        text=locs,

                        name=i,

                        showscale=False,

                        zauto=False,

                        zmax=1,

                        zmin=0,

                    )

    data.append(irace)



data = go.Data(data)



layout = go.Layout(

    autosize=False,

    geo=dict(

        countrycolor='rgb(102, 102, 102)',

        countrywidth=0.1,

        lonaxis=dict(

            gridwidth=1.5999999999999999,

            range=[-180, -50],

            showgrid=False

        ),

        projection=dict(

            type='albers usa'

        ),

        scope='usa',

        subunitcolor='rgb(102, 102, 102)',

        subunitwidth=0.5

    ),

    hovermode='closest',

    showlegend=True,

    title='<b>Race Distribution (only drivers)</b><br>By accidents',

    width= 800

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='pace')
drowsy = [0] * 4

distracted = [0] * 4

bac_pos = [0] * 4

pedestrian = [0] * 4

pedal = [0] * 4



i = 0

time_acc = accid_raw.groupby('A_DOW')

for day, st in time_acc:

    st = st.groupby('A_TOD')

    if day == 3: continue

        

    for time, t in st:

        if time == 3: continue

            

        drowsy[i] = t[t.A_DROWSY == 1].A_DROWSY.count()

        distracted[i] = t[t.A_DIST == 1].A_DIST.count()

        bac_pos[i] = t[t.A_POSBAC == 1].A_POSBAC.count()

        pedestrian[i] = t[t.A_PED == 1].A_PED.count()

        pedal[i] = t[t.A_PEDAL == 1].A_PEDAL.count()

        

        i += 1



print(pedestrian)

print(pedal)        



data = [go.Bar(

            x=['Weekday / Daytime', 'Weekday / Nighttime', 'Weekend / Daytime', 'Weekend / Nighttime'],

            y=drowsy,

            name='Drowsy Drivers'

    ),

        go.Bar(

            x=['Weekday / Daytime', 'Weekday / Nighttime', 'Weekend / Daytime', 'Weekend / Nighttime'],

            y=distracted,

            name='Distracted Drivers'

    ),

        go.Bar(

            x=['Weekday / Daytime', 'Weekday / Nighttime', 'Weekend / Daytime', 'Weekend / Nighttime'],

            y=bac_pos,

            name='BAC Positive'

    ), 

       go.Bar(

            x=['Weekday / Daytime', 'Weekday / Nighttime', 'Weekend / Daytime', 'Weekend / Nighttime'],

            y=pedal,

            name='Including Pedalcyclist'

    ),

       go.Bar(

            x=['Weekday / Daytime', 'Weekday / Nighttime', 'Weekend / Daytime', 'Weekend / Nighttime'],

            y=pedestrian,

            name='Including Pedestrian'

    )]



layout = go.Layout(

    title='Accidents by Time'

)



fig = go.Figure(data=data, layout=layout)

#py.iplot(fig, filename='basic-bar')
drowsy = [0] * 4

distracted = [0] * 4

bac_pos = [0] * 4

pedestrian = [0] * 4



i = 0

time_acc = accid_raw.groupby('A_DOW')

for day, st in time_acc:

    st = st.groupby('A_TOD')

    if day == 3: continue

        

    for time, t in st:

        if time == 3: continue

            

        t = t[t.A_PED == 1]

        pedestrian[i] = t.A_PED.count()

        

        bac_pos[i] = t[t.A_POSBAC == 1].A_POSBAC.count()

        t = t[t.A_POSBAC > 1]

        drowsy[i] = t[t.A_DROWSY == 1].A_DROWSY.count()

        t = t[t.A_DROWSY > 1]

        distracted[i] = t[t.A_DIST == 1].A_DIST.count()

        

        i += 1



drowsy = np.asarray(drowsy) / np.asarray(pedestrian)

distracted = np.asarray(distracted) / np.asarray(pedestrian)

bac_pos = np.asarray(bac_pos) / np.asarray(pedestrian)



print(distracted)

print(drowsy)

print(bac_pos)



data = [go.Bar(

            x=['Weekday / Daytime', 'Weekday / Nighttime', 'Weekend / Daytime', 'Weekend / Nighttime'],

            y=drowsy,

            name='Drowsy Drivers'

    ),

        go.Bar(

            x=['Weekday / Daytime', 'Weekday / Nighttime', 'Weekend / Daytime', 'Weekend / Nighttime'],

            y=distracted,

            name='Distracted Drivers'

    ),

        go.Bar(

            x=['Weekday / Daytime', 'Weekday / Nighttime', 'Weekend / Daytime', 'Weekend / Nighttime'],

            y=bac_pos,

            name='BAC Positive'

    )]



layout = go.Layout(

    title='Accidents Including Pedestrian<br>(Ratio)'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')
fatals = np.asarray(per_raw[per_raw.A_PERINJ == 1].groupby('A_AGE3').A_AGE3.count()[:-1])

ages = np.asarray(per_raw.groupby('A_AGE3').A_AGE3.count()[:-1])



print(ages)

print(fatals)



data = [go.Bar(

            x=['0-3', '4-7', '8-12', '13-15', '16-20', '21-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+'],

            y=fatals / ages

    )]

layout = go.Layout(

    title='Mortality by Age Groups'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')