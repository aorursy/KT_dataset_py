import numpy as np

import pandas as pd
accident_data = pd.read_csv('../input/uk-road-safety-accidents-and-vehicles/Accident_Information.csv')

accident_data
accident_data.columns
adata = accident_data[['Accident_Index', 'Local_Authority_(District)', 'Number_of_Casualties', 'Date', 'Time']].set_index('Accident_Index')

adata['DateTime'] = pd.to_datetime(adata['Date'] + ' ' + adata['Time'])  # Date and Time combined as datetime64 type

adata
london_dist = [

'Barking and Dagenham',

'Barnet',

'Bexley',

'Brent',

'Bromley',

'Camden',

'Croydon',

'Ealing',

'Enfield',

'Greenwich',

'Hackney',

'Hammersmith and Fulham',

'Haringey',

'Harrow',

'Havering',

'Hillingdon',

'Hounslow',

'Islington',

'Kensington and Chelsea',

'Kingston upon Thames',

'Lambeth',

'Lewisham',

'Merton',

'Newham',

'Redbridge',

'Richmond upon Thames',

'Southwark',

'Sutton',

'Tower Hamlets',

'Waltham Forest',

'Wandsworth',

'Westminster',

'City of London'

]

len(london_dist)
adata = adata[adata['Local_Authority_(District)'].isin(london_dist)]

adata
lunar_data = pd.read_csv('../input/moonrise-moonset-and-phase-uk-2005-2017/UK_Lunar_Data.csv')

lunar_data
ldata = lunar_data[lunar_data['Phase']=='Full Moon']

lunar_data.loc[ldata.index-1, 'Phase'] = 'FM_Previous'

lunar_data.loc[ldata.index+1, 'Phase'] = 'FM_Next'

ldata = pd.concat([ldata, lunar_data[(lunar_data['Phase']=='FM_Previous') | (lunar_data['Phase']=='FM_Next')]]).sort_index()

ldata
ldata.info()
ldata['Date'] = pd.to_datetime(ldata['Date'], dayfirst=True)

ldata['Moonset'] = pd.to_timedelta(ldata['Moonset']+':00')

ldata['Moonrise'] = pd.to_timedelta(ldata['MoonriseLate']+':00')

ldata['PhaseTime'] = pd.to_timedelta(ldata['PhaseTime']+':00')

ldata['Dusk'] = pd.to_timedelta(ldata['CivilDusk']+':00')

ldata['Dawn'] = pd.to_timedelta(ldata['CivilDawn']+':00')

ldata.drop(['MoonriseEarly', 'MoonriseLate', 'CivilDusk', 'CivilDawn'], axis=1, inplace=True)

ldata.reset_index(drop=True, inplace=True)

ldata
import plotly.graph_objects as go

fig = go.Figure()



# Full Moon Date (Current)

fm_ldata = ldata[ldata['Phase']=='Full Moon']

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Moonrise']/np.timedelta64(1, 'h'),

                         name='Moonrise', line=dict(color='darkviolet')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Moonset']/np.timedelta64(1, 'h'),

                         name='Moonset', line=dict(color='darkred')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['PhaseTime']/np.timedelta64(1, 'h'),

                         name='PhaseTime', line=dict(color='limegreen')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Dusk']/np.timedelta64(1, 'h'),

                         name='Dusk', line=dict(color='deepskyblue')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Dawn']/np.timedelta64(1, 'h'),

                         name='Dawn', line=dict(color='darkorange')))



# Previous Date

prev_ldata = ldata[ldata['Phase']=='FM_Previous']

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=prev_ldata['Moonrise']/np.timedelta64(1, 'h') - 24,

                         name='Moonrise (Previous)', line=dict(color='violet')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=prev_ldata['Dusk']/np.timedelta64(1, 'h') - 24,

                         name='Dusk (Previous)', line=dict(color='lightskyblue')))



# Next Date

next_ldata = ldata[ldata['Phase']=='FM_Next']

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=next_ldata['Moonset'] / np.timedelta64(1, 'h') + 24,

                         name='Moonset (Next)', line=dict(color='orangered')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=next_ldata['Dawn'] / np.timedelta64(1, 'h') + 24,

                         name='Dawn (Next)', line=dict(color='orange')))



fig.update_layout(xaxis=dict(title_text='Dates when Full Moon Instance occured'),

                  yaxis=dict(tick0=0, dtick=4, title_text='Timing (in Hours)'),

                  title_text='Comparison of several Astronomical event timings on Full Moon Dates',

                  xaxis_rangeslider_visible=True)

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['PhaseTime']/np.timedelta64(1, 'h'),

                         name='Full Moon Instance', mode='markers', line=dict(color='limegreen')))



fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Dawn']/np.timedelta64(1, 'h'),

                         name='Dawn', line=dict(color='darkorange')))



fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=prev_ldata['Dusk']/np.timedelta64(1, 'h') - 24,

                         name='Dusk (Previous)', fill='tonexty', line=dict(color='lightskyblue')))



fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=next_ldata['Dawn'] / np.timedelta64(1, 'h') + 24,

                         name='Dawn (Next)', line=dict(color='orange')))



fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Dusk']/np.timedelta64(1, 'h'),

                         name='Dusk', fill='tonexty', line=dict(color='deepskyblue')))



fig.update_layout(xaxis=dict(title_text='Dates when Full Moon Instance occured'),

                  yaxis=dict(tick0=0, dtick=4, title_text='Timing (in Hours)'),

                  title_text='Occurance of FullMoon Instance in two Nights',

                  xaxis_rangeslider_visible=True)

fig.show()
def get_last_night(row):

    night_start = ldata.loc[row.name-1, 'Date'] + ldata.loc[row.name-1, 'Dusk']  # Previous Dusk

    night_end = row['Date'] + row['Dawn']  # Dawn

    return (night_start, night_end)



def get_coming_night(row):

    night_start = row['Date'] + row['Dusk']  # Dusk

    night_end = ldata.loc[row.name+1, 'Date'] + ldata.loc[row.name+1, 'Dawn']  # Next Dawn

    return (night_start, night_end)



def get_full_moon_nights(row):

    if row['PhaseTime'] < row['Dawn']:

        return get_last_night(row)

        

    elif row['Dusk'] < row['PhaseTime']:

        return get_coming_night(row)

        

    else:

        mid = (row['Dawn']+row['Dusk'])/2

        if row['PhaseTime'] < mid:

            return get_last_night(row)

        else:

            return get_coming_night(row)
# Full Moon (fm) Nights

fm_nights = pd.DataFrame(list(fm_ldata.apply(get_full_moon_nights, axis=1)), columns=['NightStart', 'NightEnd'])

fm_nights
all_ldata = lunar_data[['Date', 'CivilDusk', 'CivilDawn']].rename(columns={'CivilDusk': 'Dusk', 'CivilDawn': 'Dawn'})

all_ldata['Date'] = pd.to_datetime(all_ldata['Date'], dayfirst=True)

all_ldata['Dusk'] = pd.to_timedelta(all_ldata['Dusk']+':00')

all_ldata['Dawn'] = pd.to_timedelta(all_ldata['Dawn']+':00')



def get_all_nights(row):

    night_start = row['Date'] + row['Dusk']

    night_end = all_ldata.loc[row.name+1, 'Date'] + all_ldata.loc[row.name+1, 'Dawn']  # Using next date's Dawn

    return (night_start, night_end)



all_nights = pd.DataFrame(list(all_ldata.iloc[:-1].apply(get_all_nights, axis=1)), columns=['NightStart', 'NightEnd'])

all_nights
all_nights['FullMoon'] = 0

all_nights.loc[all_nights['NightStart'].isin(fm_nights['NightStart']), 'FullMoon'] = 1

all_nights
all_nights[all_nights['FullMoon']==1]
# Interval Index formed from all_nights 

all_nights_idx = pd.IntervalIndex.from_arrays(all_nights['NightStart'],all_nights['NightEnd'],closed='both')



# Creating a mapping of adata indices with all_nights_idx, where adata['DateTime'] falls in all_nights_idx

adata_nights_idx_map = pd.Series(all_nights_idx.get_indexer(adata['DateTime']), index=adata.index)

adata_nights_idx_map
# Dropping rows where mapped value is -1 since those accidents didn't happen in night

adata_nights_idx_map = adata_nights_idx_map[adata_nights_idx_map != -1]



# Selecting all rows from adata where accidents happened in night (non -1)

adata = adata.loc[adata_nights_idx_map.index]

adata
# Finding all night periods corresonding to accidents that happned in night

nights_in_adata = all_nights.loc[adata_nights_idx_map]

nights_in_adata.index = adata_nights_idx_map.index

nights_in_adata
# Combining night periods data in adata

adata = pd.concat([adata, nights_in_adata], axis=1)

adata
# Night periods as Intervals

adata_night_idx = pd.IntervalIndex.from_arrays(adata['NightStart'],adata['NightEnd'],closed='both')
# Summarized adata

summ_adata = adata.groupby(adata_night_idx).agg(Accidents_Count=('DateTime', 'size'),

                                   Total_Casualities=('Number_of_Casualties', 'sum'),

                                   Full_Moon=('FullMoon', 'max'))  # FullMoon value is same for each night group

summ_adata
# To get adata for Full Moon Nights

summ_adata[summ_adata['Full_Moon']==1]
def plot_summ_adata(column_name):

    fig = go.Figure()



    fig.add_trace(go.Scatter(x=summ_adata.index.left.date, y=summ_adata[column_name],

                             name='Ordinary Night', line=dict(color='mediumpurple')))



    fm_summ_adata = summ_adata[summ_adata['Full_Moon']==1]

    fig.add_trace(go.Scatter(x=fm_summ_adata.index.left.date, y=fm_summ_adata[column_name],

                             name='Full Moon Night', mode='markers', line=dict(color='darkorange')))



    fig.update_layout(xaxis=dict(title_text='Date of Night\'s beginning',

                                 rangeselector=dict(buttons=list([

                                  dict(count=1, label="1m", step="month", stepmode="backward"),

                                  dict(count=6, label="6m", step="month", stepmode="backward"),

                                  dict(count=1, label="1y", step="year", stepmode="backward"),

                                  dict(count=4, label="4y", step="year", stepmode="backward"),

                                  dict(step="all")])),

                                 rangeslider=dict(visible=True,

                                                  range=['2005-01-01', '2017-12-31']),  # Span rangeslider for Date range of entire adata

                                 range=['2017-07-01', '2017-12-31'],  # Show on x-axis only date Range of last half year in adata

                                 type='date'

                                ),

                      yaxis=dict(title_text=column_name),

                      title_text='{} per Night'.format(column_name))

    fig.show()
plot_summ_adata('Accidents_Count')
plot_summ_adata('Total_Casualities')
# To test using scipy ttest_ind

from scipy import stats



# Full Moon nights

fm_summ_adata = summ_adata[summ_adata['Full_Moon']==1]



# Ordinary Nights

non_fm_summ_adata = summ_adata[summ_adata['Full_Moon']==0]
stats.ttest_ind(fm_summ_adata['Accidents_Count'], non_fm_summ_adata['Accidents_Count'])
stats.ttest_ind(fm_summ_adata['Total_Casualities'], non_fm_summ_adata['Total_Casualities'])
fig = go.Figure(data=[

    go.Bar(name='Ordinary Night', x=summ_adata.columns[:-1], y=non_fm_summ_adata.mean().iloc[:-1], marker_color='purple'),

    go.Bar(name='Full Moon Night', x=summ_adata.columns[:-1], y=fm_summ_adata.mean().iloc[:-1], marker_color='orange')

])



fig.update_layout(

    title='Comparison of Averages',

    yaxis=dict(title='Mean'),

    bargap=0.3,

    bargroupgap=0.05

)

fig.show()