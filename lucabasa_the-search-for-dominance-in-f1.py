import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

pd.set_option('max_columns', 100)
def import_all():

    data = {}

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            name = filename.replace('.csv', '')

            data[name] = pd.read_csv(os.path.join(dirname, filename))

            

    return data





def add_ids(data, key):

    

    df = data[key]

    n_lines = df.shape[0]



    df = pd.merge(df, data['races'][['raceId', 

                                     'year', 'round', 

                                     'circuitId', 'date', 'time']], 

                  on='raceId', how='left')

    if df.shape[0] != n_lines:

        raise ValueError('Merging raceId went wrong')

        

    df = pd.merge(df, data['circuits'][['circuitId', 

                                        'circuitRef', 'location', 'country']], 

                  on='circuitId', how='left')

    if df.shape[0] != n_lines:

        raise ValueError('Merging circuitId went wrong')

        

    df = pd.merge(df, data['drivers'][['driverId', 

                                       'driverRef', 'forename', 'surname', 

                                       'dob', 'nationality']].rename(columns={'nationality': 'drv_nat'}), 

                  on='driverId', how='left')

    if df.shape[0] != n_lines:

        raise ValueError('Merging driverId went wrong')

    

    if (key != 'lap_times') and (key != 'pit_stops'):

        df = pd.merge(df, data['constructors'][['constructorId', 

                                                'constructorRef', 

                                                'name', 'nationality']].rename(columns={'nationality': 'cstr_nat'}), 

                      on='constructorId', how='left')

        if df.shape[0] != n_lines:

            raise ValueError('Merging constructorId went wrong')

        

    if key == 'results':

        df = pd.merge(df, data['status'], 

                      on='statusId', how='left')

        if df.shape[0] != n_lines:

            raise ValueError('Merging statusId went wrong')

        

    return df
data = import_all()



res = add_ids(data, 'results')

qual = add_ids(data, 'qualifying')

laps = add_ids(data, 'lap_times')

pits = add_ids(data, 'pit_stops')



laps.rename(columns={'time_x': 'lap_time', 'time_y': 'time'}, inplace=True)

res.rename(columns={'time_x': 'race_time', 'time_y': 'time'}, inplace=True)

pits.rename(columns={'time_x': 'pit_time', 'time_y': 'time'}, inplace=True)



laps = pd.merge(laps, res[['raceId', 'driverId', 

                           'constructorRef', 'name', 'cstr_nat']], 

                on=['raceId', 'driverId'], how='left')

pits = pd.merge(pits, res[['raceId', 'driverId', 

                           'constructorRef', 'name', 'cstr_nat']], 

                on=['raceId', 'driverId'], how='left')
res[['lap_mins', 'lap_secs']] = res['fastestLapTime'].str.split(':', expand=True)

res[['lap_secs', 'lap_millisecs']] = res['lap_secs'].str.split('.', expand=True)

res['lap_mins'] = pd.to_numeric(res['lap_mins'], errors='coerce').fillna(99)

res['lap_secs'] = pd.to_numeric(res['lap_secs'], errors='coerce').fillna(99)

res['lap_millisecs'] = pd.to_numeric(res['lap_millisecs'], errors='coerce').fillna(99)



res['fastestLapTime_ms'] = (60 * res['lap_mins'] + res['lap_secs']) * 1000 + res['lap_millisecs']



res['race_fastestTime'] = res.groupby('raceId').fastestLapTime_ms.transform('min')

res['FastLap'] = np.where(res['race_fastestTime'] == res['fastestLapTime_ms'], 1, 0)



res.drop(['lap_mins', 'lap_secs', 'lap_millisecs'], axis=1, inplace=True)



points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}



res['points'] = res['positionOrder'].map(points).fillna(0)

#res.loc[res.FastLap == 1, 'points'] = res['points'] + 1



res['fastestLap'] = pd.to_numeric(res['fastestLap'], errors='coerce')



res['DriverName'] = res['forename'].str[0] + '. ' + res['surname']





res['net_gain'] = -(res['positionOrder'] - res['grid'])

res['abs_gain'] = abs(res['net_gain'])



res['finished'] = np.where(res.status == 'Finished', 1, 0)
def plot_frame(ax):

    ax.set_facecolor('#292525')

    ax.spines['bottom'].set_color('w')

    ax.tick_params(axis='x', colors='w')

    ax.xaxis.label.set_color('w')

    ax.spines['left'].set_color('w')

    ax.tick_params(axis='y', colors='w')

    ax.yaxis.label.set_color('w')

    return ax



def get_drv_ann(data, year, ax, adjust, count=False, measure='Pts.'):

    

    yr_data = data[data.year==year].groupby(['driverId', 'DriverName', 'name']).points.sum().sort_values(ascending=False)

    if count:

        yr_data = data[data.year==year].groupby(['driverId', 'DriverName', 'name']).resultId.count().sort_values(ascending=False)

    drv_name = yr_data.index[0][1]

    ctr_name = yr_data.index[0][2]

    pts = yr_data[0]

    value = pts / data[data.year==year].raceId.nunique()

    

    text = f'{drv_name}\n{ctr_name}, {year}\n{int(pts)} {measure}'

    

    ax.annotate(text, xy=(year, value), xycoords='data', xytext=adjust, textcoords='offset points', color='w')

    

    return ax





def get_ctr_ann(data, year, ax, adjust, count=False, measure='Pts.'):

    

    yr_data = data[data.year==year].groupby(['name']).points.sum().sort_values(ascending=False)

    if count:

        yr_data = data[data.year==year].groupby(['name']).resultId.count().sort_values(ascending=False)

    ctr_name = yr_data.index[0]

    pts = yr_data[0]

    value = pts / data[data.year==year].raceId.nunique()

    

    text = f'{ctr_name}, {year}\n{int(pts)} {measure}'

    

    ax.annotate(text, xy=(year, value), xycoords='data', xytext=adjust, textcoords='offset points', color='w')

    

    return ax





def plot_bars(bars, ax, color):

    

    colors = [color if (c == 2020) else 'w' for c in bars.index]

    bars.plot(color=colors, kind='bar', ax=ax)

    ax.set_title('Top Years vs 2020', fontsize=14, color='w')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    

    return ax
fig = plt.figure(figsize=(15, 25), facecolor='#292525')

fig.subplots_adjust(top=0.95)

fig.suptitle('Best Driver of the Season', fontsize=18, color='w')



gs = GridSpec(5, 3, figure=fig)

ax0 = fig.add_subplot(gs[0, :2])

ax1 = fig.add_subplot(gs[0, 2])

ax2 = fig.add_subplot(gs[1, 0])

ax3 = fig.add_subplot(gs[1, 1:])

ax4 = fig.add_subplot(gs[2, :2])

ax5 = fig.add_subplot(gs[2, 2])

ax6 = fig.add_subplot(gs[3, 0])

ax7 = fig.add_subplot(gs[3, 1:])

ax8 = fig.add_subplot(gs[4, :2])

ax9 = fig.add_subplot(gs[4, 2])



race_counts = res.groupby(['year']).raceId.nunique()



(res.groupby(['year', 'driverId']).points.sum().groupby('year').max() / race_counts).plot(ax=ax0, color='#15E498')

ax0.set_title('Most Points', fontsize=14, color='w')

ax0.set_ylabel('Points per GP', fontsize=12)

ax0 = get_drv_ann(res, 2002, ax0, (-80, -25))

ax0 = get_drv_ann(res, 1963, ax0, (-50, -2))

ax0 = get_drv_ann(res, 1970, ax0, (-90, 2))

ax0 = get_drv_ann(res, 1982, ax0, (6, 2))



years = (res.groupby(['year', 'driverId']).points.sum().groupby('year').max() / race_counts).sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[res.year.isin(years)].groupby(['year', 'DriverName']).points.sum().groupby(['year']).max() / race_counts).dropna()

ax1 = plot_bars(bars, ax1, '#15E498')



(res[res.positionOrder == 1].groupby(['year', 'driverId']).resultId.count().groupby('year').max() / race_counts).plot(ax=ax3, color='#C3C92E')

ax3.set_title('Most Wins', fontsize=14, color='w')

ax2.set_ylabel('Wins per GP', fontsize=12)

ax3 = get_drv_ann(res[res.positionOrder == 1], 1952, ax3, (5, -22), count=True, measure='Wins')

ax3 = get_drv_ann(res[res.positionOrder == 1], 2004, ax3, (-80, -25), count=True, measure='Wins')

ax3 = get_drv_ann(res[res.positionOrder == 1], 1982, ax3, (10, 0), count=True, measure='Wins')

ax3 = get_drv_ann(res[res.positionOrder == 1], 2012, ax3, (-20, -40), count=True, measure='Wins')



years = (res[res.positionOrder == 1].groupby(['year', 'driverId']).resultId.count().groupby('year').max() / race_counts).sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[(res.positionOrder == 1) & (res.year.isin(years))].groupby(['year', 'DriverName']).resultId.count().groupby(['year']).max() / race_counts).dropna()

ax2 = plot_bars(bars, ax2, '#C3C92E')



(res[res.positionOrder <= 3].groupby(['year', 'driverId']).resultId.count().groupby('year').max() / race_counts).plot(ax=ax4, color='#C93D2E')

ax4.set_title('Most Podiums', fontsize=14, color='w')

ax4.set_ylabel('Podiums per GP', fontsize=12)

ax4 = get_drv_ann(res[res.positionOrder <= 3], 2002, ax4, (-80, -25), count=True, measure='Podiums')

ax4 = get_drv_ann(res[res.positionOrder <= 3], 1963, ax4, (5, 0), count=True, measure='Podiums')

ax4 = get_drv_ann(res[res.positionOrder <= 3], 1982, ax4, (10, 0), count=True, measure='Podiums')

ax4 = get_drv_ann(res[res.positionOrder <= 3], 1970, ax4, (-100, 0), count=True, measure='Podiums')



years = (res[res.positionOrder <= 3].groupby(['year', 'driverId']).resultId.count().groupby('year').max() / race_counts).sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[(res.positionOrder <= 3) & (res.year.isin(years))].groupby(['year', 'DriverName']).resultId.count().groupby(['year']).max() / race_counts).dropna()

ax5 = plot_bars(bars, ax5, '#C93D2E')



(res[res.grid == 1].groupby(['year', 'driverId']).resultId.count().groupby('year').max() / race_counts).plot(ax=ax7, color='#3A3FDC')

ax7.set_title('Most Pole Positions', fontsize=14, color='w')

ax6.set_ylabel('Poles per GP', fontsize=12)

ax7 = get_drv_ann(res[res.grid == 1], 2011, ax7, (10, -10), count=True, measure='Poles')

ax7 = get_drv_ann(res[res.grid == 1], 2009, ax7, (10, -10), count=True, measure='Poles')

ax7 = get_drv_ann(res[res.grid == 1], 1992, ax7, (20, -23), count=True, measure='Poles')

ax7 = get_drv_ann(res[res.grid == 1], 1980, ax7, (25, -5), count=True, measure='Poles')



years = (res[res.grid == 1].groupby(['year', 'driverId']).resultId.count().groupby('year').max() / race_counts).sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[(res.grid == 1) & (res.year.isin(years))].groupby(['year', 'DriverName']).resultId.count().groupby(['year']).max() / race_counts).dropna()

ax6 = plot_bars(bars, ax6, '#3A3FDC')



(res[res['rank'] == '1'].groupby(['year', 'driverId']).resultId.count().groupby('year').max() / race_counts).plot(ax=ax8, color='#41DA5B')

ax8.set_title('Most Fast Laps', fontsize=14, color='w')

ax8.set_ylabel('Fast Laps per GP', fontsize=12)

ax8 = get_drv_ann(res[res['rank'] == '1'], 2009, ax8, (-80, 0), count=True, measure='FL')

ax8 = get_drv_ann(res[res['rank'] == '1'], 2008, ax8, (10, -30), count=True, measure='FL')

ax8 = get_drv_ann(res[res['rank'] == '1'], 2015, ax8, (-20, 0), count=True, measure='FL')



years = (res[res['rank'] == '1'].groupby(['year', 'driverId']).resultId.count().groupby('year').max() / race_counts).dropna().sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[(res['rank'] == '1') & (res.year.isin(years))].groupby(['year', 'DriverName']).resultId.count().groupby(['year']).max() / race_counts).dropna()

ax9 = plot_bars(bars, ax9, '#41DA5B')



for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:

    ax = plot_frame(ax)

    ax.set_xlabel('')



plt.show()
fig = plt.figure(figsize=(15, 25), facecolor='#292525')

fig.subplots_adjust(top=0.95)

fig.suptitle('Best Constructor of the Season', fontsize=18, color='w')



gs = GridSpec(5, 3, figure=fig)

ax0 = fig.add_subplot(gs[0, :2])

ax1 = fig.add_subplot(gs[0, 2])

ax2 = fig.add_subplot(gs[1, 0])

ax3 = fig.add_subplot(gs[1, 1:])

ax4 = fig.add_subplot(gs[2, :2])

ax5 = fig.add_subplot(gs[2, 2])

ax6 = fig.add_subplot(gs[3, 0])

ax7 = fig.add_subplot(gs[3, 1:])

ax8 = fig.add_subplot(gs[4, :2])

ax9 = fig.add_subplot(gs[4, 2])



race_counts = res.groupby(['year']).raceId.nunique()



(res.groupby(['year', 'constructorId']).points.sum().groupby('year').max() / race_counts).plot(ax=ax0, color='#15E498')

ax0.set_title('Most Points', fontsize=14, color='w')

ax0.set_ylabel('Points per GP', fontsize=12)

ax0 = get_ctr_ann(res, 1952, ax0, (10, -10))

ax0 = get_ctr_ann(res, 1982, ax0, (10, -10))

ax0 = get_ctr_ann(res, 2015, ax0, (-30, 10))

ax0 = get_ctr_ann(res, 1988, ax0, (-30, 10))

ax0 = get_ctr_ann(res, 2012, ax0, (-30, -20))



years = (res.groupby(['year', 'constructorId']).points.sum().groupby('year').max() / race_counts).sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[res.year.isin(years)].groupby(['year', 'name']).points.sum().groupby(['year']).max() / race_counts).dropna()

ax1 = plot_bars(bars, ax1, '#15E498')



(res[res.positionOrder == 1].groupby(['year', 'constructorId']).resultId.count().groupby('year').max() / race_counts).plot(ax=ax3, color='#C3C92E')

ax3.set_title('Most Wins', fontsize=14, color='w')

ax2.set_ylabel('Wins per GP', fontsize=12)

ax3 = get_ctr_ann(res[res.positionOrder == 1], 1988, ax3, (-50,-10), count=True, measure='Wins')

ax3 = get_ctr_ann(res[res.positionOrder == 1], 2016, ax3, (-60,0), count=True, measure='Wins')

ax3 = get_ctr_ann(res[res.positionOrder == 1], 1982, ax3, (15,0), count=True, measure='Wins')

ax3 = get_ctr_ann(res[res.positionOrder == 1], 2012, ax3, (-50,-30), count=True, measure='Wins')



years = (res[res.positionOrder == 1].groupby(['year', 'constructorId']).resultId.count().groupby('year').max() / race_counts).sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[(res.positionOrder == 1) & (res.year.isin(years))].groupby(['year', 'name']).resultId.count().groupby(['year']).max() / race_counts).dropna()

ax2 = plot_bars(bars, ax2, '#C3C92E')



(res[res.positionOrder <= 3].groupby(['year', 'constructorId']).resultId.count().groupby('year').max() / race_counts).plot(ax=ax4, color='#C93D2E')

ax4.set_title('Most Podiums', fontsize=14, color='w')

ax4.set_ylabel('Podiums per GP', fontsize=12)

ax4 = get_ctr_ann(res[res.positionOrder <= 3], 1952, ax4, (10,-10), count=True, measure='Podiums')

ax4 = get_ctr_ann(res[res.positionOrder <= 3], 1961, ax4, (10,0), count=True, measure='Podiums')

ax4 = get_ctr_ann(res[res.positionOrder <= 3], 1982, ax4, (30,0), count=True, measure='Podiums')

ax4 = get_ctr_ann(res[res.positionOrder <= 3], 2012, ax4, (10,-15), count=True, measure='Podiums')

ax4 = get_ctr_ann(res[res.positionOrder <= 3], 2004, ax4, (-30,5), count=True, measure='Podiums')

ax4 = get_ctr_ann(res[res.positionOrder <= 3], 2015, ax4, (-20,10), count=True, measure='Podiums')



years = (res[res.positionOrder <= 3].groupby(['year', 'constructorId']).resultId.count().groupby('year').max() / race_counts).sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[(res.positionOrder <= 3) & (res.year.isin(years))].groupby(['year', 'name']).resultId.count().groupby(['year']).max() / race_counts).dropna()

ax5 = plot_bars(bars, ax5, '#C93D2E')



(res[res.grid == 1].groupby(['year', 'constructorId']).resultId.count().groupby('year').max() / race_counts).plot(ax=ax7, color='#3A3FDC')

ax7.set_title('Most Pole Positions', fontsize=14, color='w')

ax6.set_ylabel('Poles per GP', fontsize=12)

ax7 = get_ctr_ann(res[res.grid == 1], 2009, ax7, (10,0), count=True, measure='Poles')

ax7 = get_ctr_ann(res[res.grid == 1], 1972, ax7, (10,-10), count=True, measure='Poles')

ax7 = get_ctr_ann(res[res.grid == 1], 1956, ax7, (5,-30), count=True, measure='Poles')

ax7 = get_ctr_ann(res[res.grid == 1], 2016, ax7, (-80,0), count=True, measure='Poles')



years = (res[res.grid == 1].groupby(['year', 'constructorId']).resultId.count().groupby('year').max() / race_counts).sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[(res.grid == 1) & (res.year.isin(years))].groupby(['year', 'name']).resultId.count().groupby(['year']).max() / race_counts).dropna()

ax6 = plot_bars(bars, ax6, '#3A3FDC')



(res[res['rank'] == '1'].groupby(['year', 'constructorId']).resultId.count().groupby('year').max() / race_counts).plot(ax=ax8, color='#41DA5B')

ax8.set_title('Most Fast Laps', fontsize=14, color='w')

ax8.set_ylabel('Fast Laps per GP', fontsize=12)

ax8 = get_ctr_ann(res[res['rank'] == '1'], 2010, ax8, (10,-10), count=True, measure='FL')

ax8 = get_ctr_ann(res[res['rank'] == '1'], 2012, ax8, (20,0), count=True, measure='FL')

ax8 = get_ctr_ann(res[res['rank'] == '1'], 2004, ax8, (-10,5), count=True, measure='FL')

ax8 = get_ctr_ann(res[res['rank'] == '1'], 2008, ax8, (-10,5), count=True, measure='FL')

ax8 = get_ctr_ann(res[res['rank'] == '1'], 2015, ax8, (-20,5), count=True, measure='FL')



years = (res[res['rank'] == '1'].groupby(['year', 'constructorId']).resultId.count().groupby('year').max() / race_counts).dropna().sort_values()[-6:].index.to_list()

if 2020 not in years:

    years += [2020]

bars = (res[(res['rank'] == '1') & (res.year.isin(years))].groupby(['year', 'name']).resultId.count().groupby(['year']).max() / race_counts).dropna()

ax9 = plot_bars(bars, ax9, '#41DA5B')



for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:

    ax = plot_frame(ax)

    ax.set_xlabel('')



plt.show()
def annotate_season(data, ax, tp='Driver'):

    

    text = f'Top {tp} \n'

    for name, points in data.iteritems():

        text += f'{name} {(int(points))}\n'

    

    ax.text(0.05, 0.65, text, transform=ax.transAxes, color='w', fontsize=13)

    

    return ax





def plot_season(res, year, n_drivers, n_ctr, drv_colors, ctr_colors):

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), facecolor='#292525')

    fig.subplots_adjust(top=0.92)

    fig.suptitle(f'{year} Season', fontsize=18, color='w')

    

    drivers = res[res.year==year].groupby('DriverName').points.sum().sort_values()[-n_drivers:].index



    tmp = res[(res.year==year)].sort_values(by=['round'])

    

    to_print = tmp.groupby('DriverName').points.sum().sort_values(ascending=False)[:n_drivers]



    tmp['tot_points'] = tmp.groupby('driverId').points.cumsum()

    tmp = tmp.set_index('raceId')

    i = 0

    for driver in tmp.DriverName.unique():

        if driver in drivers:

            color = drv_colors[i]

            i += 1

        else:

            color = '#7C7373'

        tmp[tmp.DriverName == driver].tot_points.plot(ax=ax[0], color=color)

        

    ax[0] = annotate_season(to_print, ax[0])

           

    constr = res[res.year==year].groupby('name').points.sum().sort_values()[-n_ctr:].index



    tmp = res[(res.year==year)].groupby(['raceId', 'constructorId', 'name'], as_index=False).points.sum().sort_values(by=['raceId'])

    

    to_print = tmp.groupby('name').points.sum().sort_values(ascending=False)[:n_ctr]



    tmp['tot_points'] = tmp.groupby('constructorId').points.cumsum()

    tmp = tmp.set_index('raceId')

    i = 0

    for ctr in tmp.name.unique():

        if ctr in constr:

            color = ctr_colors[i]

            i += 1

        else:

            color = '#7C7373'

        tmp[tmp.name == ctr].tot_points.plot(ax=ax[1], color=color)

        

    ax[1] = annotate_season(to_print, ax[1], tp='Constructor')

        

    ax[0].set_title('Driver Title', fontsize=14, color='w')

    ax[1].set_title('Constructor Title', fontsize=14, color='w')

    

    for axes in ax:

        axes = plot_frame(axes)

        axes.set_xlabel('')

        axes.set_xticks([])

        

    plt.show()

    

    

clrs = {'Ferrari': '#ff2800', 

        'Red Bull': 'b', 

        'McLaren': '#f98e1d', 

        'Williams': 'w', 

        'Mercedes': '#00D2BE', 

        'Brawn': '#B8FD6E', 

        'Lotus': '#FFB800', 

        'BRM': 'g'}
plot_season(res, 1963, n_drivers=3, n_ctr=2, 

            drv_colors=[clrs['BRM'], clrs['BRM'], clrs['Lotus']], 

            ctr_colors=[clrs['BRM'], clrs['Lotus']])
plot_season(res, 1982, n_drivers=3, n_ctr=3, 

            drv_colors=[clrs['Ferrari'], clrs['Williams'], clrs['McLaren']], 

            ctr_colors=[clrs['McLaren'], clrs['Ferrari'], clrs['Williams']])
plot_season(res, 1988, n_drivers=2, n_ctr=1, 

            drv_colors=[clrs['McLaren'], clrs['McLaren']], 

            ctr_colors=[clrs['McLaren']])
plot_season(res, 2002, n_drivers=2, n_ctr=1, 

            drv_colors=[clrs['Ferrari'], clrs['Ferrari']], 

            ctr_colors=[clrs['Ferrari']])
plot_season(res, 2009, n_drivers=4, n_ctr=2, 

            drv_colors=[clrs['Brawn'], clrs['Red Bull'], clrs['Red Bull'], clrs['Brawn']], 

            ctr_colors=[clrs['Red Bull'], clrs['Brawn']])
plot_season(res, 2012, n_drivers=2, n_ctr=3, 

            drv_colors=[clrs['Ferrari'], clrs['Red Bull']], 

            ctr_colors=[clrs['McLaren'], clrs['Red Bull'], clrs['Ferrari']])
plot_season(res, 2013, n_drivers=2, n_ctr=3, 

            drv_colors=[clrs['Red Bull'], clrs['Ferrari']], 

            ctr_colors=[clrs['Ferrari'], clrs['Red Bull'], clrs['Mercedes']])
plot_season(res, 2015, n_drivers=3, n_ctr=2, 

            drv_colors=[clrs['Mercedes'], clrs['Ferrari'], clrs['Mercedes']], 

            ctr_colors=[clrs['Ferrari'], clrs['Mercedes']])
plot_season(res, 2020, n_drivers=3, n_ctr=2, 

            drv_colors=[clrs['Mercedes'], clrs['Red Bull'], clrs['Mercedes']], 

            ctr_colors=[clrs['Red Bull'], clrs['Mercedes']])
fig, ax = plt.subplots(1, 1, figsize=(15, 7), facecolor='#292525')

fig.suptitle(f'Position changes per Grand Prix', fontsize=18, color='w')



(res.groupby(['year', 'circuitRef']).abs_gain.sum() / res.groupby(['year', 'circuitRef']).size()).groupby('year').mean().plot(label='Mean', color='w')

(res.groupby(['year', 'circuitRef']).abs_gain.sum() / res.groupby(['year', 'circuitRef']).size()).groupby('year').min().plot(label='Min', color='g')

(res.groupby(['year', 'circuitRef']).abs_gain.sum() / res.groupby(['year', 'circuitRef']).size()).groupby('year').max().plot(label='Max', color='r')



leg = ax.legend(facecolor="#292525")

for text in leg.get_texts():

    text.set_color("w")



ax = plot_frame(ax)
fig, ax = plt.subplots(1, 1, figsize=(15, 7), facecolor='#292525')

fig.suptitle(f'Proportion of Drivers that finished the race per Grand Prix', fontsize=18, color='w')



res.groupby(['year', 'circuitRef']).finished.mean().groupby('year').mean().plot(color='w', label='Mean')

res.groupby(['year', 'circuitRef']).finished.mean().groupby('year').max().plot(color='r', label='Max')

res.groupby(['year', 'circuitRef']).finished.mean().groupby('year').min().plot(color='g', label='Min')



leg = ax.legend(facecolor="#292525")

for text in leg.get_texts():

    text.set_color("w")



ax = plot_frame(ax)
laps = laps.sort_values(by=['raceId', 'driverId', 'lap'])

laps['pos_change'] = -laps.groupby(['raceId', 'driverId']).position.diff().fillna(0)

laps['abs_change'] = abs(laps['pos_change'])
fig, ax = plt.subplots(1, 1, figsize=(15, 7), facecolor='#292525')

fig.suptitle(f'Proportion of Laps with at least a position change', fontsize=18, color='w')



tmp = laps.groupby(['year', 'raceId', 'circuitRef', 'lap'], as_index=False).abs_change.sum()

tmp['lap_with_change'] = np.sign(tmp.abs_change)



tmp.groupby(['year', 'raceId', 'circuitRef']).lap_with_change.mean().groupby('year').agg(['mean', 'max', 'min']).plot(ax=ax, color=['w', 'r', 'g'])



leg = ax.legend(facecolor="#292525")

leg.get_texts()[0].set_text('Mean')

leg.get_texts()[1].set_text('Max')

leg.get_texts()[2].set_text('Min')

for text in leg.get_texts():

    text.set_color("w")

    

text = f'Indianapolis \n2005 \n8% of laps'

ax.annotate(text, xy=(2005, 0.08), xycoords='data', xytext=(20,-20), textcoords='offset points', color='w')

text = f'Bahrain \n2013 \n91% of laps'

ax.annotate(text, xy=(2013, 0.91), xycoords='data', xytext=(10,-3), textcoords='offset points', color='w')



ax.set_ylim((0,1))    

ax = plot_frame(ax)
fig, ax = plt.subplots(1, 1, figsize=(15, 7), facecolor='#292525')

fig.suptitle(f'Proportion of Laps with at least a position change in the top 3', fontsize=18, color='w')



tmp = laps[laps.position <= 3].groupby(['year', 'raceId', 'circuitRef', 'lap'], as_index=False).abs_change.sum()

tmp['lap_with_change'] = np.sign(tmp.abs_change)



tmp.groupby(['year', 'raceId', 'circuitRef']).lap_with_change.mean().groupby('year').agg(['mean', 'max', 'min']).plot(ax=ax, color=['w', 'r', 'g'])



leg = ax.legend(facecolor="#292525")

leg.get_texts()[0].set_text('Mean')

leg.get_texts()[1].set_text('Max')

leg.get_texts()[2].set_text('Min')

for text in leg.get_texts():

    text.set_color("w")

    

text = f'Belgium \n2011 \n43% of laps'

ax.annotate(text, xy=(2011, 0.43), xycoords='data', xytext=(-30,20), textcoords='offset points', color='w')





ax.set_ylim((0,1))    

ax = plot_frame(ax)
laps['seconds'] = laps['milliseconds'] / 1000

laps['DriverName'] = laps['forename'].str[0] + '. ' + laps['surname']



clrs = {'Ferrari': '#ff2800', 

        'Red Bull': 'b', 

        'McLaren': '#f98e1d', 

        'Williams': 'w', 

        'Mercedes': '#00D2BE', 

        'Brawn': '#B8FD6E', 

        'Lotus': '#FFB800', 

        'BRM': 'g', 

        'Jordan': '#F9D71C'}



def plot_race(data, country, year, colors=None, raceid=None):

    fig, ax = plt.subplots(1, 1, figsize=(13, 10), facecolor='#292525')

    fig.subplots_adjust(top=0.94)

    if raceid is None:

        tmp = data[(data.country == country) & (data.year == year)]

    else:

        tmp = data[data.raceId==raceid]

        

    fig.suptitle(f'Grand Prix of {country}, {year}', fontsize=18, color='w')

        

    last_lap = tmp.groupby('DriverName', as_index=False).lap.max()

    order = pd.merge(tmp, last_lap, on=['DriverName', 'lap']).sort_values(by='position').DriverName.values

    

    tmp = tmp[['DriverName', 'lap', 'seconds']].copy()



    tmp = tmp.set_index(['DriverName', 'lap']).unstack().cumsum(axis=1).reset_index()

    tmp = tmp.set_index('DriverName').reindex(order)

    tmp.columns = tmp.columns.get_level_values(1)

    basis = tmp.median()

    

    if colors is None:

        (- (tmp - basis)).T.plot(ax=ax)

    else:

        cols = []

        for driver in tmp.index:

            if driver in colors.keys():

                cols.append(colors[driver])

            else:

                cols.append('#7C7373')

        (- (tmp - basis)).T.plot(ax=ax, color=cols)

    

    ax.set_xlim((tmp.columns[0], tmp.columns[-1]))

    ax.set_xlabel('Lap', fontsize=14)

    ax.set_ylabel('Delta time (seconds)', fontsize=14)

    

    leg = ax.legend(facecolor="#292525", loc='center left', bbox_to_anchor=(1, 0.5))

    for text in leg.get_texts():

        text.set_color("w")

    ax = plot_frame(ax)

    plt.show()
plot_race(laps, 'Belgium', 2011, colors={'S. Vettel': clrs['Red Bull'], 

                                         'J. Button': clrs['McLaren'], 

                                         'M. Webber': clrs['Red Bull'], 

                                         'F. Alonso': clrs['Ferrari']})
plot_race(laps, 'Belgium', 1998, colors={'M. Schumacher': clrs['Ferrari'], 

                                         'D. Coulthard': clrs['McLaren'], 

                                         'D. Hill': clrs['Jordan']})
plot_race(laps, 'Bahrain', 2013)
plot_race(laps, 'Bahrain', 2014, {'L. Hamilton': clrs['Mercedes'], 

                                  'N. Rosberg': clrs['Mercedes']})
plot_race(laps, 'Germany', 2019, {'M. Verstappen': clrs['Red Bull'], 

                                  'S. Vettel': clrs['Ferrari'], 

                                  'L. Hamilton': clrs['Mercedes']})
plot_race(laps, 'Italy', 2020, raceid=1038, colors={'P. Gasly': clrs['Red Bull'], 

                                                  'C. Sainz': clrs['McLaren'], 

                                                  'L. Hamilton': clrs['Mercedes']})
plot_race(laps, 'USA', 2005, {'M. Schumacher': clrs['Ferrari']})
plot_race(laps, 'Singapore', 2015)
plot_race(laps, 'Belgium', 2020, raceid=None)