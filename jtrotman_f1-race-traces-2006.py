YEAR = 2006

DRIVER_LS = {2:0,3:1,4:0,8:0,9:2,11:0,13:1,14:0,15:1,17:0,18:0,21:1,22:1,23:0,24:0,26:1,27:0,29:1,30:0,31:1,32:1,33:1,34:2,35:1,36:3,37:2,38:2}

DRIVER_C = {2:"#BABABA",3:"#007FFE",4:"#B4B400",8:"#7F7F7F",9:"#BABABA",11:"#580000",13:"#FF0000",14:"#0000B0",15:"#E300E3",17:"#007FFE",18:"#00B000",21:"#B4B400",22:"#00B000",23:"#E300E3",24:"#7B68EE",26:"#7B68EE",27:"#000000",29:"#580000",30:"#FF0000",31:"#7F7F7F",32:"#0000B0",33:"#000000",34:"#580000",35:"#BABABA",36:"#580000",37:"#7F7F7F",38:"#0000B0"}

TEAM_C = {1:"#7F7F7F",2:"#BABABA",3:"#007FFE",4:"#B4B400",5:"#7B68EE",6:"#FF0000",7:"#E300E3",8:"#580000",9:"#0000B0",11:"#00B000",13:"#000000",14:"#000000"}

LINESTYLES = ['-', '-.', '--', ':', '-', '-']
import os, sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import HTML, display

import urllib



def read_csv(name, **kwargs):

    df = pd.read_csv(f'../input/formula-1-race-data-19502017/{name}', **kwargs)

    return df



def races_subset(df, races_index):

    df = df[df.raceId.isin(races_index)].copy()

    df['round'] = df.raceId.map(races['round'])

    df['round'] -= df['round'].min()

    return df.set_index('round').sort_index()



IMG_ATTRS = 'style="display: inline-block;" width=16 height=16'

YT_IMG = f'<img {IMG_ATTRS} src="https://youtube.com/favicon.ico">'

WK_IMG = f'<img {IMG_ATTRS} src="https://wikipedia.org/favicon.ico">'

GM_IMG = f'<img {IMG_ATTRS} src="https://maps.google.com/favicon.ico">'



# Read data

circuits = read_csv('circuits.csv', encoding='ISO-8859-1', index_col=0)

constructorResults = read_csv('constructorResults.csv', index_col=0)

constructors = read_csv('constructors.csv', index_col=0)

constructorStandings = read_csv('constructorStandings.csv', index_col=0)

drivers = read_csv('drivers.csv', encoding='ISO-8859-1', index_col=0)

driverStandings = read_csv('driverStandings.csv', index_col=0)

lapTimes = read_csv('lapTimes.csv')

pitStops = read_csv('pitStops.csv')

qualifying = read_csv('qualifying.csv', index_col=0)

races = read_csv('races.csv', index_col=0)

results = read_csv('results.csv', index_col=0)

seasons = read_csv('seasons.csv', index_col=0)

status = read_csv('status.csv', index_col=0)



def url_extract(s):

    return (s.str.split('/') 

             .str[-1].fillna('') 

             .apply(urllib.parse.unquote) 

             .str.replace('_', ' ') 

             .str.replace('\s*\(.*\)', ''))



# Fix circuit data

idx = circuits.url.str.contains('%').fillna(False)

circuits.loc[idx, 'name'] = url_extract(circuits[idx].url)

circuits.location.replace({ 'MontmelÌ_':'Montmeló',

                            'SÌ£o Paulo':'São Paulo',

                            'NÌ_rburg':'Nürburg'}, inplace=True)



# Fix driver data

idx = drivers.url.str.contains('%').fillna(False)

t = url_extract(drivers.url)

drivers.loc[idx, 'forename'] = t[idx].str.split(' ').str[0]

drivers.loc[idx, 'surname'] = t[idx].str.split(' ').str[1:].str.join(' ')



# Fix Montoya (exception not fixed by above code)

drivers.loc[31, 'forename'] = 'Juan Pablo'

drivers.loc[31, 'surname'] = 'Montoya'



idx = drivers.surname.str.contains('Schumacher').fillna(False)

drivers['display'] = drivers.surname

drivers.loc[idx, 'display'] = drivers.loc[idx, 'forename'].str[0] + ". " + drivers.loc[idx, 'surname']



# For display in HTML tables

drivers['Driver'] = drivers['forename'] + " " + drivers['surname']

drivers['Driver'] = drivers.apply(lambda r: '<a href="{url}">{Driver}</a>'.format(**r), 1)

constructors['label'] = constructors['name']

constructors['name'] = constructors.apply(lambda r: '<a href="{url}">{name}</a>'.format(**r), 1)



# Join fields

results['status'] = results.statusId.map(status.status)

results['Team'] = results.constructorId.map(constructors.name)

results['score'] = results.points>0

results['podium'] = results.position<=3



# Cut data to one year

races = races.query('year==@YEAR').sort_values('round').copy()

results = results[results.raceId.isin(races.index)].copy()

lapTimes = lapTimes[lapTimes.raceId.isin(races.index)].copy()

driverStandings = races_subset(driverStandings, races.index)

constructorStandings = races_subset(constructorStandings, races.index)





lapTimes = lapTimes.merge(results[['raceId', 'driverId', 'positionOrder']], on=['raceId', 'driverId'])

lapTimes['seconds'] = lapTimes.pop('milliseconds') / 1000



def format_standings(df, key):

    df = df.sort_values('position')

    gb = results.groupby(key)

    df['Position'] = df.positionText

    df['points'] = df.points.astype(int)

    df['scores'] = gb.score.sum().astype(int)

    df['podiums'] = gb.podium.sum().astype(int)

    for c in [ 'scores', 'points', 'podiums', 'wins' ]:

        df.loc[df[c] <= 0, c] = ''

    return df



# Drivers championship table

def drivers_standings(df):

    df = df.join(drivers)

    df = format_standings(df, 'driverId')

    df['Team'] = results.groupby('driverId').Team.last()

    use = ['Position', 'Driver',  'Team', 'points', 'wins', 'podiums', 'scores', 'nationality' ]

    df = df[use].set_index('Position').fillna('')

    df.columns = df.columns.str.capitalize()

    return df



# Constructors championship table

def constructors_standings(df):

    df = df.join(constructors)

    df = format_standings(df, 'constructorId')

    

    # add drivers for team

    tmp = results.join(drivers.drop('number', 1), on='driverId')

    df = df.join(tmp.groupby('constructorId').Driver.unique().str.join(', ').to_frame('Drivers'))



    use = ['Position', 'name', 'points', 'wins', 'podiums', 'scores', 'nationality', 'Drivers' ]

    df = df[use].set_index('Position').fillna('')

    df.columns = df.columns.str.capitalize()

    return df



# Race results table

def format_results(df):

    df['Team'] = df.constructorId.map(constructors.name)

    df['Position'] = df.positionOrder

    df['number'] = df.number.map(int)

    df['points'] = df.points.map(int)

    df.loc[df.points <= 0, 'points'] = ''

    use = ['Driver', 'Team', 'number', 'grid', 'Position', 'points', 'laps', 'time', 'status' ]

    df = df[use].set_index('Position').fillna('')

    df.columns = df.columns.str.capitalize()

    return df
plt.rc("figure", figsize=(16, 12))

plt.rc("font", size=(14))

plt.rc("axes", xmargin=0)



display(HTML(

    f'<h1 id="drivers">Formula One Drivers\' World Championship &mdash; {YEAR}</h1>'

))



# Championship position traces

champ = driverStandings.groupby("driverId").position.last().to_frame("Pos")

champ = champ.join(drivers)

order = np.argsort(champ.Pos)



color = [DRIVER_C[d] for d in champ.index[order]]

style = [LINESTYLES[DRIVER_LS[d]] for d in champ.index[order]]

labels = champ.Pos.astype(str) + ". " + champ.display



chart = driverStandings.pivot("raceId", "driverId", "points")

# driverStandings may have a subset of races (i.e. season in progress) so reindex races

chart.index = races.reindex(chart.index).name.str.replace("Grand Prix", "GP").rename("Race")

chart.columns = labels



chart.iloc[:, order].plot(title=f"F1 Drivers\' World Championship — {YEAR}", color=color, style=style)

plt.xticks(range(chart.shape[0]), chart.index, rotation=45)

plt.grid(axis="x", linestyle="--")

plt.ylabel("Points")

legend_opts = dict(bbox_to_anchor=(1.02, 0, 0.2, 1),

                   loc="upper right",

                   ncol=1,

                   shadow=True,

                   edgecolor="black",

                   mode="expand",

                   borderaxespad=0.)

plt.legend(**legend_opts)

plt.tight_layout()

plt.show()



display(HTML(f"<h2>Results</h2>"))

display(drivers_standings(driverStandings.loc[driverStandings.index.max()].set_index("driverId")).style)
display(HTML(

    f'<h1 id="constructors">Formula One Constructors\' World Championship &mdash; {YEAR}</h1>'

))



# Championship position traces

champ = constructorStandings.groupby("constructorId").position.last().to_frame("Pos")

champ = champ.join(constructors)

order = np.argsort(champ.Pos)



color = [TEAM_C[c] for c in champ.index[order]]

labels = champ.Pos.astype(str) + ". " + champ.label



chart = constructorStandings.pivot("raceId", "constructorId", "points")

# constructorStandings may have a subset of races (i.e. season in progress) so reindex races

chart.index = races.reindex(chart.index).name.str.replace("Grand Prix", "GP").rename("Race")

chart.columns = labels



chart.iloc[:, order].plot(title=f"F1 Constructors\' World Championship — {YEAR}", color=color)

plt.xticks(range(chart.shape[0]), chart.index, rotation=45)

plt.grid(axis="x", linestyle="--")

plt.ylabel("Points")

plt.legend(**legend_opts)

plt.tight_layout()

plt.show()



display(HTML(f"<h2>Results</h2>"))

display(constructors_standings(constructorStandings.loc[constructorStandings.index.max()].set_index("constructorId")).style)
# Show race traces

for rid, times in lapTimes.groupby("raceId"):



    race = races.loc[rid]

    circuit = circuits.loc[race.circuitId]

    title = "Round {round} — F1 {name} — {year}".format(**race)

    qstr = race["name"].replace(" ", "+")

    

    res = results.query("raceId==@rid").set_index("driverId")

    res = res.join(drivers.drop("number", 1))



    map_url = "https://www.google.com/maps/search/{lat}+{lng}".format(**circuit)

    vid_url = f"https://www.youtube.com/results?search_query=f1+{YEAR}+{qstr}"



    lines = [

        '<h1 id="race{round}">R{round} — {name}</h1>'.format(**race),

        '<p><b>{date}</b> — '.format(img=WK_IMG, **race),

        '<b>Circuit:</b> <a href="{url}">{name}</a>, {location}, {country}'.format(**circuit),

        '<br><a href="{url}">{img} Wikipedia race report</a>'.format(img=WK_IMG, **race),

        f'<br><a href="{map_url}">{GM_IMG} Map Search</a>',

        f'<br><a href="{vid_url}">{YT_IMG} YouTube Search</a>',

    ]

    

    display(HTML("\n".join(lines)))

    

    chart = times.pivot_table("seconds", "lap", "driverId")



    # reference laptime series

    basis = chart.median(1).cumsum()



    labels = res.loc[chart.columns].apply(lambda r: "{positionOrder:2.0f}. {display}".format(**r), 1)

    order = np.argsort(labels)

    show = chart.iloc[:, order]

    

    color = [DRIVER_C[d] for d in show.columns]

    style = [LINESTYLES[DRIVER_LS[d]] for d in show.columns]



    show = (basis - show.cumsum().T).T

    show.columns = labels.values[order]



    # fix large outliers; only applies to 1 race - Aus 2016

    show[show>1000] = np.nan

    

    xticks = np.arange(0, len(chart)+1, 2)

    if len(chart) % 2:  # odd number of laps: nudge last tick to show it

        xticks[-1] += 1



    show.plot(title=title, style=style, color=color)

    if show.min().min() < -180:

        plt.ylim(-180, show.max().max()+3)

    plt.ylabel("Time Delta (s)")

    plt.xticks(xticks, xticks)

    plt.grid(linestyle="--")

    plt.legend(bbox_to_anchor=(0, -0.2, 1, 1),

               loc=(0, 0),

               ncol=6,

               shadow=True,

               edgecolor="black",

               mode="expand",

               borderaxespad=0.)

    plt.tight_layout()

    plt.show()

    

    display(HTML(f"<h2>Results</h2>"))

    display(format_results(res).style)