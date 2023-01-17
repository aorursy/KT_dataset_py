import numpy as np

import pandas as pd



import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg' 



import warnings

warnings.filterwarnings("ignore")

import os

import gc
print('%-33s %d' % ('Input files available:', len(os.listdir('../input'))))

for i in range(34):

    print('-',end='')

print('-')

for file in os.listdir("../input/"):

    unit = 'MB'

    size = os.stat('../input/' + file).st_size

    if round(size / 2**20, 2) < 0.5:

        size = round(size / 2**10, 2)

        unit = 'KB'

    else:

        size = round(size / 2**20, 2)

    print('%-25s %6.2f %2s' % (file, size, unit))
#Source kernel: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df



def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
wowah = import_data('../input/wowah_data.csv')

zones = pd.read_csv('../input/zones.csv', encoding='iso-8859-1')

location_coords = pd.read_csv('../input/location_coords.csv', encoding='iso-8859-1')

locations = pd.read_csv('../input/locations.csv', encoding='iso-8859-1')
wowah.rename({'char': 'player', 

              ' level': 'level',

              ' race': 'race',

              ' charclass': 'class',

              ' zone': 'zone',

              ' guild': 'guild',

              ' timestamp': 'timestamp'}, axis=1, inplace=True)

zones['Zone_Name'].replace({'Dalaran<U+7AF6><U+6280><U+5834>': 'Dalaran Arena'}, inplace=True)

wowah['zone'].replace({'Dalaran競技場': 'Dalaran Arena'}, inplace=True)



def time_transform(x):

    y = x.split()[0]

    return y[:-2] + '20' + y[-2:]



wowah['date'] = wowah['timestamp'].apply(time_transform)

wowah['time'] = wowah['timestamp'].apply(lambda x: x.split()[1][:-4] + '0')



zones_dict = zones[['Zone_Name', 'Type']].set_index(['Zone_Name']).T.to_dict('records')[0]

wowah['zone_type'] = wowah['zone'].map(zones_dict)



wowah['class_id'] = np.array(pd.factorize(wowah['class'])[0])

wowah['race_id'] = np.array(pd.factorize(wowah['race'])[0])

wowah['class_id'] = wowah['class_id'].astype(str)

wowah['race_id'] = wowah['race_id'].astype(str)

wowah['sym'] = pd.Series(index = wowah.index, data='#')

wowah['char'] = wowah[['race_id', 'class_id', 'sym', 'player']].astype(str).sum(axis=1)

wowah.drop(['race_id', 'class_id', 'sym'], axis=1, inplace=True)



zones['Min_rec_level'].iloc[27] = 25.0

zones['Max_rec_level'].iloc[27] = 30.0
print('Records dataframe size:', wowah.shape)

print('Data on {:.0f} players and {:.0f} their charachters available'.format(len(wowah['player'].unique()), len(wowah['char'].unique())))

wowah.head()
loc_df = locations[locations['Location_Type'].isin(['Dungeon', 'Raid'])].pivot_table(columns='Game_Version',

                                                                            index='Location_Type',

                                                                            values='Location_Name',

                                                                            aggfunc=lambda x: x.count())[['WoW','TBC','WLK','CAT','MoP','WoD']]



import plotly.graph_objects as go



fig = go.Figure(data=[

    go.Bar(name='Dungeons', x=loc_df.columns, y=loc_df.loc['Dungeon',:]),

    go.Bar(name='Raids', x=loc_df.columns, y=loc_df.loc['Raid',:])

])



fig.update_layout(title='Dungeons & Raids/Expansion', barmode='group')

fig.show()
del loc_df

gc.collect()
from plotly.subplots import make_subplots



race_stats = wowah.drop_duplicates(['char']).groupby(['race'], as_index=False)['char'].count().sort_values(['char'], ascending=False)

class_stats = wowah.drop_duplicates(['char']).groupby(['class'], as_index=False)['char'].count().sort_values(['char'], ascending=False)



race_colors = ['#dc1c13', '#ea4c46', '#f07470', '#f1959b', '#f6bdc0']

class_colors = ['#dc1c13', '#E3342D', '#ea4c46', '#ED605B', '#f07470',

                '#F17D7B', '#F18586', '#f1959b', '#F4A9AE', '#f6bdc0']



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=race_stats['race'],

                     values=race_stats['char'],

                     name="",

                     marker=dict(colors=race_colors, line=dict(color='#ffffff', width=0.5)),

                     showlegend=False

                    ),

              1, 1)

fig.add_trace(go.Pie(labels=class_stats['class'],

                     values=class_stats['char'],

                     name="",

                     marker=dict(colors=class_colors, line=dict(color='#ffffff', width=0.5)),

                     textfont=dict(size=11),

                     showlegend=False),

              1, 2)



fig.update_traces(hole=.4, hoverinfo="value+percent", textinfo="label")



fig.update_layout(

    title_text="Popularity Charts",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Race', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Class', x=0.825, y=0.5, font_size=20, showarrow=False)]

)

fig.show()
race_class_mix = wowah.drop_duplicates(['char']).pivot_table(values='char',

                                                             index='race',

                                                             columns='class',

                                                     aggfunc=lambda x: x.value_counts().count()).fillna(0).astype(int)



_, ax = plt.subplots(1, 1, figsize=(14, 5.5))

sns.set_context("paper", font_scale=1.4) 

sns.heatmap(race_class_mix, annot=True, cmap='Reds', fmt='g', ax=ax)

plt.title('Race/Class Combinations')

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

ax.set_yticklabels(ax.get_yticklabels(), va='center')

ax.set(ylabel='', xlabel='')

plt.show()
del race_stats

del class_stats

del race_class_mix

gc.collect()
activity = wowah.groupby('date')['char'].nunique().to_frame('char').reset_index()



all_dates = pd.Series(pd.date_range('01/01/08', freq='D', periods=365))

all_dates = all_dates.dt.strftime('%m/%d/%Y')



missing_dates = list(set(all_dates) - set(activity['date'].unique()))

print('Missing dates:', *(missing_dates), sep='\n')
add_df = pd.DataFrame(columns=['date', 'char'])

add_df['date'] = missing_dates

add_df['char'] = 0



activity = pd.concat([activity, add_df])

activity['date'] = pd.to_datetime(activity['date'])

activity.sort_values(by=['date'], inplace=True)

activity.reset_index(drop=True, inplace=True)



#Source: https://community.plot.ly/t/colored-calendar-heatmap-in-dash/10907/9

import datetime



start = activity['date'].iloc[0]

end = activity['date'].iloc[-1]



d1 = datetime.date(start.year, start.month, start.day)

d2 = datetime.date(end.year, end.month, end.day)



delta = d2 - d1



dates_in_year = [d1 + datetime.timedelta(i) for i in range(delta.days+1)] 

#gives me a list with datetimes for each day a year



weekdays_in_year = [6 - i.weekday() for i in dates_in_year] 

#gives [1,2,3,4,5,6,0,1,2,3,4,5,6,…] (ticktext in xaxis dict translates this to weekdays



weeknumber_of_dates = [i.strftime("%Gww%V")[2:] for i in dates_in_year] 

#gives [1,1,1,1,1,1,1,2,2,2,2,2,2,2,…] name is self-explanatory



z = activity['char']



text = [str(i).replace('-','/') for i in dates_in_year] 

#gives something like list of strings like ‘2018-01-25’ for each date. Used in data trace to make good hovertext.

colorscale=[[False, '#eeeeee'], [True, '#76cf63']]



data = [go.Heatmap(x = weeknumber_of_dates,

                   y = weekdays_in_year,

                   z = z,

                   text=text,

                   hoverinfo='text+z',

                   xgap=3, # this

                   ygap=3, # and this is used to make the grid-like apperance

                   showscale=False,

                   colorscale='Hot',

                   reversescale=True)]

layout = go.Layout(title='Players Activity throughout the Year',

                   height=230,

                   width=1000,

                   yaxis=dict(showline = False, 

                              showgrid = False, 

                              zeroline = False,

                              tickmode='array',

                              ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][::-1],

                              tickvals=list(range(7))),

                   xaxis=dict(showline = False, 

                              showgrid = False, 

                              zeroline = False,

                              tickmode='array',

                              ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',

                                        'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec'],

                              tickvals=[0, 4, 8, 13, 17, 21, 26, 30, 35, 39, 43, 48]),

                   font={'size':10, 'color':'#9e9e9e'},

                   plot_bgcolor=('#fff'),

                   margin=dict(t=40))

fig = go.Figure(data=data, layout=layout)

fig.show()
wowah['level_bins'] = pd.cut(wowah['level'], [0, 15, 60, 70, 80])

lvl_act = wowah.pivot_table(index='date',

                            columns=['level_bins'],

                            values='char',

                            aggfunc=lambda x: x.value_counts().count()).fillna(0).astype(int)

lvl_act.columns = ['1-15', '15-60', '60-70', '70-80']

lvl_act.reset_index(inplace=True)



all_dates = pd.Series(pd.date_range('01/01/08', freq='D', periods=365))

all_dates = all_dates.dt.strftime('%m/%d/%Y')

missing_dates = list(set(all_dates) - set(lvl_act['date'].unique()))



add_df = pd.DataFrame(columns=['date'])

add_df['date'] = missing_dates



lvl_act = pd.concat([lvl_act, add_df])

lvl_act['date'] = pd.to_datetime(lvl_act['date'])

lvl_act.sort_values(by=['date'], inplace=True)

lvl_act.reset_index(drop=True, inplace=True)

lvl_act['date'] = lvl_act['date'].dt.strftime('%m/%d/%Y')



fig = go.Figure()

colormap = ['purple', 'orange', 'green', 'blue', 'purple']

columns = ['1-15', '15-60', '60-70', '70-80']

for color, column in zip(colormap, columns):

    fig.add_trace(go.Scatter(

                    x=lvl_act['date'],

                    y=lvl_act[column],

                    name=column,

                    line_color=color,

                    hoverinfo='name+x+y',

                    opacity=0.8))

fig.update_layout(title_text="Players Activity over Year/Levels", 

                 xaxis=dict(

                     tickmode='array',

                     ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],

                     tickvals=[1, 31, 60, 91, 121, 152, 182, 213, 244, 274, 304, 335]),

                  xaxis_rangeslider_visible=True)

fig.show()
del activity

del add_df

del lvl_act

gc.collect()
tmp_df = wowah.groupby(by=['time', 'date'])['char'].nunique().to_frame('char').reset_index()

day_activity = round(tmp_df.groupby(['time'], as_index=False)['char'].mean())



for zone_type in ['Arena', 'Battleground', 'City', 'Dungeon']:

    day_activity[zone_type] = np.array(round((wowah[wowah['zone_type'] == zone_type] \

                                                .groupby(by=['time', 'date'])['char'] \

                                                .nunique() \

                                                .to_frame('char') \

                                                .reset_index()).groupby(['time'])['char'].mean()))

day_activity['Zone'] = np.array(round((wowah[(wowah['zone_type'] == 'Zone') | 

                                               (wowah['zone_type'] == 'Sea') | 

                                               (wowah['zone_type'] == 'Transit')] \

                                                .groupby(by=['time', 'date'])['char'] \

                                                .nunique() \

                                                .to_frame('char') \

                                                .reset_index()).groupby(['time'])['char'].mean()))

day_activity.rename({'char': 'Total', 'time': 'Time'}, axis=1, inplace=True)



fig = go.Figure()

colormap = ['red', 'orange', 'green', 'blue', 'purple']

columns = list(day_activity.columns[1:])

for color, column in zip(colormap, columns):

    fig.add_trace(go.Scatter(

                    x=day_activity['Time'],

                    y=day_activity[column],

                    name=column,

                    line_color=color,

                    hoverinfo='name+x+y',

                    opacity=0.8))

fig.update_layout(title_text="Players Activity on average Day")

fig.show()
del tmp_df

del day_activity

gc.collect()
print('Number of guilds on the server:', len(wowah['guild'].unique())-1)



chars_2008 = wowah.groupby(['char'], as_index=False)['level', 'guild', 'class'].first()

chars_2008 = chars_2008[(chars_2008['level'] == 1) | ((chars_2008['level'] == 55) & (chars_2008['class'] == 'Death Knight'))]



first_joined = wowah[(wowah['char'].isin(chars_2008['char'])) & (wowah['guild'] != -1)].groupby(['char'], as_index=False)['guild', 'level'].first()

print('Number of players ever joined the guild:', len(first_joined))



never_joined = wowah[wowah['char'].isin(chars_2008['char'])].groupby(['char'], as_index=False)['guild', 'level'].max()

print('Number of players never joined the guild:', len(never_joined))

never_joined = never_joined[never_joined['guild'] == -1]
fig = make_subplots(rows=2, cols=1, specs=[[{'type':'xy'}], [{'type':'xy'}]])



fig.add_trace(go.Histogram(x=first_joined['level'],

                           y=first_joined['char'],

                           name='First Lvl Joined'), 1, 1)

fig.add_trace(go.Histogram(x=never_joined['level'],

                           y=never_joined['char'],

                           name='Max Lvl Never Joined'), 2, 1)



fig.update_layout(title_text="Guild/No-Guild Players Stats", 

                  height=1000)

fig.update_traces(hoverinfo='x+y')

fig.show()
wowah['zone_type'].replace({'Zone': 'Levelling',

                            'Sea': 'Levelling',

                            'Transit': 'Levelling',

                            'Event': 'Dungeon', 

                            'Battleground': 'PVP', 

                            'Arena': 'PVP'}, inplace=True)



guild_act = wowah[wowah['guild'] != -1 & (wowah['level'] >= 15)].pivot_table(index='date',

                                        columns=['guild', 'zone_type'],

                                        values='char',

                                        aggfunc=lambda x: x.value_counts().count()).fillna(0).astype(int).sum(axis=0).reset_index().groupby(['zone_type'], as_index=False)[0].sum()



no_guild_act = wowah[wowah['guild'] == -1 & (wowah['level'] >= 15)].pivot_table(index='date',

                                        columns=['guild', 'zone_type'],

                                        values='char',

                                        aggfunc=lambda x: x.value_counts().count()).fillna(0).astype(int).sum(axis=0).reset_index().groupby(['zone_type'], as_index=False)[0].sum()



act_colors = ['9d44d1', 'ff0000', '007fd7', 'ffb100']



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=guild_act['zone_type'],

                     values=guild_act[0],

                     name="",

                     marker=dict(colors=act_colors, line=dict(color='#ffffff', width=0.5)),

                     showlegend=False

                    ),

              1, 1)



fig.add_trace(go.Pie(labels=no_guild_act['zone_type'],

                     values=no_guild_act[0],

                     name="",

                     marker=dict(colors=act_colors, line=dict(color='#ffffff', width=0.5)),

                     showlegend=False

                    ),

              1, 2)

fig.update_traces(hole=.4, hoverinfo="percent", textinfo="label")



fig.update_layout(

    title_text="Players Activity Comparison [15+ Level]",

    annotations=[dict(text='Guild', x=0.175, y=0.5, font_size=20, showarrow=False),

                 dict(text='No-Guild', x=0.855, y=0.5, font_size=20, showarrow=False)]

)

fig.show()
del first_joined

del never_joined

del guild_act

del no_guild_act

gc.collect()
char_creation = wowah[wowah['level'] == 1].groupby(['date'], as_index=False)['char'].count()

lvl_70 = wowah[(wowah['char'].isin(chars_2008['char'])) & (wowah['level'] == 70)].drop_duplicates(['char']).groupby(['date'], as_index=False)['char'].count()

lvl_70['date'] = pd.to_datetime(lvl_70['date'])

lvl_70 = lvl_70[lvl_70['date'] < datetime.date(2008, 10, 13)]

lvl_70['date'] = lvl_70['date'].dt.strftime('%m/%d/%Y')

lvl_80 = wowah[(wowah['char'].isin(chars_2008['char'])) & (wowah['level'] == 80)].drop_duplicates(['char']).groupby(['date'], as_index=False)['char'].count()



print('Chars started in 2008:', len(chars_2008))

print('Reached level 70 (TBC): {:d} ({:.2f}%)'.format(lvl_70['char'].sum(), lvl_70['char'].sum() / len(chars_2008) * 100))

print('Reached level 80 (WLK): {:d} ({:.2f}%)'.format(lvl_80['char'].sum(), lvl_80['char'].sum() / len(chars_2008) * 100))
lvl_70_all = wowah[wowah['level'] == 70].drop_duplicates(['char'])

lvl_70_all['date'] = pd.to_datetime(lvl_70_all['date'])

lvl_70_all = lvl_70_all[lvl_70_all['date'] < datetime.date(2008, 10, 13)]

lvl_70_all['date'] = lvl_70_all['date'].dt.strftime('%m/%d/%Y')

lvl_80_all = wowah[wowah['level'] == 80].drop_duplicates(['char'])



print('TBC endgame community: {:d} ({:.2f}% of total) / 2008 players share: {:.2f}%' \

      .format(len(lvl_70_all), len(lvl_70_all) / len(wowah['char'].unique()) * 100, lvl_70['char'].sum() / len(lvl_70_all) * 100))

print('WLK endgame community: {:d} ({:.2f}% of total) / 2008 players share: {:.2f}%' \

      .format(len(lvl_80_all), len(lvl_80_all) / len(wowah['char'].unique()) * 100, lvl_80['char'].sum() / len(lvl_80_all) * 100))
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=char_creation['date'],

                y=char_creation['char'],

                name='Chars Started',

                line_color='red',

                hoverinfo='x+y',

                opacity=0.8))

fig.add_trace(go.Scatter(

                x=lvl_70['date'],

                y=lvl_70['char'],

                name='Reached Lvl 70 (TBC)',

                line_color='green',

                hoverinfo='x+y',

                opacity=0.8))

fig.add_trace(go.Scatter(

                x=lvl_80['date'],

                y=lvl_80['char'],

                name='Reached Lvl 80 (WLK)',

                line_color='blue',

                hoverinfo='x+y',

                opacity=0.8))

fig.update_layout(title_text="Chars Created/Reached Lvl Cap Chart", 

                 xaxis=dict(

                     tickmode='array',

                     ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],

                     tickvals=[1, 31, 60, 91, 121, 152, 182, 213, 244, 274, 304, 335]),

                  xaxis_rangeslider_visible=True

                 )

fig.show()
del char_creation

del lvl_70

del lvl_80

del lvl_70_all

del lvl_80_all

gc.collect()
for race in wowah['race'].unique():

    levelling_zones = wowah[(wowah['zone_type'] == 'Levelling') & (wowah['race'] == race)].pivot_table(index='level',

                                        columns=['zone_type'],

                                        values='zone',

                                        aggfunc=lambda x: x.value_counts().index[0])

    levelling_zones.columns = ['Levelling']

    levelling_zones.reset_index(inplace=True)

    

    zones_min_rec = zones[zones['Type'].isin(['Zone', 'Transit', 'Sea'])][['Zone_Name', 'Min_rec_level']].set_index(['Zone_Name']).T.to_dict('records')[0]

    zones_max_rec = zones[zones['Type'].isin(['Zone', 'Transit', 'Sea'])][['Zone_Name', 'Max_rec_level']].set_index(['Zone_Name']).T.to_dict('records')[0]



    levelling_zones['Min_Rec'] = levelling_zones['Levelling'].map(zones_min_rec).astype(int)

    levelling_zones['Max_Rec'] = levelling_zones['Levelling'].map(zones_max_rec).astype(int)



    levelling_zones['Under'] = -(levelling_zones['level'] < levelling_zones['Min_Rec']).astype(int)

    levelling_zones['Over'] = (levelling_zones['level'] > levelling_zones['Max_Rec']).astype(int)

    levelling_zones['Recommended'] = levelling_zones['Under'] + levelling_zones['Over']

    levelling_zones['Recommended'].replace({-1: 'Under Recommended', 0: 'Recommended', 1: 'Over Recommended'}, inplace=True)



    fig = go.Figure()



    colorsIdx = {'Under Recommended': 'red', 'Recommended': 'green', 'Over Recommended': 'darkgray'}

    cols = levelling_zones['Recommended'].map(colorsIdx)



    fig.add_trace(go.Scatter(

                    x=levelling_zones['level'],

                    y=levelling_zones['Levelling'],

                    name='',

                    mode='markers',

                    text = levelling_zones['Recommended'],

                    marker=dict(color=cols),

                    hoverinfo='x+y+text',

                    opacity=0.8))

    fig.update_layout(title_text="Popular " + race + " Levelling Zone/Level")

    fig.update_xaxes(nticks=20)

    fig.update_yaxes(tickfont=dict(size=10))

    fig.show()
del levelling_zones

del zones_min_rec

del zones_max_rec

gc.collect()
fastest_80 = []

classes = wowah['class'].unique()

for char_class in classes:

    fastest_80.append(wowah[(wowah['char'].isin(chars_2008['char'])) & (wowah['level'] == 80) & (wowah['class'] == char_class)]['char'].iloc[0])



fig = go.Figure()

colormap = ['gray', 'black', 'red', 'orange', 'goldenrod', 'green', 'blue', 'hotpink', 'purple', 'turquoise']

for color, char in zip(colormap, fastest_80):

    hist_80 = wowah[wowah['char'] == char].reset_index(drop=True)

    hist_80['playtime'] = (hist_80.index + 1) / 6

    hist_80 = hist_80[['level', 'playtime']]

    fig.add_trace(go.Scatter(

                    x=hist_80['playtime'],

                    y=hist_80['level'],

                    name=wowah[wowah['char'] == char]['class'].iloc[0],

                    line_color=color,

                    hoverinfo='name+x+y',

                    opacity=0.7))

fig.update_layout(title="Fastest Players Levelling/Playtime (Hours)")

fig.show()
del fastest_80

del hist_80

gc.collect()
lvl_70_chars = wowah[(wowah['char'].isin(chars_2008['char'])) & (wowah['level'] == 70)]['char']

lvl_70_levelling = wowah[wowah['char'].isin(lvl_70_chars) & (wowah['zone_type'] != 'City')]

lvl_70_levelling = lvl_70_levelling[lvl_70_levelling['level'] < 70]



act_list = []

for act in ['PVP', 'Dungeon', 'Levelling']:

    tmp_df = lvl_70_levelling[lvl_70_levelling['zone_type'] == act].pivot_table(index='date', 

                                                                                columns='char', 

                                                                                values='time', 

                                                                            aggfunc=lambda x: x.value_counts().count())

    act_list.append(tmp_df.fillna(0).sum().sum() / (tmp_df.shape[0] * tmp_df.shape[1] - tmp_df.isna().sum().sum()) / 6)



fastest_70 = []

classes = wowah['class'].unique()

for char_class in classes:

    fastest_70.append(wowah[(wowah['char'].isin(chars_2008['char'])) & (wowah['level'] == 70) & (wowah['class'] == char_class)]['char'].iloc[0])



fast_lvl_70_levelling = wowah[wowah['char'].isin(fastest_70) & (wowah['zone_type'] != 'City')]

fast_lvl_70_levelling = fast_lvl_70_levelling[fast_lvl_70_levelling['level'] < 70]



fast_act_list = []

for act in ['PVP', 'Dungeon', 'Levelling']:

    tmp_df = fast_lvl_70_levelling[fast_lvl_70_levelling['zone_type'] == act].pivot_table(index='date', 

                                                                                          columns='char', 

                                                                                          values='time', 

                                                                            aggfunc=lambda x: x.value_counts().count())

    fast_act_list.append(tmp_df.fillna(0).sum().sum() / (tmp_df.shape[0] * tmp_df.shape[1] - tmp_df.isna().sum().sum()) / 6)



categories = ['PVP', 'PVE', 'Levelling']



fig = go.Figure()



fig.add_trace(go.Scatterpolar(

      r=act_list,

      theta=categories,

      fill='toself',

      name='Average'

))



fig.add_trace(go.Scatterpolar(

      r=fast_act_list,

      theta=categories,

      fill='toself',

      name='Fastest'

))



fig.update_layout(

    title='Levelling Players Everyday Activities',

  polar=dict(

    radialaxis=dict(

      visible=True,

      range=[0, 8]

    )),

  showlegend=True

)



fig.show()
del lvl_70_chars

del lvl_70_levelling

del fastest_70

del fast_lvl_70_levelling

del act_list

del fast_act_list

gc.collect()
dungeon_stats = wowah[wowah['zone_type'] == 'Dungeon'].pivot_table(index='level',

                                        columns=['zone_type'],

                                        values='zone',

                                        aggfunc=lambda x: x.value_counts().index[0])

dungeon_stats.columns = ['Dungeon']

dungeon_stats.reset_index(inplace=True)

dungeon_stats = dungeon_stats.iloc[1:,:]



zones_min_rec = zones[zones['Type'] == 'Dungeon'][['Zone_Name', 'Min_rec_level']].set_index(['Zone_Name']).T.to_dict('records')[0]

zones_max_rec = zones[zones['Type'] == 'Dungeon'][['Zone_Name', 'Max_rec_level']].set_index(['Zone_Name']).T.to_dict('records')[0]



dungeon_stats['Min_Rec'] = dungeon_stats['Dungeon'].map(zones_min_rec).astype(int)

dungeon_stats['Max_Rec'] = dungeon_stats['Dungeon'].map(zones_max_rec).astype(int)



dungeon_stats['Under'] = -(dungeon_stats['level'] < dungeon_stats['Min_Rec']).astype(int)

dungeon_stats['Over'] = (dungeon_stats['level'] > dungeon_stats['Max_Rec']).astype(int)

dungeon_stats['Recommended'] = dungeon_stats['Under'] + dungeon_stats['Over']



#dungeon_stats['Recommended'] = (dungeon_stats['level'] >= dungeon_stats['Min_Rec']) & (dungeon_stats['level'] <= dungeon_stats['Max_Rec'])



fig = go.Figure()



colorsIdx = {-1: 'red', 0: 'green', 1: 'darkgray'}

cols = dungeon_stats['Recommended'].map(colorsIdx)



fig.add_trace(go.Scatter(

                    x=dungeon_stats['level'],

                    y=dungeon_stats['Dungeon'],

                    name='',

                    mode='markers',

    marker=dict(color=cols),

                    hoverinfo='x+y',

                    opacity=0.8))

fig.update_layout(title_text="Popular Dungeon/Level")

#fig.update_layout(title_text="Popular Dungeon/Level", width=740, height=480)

fig.update_xaxes(nticks=10)

#fig.update_yaxes(tickfont=dict(size=10))

fig.show()
del dungeon_stats

del zones_min_rec

del zones_max_rec

gc.collect()
#had to put a specific number of a row, where TBC era ends, as other, more elegant options take hours to compute

tbc_cap = wowah[wowah['level'] == 70].reset_index().iloc[: 6034346, :]

tbc_act = tbc_cap.pivot_table(index='date',

                    columns=['zone_type'],

                    values='char',

                    aggfunc=lambda x: x.value_counts().count()).fillna(0).astype(int).sum(axis=0).reset_index().groupby(['zone_type'], as_index=False)[0].sum()



wlk_cap = wowah[wowah['level'] == 80]

wlk_act = wlk_cap.pivot_table(index='date',

                    columns=['zone_type'],

                    values='char',

                    aggfunc=lambda x: x.value_counts().count()).fillna(0).astype(int).sum(axis=0).reset_index().groupby(['zone_type'], as_index=False)[0].sum()

tbc_act['zone_type'].replace({'Levelling': 'The World'}, inplace=True)

wlk_act['zone_type'].replace({'Levelling': 'The World'}, inplace=True)



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=tbc_act['zone_type'],

                     values=tbc_act[0],

                     name="",

                     marker=dict(colors=act_colors, line=dict(color='#ffffff', width=0.5)),

                     showlegend=False, textfont=dict(size=11)

                    ),

              1, 1)



fig.add_trace(go.Pie(labels=wlk_act['zone_type'],

                     values=wlk_act[0],

                     name="",

                     marker=dict(colors=act_colors, line=dict(color='#ffffff', width=0.5)),

                     showlegend=False, textfont=dict(size=11)

                    ),

              1, 2)

# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="percent", textinfo="label")



fig.update_layout(

    title_text="Players Endgame Activity",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='TBC', x=0.185, y=0.5, font_size=20, showarrow=False),

                 dict(text='WLK', x=0.82, y=0.5, font_size=20, showarrow=False)]

)

fig.show()
del tbc_cap

del wlk_cap

del tbc_act

del wlk_act

gc.collect()
wowah['zone_type'] = wowah['zone'].map(zones_dict)

bgs = wowah[wowah['zone_type'].isin(['Battleground'])].groupby(['zone'], as_index=False)['char'].count().sort_values(by=['char'], ascending=False)

arenas = wowah[wowah['zone_type'].isin(['Arena'])].groupby(['zone'], as_index=False)['char'].count().sort_values(by=['char'], ascending=False)



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]])

fig.add_trace(go.Bar(x=arenas['zone'],

                     y=arenas['char'],

                     name="Arenas",

                     showlegend=False

                    ),

              1, 1)

fig.add_trace(go.Bar(x=bgs['zone'],

                     y=bgs['char'],

                     name="Battlegrounds",

                     showlegend=False),

              1, 2)



fig.update_layout(title="Arenas & Battlegrounds Popularity")

fig.show()
del bgs

del arenas

gc.collect()