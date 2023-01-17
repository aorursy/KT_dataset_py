import numpy as np

import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

from os import listdir



pd.set_option('display.max_columns',100)



listdir('../input/csgo-professional-matches/')
base_dir = '../input/csgo-professional-matches/'



results_df = pd.read_csv(base_dir+'results.csv',low_memory=False)

picks_df = pd.read_csv(base_dir+'picks.csv',low_memory=False)

economy_df = pd.read_csv(base_dir+'economy.csv',low_memory=False)

players_df = pd.read_csv(base_dir+'players.csv',low_memory=False)
results_df.head()
picks_df.head()
economy_df.head()
players_df.head()
min_rank = 30

results_df = results_df[(results_df.rank_1<min_rank)&(results_df.rank_2<min_rank)]



picks_df     = picks_df  [picks_df  .match_id.isin(results_df.match_id.unique())]

economy_df   = economy_df[economy_df.match_id.isin(results_df.match_id.unique())]

players_df   = players_df[players_df.match_id.isin(results_df.match_id.unique())]
winner_1 = results_df[results_df.result_1>=results_df.result_2].result_1.values

loser_1  = results_df[results_df.result_1>=results_df.result_2].result_2.values



winner_2 = results_df[results_df.result_1<results_df.result_2].result_2.values

loser_2  = results_df[results_df.result_1<results_df.result_2].result_1.values



winner = np.concatenate((winner_1,winner_2))

loser = np.concatenate((loser_1,loser_2))

scores_df = pd.DataFrame(np.vstack((winner,loser)).T,columns=['winner','loser'])
gb = scores_df.groupby(by=['winner','loser'])['winner'].count()/scores_df.shape[0]

overtime_percentage = str(round(gb[gb.index.get_level_values(0)!=16].sum()*100,1))+'%'



gb = round(gb[gb>10**-3]*100,1)



index_plot = np.array(gb.index.get_level_values(0).astype('str'))+'-'+np.array(

    gb.index.get_level_values(1).astype('str'))



fig = go.Figure()

fig.add_trace(go.Scatter(x=index_plot,y=gb.values, name='results'))

fig.update_layout(xaxis_type='category',title='Scores distribution',xaxis_title='Score',yaxis_title='Percentage of matches (%)')
overtime_percentage
ct_1 = results_df[['date','_map','ct_1']].rename(columns={'ct_1':'ct'})

ct_2 = results_df[['date','_map','ct_2']].rename(columns={'ct_2':'ct'})

ct = pd.concat((ct_1,ct_2))
t_1 = results_df[['date','_map','t_1']].rename(columns={'t_1':'t'})

t_2 = results_df[['date','_map','t_2']].rename(columns={'t_2':'t'})

t = pd.concat((t_1,t_2))
t = t.sort_values('date')

ct = ct.sort_values('date')
maps = ['Cache','Cobblestone','Dust2','Inferno','Mirage','Nuke','Overpass','Train','Vertigo']
series_t, series_ct, how_ct = {},{},{}

for i, key in enumerate(maps):

    t_map = t[t._map == maps[i]]

    ct_map = ct[ct._map == maps[i]]

    y_t = t_map.t.rolling(min_periods = 20, window= 200, center=True).sum().values

    y_ct = ct_map.ct.rolling(min_periods = 20, window= 200, center=True).sum().values

    

    series_t[key] = pd.Series(data=y_t,index=t_map.date)

    series_ct[key] = pd.Series(data=y_ct,index=ct_map.date)

    

    how_ct[key] = series_ct[key]/(series_ct[key]+series_t[key])//0.001/10
def add_trace(_map):

    fig.add_trace(go.Scatter(x=how_ct[_map].index, y=how_ct[_map].values, name=_map))
fig = go.Figure()

for _map in maps:

    add_trace(_map)

fig.add_trace(go.Scatter(x=['2015-11-01', '2020-03-12'], y=[50,50],

                         mode='lines',line=dict(color='grey'),showlegend=False))

fig.update_layout(title='Distribution of rounds between CT and T sides',

                  yaxis_title='Percentage of round won on the CT-side (%)')

fig.show()
print('Total number of matches played on the map:')

results_df.groupby('_map').date.count()
majors = [{'tournament':'01. Cluj-Napoca 2015','start_date':'2015-10-28'},

          {'tournament':'02. Columbus 2016','start_date':'2016-03-29'},

          {'tournament':'03. Cologne 2016','start_date':'2016-07-05'},

          {'tournament':'04. Atlanta 2017','start_date':'2017-01-22'},

          {'tournament':'05. Krakow 2017','start_date':'2017-07-16'},

          {'tournament':'06. Boston 2018','start_date':'2018-01-26'},

          {'tournament':'07. London 2018','start_date':'2018-09-20'},

          {'tournament':'08. Katowice 2019','start_date':'2019-02-28'},

          {'tournament':'09. Berlin 2019','start_date':'2019-09-05'}]
def create_col_time_period(df):

    df['time_period'] = ''

    

    for major_start in majors:

        df.loc[(df['date']>=major_start['start_date']),'time_period'] = major_start['tournament']

    

    return df
results_df = create_col_time_period(results_df)

economy_df = create_col_time_period(economy_df)

picks_df = create_col_time_period(picks_df)

players_df = players_df.merge(results_df[['match_id','time_period']],'left',on='match_id')
results_df_team_1 = results_df[['time_period','team_1','_map','ct_1','t_2','ct_2','t_1']

                      ].rename(columns={'team_1':'team'})

results_df_team_2 = results_df[['time_period','team_2','_map','ct_1','t_2','ct_2','t_1']

                      ].rename(columns={'team_2':'team'})

results_df_teams = pd.concat((results_df_team_1,results_df_team_2))[['time_period','team','_map']]
gb = results_df_teams.groupby(['time_period','_map']).team.count()

gb_text = round(gb*100/gb.groupby('time_period').sum(),1).reset_index().rename(columns={'team':'percentage'})

gb_text.percentage = gb_text.percentage.astype(str)+'%'

gb = gb.reset_index()
fig = go.Figure()

for _map in maps:

    fig.add_bar(name=_map,x=gb[gb._map==_map].time_period,y=gb[gb._map==_map].team,

                text=gb_text[gb_text._map==_map].percentage,textposition='inside')



fig.update_layout(barmode='stack',legend=dict(traceorder='normal'),yaxis_title='Number of maps played',font=dict(size=10))

fig.show()
results_df_team_1_ct = results_df_team_1.rename(columns={'ct_1':'ct_team','t_2':'t_opponent'}).drop(columns=['ct_2','t_1'])

results_df_team_2_ct = results_df_team_2.rename(columns={'ct_2':'ct_team','t_1':'t_opponent'}).drop(columns=['ct_1','t_2'])

results_df_ct = pd.concat((results_df_team_1_ct,results_df_team_2_ct),sort=True)



results_df_team_1_t = results_df_team_1.rename(columns={'t_1':'t_team','ct_2':'ct_opponent'}).drop(columns=['ct_1','t_2'])

results_df_team_2_t = results_df_team_2.rename(columns={'t_2':'t_team','ct_1':'ct_opponent'}).drop(columns=['ct_2','t_1'])

results_df_t = pd.concat((results_df_team_1_t,results_df_team_2_t),sort=True)
results_df_ct['side_diff'] = results_df_ct['ct_team']-results_df_ct['t_opponent']

results_df_ct['side_sum'] = results_df_ct['ct_team']+results_df_ct['t_opponent']



results_df_t['side_diff'] = results_df_t['t_team']-results_df_t['ct_opponent']

results_df_t['side_sum']  = results_df_t['t_team'] +results_df_t['ct_opponent']



results_df_ct.head()
def groupby_time_map_team(results_df_side):

    gb = results_df_side.groupby(['time_period','_map','team'])['side_diff','side_sum'].sum()

    gb['side_diff_per_game'] = gb['side_diff']/(gb['side_sum']/15)

    gb = gb.sort_values(['time_period','_map','side_diff_per_game'],ascending=[1,1,0])



    for major in majors:

        col = major['tournament']

        _filter = (gb.side_sum > gb.loc[col].side_sum.mean()*3/4)

        gb.loc[col] = gb.loc[_filter][gb.loc[_filter].index.get_level_values(0)==col]



    gb.dropna(inplace=True)    



    return gb
gb_ct = groupby_time_map_team(results_df_ct)

gb_t = groupby_time_map_team(results_df_t)
def plot_ranking_teams_sides(gb):

    rankings_teams = {}

    for _map in maps:

        rankings_teams[_map] = pd.DataFrame(index=range(1,6),)

        rankings_teams[_map].index.name = 'ranking'

        rankings_teams[_map].style.set_caption(_map)



        for major in majors:

            col = major['tournament']

            try:

                rankings_teams[_map][col] = gb.loc[col,_map]['side_diff_per_game'][:5].index

            except:

                pass

        print('\n'+_map+':')

        display(rankings_teams[_map])
print('T-side Rankings:\n')

plot_ranking_teams_sides(gb_t)
print('CT-side Rankings:\n')

plot_ranking_teams_sides(gb_ct)
economy_df.head()
money_columns = ['2_t1','3_t1','4_t1','5_t1','6_t1','7_t1','8_t1','9_t1','10_t1','11_t1','12_t1','13_t1','14_t1'

                ,'15_t1','17_t1','18_t1','19_t1','20_t1','21_t1','22_t1','23_t1','24_t1','25_t1','26_t1','27_t1',

                 '28_t1','29_t1','30_t1',

                '2_t2','3_t2','4_t2','5_t2','6_t2','7_t2','8_t2','9_t2','10_t2','11_t2','12_t2','13_t2','14_t2'

                ,'15_t2','17_t2','18_t2','19_t2','20_t2','21_t2','22_t2','23_t2','24_t2','25_t2','26_t2','27_t2',

                 '28_t2','29_t2','30_t2']



economy_categories = {0:{'name':'eco','start':0,'end':5000},

                      1:{'name':'forcedPistols','start':5000,'end':10000},

                      2:{'name':'forcedSMGs','start':10000,'end':15000},

                      3:{'name':'forcedBuy','start':15000,'end':20000},

                      4:{'name':'fullBuy','start':20000,'end':50000}

                      }
for col in money_columns:

    for key, category in economy_categories.items():

        economy_df.loc[(economy_df[col]>category['start']) & (economy_df[col]<=category['end']),col] = key

    for key, category in economy_categories.items():

        economy_df.loc[economy_df[col]==key,col] = category['name']
def get_economy_stats(category):



    wins_by_side_t1 = pd.DataFrame([[0,0,0],[0,0,0]],index=['ct','t'],columns=['sum','count','mean'])

    wins_by_side_t2 = pd.DataFrame([[0,0,0],[0,0,0]],index=['ct','t'],columns=['sum','count','mean'])



    for _round in range(2,16):

        gb_1 = economy_df[economy_df[str(_round)+'_t1']==category].rename(columns={'t1_start':'side'}).groupby('side')[str(_round)+'_winner']

        gb_1 = gb_1.agg(['sum','count','mean'])



        gb_3 = economy_df[economy_df[str(_round+15)+'_t1']==category].rename(columns={'t2_start':'side'}).groupby('side')[str(_round+15)+'_winner']

        gb_3 = gb_3.agg(['sum','count','mean'])



        gb_1 = gb_1.reindex(['ct','t'], fill_value=0)

        gb_3 = gb_3.reindex(['ct','t'], fill_value=0)



        wins_by_side_t1 = wins_by_side_t1 + gb_1 + gb_3



    wins_by_side_t1['sum'] = 2*wins_by_side_t1['count']-wins_by_side_t1['sum']



    for _round in range(2,16):

        gb_2 = economy_df[economy_df[str(_round)+'_t2']==category].rename(columns={'t2_start':'side'}).groupby('side')[str(_round)+'_winner']

        gb_2 = gb_2.agg(['sum','count','mean'])



        gb_4 = economy_df[economy_df[str(_round+15)+'_t2']==category].rename(columns={'t1_start':'side'}).groupby('side')[str(_round+15)+'_winner']

        gb_4 = gb_4.agg(['sum','count','mean'])



        gb_2 = gb_2.reindex(['ct','t'], fill_value=0)

        gb_4 = gb_4.reindex(['ct','t'], fill_value=0)



        wins_by_side_t2 = wins_by_side_t2 + gb_2 + gb_4



    wins_by_side_t2['sum'] = wins_by_side_t2['sum']-wins_by_side_t2['count']



    wins_by_side = wins_by_side_t1 + wins_by_side_t2



    wins_by_side['mean'] = wins_by_side['sum']/wins_by_side['count']//0.001/10

    wins_by_side['num_per_game'] = wins_by_side['count']/economy_df.shape[0]//0.1/10

    wins_by_side = wins_by_side[['mean','num_per_game']]

    

    return wins_by_side
economy_stats = {}

mean_victories_df = pd.DataFrame(index=['ct','t'])

num_per_game_df = pd.DataFrame(index=['ct','t'])



for category in economy_categories.values():

    cat = category['name']

    economy_stats[cat] = get_economy_stats(cat)

    mean_victories_df[cat] = economy_stats[cat]['mean']

    num_per_game_df[cat] = economy_stats[cat]['num_per_game']



print('\nVictory probability (%):')

display(mean_victories_df)

print('\nNumber per game:')

display(num_per_game_df)
gb_team_1_first_pistol   = economy_df.rename(columns={'team_1':'team','t1_start':'side'}).groupby(['side','time_period','team'])['1_winner'].agg(['mean','count'])

gb_team_1_second_pistol  = economy_df.rename(columns={'team_1':'team','t2_start':'side'}).groupby(['side','time_period','team'])['16_winner'].agg(['mean','count'])



gb_team_2_first_pistol   = economy_df.rename(columns={'team_2':'team','t2_start':'side'}).groupby(['side','time_period','team'])['1_winner'].agg(['mean','count'])

gb_team_2_second_pistol  = economy_df.rename(columns={'team_2':'team','t1_start':'side'}).groupby(['side','time_period','team'])['16_winner'].agg(['mean','count'])
gb = (2-gb_team_1_first_pistol['mean'])*gb_team_1_first_pistol['count']+(

    2-gb_team_1_second_pistol['mean'])*gb_team_1_second_pistol['count']+(

    gb_team_2_first_pistol['mean']-1)*gb_team_2_first_pistol['count']+(

    gb_team_2_second_pistol['mean']-1)*gb_team_2_second_pistol['count']



total_pistols = (gb_team_1_first_pistol['count']+gb_team_1_second_pistol['count']+gb_team_2_first_pistol['count']+gb_team_2_second_pistol['count'])



for major in majors[3:]:

    col = major['tournament']

    

    _filter = total_pistols > total_pistols.loc[:,col].quantile(0.3)

    

    gb.loc[:,col] = gb.loc[_filter,col]

    total_pistols.loc[:,col] = total_pistols.loc[_filter,col]

    

    gb.dropna(inplace=True)

    total_pistols.dropna(inplace=True)



mean_pistols = pd.DataFrame(gb/total_pistols)

mean_pistols.dropna(inplace=True)

mean_pistols.sort_values(['side','time_period',0],ascending=[1,1,0],inplace=True)
def get_rankings_pistols_side(side):

    ranking_pistols_side = pd.DataFrame(index=range(1,8))

    ranking_pistols_side.index.name = 'ranking'



    for major in majors[3:]:

        col = major['tournament']

        ranking_pistols_side[col] = mean_pistols.loc[side,col][0][:7].index

    

    return ranking_pistols_side
print('\nRankings Pistols CT-side:')

display(get_rankings_pistols_side('ct'))

print('\nRankings Pistols T-side:')

display(get_rankings_pistols_side('t'))
players_df.head()
all_maps_columns = ['date','time_period','country','player_name','team','opponent','player_id',

                    'match_id','event_id','event_name','best_of']

each_map_columns = ['kills','assists','deaths','hs','flash_assists','kast','kddiff','adr','fkdiff','rating']
map1_columns = ['map_1']+['m1_'+ x for x in each_map_columns]

map2_columns = ['map_2']+['m2_'+ x for x in each_map_columns]

map3_columns = ['map_3']+['m3_'+ x for x in each_map_columns]
out_columns = all_maps_columns+['_map']+each_map_columns



players_df_by_map_columns = pd.DataFrame(columns=out_columns)
#Countries that contribute the most to the professional scene by number of matches

players_df.groupby('country')['country'].count().sort_values(ascending=False)[:30]
curr_map = {}

curr_map[0] = players_df[(all_maps_columns+map1_columns)]

curr_map[1] = players_df[(all_maps_columns+map2_columns)]

curr_map[2] = players_df[(all_maps_columns+map3_columns)]



curr_map[0].columns = out_columns

curr_map[1].columns = out_columns

curr_map[2].columns = out_columns



all_maps = pd.concat(   (   pd.concat(   (curr_map[0],curr_map[1])    ), curr_map[2]   )   )
gb2 = all_maps.groupby(['time_period','player_id','_map'])

threshold_maps_played = 7

all_maps2 = gb2.filter(lambda x:x.player_name.count()>threshold_maps_played)

all_maps2.head()
gb = all_maps2.groupby(['time_period','_map','player_name'],sort=False)['rating','kddiff'].mean()

rankings = gb.sort_values(['time_period','_map','rating'],ascending=[1,1,0])
rankings_players = {}

for _map in maps:

    rankings_players[_map] = pd.DataFrame(index=range(1,21))

    rankings_players[_map].index.name = 'ranking'

    

    for major in majors:

        col = major['tournament']

        try:

            rankings_players[_map][col] = rankings.loc[col,_map].rating[:20].index

        except:

            pass
for _map in maps:

    print('\n'+_map+':')

    display(rankings_players[_map])
ranking_players_df = pd.DataFrame()

ranking_players_df['player'] = players_df.player_name.unique()

ranking_players_df.set_index('player',inplace=True)



for major in majors:

    col = major['tournament']

    ranking_players_df[col] = 0

    

for _map in maps:

    for col in rankings_players[_map].columns:

        for i in range(1,21):

            ranking_players_df.loc[rankings_players[_map][col][i],col] += 21-i

            

rankings_players_again = {}



rankings_players_again = pd.DataFrame(index=range(1,21))

rankings_players_again.index.name = 'ranking'



for major in majors:

    col = major['tournament']

    rankings_players_again[col] = ranking_players_df[col].sort_values(ascending=False)[:20].index
rankings_players_again
results_df_rank_part_1 = results_df[['match_id','team_1','rank_1']].rename(columns={'team_1':'team','rank_1':'team_rank'})

results_df_rank_part_2 = results_df[['match_id','team_2','rank_2']].rename(columns={'team_2':'team','rank_2':'team_rank'})

results_df_rank = pd.concat((results_df_rank_part_1,results_df_rank_part_2))
all_maps3 = all_maps2.merge(results_df_rank,'left',on=['match_id','team'])
players_series = all_maps3.groupby('player_name').team_rank.min()

players_list = list(players_series[players_series<=3].index)
gb = all_maps3.groupby(['time_period','player_name','country'])['kills','deaths','team_rank']

gb = gb.mean()[gb.count()['kills']>100]

gb = gb[gb.index.get_level_values(1).isin(players_list)]



gb['kills'] = gb['kills'].round(1)

gb['deaths'] = gb['deaths'].round(1)

gb['team_rank'] = gb['team_rank'].round(0).astype('int')

gb.reset_index(inplace=True)
gb['region'] = ''

gb.loc[(gb['country']=='Ukraine') | (gb['country']=='Russia') | (gb['country']=='Kazakhstan'),'region'] = 'CIS'

gb.loc[(gb['country']=='Brazil'),'region'] = 'Brazil'

gb.loc[(gb['country']=='France') | (gb['country']=='Belgium'),'region'] = 'France/Belgium'

gb.loc[(gb['country']=='United States') | (gb['country']=='Canada'),'region'] = 'North America'

gb.loc[(gb['country']=='Denmark'),'region'] = 'Denmark'

gb.loc[(gb['country']=='Sweden'),'region'] = 'Sweden'

gb.loc[(gb['country']=='Poland'),'region'] = 'Poland'

gb.loc[gb.country.isin(['Netherlands','Slovakia','Bosnia and Herzegovina',

                        'Norway','Czech Republic','Spain','Estonia','United Kingdom','Portugal','Turkey','Bulgaria', 'Finland']),'region'] = 'Rest of Europe'



gb = gb.sort_values(['time_period','region'])
gb['size'] = (100/(gb['team_rank']+2)).round(1)
fig = px.scatter(gb, x="deaths", y="kills", animation_frame="time_period", animation_group="player_name",

           size="size", color="region", hover_name="player_name", hover_data=["team_rank"],

                 range_x=[14,24],range_y=[14,24]

                )

#.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 1300

fig.update_layout(xaxis_title='Deaths', yaxis_title = 'Kills')

fig.add_shape(type="line", x0=14, y0=14, x1=24, y1=24, line=dict(width=4, dash="dot"))

fig.update_shapes(dict(xref='x', yref='y'))

fig.show()
picks_df.head()
gb = picks_df.groupby('system').system.count()

gb = gb[gb>10]

gb