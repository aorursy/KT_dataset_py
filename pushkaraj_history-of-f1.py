from IPython.display import Image

import os

Image('../input/ferrari/1950to2019.jpg')
# import necessary libraries

import pandas as pd

import numpy as np

import os

import warnings

warnings.filterwarnings('ignore')



import sqlite3

from sqlalchemy import create_engine



import matplotlib.pyplot as plt



input_files=os.listdir('/kaggle/input/formula-1-world-championship-1950-2020')
# Set up a connection with database file to interact with the data base. If this file does not exist then it gets automatically created.

conn=sqlite3.connect("f1_info.db")
engine=create_engine('sqlite:///f1_info.db', echo=False)
def preprocess_cat_cols(df):

    cat_col=df.select_dtypes(include='object').columns

    for col in cat_col:

        for val in df[col].tolist():

            try:

                if '\\N' in val:

                    df[col].replace({'\\N':'nan'}, inplace=True)

                    break

            except:

                print('Column:',col,'Value:',val)

        df[col].str.strip()

        

        

def duplicate_index(df):

    dup=df.duplicated()

    indices=dup[dup==True].index

    return indices
input_files
ckt=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('circuits.csv')])

#Column alt has nothing but '\N' so we drop that column

preprocess_cat_cols(ckt)

print(ckt.head())

ckt.to_sql('circuit', con=engine, if_exists='replace')
input_files.index('constructors.csv')
constr=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('constructors.csv')])

preprocess_cat_cols(constr)

print(constr.head())

constr.to_sql('constructors', con=engine, if_exists='replace')
input_files.index('constructor_results.csv')
constr_rsl=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('constructor_results.csv')])

preprocess_cat_cols(constr_rsl)

print(constr_rsl.head())

constr_rsl.to_sql('constructor_results', con=engine, if_exists='replace')
input_files.index('constructor_standings.csv')
constr_std=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('constructor_standings.csv')])

preprocess_cat_cols(constr_std)

print(constr_std)

constr_std.to_sql('constr_std', con=engine, if_exists='replace')
input_files.index('drivers.csv')
drivers=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('drivers.csv')])



drivers['name']=drivers['forename']+' '+drivers['surname']

drivers.drop(['forename','surname'],axis=1,inplace=True)



preprocess_cat_cols(drivers)

print(drivers.head())

drivers.to_sql('drivers', con=engine, if_exists='replace')
input_files.index('driver_standings.csv')
drivers_std=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('driver_standings.csv')])

preprocess_cat_cols(drivers_std)

print(drivers_std)

drivers_std.to_sql('drivers_std', con=engine, if_exists='replace')
input_files.index('lap_times.csv')
lap_time=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('lap_times.csv')])

preprocess_cat_cols(lap_time)

print(lap_time.head())

lap_time.to_sql('lap_time', con=engine, if_exists='replace')
input_files.index('pit_stops.csv')
pit_stops=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('pit_stops.csv')])

preprocess_cat_cols(pit_stops)

print(pit_stops.head())

pit_stops.to_sql('pit_stops', con=engine, if_exists='replace')
input_files.index('qualifying.csv')
quali=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('qualifying.csv')])

preprocess_cat_cols(quali)

print(quali.head())

quali.to_sql('quali', con=engine, if_exists='replace')
input_files.index('races.csv')
races=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('races.csv')])

preprocess_cat_cols(races)

print(races)

races.to_sql('races', con=engine, if_exists='replace')
input_files.index('results.csv')
results=pd.read_csv('/kaggle/input/formula-1-world-championship-1950-2020/'+input_files[input_files.index('results.csv')])

results.position.replace({'\\N':1000},inplace=True)

results['position']=results.position.astype('int32').tolist()

grouped=results.groupby(by='raceId')['position']

values=grouped.transform(lambda x: len(x))

indices_to_replace=results[results.position==1000].index.tolist()

values_to_replace=values[indices_to_replace]

results['position'].iloc[indices_to_replace]=values_to_replace



preprocess_cat_cols(results)

print(results.head())

results.to_sql('results', con=engine, if_exists='replace')
# import necessary libraries

import pandas as pd

import numpy as np

from collections import Counter

import sqlite3

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

#plt.style.use('dark_background')

plt.style.use('ggplot')
query="""

        select name,r.constructorId,count(raceId) wins from constructors c

        join results r on c.constructorId=r.constructorId

        where r.position=1

        group by r.constructorId

        order by wins desc

"""
team_race_wins=pd.read_sql_query(query,conn).copy()

fig,ax=plt.subplots(figsize=(15,7))

ax=team_race_wins.wins.plot.bar(x='name',y='wins', color='tab:red')

ax.set_xticks(range(team_race_wins.shape[0]))

ax.set_xticklabels(team_race_wins.name, fontsize=12)

plt.xlabel('Constructors that have won atleast one race', fontsize=12)

plt.ylabel('Number of race wins', fontsize=12);
query="""

    select ssq.year, cs.name, max(ssq.total_pts) win_pts from

    (select sq.constructorId, sq.year, sum(sq.pts) total_pts from

    (select rs.constructorId,r.year,sum(rs.points) pts from results rs

    join races r on rs.raceId=r.raceId

    group by rs.raceId, rs.constructorId) sq

    group by sq.constructorId,sq.year) ssq

    join constructors cs

    on ssq.constructorId=cs.constructorId

    group by ssq.year

"""
q=pd.read_sql_query(query,conn)

constr_champs_by_year=q.copy()



constr_champs_by_team=constr_champs_by_year.name.value_counts()



fig,ax=plt.subplots(figsize=(15,7))

ax=constr_champs_by_team.plot.bar(color='tab:red')

ax.set_xticks(range(team_race_wins.shape[0]))

ax.set_xticklabels(team_race_wins.name, fontsize=12)

plt.ylabel('Number of constructors championships', fontsize=12)

plt.xlabel('Constructors', fontsize=15);
query="""

    select cs.name, driver_champs from

    (select sqq.constructorId, count(sqq.constructorId) driver_champs from

    (select sq.year, sq.driverId, sq.constructorId, max(pts) season_pts from

    (select rs.driverId, rs.constructorId, r.year, sum(rs.points) pts from results rs

    join races r on rs.raceId=r.raceId

    group by rs.driverId, r.year) sq

    group by sq.year) sqq

    group by sqq.constructorId) sqqq join

    constructors cs on sqqq.constructorId=cs.constructorId

    order by driver_champs desc

"""
q=pd.read_sql_query(query,conn)

driver_champs_by_team=q.copy()



fig,ax=plt.subplots(figsize=(13,8))

driver_champs_by_team.plot.bar(x='name',y='driver_champs',ax=ax, color='tab:red')

ax.set_xlabel('Constructors',fontsize=14)

ax.set_ylabel('Number of driver championships won by constructor',fontsize=14);
query="""

    select sssq.name, sq2.num_seasons ,sssq.num_champ from

    (select ssq.name, count(ssq.name) num_champ from

    (select sq.year, sq.name, max(sq.season_points) win_points from

    (select r.year, cs.name, sum(rs.points) season_points from results rs join

    races r on rs.raceId=r.raceId join

    constructors cs on rs.constructorId=cs.constructorId

    group by cs.name, r.year) sq

    group by sq.year) ssq

    group by ssq.name) sssq join 

    

    (select sq1.name, count(sq1.year) num_seasons from

    (select r.year, cs.name from results rs join

    races r on rs.raceId=r.raceId join

    constructors cs on rs.constructorId=cs.constructorId

    group by cs.name, r.year) sq1

    group by sq1.name) sq2

    

    on

    

    sssq.name=sq2.name

"""

q=pd.read_sql_query(query,conn)

const_champs_seasons=q.copy()

const_champs_seasons['perc_season_champs']=round((const_champs_seasons['num_champ']/const_champs_seasons['num_seasons'])*100,2)

const_champs_seasons.sort_values(by=['perc_season_champs'],ascending=False,inplace=True)



fig=plt.figure(figsize=(17,17))

fig.tight_layout()



ax1=fig.add_subplot(221)

ax2=fig.add_subplot(222)

ax3=fig.add_subplot(212)



const_champs_seasons.plot.barh(x='name',y='num_seasons',ax=ax1, color='darkorange' ,label='Number of seasons participated')

const_champs_seasons.plot.barh(x='name',y='num_champ',ax=ax2, sharey=ax1, color='tab:red' ,label='Number of championships won')

const_champs_seasons.plot.bar(x='name',y='perc_season_champs',ax=ax3, color='turquoise' ,label='Percentage of championship wins');
champ_teams=constr_champs_by_team.index

ohe_teams=pd.get_dummies(constr_champs_by_year.name)

champ_teams_by_year=dict()

for team in champ_teams:    

    champ_teams_by_year[team]=np.cumsum(ohe_teams[team])



# plot the chart

fig,ax=plt.subplots(figsize=(15,9))

for team,performance in champ_teams_by_year.items():

    ax.plot(performance)

    ax.scatter(range(len(performance)),performance,label=team)

ax.set_ylabel('Number of championships', fontsize=14)

ax.set_xlabel('Season Year', fontsize=14)

ax.set_xticks(range(constr_champs_by_year.shape[0]))

ax.set_xticklabels(constr_champs_by_year.year.tolist(), rotation='vertical', fontsize=15)

plt.legend();
query="""

    select sqq.year, cs.name from

    (select sq.year, sq.driverId, sq.constructorId, max(pts) season_pts from

    (select rs.driverId, rs.constructorId, r.year, sum(rs.points) pts from results rs

    join races r on rs.raceId=r.raceId

    group by rs.driverId, r.year) sq

    group by sq.year) sqq join

    constructors cs on

    cs.constructorId=sqq.constructorId

    order by sqq.year

"""
q=pd.read_sql_query(query,conn)

drivers_champs_by_team_yearly=q.copy()



champ_teams_for_drivers=drivers_champs_by_team_yearly.name.tolist()

ohe_champ_teams_for_drivers=pd.get_dummies(drivers_champs_by_team_yearly)

champ_teams_for_drivers=dict()

for team in ohe_champ_teams_for_drivers.columns[1:]:

    champ_teams_for_drivers[team.split('_')[1]]=np.cumsum(ohe_champ_teams_for_drivers[team])



# plot the graph

fig,ax=plt.subplots(figsize=(15,9))

for team,driver_champs in champ_teams_for_drivers.items():

    ax.plot(driver_champs, label=team)

    ax.scatter(range(drivers_champs_by_team_yearly.shape[0]),driver_champs)

ax.set_xticks(range(drivers_champs_by_team_yearly.shape[0]))

ax.set_xticklabels(drivers_champs_by_team_yearly.year.tolist(), rotation='vertical', fontsize=15)

plt.legend();
query="""

    select sssq.name, sq2.num_seasons ,sssq.num_champ from

    (select ssq.name, count(ssq.name) num_champ from

    (select sq.year, sq.name, max(sq.season_points) win_points from

    (select r.year, dr.name, sum(rs.points) season_points from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    group by dr.name, r.year) sq

    group by sq.year) ssq

    group by ssq.name) sssq join 

    

    (select sq1.name, count(sq1.year) num_seasons from

    (select r.year, dr.name from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    group by dr.name, r.year) sq1

    group by sq1.name) sq2

    

    on

    

    sssq.name=sq2.name

"""
q=pd.read_sql_query(query,conn)

dr_champs_seasons=q.copy()

dr_champs_seasons['perc_season_champs']=round((dr_champs_seasons['num_champ']/dr_champs_seasons['num_seasons'])*100,2)

dr_champs_seasons.sort_values(by=['perc_season_champs'],ascending=False,inplace=True)



fig=plt.figure(figsize=(17,17))

fig.tight_layout()



ax1=fig.add_subplot(221)

ax2=fig.add_subplot(222)

ax3=fig.add_subplot(212)



dr_champs_seasons.plot.barh(x='name',y='num_seasons',ax=ax1)

ax1.set_xticks(range(1,max(dr_champs_seasons['num_seasons'])+1))

ax1.set_title('Number season participated')



dr_champs_seasons.plot.barh(x='name',y='num_champ',ax=ax2)

ax2.set_title('Number of driver championships won')



dr_champs_seasons.plot.bar(x='name',y='perc_season_champs',ax=ax3)

ax3.set_title('Percentage of championships won');
query="""

    select sq.name, count(*) num_wins from

    (select dr.name from results rs join

    drivers dr on rs.driverId=dr.driverId

    where rs.position==1) sq

    group by sq.name

    order by num_wins desc limit(25)

"""
dr_race_wins=pd.read_sql_query(query,conn)

fig,ax=plt.subplots(figsize=(14,6))

dr_race_wins.plot.bar(x='name',y='num_wins',ax=ax)

ax.set_title('Drivers by number race wins')

ax.set_ylabel('Number of race wins');
query="""

    select dr.name, count(*) num_race_wins, sq.debut_season from results rs

    

    join

    

    races r on rs.raceId=r.raceId

    

    join

    

    drivers dr on rs.driverId=dr.driverId

    

    join

    

    (select rs.driverId, min(r.year) debut_season from results rs join

    races r on rs.raceId=r.raceId

    group by rs.driverId) sq

    

    on

    

    rs.driverId=sq.driverId and r.year=sq.debut_season

    

    where rs.position=1

    

    group by dr.name

    

    order by num_race_wins desc

"""
dr_debut_wins=pd.read_sql_query(query,conn)

fig,ax=plt.subplots(figsize=(13,6))

dr_debut_wins.plot.bar(x='name',y='num_race_wins',ax=ax)

ax.set_title('Drivers that won races in their debut season')



xlocs, xlabs = plt.xticks()

for i,v in enumerate(dr_debut_wins.debut_season):

    plt.text(xlocs[i]-0.2,dr_debut_wins.num_race_wins[i]+0.05, str(v))
query="""

    select dr.name, count(*) num_podiums, sq.debut_season from results rs

    

    join

    

    races r on rs.raceId=r.raceId

    

    join

    

    drivers dr on rs.driverId=dr.driverId

    

    join

    

    (select rs.driverId, min(r.year) debut_season from results rs join

    races r on rs.raceId=r.raceId

    group by rs.driverId) sq

    

    on

    

    rs.driverId=sq.driverId and r.year=sq.debut_season

    

    where rs.position<4

    

    group by dr.name

    

    order by num_podiums desc

"""
dr_debut_podiums=pd.read_sql_query(query,conn)

fig,ax=plt.subplots(figsize=(20,6))

dr_debut_podiums.plot.bar(x='name',y='num_podiums',ax=ax)

ax.set_title('Drivers with number of podium finishes in a debut season');



xlocs, xlabs = plt.xticks()

for i,v in enumerate(dr_debut_podiums.debut_season):

    plt.text(xlocs[i]-0.2,dr_debut_podiums.num_podiums[i]+0.05, str(v), rotation=35)
query="""

    select sq.name, count(*) num_pole_positions from

    (select dr.name from results rs join

    drivers dr on rs.driverId=dr.driverId

    where rs.grid==1) sq

    group by sq.name

    order by num_pole_positions desc limit(25)

"""
q=pd.read_sql_query(query,conn)

fig,ax=plt.subplots(figsize=(14,6))

q.plot.bar(x='name',y='num_pole_positions',ax=ax)

ax.set_title('Drivers by number of pole positions')

ax.set_ylabel('Number of pole positions');
query="""

    select dr.name, count(*) num_poles, sq.debut_season from results rs

    

    join

    

    races r on rs.raceId=r.raceId

    

    join

    

    drivers dr on rs.driverId=dr.driverId

    

    join

    

    (select rs.driverId, min(r.year) debut_season from results rs join

    races r on rs.raceId=r.raceId

    group by rs.driverId) sq

    

    on

    

    rs.driverId=sq.driverId and r.year=sq.debut_season

    

    where rs.grid=1

    

    group by dr.name

    

    order by num_poles desc

"""
dr_debut_poles=pd.read_sql_query(query,conn)



fig,ax=plt.subplots(figsize=(13,6))

dr_debut_poles.plot.bar(x='name',y='num_poles',ax=ax)

ax.set_title('Drivers number of pole positions debut season')



xlocs, xlabs = plt.xticks()

for i,v in enumerate(dr_debut_poles.debut_season):

    plt.text(xlocs[i]-0.2,dr_debut_poles.num_poles[i]+0.05, str(v))
query="""

    select dr.name driver_name, ckt.location circuit_name, max(sq.ckt_wins) most_ckt_wins from

    

    (select rs.driverId, r.circuitId, count(*) ckt_wins from results rs join

    races r on rs.raceId=r.raceId join

    circuit ckt on r.circuitId=ckt.circuitId

    where rs.position=1

    group by rs.driverId, r.circuitId) sq

    

    join

    

    drivers dr on sq.driverId=dr.driverId

    

    join

    

    circuit ckt on sq.circuitId=ckt.circuitId

    

    group by sq.driverId

    having most_ckt_wins>1

    order by most_ckt_wins desc

"""
dr_most_ckt_wins=pd.read_sql_query(query,conn)



fig,ax=plt.subplots(figsize=(17,6))



dr_most_ckt_wins.plot.bar(x='driver_name',y='most_ckt_wins',ax=ax)



xlocs, xlabs = plt.xticks()

for i,v in enumerate(dr_most_ckt_wins.circuit_name):

    plt.text(xlocs[i]-0.2,dr_most_ckt_wins.most_ckt_wins[i]+0.05, str(v), rotation=45);
query="""

    select dr.name driver_name, ckt.location circuit_name, max(sq.ckt_poles) most_ckt_poles from

    

    (select rs.driverId, r.circuitId, count(*) ckt_poles from results rs join

    races r on rs.raceId=r.raceId join

    circuit ckt on r.circuitId=ckt.circuitId

    where rs.grid=1

    group by rs.driverId, r.circuitId) sq

    

    join

    

    drivers dr on sq.driverId=dr.driverId

    

    join

    

    circuit ckt on sq.circuitId=ckt.circuitId

    

    group by sq.driverId

    having most_ckt_poles>1

    

    order by most_ckt_poles desc

"""
dr_most_ckt_poles=pd.read_sql_query(query,conn)



fig,ax=plt.subplots(figsize=(17,6))



dr_most_ckt_poles.plot.bar(x='driver_name',y='most_ckt_poles',ax=ax)



xlocs, xlabs = plt.xticks()

for i,v in enumerate(dr_most_ckt_poles.circuit_name):

    plt.text(xlocs[i]-0.2,dr_most_ckt_poles.most_ckt_poles[i]+0.05, str(v), rotation=45);
query="""

    select count(distinct(r.circuitId)) distinct_ckt_wins, dr.name from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    where rs.position=1

    group by rs.driverId

    order by distinct_ckt_wins desc

"""
dr_ckt_wins=pd.read_sql_query(query,conn)

dr_ckt_wins=dr_ckt_wins[dr_ckt_wins.distinct_ckt_wins>=10]

fig,ax=plt.subplots(figsize=(15,6))

dr_ckt_wins.plot.bar(x='name',y='distinct_ckt_wins',ax=ax)

ax.set_title('Drivers by number of distinct circuit wins');
query="""

    select count(distinct(r.circuitId)) distinct_ckt_poles, dr.name from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    where rs.grid=1

    group by rs.driverId

    order by distinct_ckt_poles desc

"""
dr_ckt_poles=pd.read_sql_query(query,conn)

dr_ckt_poles=dr_ckt_poles[dr_ckt_poles.distinct_ckt_poles>=10]

fig,ax=plt.subplots(figsize=(15,6))

dr_ckt_poles.plot.bar(x='name',y='distinct_ckt_poles',ax=ax)

ax.set_title('Drivers by number of distinct circuit poles');
query="""

    select count(distinct(r.year)) seasons_with_win, dr.name from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    where rs.position=1

    group by dr.name

    having seasons_with_win>1

    order by seasons_with_win desc

"""
dr_season_with_wins=pd.read_sql_query(query,conn)

fig,ax=plt.subplots(figsize=(15,6))

dr_season_with_wins.plot.bar(x='name',y='seasons_with_win',ax=ax)

ax.set_title('Drivers by number of season with race wins');
query="""

    select dr.name, rs.position from drivers dr join

    results rs on dr.driverId=rs.driverId

    where dr.name in

    (select distinct(dr.name) from drivers dr join

    results rs on dr.driverId=rs.driverId

    where rs.position<4)

"""
q=pd.read_sql_query(query,conn)

q.rename(columns={'position':'points'}, inplace=True)

avg_pts_df=q.groupby('name').filter(lambda x: len(x)>6)



oop=avg_pts_df[avg_pts_df.points>10].index

avg_pts_df.points[oop]=0

avg_pts_df['points'].replace({1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1},inplace=True)

avg_pts_df=pd.DataFrame(avg_pts_df.groupby(by='name',as_index=False).mean())

avg_pts_df.sort_values(by='points',ascending=False, inplace=True)



fig,ax=plt.subplots(figsize=(15,6))

avg_pts_df[:20].plot.bar(x='name',y='points',ax=ax)

ax.set_ylabel('Average points score')

ax.set_xlabel('Driver name')

ax.set_title('Top 20 drivers by average points scored according to 2019 points system');
query="""

    select r.name track, count(*) races_held from races r

    group by track

    order by races_held desc

"""
q=pd.read_sql_query(query,conn)

q.rename(columns={'races_held':'Number of races'},inplace=True)



reace_tracks=q.copy()



fig,ax=plt.subplots(figsize=(14,7))

reace_tracks.plot.bar(x='track',y='Number of races',ax=ax);
query="""

    select r.circuitId, ssq.name, ssq.cnt, count(r.circuitId) race_count from races r join

    (select ckt.circuitId, ckt.name, count(sq.circuitId) cnt from circuit ckt join

    (select rs.raceId, r.circuitId from races r join

    results rs on r.raceId=rs.raceId

    where rs.grid=1 and rs.position=1) sq

    on ckt.circuitId=sq.circuitId

    group by sq.circuitId) ssq

    on r.circuitId=ssq.circuitId

    group by r.circuitId 

"""
q=pd.read_sql_query(query,conn)



q['percentage_win']=round((q['cnt']/q['race_count'])*100,2)

q.drop(['circuitId','cnt'],axis=1,inplace=True)

q=q.sort_values(by=['percentage_win'],ascending=False)





fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(13,6))

fig.tight_layout()

q.plot(x='name',y='race_count',ax=ax, rot=90, color='k',label='Total number of race events on')

q.plot.bar(x='name',y='percentage_win',ax=ax, rot=90,label='Probability of winning from the pole position');
query="""

    select ssq.name, count(r.raceId) total_races, ssq.cnt from races r join

    (select ckt.name, sq.circuitId, count(sq.raceId) cnt from circuit ckt join

    (select rs.raceId, r.circuitId from races r join

    results rs on r.raceId=rs.raceId

    where rs.grid=1 and rs.position!=1) sq

    on ckt.circuitId=sq.circuitId

    group by ckt.name) ssq

    on r.circuitId=ssq.circuitId

    group by ssq.name

"""
q=pd.read_sql_query(query,conn)

q['percentage_lose']=q['cnt']/q['total_races']

q.drop(['cnt'],axis=1,inplace=True)

q=q.sort_values(by=['percentage_lose'],ascending=False)



q['scaled_total_races']=q['total_races']/max(q['total_races'])



fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(13,6))

fig.tight_layout()

q.plot(x='name',y='scaled_total_races',ax=ax, rot=90, color='k', label='Relative total number of races on a track (scaled)')

q.plot.bar(x='name',y='percentage_lose',ax=ax, rot=90, label='Probability of not winning from a pole position on a track');
query="""

    select dr.name, sq.races from drivers dr

    join

    (select driverId, count(*) races from results rs join

    races r on rs.raceId=r.raceId

    where r.year>1999 and r.year<2010

    group by driverId) sq

    on dr.driverId=sq.driverId

    order by sq.races desc

"""
q=pd.read_sql_query(query,conn)

q.rename(columns={'name':'driver'},inplace=True)



v8_drivers_races=q.copy()



fig,ax=plt.subplots(figsize=(15,5))



v8_drivers_races.plot.bar(x='driver',y='races',ax=ax)

ax.set_title('Number of race appearnces by drivers in 2000-2009')

ax.set_xlabel('Drivers',fontsize=14)

ax.set_ylabel('Number of races',fontsize=14);
query="""

    select dr.name, sq.races from drivers dr

    join

    (select driverId, count(*) races from results rs join

    races r on rs.raceId=r.raceId

    where r.year>2009 and r.year<2020

    group by driverId) sq

    on dr.driverId=sq.driverId

    order by sq.races desc

"""
q=pd.read_sql_query(query,conn)

q.rename(columns={'name':'driver'},inplace=True)



v6_drivers_races=q.copy()

fig,ax=plt.subplots(figsize=(15,5))

v6_drivers_races.plot.bar(x='driver',y='races',ax=ax)

ax.set_title('Number of race appearnces by drivers in 2010-2019')

ax.set_xlabel('Drivers',fontsize=14)

ax.set_ylabel('Number of races',fontsize=14);
fig,ax=plt.subplots(figsize=(10,5))

ax.bar(x=[1,2],height=[v8_drivers_races.shape[0],v6_drivers_races.shape[0]],width=0.5)

ax.set_xticks([1,2])

ax.set_xticklabels(['2000-2009','2010-2019'])

ax.set_ylabel('Number of unique drivers',fontsize=12);
query="""

    select dr.name, count(*) wins from results rs join

    drivers dr on rs.driverId=dr.driverId

    where rs.raceId in (select distinct(r.raceId) from races r

    where r.year>1999 and r.year<2010) and rs.position=1

    group by dr.name

    order by wins desc

"""
q=pd.read_sql_query(query,conn)

q.rename(columns={'name':'driver'},inplace=True)



winners_v8=q.copy()



fig,ax=plt.subplots(figsize=(14,5))

winners_v8.plot.bar(x='driver',y='wins',ax=ax)

ax.set_title('Number of race wins by drivers in 2000-2009',fontsize=14)

ax.set_xlabel('Drivers',fontsize=14)

ax.set_ylabel('Number of race wins',fontsize=14);
query="""

    select dr.name, count(*) wins from results rs join

    drivers dr on rs.driverId=dr.driverId

    where rs.raceId in (select distinct(r.raceId) from races r

    where r.year>2009 and r.year<2020) and rs.position=1

    group by dr.name

    order by wins desc

"""
q=pd.read_sql_query(query,conn)

q.rename(columns={'name':'driver'},inplace=True)



winners_v6=q.copy()



fig,ax=plt.subplots(figsize=(14,5))

winners_v6.plot.bar(x='driver',y='wins',ax=ax)

ax.set_title('Number of race wins by drivers in 2010-2019',fontsize=14)

ax.set_xlabel('Drivers',fontsize=14)

ax.set_ylabel('Number of race wins',fontsize=14);
fig,ax=plt.subplots(figsize=(10,5))

ax.bar(x=[1,2],height=[winners_v8.shape[0],winners_v6.shape[0]],width=0.5)

ax.set_xticks([1,2])

ax.set_xticklabels(['2000-2009','2010-2019'])

ax.set_ylabel('Number of drivers that one races',fontsize=12);
v8_driver_perc_win=pd.merge(winners_v8,v8_drivers_races,how='inner',on='driver')



v8_driver_perc_win['Percentage Win']=round((v8_driver_perc_win['wins']/v8_driver_perc_win['races'])*100,2)

v8_driver_perc_win.drop(['wins','races'],axis=1,inplace=True)



v8_driver_perc_win=v8_driver_perc_win.sort_values(by='Percentage Win', ascending=False)



fig,ax=plt.subplots(figsize=(14,5))

v8_driver_perc_win.plot.bar(x='driver',y='Percentage Win',ax=ax)

ax.set_title("Drivers by percentages of race wins (2000-2009)",fontsize=14)

ax.set_ylabel('Win percentage',fontsize=14)

ax.set_xlabel('Drivers',fontsize=14);
v6_driver_perc_win=pd.merge(winners_v6,v6_drivers_races,how='inner',on='driver')



v6_driver_perc_win['Percentage Win']=round((v6_driver_perc_win['wins']/v6_driver_perc_win['races'])*100,2)

v6_driver_perc_win.drop(['wins','races'],axis=1,inplace=True)



v6_driver_perc_win=v6_driver_perc_win.sort_values(by='Percentage Win', ascending=False)



fig,ax=plt.subplots(figsize=(14,5))

v6_driver_perc_win.plot.bar(x='driver',y='Percentage Win',ax=ax)

plt.title("Drivers by percentages of race wins (2010-2019)",fontsize=14)

ax.set_ylabel('Win percentage',fontsize=14)

ax.set_xlabel('Drivers',fontsize=14);
query="""

    select sq.year, sq.name, max(sq.season_score) winning_score from 

    (select r.year, dr.name, sum(rs.points) season_score from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    where r.year>1999 and r.year<2010

    group by rs.driverId, r.year) sq

    group by sq.year

"""
q=pd.read_sql_query(query,conn)



v8_champions=q.copy()



v8_champions.rename(columns={'year':'championship','name':'driver'},inplace=True)



fig,ax=plt.subplots(figsize=(13,5))

v8_champions[['driver','championship']].groupby(by='driver',as_index=False).count().plot.bar(x='driver',y='championship',ax=ax)

ax.set_title("Driver's championships in 2000-2009",fontsize=14)

ax.set_xlabel('Drivers',fontsize=14)

ax.set_ylabel('Number of championships',fontsize=14);
query="""

    select sq.year, sq.name, max(sq.season_score) winning_score from 

    (select r.year, dr.name, sum(rs.points) season_score from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    where r.year>2009 and r.year<2020

    group by rs.driverId, r.year) sq

    group by sq.year

"""
q=pd.read_sql_query(query,conn)



v6_champions=q.copy()



v6_champions.rename(columns={'year':'championship','name':'driver'},inplace=True)



fig,ax=plt.subplots(figsize=(13,5))

v6_champions[['driver','championship']].groupby(by='driver',as_index=False).count().plot.bar(x='driver',y='championship',ax=ax)

ax.set_title("Driver's championships in (2010-2019)",fontsize=14)

ax.set_xlabel('Drivers',fontsize=14)

ax.set_ylabel('Number of championships',fontsize=14);
query="""

    select dr.name, count(qu.position) pole_position from quali qu join

    races r on qu.raceId=r.raceId join

    drivers dr on qu.driverId=dr.driverId

    where r.year>1999 and r.year<2010 and qu.position=1

    group by dr.name

    order by pole_position desc

"""
q=pd.read_sql_query(query,conn)

q.rename({'name':'driver'})



fig,ax=plt.subplots(figsize=(14,5))

q.plot.bar(x='name',y='pole_position',ax=ax)

ax.set_title('Number of pole positions by driver in 2000-2009',fontsize=14)

ax.set_xlabel('Drivers',fontsize=14)

ax.set_ylabel('Number of poles',fontsize=14);
query="""

    select dr.name, count(qu.position) pole_position from quali qu join

    races r on qu.raceId=r.raceId join

    drivers dr on qu.driverId=dr.driverId

    where r.year>2009 and r.year<2020 and qu.position=1

    group by dr.name

    order by pole_position desc

"""
q=pd.read_sql_query(query,conn)

q.rename({'name':'driver'})



fig,ax=plt.subplots(figsize=(14,5))

q.plot.bar(x='name',y='pole_position',ax=ax)

ax.set_title('Number of pole positions by driver in 2010-2019')

ax.set_xlabel('Drivers',fontsize=14)

ax.set_ylabel('Number of poles',fontsize=14);
query="""

    select dr.name, min(r.year) debut_year, ckt.name from results rs join

    drivers dr on rs.driverId=dr.driverId join

    races r on r.raceId=rs.raceId join

    circuit ckt on r.circuitId=ckt.circuitId

    where dr.name='Lewis Hamilton' or dr.name='Sebastian Vettel'

    group by dr.name

"""

q=pd.read_sql_query(query,conn)

q
query="""

    select r.round, r.year, dr.name driver, 

    sum(rs.points) over (partition by r.year, dr.name order by r.round) as season_score

    from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    where (dr.name='Lewis Hamilton' or dr.name='Sebastian Vettel') and r.year>=2007 and r.year<2020

    order by r.year, r.round

"""
q=pd.read_sql_query(query,conn)



lewise_seb=q.copy()

years_of_champs=lewise_seb.year.tolist()

index=[]

yrs=sorted(list(set(years_of_champs)))

for ind in yrs:

    index.append(years_of_champs.index(ind))

    

fig,ax=plt.subplots(figsize=(15,10))

lewise_seb.groupby(by='driver')['season_score'].plot.line(ax=ax)

for ind in index:

    ax.plot([ind]*2,[0,400], ls='--',label='Season:'+str(years_of_champs[ind]),c='black', alpha=0.7)

ax.set_xlabel('Races',fontsize=14)

ax.set_ylabel('Cummulative Score of the season',fontsize=14)

ax.set_title('Season performances of Lewis and Sebastian from 2007-2019',fontsize=14)

plt.legend();
query="""

    select r.year, dr.name, round((sum(rs.points)/sq.max_season_race_points_possible)*100,2) percentage_of_maximum_race_points_possible from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId join

    (select r.year, (round(max(rs.points)/5)*5)*count(distinct(rs.raceId)) max_season_race_points_possible from results rs join

    races r on rs.raceId=r.raceId join

    drivers dr on rs.driverId=dr.driverId

    where r.year>2006 and r.year<2020

    group by r.year) sq on r.year=sq.year

    where (r.year>2006 and r.year<2020) and (dr.name='Lewis Hamilton' or dr.name='Sebastian Vettel')

    group by r.year, rs.driverId

"""
q=pd.read_sql_query(query,conn)



season_winning_score_precentage=q



fig,ax=plt.subplots(figsize=(10,5))



ax.plot(range(2007,2020),list(season_winning_score_precentage.percentage_of_maximum_race_points_possible[season_winning_score_precentage.name=='Lewis Hamilton']), label='Lewis Hamilton')

ax.plot(range(2007,2020),list(season_winning_score_precentage.percentage_of_maximum_race_points_possible[season_winning_score_precentage.name=='Sebastian Vettel']), label='Sebastian Vettel')

ax.set_ylabel('Percentage of maximum possible points scored by a driver',fontsize=10)

ax.set_xlabel('Seasons 2007-2019',fontsize=14)

plt.legend();
query="""

    select dr.name, rs.position from results rs join

    drivers dr on rs.driverId=dr.driverId join

    races r on rs.raceId=r.raceId

    where dr.name in ('Sebastian Vettel','Lewis Hamilton')

    group by r.year, r.round, dr.name

"""
q=pd.read_sql_query(query,conn)



q.rename(columns={'position':'points'},inplace=True)



oop=q[q.points>10].index

q.points.iloc[oop]=0



q.points.replace({1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1},inplace=True)



lew_pts=np.cumsum(q[q.name=='Lewis Hamilton'].points)

seb_pts=np.cumsum(q[q.name=='Sebastian Vettel'].points)



fig,ax=plt.subplots(figsize=(15,6))

ax.plot(lew_pts,label='Lewis Hamilton')

ax.plot(seb_pts,label='Sebastian Vettel')

ax.set_xlabel('Races in sequence from 2007',fontsize=14)

ax.set_ylabel('Cummulative points scored by driver ( as per 2019 pointing system)',fontsize=10)

plt.legend();