# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



!pip install psycopg2



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle # data processing, pickle file I/O

import json # data processing, json file I/O

import random # data quality testing

import psycopg2 # establish aws Redshift connection

import sqlalchemy # copy pd dataframe to Redshift 

from sqlalchemy.types import * # load staging_tables



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
match_df = pd.read_pickle('../input/league-of-legendslol-ranked-games-2020-ver1/match_data_version1.pickle')

winner_df =  pd.read_pickle('../input/league-of-legendslol-ranked-games-2020-ver1/match_winner_data_version1.pickle')

loser_df = pd.read_pickle('../input/league-of-legendslol-ranked-games-2020-ver1/match_loser_data_version1.pickle')
match_df.info()
match_df['gameVersion'].iloc[0]
winner_df.info()
loser_df.info() 
meta_champs = pd.read_json('../input/league-of-legendslol-data-dragon-en-us10151/en_US-10.15.1/meta_champion.json')
meta_champs.iloc[0]['data']
with open('../input/league-of-legendslol-data-dragon-en-us10151/en_US-10.15.1/meta_item.json') as f:

    data = json.load(f)

meta_items = pd.read_json(json.dumps(data['data']), orient='index')
meta_items.info()
meta_items.iloc[0]
loser_df['win'].isna().sum()
loser_df[loser_df['win'].isna() == True]
missingVals = loser_df[loser_df['win'].isna() == True]['gameId'].tolist()

winner_df[winner_df['gameId'].isin(missingVals)]
match_df[match_df['gameId'].isin(missingVals)]
randomlist = []

for i in range(10):

    n = random.randint(0, 108828)

    randomlist.append(n)

print(randomlist)
match_df[match_df['gameId'].isin(loser_df.iloc[randomlist]['gameId']).tolist()]
avg_gameDuration = np.average(match_df[match_df['gameId'].isin(loser_df.iloc[randomlist]['gameId']).tolist()]['gameDuration'].values)

print('if unit in mins: {:.2f} mins'.format(avg_gameDuration))

print('if unit in sec: {:.2f} mins'.format(avg_gameDuration/60))

print('if unit in milisec: {:.2f} mins'.format(avg_gameDuration * 1.6666666666667E-5))
avg_NaNgameDura = np.average(match_df[match_df['gameId'].isin(missingVals)]['gameDuration'].values)

print('NaN games avg: {:.2f} mins'.format(avg_NaNgameDura/60))
winner_df = winner_df[~winner_df['gameId'].isin(missingVals)]

loser_df = loser_df[~loser_df['gameId'].isin(missingVals)]

match_df = match_df[~match_df['gameId'].isin(missingVals)]
match_df['gameMode'].unique()
gameId_CLASSIC =  match_df[match_df['gameMode'] == 'CLASSIC'].gameId.tolist()
winner_df = winner_df[winner_df['gameId'].isin(gameId_CLASSIC)]

loser_df = loser_df[loser_df['gameId'].isin(gameId_CLASSIC)]

match_df = match_df[match_df['gameId'].isin(gameId_CLASSIC)]
match_df.gameMode.unique()
gameId_remake = match_df.query('gameDuration <= 15*60').gameId.values.tolist()
winner_df = winner_df[~winner_df['gameId'].isin(gameId_remake)]

loser_df = loser_df[~loser_df['gameId'].isin(gameId_remake)]

match_df = match_df[~match_df['gameId'].isin(gameId_remake)]
print('now the shortest game in df is {:.2f} mins'.format(match_df.gameDuration.min() / 60))
# kaggle's add-in is used to store Postgres database's access info

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

dbname= user_secrets.get_secret("dbname")

host = user_secrets.get_secret("host")

password = user_secrets.get_secret("password")

port = user_secrets.get_secret("port")

user = user_secrets.get_secret("user")



conn = psycopg2.connect("host={} dbname={} user={} password={} port={}".format(host, dbname, user, password, port))

conn.autocommit = True

cur = conn.cursor()
# in the meta_dfs, we only need the key-name mapping, all other columns are dropped

meta_items = meta_items[['name']]

meta_champs['key'] = meta_champs.apply(lambda row: row.data['key'], axis=1)

meta_champs = meta_champs[['key']]
# column name/datatype of each staging table defined, used in loading_staging_tables

match_dict = {'gameCreation': Float(), 'gameDuration': Float(), 'gameId': Float(), 'gameMode': String(), 'gameType': String(), 'gameVersion': String(), \

             'mapId': Float(), 'participantIdentities': JSON(), 'participants': JSON(), 'platformId': String(), 'queueId': Float(), 'sessionId': Float(), \

             'status.message': String(), 'status.status_code': Float()}

winner_dict = {'teamId': Integer(), 'win': String(), 'firstBlood': Boolean(), 'firstTower': Boolean(), 'firstInhibitor': Boolean(), 'firstBaron': Boolean(), \

              'firstDragon': Boolean(), 'firstRiftHerald': Boolean(), 'towerKills': Integer(), 'inhibitorKills': Integer(), 'baronKills': Integer(), \

              'dragonKills': Integer(), 'vilemawKills': Integer(), 'riftHeraldKills': Integer(), 'dominionVictoryScore': Integer(), 'bans': JSON(), \

              'gameId': Float()}

loser_dict = winner_dict
# drop tables outlined in the 'data_modeling.pdf', in case a restart is needed

def drop_tables(cur):

    games_table_drop = "DROP TABLE IF EXISTS games"

    champions_table_drop = "DROP TABLE IF EXISTS champions"

    items_table_drop = "DROP TABLE IF EXISTS items"

    objectives_visions_table_drop = "DROP TABLE IF EXISTS objectives_visions"

    champion_key_table_drop = "DROP TABLE IF EXISTS champion_key"

    item_key_table_drop = "DROP TABLE IF EXISTS item_key"

    

    # execute all queries defined

    drop_table_queries = [games_table_drop, champions_table_drop, items_table_drop, objectives_visions_table_drop, champion_key_table_drop, item_key_table_drop]

    for query in drop_table_queries:

        cur.execute(query)
# drop tables outlined in the 'data_modeling.pdf', in case a restart is needed

def drop_staging_tables(cur):

    staging_match_table_drop = "DROP TABLE IF EXISTS staging_match"

    staging_winner_table_drop = "DROP TABLE IF EXISTS staging_winner"

    staging_loser_table_drop = "DROP TABLE IF EXISTS staging_loser"

    staging_meta_champs_table_drop = "DROP TABLE IF EXISTS staging_meta_champs"

    staging_meta_items_table_drop = "DROP TABLE IF EXISTS staging_meta_items"

    

    # execute all queries defined

    drop_table_queries = [staging_match_table_drop, staging_winner_table_drop, staging_loser_table_drop, staging_meta_champs_table_drop, staging_meta_items_table_drop]

    for query in drop_table_queries:

        cur.execute(query)
# create and insert staging tables

def load_staging_tables(conn):

    

    conn = sqlalchemy.create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname))

    

    print('loading staging_match')

    match_df.to_sql('staging_match', conn, index=False, if_exists='replace', dtype=match_dict)

    

    print('loading staging_winner')

    winner_df.to_sql('staging_winner', conn, index=False, if_exists='replace', dtype=winner_dict)

    

    print('loading staging_loser')

    loser_df.to_sql('staging_loser', conn, index=False, if_exists='replace', dtype=loser_dict)

    

    print('loading staging_meta_champs')

    meta_champs.to_sql('staging_meta_champs', conn, index=True, if_exists='replace')

    

    print('creating staging_meta_items')

    meta_items.to_sql('staging_meta_items', conn, index=True, if_exists='replace')
# create tables outlined in the 'data_modeling.pdf'

def create_tables(cur):

    games_table_create = ("""CREATE TABLE IF NOT EXISTS games(game_id bigint PRIMARY KEY, game_duration float NOT NULL, game_version varchar NOT NULL, participants varchar[10] NOT NULL)

    """)

    champions_table_create = ("""CREATE TABLE IF NOT EXISTS champions(game_id bigint PRIMARY KEY, champ_1 int NOT NULL, champ_2 int NOT NULL, champ_3 int NOT NULL, champ_4 int NOT NULL, champ_5 int NOT NULL, champ_6 int NOT NULL, champ_7 int NOT NULL, champ_8 int NOT NULL, champ_9 int NOT NULL, champ_10 int NOT NULL)

    """)

    items_table_create = ("""CREATE TABLE IF NOT EXISTS items(game_id bigint PRIMARY KEY, build_1 int[6] NOT NULL, build_2 int[6] NOT NULL, build_3 int[6] NOT NULL, build_4 int[6] NOT NULL, build_5 int[6] NOT NULL, build_6 int[6] NOT NULL, build_7 int[6] NOT NULL, build_8 int[6] NOT NULL, build_9 int[6] NOT NULL, build_10 int[6] NOT NULL)

    """)

    objectives_visions_table_create = ("""CREATE TABLE IF NOT EXISTS objectives_visions(game_id bigint PRIMARY KEY, win_dragon_soul boolean NOT NULL, win_baron_nashor boolean NOT NULL, win_ward_placed int NOT NULL, win_ward_destroyed int NOT NULL, lose_dragon_soul boolean NOT NULL, lose_baron_nashor boolean NOT NULL, lose_ward_placed int NOT NULL, lose_ward_destroyed int NOT NULL)

    """)

    champion_key_table_create = ("""CREATE TABLE IF NOT EXISTS champion_key(champion_key bigint PRIMARY KEY, champion_name varchar NOT NULL)

    """)

    item_key_table_create = ("""CREATE TABLE IF NOT EXISTS item_key(item_key bigint PRIMARY KEY, item_name varchar NOT NULL)

    """)



    # execute all queries defined

    create_table_queries = [games_table_create, champions_table_create, items_table_create, objectives_visions_table_create, champion_key_table_create, item_key_table_create]

    for query in create_table_queries:

        cur.execute(query)
print('drop staging tables')

drop_staging_tables(cur)

print('dropping fact/dimension tables')

drop_tables(cur)

print('creating staging tables')

load_staging_tables(cur)

print('creating fact/dimension tables')

create_tables(cur)
# list all tables created

cur.execute("""SELECT table_name FROM information_schema.tables

       WHERE table_schema = 'public'""")

for table in cur.fetchall():

    print(table)
pd.options.display.max_rows = 75

cur.execute("""SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_name LIKE 'staging_%'""")

pd.DataFrame(cur.fetchall(), columns=['table_name', 'column_name', 'data_type'])
champions_table_value = """

SELECT tb2.game_id, tb2.champ_ids[1], tb2.champ_ids[2], tb2.champ_ids[3], tb2.champ_ids[4],tb2.champ_ids[5],

tb2.champ_ids[6], tb2.champ_ids[7], tb2.champ_ids[8], tb2.champ_ids[9], tb2.champ_ids[10]

FROM

(SELECT tb.game_id AS game_id, array_agg(tb.c ORDER BY tb.i ASC)::jsonb[]::int[] AS champ_ids FROM 

(SELECT "gameId" AS game_id, 

json_array_elements(participants) -> 'championId' AS c, 

cast(json_array_elements(participants) -> 'participantId' as jsonb)::int AS i FROM staging_match) AS tb 

GROUP BY tb.game_id) AS tb2 ORDER BY tb2.game_id

"""
champions_table_insert = """INSERT INTO champions(game_id, champ_1, champ_2, champ_3, champ_4, champ_5, 

champ_6, champ_7, champ_8, champ_9, champ_10) {}""".format(champions_table_value)
cur.execute(champions_table_value)

a = pd.DataFrame(cur.fetchall())

a[a.isnull().any(axis=1)]
game_with_missing_champ_ids = list(a[a.isnull().any(axis=1)][0].array)

for game in range(len(game_with_missing_champ_ids)):

    total_participants = len(match_df[match_df['gameId'].isin(game_with_missing_champ_ids)].iloc[game].participants)

    print('game {} has: {} participants'.format(game_with_missing_champ_ids[game], total_participants))
# delete game with not 10 participants

cur.execute("""DELETE FROM staging_match WHERE "gameId" IN %s""", (tuple(game_with_missing_champ_ids),))
cur.execute(champions_table_value)

a = pd.DataFrame(cur.fetchall())

a[a.isnull().any(axis=1)]
cur.execute(champions_table_insert)
cur.execute("""SELECT * FROM champions LIMIT 3""")

cur.fetchall()
items_table_value = """

SELECT game_id, i[1:1], i[2:2], i[3:3], i[4:4], i[5:5], i[6:6], i[7:7], i[8:8], i[9:9], i[10:] FROM

(SELECT game_id AS game_id, ((array_agg(array[i0,i1,i2,i3,i4,i5,i6])))::jsonb[]::int[] AS i FROM

(SELECT game_id AS game_id, p ->'item0' AS i0, p -> 'item1' AS i1, p ->'item2' AS i2, p ->'item3' AS i3, 

p ->'item4' AS i4,p ->'item5' AS i5,p ->'item6' AS i6

FROM  (SELECT "gameId" AS game_id, json_array_elements(participants) -> 'stats' AS p FROM staging_match) AS tb1) AS tb2 

GROUP BY game_id) AS tb3 ORDER BY game_id

"""
items_table_insert = """INSERT INTO items(game_id, build_1, build_2, build_3, build_4, build_5, 

build_6, build_7, build_8, build_9, build_10) {}""".format(items_table_value)
cur.execute(items_table_insert)
cur.execute("""SELECT * FROM items LIMIT 3""")

cur.fetchall()
objectives_visions_table_value = """

SELECT game_id, 

CASE WHEN wdk >= 4 THEN TRUE ELSE FALSE END AS win_dragon_soul,

CASE WHEN wbk > 0 THEN TRUE ELSE FALSE END AS win_baron_nashor,

wwp AS win_ward_placed, wwk AS win_ward_killed,

CASE WHEN ldk >= 4 THEN TRUE ELSE FALSE END AS lose_dragon_soul,

CASE WHEN lbk > 0 THEN TRUE ELSE FALSE END AS lose_baron_nashor,

lwp AS lose_ward_placed, lwk AS lose_ward_killed

FROM 

(SELECT game_id AS game_id,

avg(wdk)::int AS wdk, avg(wbk)::int AS wbk, 

sum(wp::jsonb::int) FILTER (WHERE win::jsonb::boolean IS TRUE) AS wwp,

sum(wk::jsonb::int) FILTER (WHERE win::jsonb::boolean IS TRUE) AS wwk,

avg(ldk)::int AS ldk, avg(lbk)::int AS lbk,

sum(wp::jsonb::int) FILTER (WHERE win::jsonb::boolean IS FALSE) AS lwp,

sum(wk::jsonb::int) FILTER (WHERE win::jsonb::boolean IS FALSE) AS lwk

FROM (SELECT m."gameId" AS game_id,

json_array_elements(participants) #> '{stats, win}' AS win,

w."baronKills" AS wbk, w."dragonKills" AS wdk,

json_array_elements(participants) #> '{stats, wardsPlaced}' AS wp,

json_array_elements(participants) #> '{stats, wardsKilled}' AS wk,  

l."baronKills" AS lbk, l."dragonKills" AS ldk

FROM staging_match AS m  

INNER JOIN staging_winner AS w ON (m."gameId" = w."gameId") 

INNER JOIN staging_loser AS l ON (m."gameId" = l."gameId")

ORDER BY game_id

) AS tb1 GROUP BY game_id) AS tb3

"""
objectives_visions_table_insert = """INSERT INTO objectives_visions(game_id, win_dragon_soul, win_baron_nashor, win_ward_placed, win_ward_destroyed, 

lose_dragon_soul, lose_baron_nashor, lose_ward_placed, lose_ward_destroyed) 

{}""".format(objectives_visions_table_value)
cur.execute(objectives_visions_table_insert)
cur.execute("""SELECT * FROM objectives_visions LIMIT 3""")

cur.fetchall()
champion_key_table_value = """SELECT key::int, index FROM staging_meta_champs ORDER BY key::int"""
champion_key_table_insert = """INSERT INTO champion_key(champion_key, champion_name) {}""".format(champion_key_table_value)
cur.execute(champion_key_table_insert)
cur.execute("""SELECT * FROM champion_key LIMIT 10""")

cur.fetchall()
item_key_table_value = """SELECT index, name FROM staging_meta_items ORDER BY index"""
item_key_table_insert = """INSERT INTO item_key(item_key, item_name) {}""".format(item_key_table_value)
cur.execute(item_key_table_insert)
games_table_value = """

SELECT game_id, game_duration, game_version, array_agg(a) AS participants FROM

(SELECT "gameId" AS game_id, "gameDuration" AS game_duration, "gameVersion" AS game_version, 

json_array_elements("participantIdentities") #> '{player, accountId}' AS a

FROM staging_match) AS tb1

GROUP BY game_id, game_duration, game_version

ORDER BY game_id

"""
games_table_insert = """INSERT INTO games(game_id, game_duration, game_version, participants) {}""".format(games_table_value)
cur.execute(games_table_insert)
cur.execute("""SELECT * FROM games LIMIT 3""")

cur.fetchall()
# print data type of 'games'

cur.execute("""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'games' """)

pd.DataFrame(cur.fetchall(), columns=['column_name', 'data_type'])
# print data type of 'champions'

cur.execute("""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'champions' """)

pd.DataFrame(cur.fetchall(), columns=['column_name', 'data_type'])
# print data type of 'items'

cur.execute("""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'items' """)

pd.DataFrame(cur.fetchall(), columns=['column_name', 'data_type'])
# print data type of 'objectives_visions'

cur.execute("""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'objectives_visions' """)

pd.DataFrame(cur.fetchall(), columns=['column_name', 'data_type'])
# print data type of 'champion_key'

cur.execute("""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'champion_key' """)

pd.DataFrame(cur.fetchall(), columns=['column_name', 'data_type'])
# print data type of 'item_key'

cur.execute("""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'item_key' """)

pd.DataFrame(cur.fetchall(), columns=['column_name', 'data_type'])
# count rows of table

count_rows = """

SELECT

(SELECT count(*) FROM games) AS g,

(SELECT count(*) FROM champions) AS c,

(SELECT count(*) FROM items) AS i,

(SELECT count(*) FROM objectives_visions) AS o

"""

cur.execute(count_rows)

cur.fetchall()
random_games_samples = """

SELECT "gameDuration" AS sm_gd, gd, "gameVersion" AS sm_gv, gv FROM staging_match, 

(SELECT g.game_id AS game_id, g.game_duration AS gd, g.game_version AS gv FROM games AS g ORDER BY random() LIMIT 3) AS tb1

WHERE "gameId" IN (game_id)

"""

cur.execute(random_games_samples)

cur.fetchall()
# print two random rows of 'games'

cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_name = 'games' """)

columns = cur.fetchall()

cur.execute("""SELECT * FROM games ORDER BY random() LIMIT 2""")

pd.DataFrame(cur.fetchall(), columns=columns)
# print two random rows of 'champions'

cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_name = 'champions' """)

columns = cur.fetchall()

cur.execute("""SELECT * FROM champions ORDER BY random() LIMIT 2""")

pd.DataFrame(cur.fetchall(), columns=columns)
# print two random rows of 'items'

cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_name = 'items' """)

columns = cur.fetchall()

cur.execute("""SELECT * FROM items ORDER BY random() LIMIT 2""")

pd.DataFrame(cur.fetchall(), columns=columns)
# print two random rows of 'objectives_visions'

cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_name = 'objectives_visions' """)

columns = cur.fetchall()

cur.execute("""SELECT * FROM objectives_visions ORDER BY random() LIMIT 2""")

pd.DataFrame(cur.fetchall(), columns=columns)
# print two random rows of 'champion_key'

cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_name = 'champion_key' """)

columns = cur.fetchall()

cur.execute("""SELECT * FROM champion_key ORDER BY random() LIMIT 2""")

pd.DataFrame(cur.fetchall(), columns=columns)
# print two random rows of 'item_key'

cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_name = 'item_key' """)

columns = cur.fetchall()

cur.execute("""SELECT * FROM item_key ORDER BY random() LIMIT 2""")

pd.DataFrame(cur.fetchall(), columns=columns)