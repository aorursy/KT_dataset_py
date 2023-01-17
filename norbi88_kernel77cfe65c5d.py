import sqlite3
# Create a SQL connection to our SQLite database

con = sqlite3.connect("/kaggle/input/soccer/database.sqlite")



cur = con.cursor()



# The result of a "cursor.execute" can be iterated over by row

for row in cur.execute('SELECT match_api_id FROM Match limit 5;'):

    print(row)



# Be sure to close the connection

con.close()

con = sqlite3.connect('/kaggle/input/soccer/database.sqlite')



cur = con.cursor()
cur.execute('SELECT match_api_id FROM Match limit 5;')
df_player = pd.read_sql_query('SELECT * FROM Player;', con)
df_player['weight'].value_counts()
df_player_attributes = pd.read_sql_query('select * from Player_Attributes;', con)
import pandas as pd
df_player.head()
df_player_low = df_player.query('weight < 190 ', )
import pandasql as psql
df_player_low.head()