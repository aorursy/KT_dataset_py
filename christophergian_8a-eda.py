import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Read sqlite query results into a pandas DataFrame
def qry(q, connection = sqlite3.connect("../input/database.sqlite")):
    df = pd.read_sql_query(q, connection)
    connection.close
    return df
tables = qry("""
    SELECT name FROM sqlite_master
""")

# Get Col + info
columns_in_table = pd.DataFrame()
for i in get_tables.name:
    df_i = qry("PRAGMA TABLE_INFO(" + i + ");")
    df_i['table_name'] = i
    columns_in_table = columns_in_table.append(df_i)
tables = tables.name
print(tables)
print("# OF COLS")
columns_in_table.groupby('table_name').size()
# work w dataframes
df_user = qry("SELECT * FROM USER")
df_grade = qry("SELECT * FROM grade")
df_method = qry("SELECT * FROM method")
df_ascent = qry("SELECT * FROM ascent")

# get all desc. tabs..
desc_user = df_user.describe().T
desc_user['table'] = 'user'

desc_grade = df_grade.describe().T
desc_grade['table'] = 'grade'

desc_method = df_method.describe().T
desc_method['table'] = 'method'

desc_ascent = df_ascent.describe().T
desc_ascent['table'] = 'ascent'

pd.set_option('display.float_format', lambda x: '%.4f' % x)
desc_user.append(desc_grade).append(desc_method).append(desc_ascent)
df_user = df_user.rename(columns = {'id':'user_id'})
df_ascent = df_ascent.rename(columns = {'id':'ascent_id'})
df_grade = df_grade.rename(columns = {'id':'grade_id'})
top_20 = df_user.groupby('country').size().sort_values(ascending = False).head(20)
plt.figure(figsize = (20,5))
plt.title('Top 20 nations by # of Climbers on 8A.nu')
top_20.plot(kind = 'barh')
plt.show()
filter_user_table = df_user[['user_id','country','city','sex']]
filter_user_table = filter_user_table[filter_user_table.country.isin(top_20.index.tolist())]
filter_grade_table = df_grade[['grade_id','score']].query("score != 0")


explore_1 = (df_ascent[['ascent_id','user_id','grade_id']]
 .merge(filter_grade_table, left_on = 'grade_id', right_on = 'grade_id', how = 'left')
 .merge(filter_user_table, left_on = 'user_id',right_on ='user_id', how = 'right')
 )

explore_1 = (explore_1
 #.query("country == 'USA'")
 .dropna()
 .pivot_table(index = ['country'], values= 'score',aggfunc=[np.mean, len]))
explore_1.columns = ['_'.join(i) for i in explore_1.columns]

explore_1 = explore_1.reset_index().sort_values('mean_score', ascending = False)

explore_1['len_score'].sum()
explore_1.set_index('country')[['mean_score']].sort_values('mean_score', ascending = False).plot(kind = 'bar')
explore_1.set_index('country')
df_grade_list = df_grade[['usa_routes','score']].query("usa_routes != '' & score != 0")
df_grade_list['usa_routes'] = df_grade_list.usa_routes.str.replace("[a-z]","")
df_grade_list = df_grade_list[~df_grade_list.usa_routes.str.contains("/")]
df_grade_list = df_grade_list.groupby('usa_routes').agg([np.min,np.max]).reset_index()
df_grade_list.columns = ["_".join(i) for i in df_grade_list.columns]
rows = [0]+ np.arange(7,15).tolist() + np.arange(1,7).tolist()
df_grade_list.iloc[rows,:].rename(columns = {'score_amin':'min','score_amax':'max','usa_routes_':'grade'})
