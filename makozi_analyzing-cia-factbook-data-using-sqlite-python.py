import sqlite3

import pandas as pd

conn= sqlite3.connect('../input/cia-factbook-data/factbook-data/factbook.db')

q= "SELECT  * FROM sqlite_master WHERE type='table';"

pd.read_sql_query(q,conn)
cursor= conn.cursor()

cursor.execute(q).fetchall()
q1= "SELECT * FROM facts limit 5"

pd.read_sql_query(q1,conn)
q2= '''

    select min(population) min_pop, max(population) max_pop, min(population_growth) min_pop_growth, max(population_growth) max_pop_growth from facts

'''

pd.read_sql_query(q2, conn)
q3= '''

    SELECT * FROM facts WHERE population==(SELECT max(population) FROM facts)

'''



pd.read_sql_query(q3,conn)

q4= ''' 



select * from facts where population==(select min(population) from facts)

'''



pd.read_sql_query(q4,conn)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)



q5 = '''

select population, population_growth, birth_rate, death_rate

from facts

where population != (select max(population) from facts)

and population != (select min(population) from facts);

'''

pd.read_sql_query(q5, conn).hist(ax=ax)
q6='''

select name, cast(population as float)/cast(area as float) density from facts order by density desc limit 20

'''

pd.read_sql_query(q6, conn)
q7 = '''select population, population_growth, birth_rate, death_rate

from facts

where population != (select max(population) from facts)

and population != (select min(population) from facts);

'''

pd.read_sql_query(q7, conn)