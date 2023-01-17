import pandas as pd

import sqlite3

from datetime import timedelta

import warnings

warnings.filterwarnings("ignore")



#load data (make sure you have downloaded database.sqlite)

con = sqlite3.connect('../input/database.sqlite')

cursor = con.cursor()

tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'",con)



for r in tables.itertuples():

    nome = r[1]

    meta = pd.read_sql_query("PRAGMA table_info('"+nome+"')",con)

    print(meta)

    meta.head()