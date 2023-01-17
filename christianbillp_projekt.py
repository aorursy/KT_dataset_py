%matplotlib inline

import pandas as pd

import sqlite3
conn = sqlite3.connect("../input/database.sqlite")

table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)

table_names
ds = {}

for name in table_names.values.tolist():

    ds[name[0]] = pd.read_sql_query("SELECT * FROM {}".format(name[0]), conn)
list(ds.keys())
ds['Player'].tail()
ds['Player']['height'].tail()
ds['Player'].tail(100).plot();