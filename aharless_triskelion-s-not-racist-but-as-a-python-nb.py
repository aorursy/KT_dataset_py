import numpy as np 

import pandas as pd

import sqlite3



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
path = "../input/"  #Insert path here

database = path + 'database.sqlite'



conn = sqlite3.connect(database)



tables = pd.read_sql("""SELECT *

                        FROM sqlite_master

                        WHERE type='table';""", conn)

tables
query = '''

SELECT body FROM May2015

WHERE LENGTH(body) < 255

AND LENGTH(body) > 30

AND body LIKE '%not racist, but%'

LIMIT 2500;

'''



nrb = pd.read_sql( query, conn )
nrb
nrb.to_csv("not_racist_but.csv", index=False)