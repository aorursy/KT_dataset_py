dbname = 'database.sqlite'
import numpy as np 

import pandas as pd

import sqlite3



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
path = "../input/"  #Insert path here

database = path + dbname



conn = sqlite3.connect(database)



tables = pd.read_sql("""SELECT *

                        FROM sqlite_master

                        WHERE type='table';""", conn)

tables
query = '''

SELECT addr_state,

REPLACE(SUBSTR(QUOTE(

    ZEROBLOB((COUNT(*)/1000+1)/2)

), 3, COUNT(*)/1000), '0', '*')

AS total_loans

FROM loan

GROUP BY addr_state

ORDER BY COUNT(*) DESC;

'''
result = pd.read_sql( query, conn )
result
result.to_csv("result.csv", index=False)