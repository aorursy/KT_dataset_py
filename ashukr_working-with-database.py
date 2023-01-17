import sqlite3

conn = sqlite3.connect('../input/database.sqlite')
cursor = conn.cursor()
type(cursor)
for row in cursor.execute("SELECT name FROM sqlite_master"):

    print(row)
cursor.execute("SELECT name FROM sqlite_master").fetchall()
sample_data =cursor.execute("SELECT * FROM Iris LIMIT 3").fetchall()
print(type(sample_data))

sample_data
[row[0] for row in cursor.description]
import   pandas as pd
iris_data = pd.read_sql_query("SELECT * FROM Iris",conn)
iris_data.head()
iris_data.dtypes
setosa_data = pd.read_sql_query("SELECT * FROM Iris WHERE Species == 'Iris-setosa'",conn)
setosa_data
iris_