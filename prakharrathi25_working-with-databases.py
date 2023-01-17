import os

data_iris_folder_content = os.listdir("data/iris")
error_message = "Error: sqlite file not available, check instructions above to download it"

assert "database.sqlite" in data_iris_folder_content, error_message
import sqlite3
conn = sqlite3.connect('data/iris/database.sqlite')
cursor = conn.cursor()
type(cursor)
for row in cursor.execute("SELECT name FROM sqlite_master"):

    print(row)
cursor.execute("SELECT name FROM sqlite_master").fetchall()
sample_data = cursor.execute("SELECT * FROM Iris LIMIT 20").fetchall()
print(type(sample_data))

sample_data
[row[0] for row in cursor.description]
import pandas as pd
iris_data = pd.read_sql_query("SELECT * FROM Iris", conn)
iris_data.head()
iris_data.dtypes
iris_setosa_data = pd.read_sql_query("SELECT * FROM Iris WHERE Species == 'Iris-setosa'", conn)
iris_setosa_data

print(iris_setosa_data.shape)

print(iris_data.shape)
