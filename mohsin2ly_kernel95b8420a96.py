import sqlite3
import datetime
import pandas as pd
from os import listdir

# from subprocess import check_output
# print(check_output(["ls", "-a", "../input"]).decode("utf8"))

path = '../input'
 
files = listdir(path)
for name in files:
    print(name)

start = datetime.datetime.now()
print(start)

conn = sqlite3.connect('../input/database.sqlite')
cursor = conn.cursor()
cursor.execute('select count* from country_names')
