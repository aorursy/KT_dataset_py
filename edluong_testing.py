# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# borrowed from https://www.kaggle.com/samaxtech/designing-and-creating-a-database-sql#Importing-Data-into-SQLite
import sqlite3
#run_query(q): Takes a SQL query as an argument and returns a pandas dataframe by using the connection as a SQLite built-in context manager. 
def run_query(q):
    with sqlite3.connect('testing') as conn:
        return pd.read_sql_query(q, conn)
    
#run_command(c): Takes a SQL command as an argument and executes it using the sqlite module.
def run_command(c):
    with sqlite3.connect('testing') as conn:
        conn.execute('PRAGMA foreign_keys = ON;') #Enables enforcement of foreign key restraints.
        conn.isolation_level = None
        conn.execute(c)
    
#show_tables(): calls the run_query() function to return a list of all tables and views in the database.
def show_tables():
    q = '''SELECT
            name,
            type
        FROM sqlite_master
        WHERE type IN ("table","view");
        '''
    return run_query(q)    
import pandas as pd


df = pd.read_csv("/kaggle/input/democratvsrepublicantweets/ExtractedTweets.csv")

conn = sqlite3.connect('testing')
df.to_sql('tweets', conn, index=False, if_exists='replace')
show_tables()
cursor = conn.cursor()
cursor.execute("select * from tweets")
books = cursor.fetchall()
for i in books:
    print(i)