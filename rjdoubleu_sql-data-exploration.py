# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        f = os.path.join(dirname, filename)
        files.append(f)
        print(f)

# Any results you write to the current directory are saved as output.
!pip install csv-to-sqlite
def makeExecutable(ls):
    executable = str(ls).replace(',', '').replace('[', '').replace(']','').replace(' ', ' -f ')
    return executable

files_str = makeExecutable(files)

!echo $files_str
!csv-to-sqlite -f $files_str -o 'coronavirus.db'
import sqlite3 
  
# connecting to the database 
def makeConnection(db_name):
    return sqlite3.connect(db_name)
    
# fetch all the tables from the db
def sql_fetch(con):

    cursorObj = connection.cursor()

    cursorObj.execute('SELECT name FROM sqlite_master WHERE TYPE="table"')

    print(cursorObj.fetchall())

print('Tables:')
connection = makeConnection("coronavirus.db")
sql_fetch(connection)
pd.read_sql_query('SELECT * FROM time_series_covid_19_confirmed LIMIT 5', connection)
corupt_files = ["/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv",
                "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv",
                "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv"]

fixed_files = []

for file in corupt_files:
    df = pd.read_csv(file)
    col_map = {col:col.replace('/', '_') for col in df.columns}
    df = df.rename(columns=col_map)
    fixed = file[file.rindex('/')+1:]
    df.to_csv(fixed, index=False)
    fixed_files.append(fixed)
pd.read_csv(fixed).head()
leftovers = list(set(files) - set(corupt_files))
new_files = leftovers + fixed_files
new_files_str = makeExecutable(new_files)
new_files_str
! rm coronavirus.db
! csv-to-sqlite -f $new_files_str -o 'coronavirus.db'
connection = makeConnection("coronavirus.db")
conf_df = pd.read_sql_query('SELECT * FROM time_series_covid_19_confirmed LIMIT 5', connection)
conf_df
pd.read_sql_query('SELECT SUM("'+ df.columns[-1] +'") AS "Worldwide Death Toll"' +
                  ' FROM time_series_covid_19_deaths', 
                  connection)
pd.read_sql_query('SELECT Country_Region, "'+ df.columns[-1] +'" AS "Death Toll"' +
                  ' FROM time_series_covid_19_deaths' +
                  ' ORDER BY "' + df.columns[-1] + '" DESC' +
                  ' LIMIT 10', 
                  connection)
pd.read_sql_query('SELECT ROUND('+
                      '1.0 * SUM(' +
                          ' CASE WHEN "' + df.columns[-1] +'" > 0 THEN 1' +
                          ' ELSE 0 ' +
                          ' END) / COUNT("'+ df.columns[-1] +'"), 2)' +  
                          ' AS "Countries/Regions With Confirmed Cases (%)"' +
                  ' FROM time_series_covid_19_confirmed', 
                  connection)
pd.read_sql_query(' WITH deaths AS (SELECT Country_Region, "'+ df.columns[-1] +'" AS "Deaths"' +
                  ' FROM time_series_covid_19_deaths),' +
                  ' cases AS (SELECT Country_Region, "'+ df.columns[-1] +'" AS "Cases"' +
                  ' FROM time_series_covid_19_confirmed cases)' +
                  ' SELECT cases.Country_Region, SUM(cases.Cases) AS "Cases", SUM(deaths.Deaths) AS "Deaths",' +
                  ' ROUND(1.0 * deaths.Deaths/cases.Cases, 2) AS "Severity Score"'
                  ' FROM cases'
                  ' JOIN deaths' +
                  ' ON cases.Country_Region = deaths.Country_Region' +
                  ' GROUP BY cases.Country_Region'
                  ' ORDER BY cases.Cases DESC' + 
                  ' LIMIT 10',
                  connection)
pd.read_sql_query('SELECT Country_Region, SUM("'+ df.columns[-1] +'") AS "Cases"'
                  ' FROM time_series_covid_19_confirmed' +
                  ' GROUP BY Country_Region' +
                  ' ORDER BY "'+ df.columns[-1] +'"' +
                  ' LIMIT 10',
                  connection)
pd.read_sql_query('WITH total_cases AS ('
                  ' SELECT SUM("'+ df.columns[-1] +'") AS "Cases"' +
                  ' FROM time_series_covid_19_confirmed' +
                  ' GROUP BY Country_Region)' +
                  ' SELECT ROUND(AVG(Cases)) AS "Average Cases per Country"' +
                  ' FROM total_cases',
                  connection)
