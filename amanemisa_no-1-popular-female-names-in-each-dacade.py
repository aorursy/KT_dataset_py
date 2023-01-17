import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
con = sqlite3.connect('../input/database.sqlite')

NationalNames = pd.read_csv('../input/NationalNames.csv')
popular_female_dacade = pd.read_sql_query("""

WITH name_dacade AS (

SELECT 

CASE WHEN year like '188%' THEN '1880-1889'

     WHEN year like '189%' THEN '1890-1899'

     WHEN year like '190%' THEN '1900-1909'

     WHEN year like '191%' THEN '1910-1919'

     WHEN year like '192%' THEN '1920-1929'

     WHEN year like '193%' THEN '1930-1939'

     WHEN year like '194%' THEN '1940-1949'

     WHEN year like '195%' THEN '1950-1959'

     WHEN year like '196%' THEN '1960-1969'

     WHEN year like '197%' THEN '1970-1979'

     WHEN year like '198%' THEN '1980-1989'

     WHEN year like '199%' THEN '1990-1999'

     WHEN year like '200%' THEN '2000-2009'

     WHEN year like '201%' THEN '2010-2019'

END AS dacade,

Name, SUM(Count) AS Total_Count

FROM NationalNames

WHERE Gender = 'F'

GROUP BY dacade, Name)

SELECT dacade, Name, MAX(Total_Count) AS Total_Count

FROM name_dacade

GROUP BY dacade""", con)

popular_female_dacade
Mary_Lisa_Jennifer_Jessica_Emily_year = pd.read_sql_query("""

SELECT year,Name, Count

FROM NationalNames

WHERE Gender = 'F' AND Name IN ('Mary','Lisa','Jennifer','Jessica','Emily')

""", con)
fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(1,1,1)

Mary_Lisa_Jennifer_Jessica_Emily_year[Mary_Lisa_Jennifer_Jessica_Emily_year['Name']=='Mary'].plot(x='Year', y='Count',color = 'red',ax=ax,label='Mary')

Mary_Lisa_Jennifer_Jessica_Emily_year[Mary_Lisa_Jennifer_Jessica_Emily_year['Name']=='Lisa'].plot(x='Year', y='Count',color = 'green',ax=ax,label='Lisa')

Mary_Lisa_Jennifer_Jessica_Emily_year[Mary_Lisa_Jennifer_Jessica_Emily_year['Name']=='Jennifer'].plot(x='Year', y='Count',color = 'blue',ax=ax,label='Jennifer')

Mary_Lisa_Jennifer_Jessica_Emily_year[Mary_Lisa_Jennifer_Jessica_Emily_year['Name']=='Jessica'].plot(x='Year', y='Count', color ='orange',ax=ax,label = 'Jessica')

Mary_Lisa_Jennifer_Jessica_Emily_year[Mary_Lisa_Jennifer_Jessica_Emily_year['Name']=='Emily'].plot(x='Year', y='Count',color = 'black',ax=ax,label='Emily')

fig.suptitle('Name trends from 1880 to 2014', fontsize=12)
Mary_Lisa_Jennifer_Jessica_Emily_dacade = pd.read_sql_query("""

SELECT 

    CASE WHEN year like '188%' THEN '1880-1889'

     WHEN year like '189%' THEN '1890-1899'

     WHEN year like '190%' THEN '1900-1909'

     WHEN year like '191%' THEN '1910-1919'

     WHEN year like '192%' THEN '1920-1929'

     WHEN year like '193%' THEN '1930-1939'

     WHEN year like '194%' THEN '1940-1949'

     WHEN year like '195%' THEN '1950-1959'

     WHEN year like '196%' THEN '1960-1969'

     WHEN year like '197%' THEN '1970-1979'

     WHEN year like '198%' THEN '1980-1989'

     WHEN year like '199%' THEN '1990-1999'

     WHEN year like '200%' THEN '2000-2009'

     WHEN year like '201%' THEN '2010-2019'

END AS dacade, 

Name, SUM(Count) AS dacade_total

FROM NationalNames

WHERE Gender = 'F' AND Name IN ('Mary','Lisa','Jennifer','Jessica','Emily')

GROUP BY dacade, Name

""", con)
fig = plt.figure(figsize=(9,6))

ax1 = fig.add_subplot(1,5,1)

Mary_Lisa_Jennifer_Jessica_Emily_dacade[Mary_Lisa_Jennifer_Jessica_Emily_dacade['Name']=='Mary'].plot(x='dacade', y='dacade_total',kind = 'bar',color = 'red',ax=ax1,label='Mary',xlim=None, ylim=[0,800000])

ax2 = fig.add_subplot(1,5,2)

Mary_Lisa_Jennifer_Jessica_Emily_dacade[Mary_Lisa_Jennifer_Jessica_Emily_dacade['Name']=='Lisa'].plot(x='dacade', y='dacade_total',kind = 'bar',color = 'green',ax=ax2,label='Lisa',xlim=None, ylim=[0,800000])

ax3 = fig.add_subplot(1,5,3)

Mary_Lisa_Jennifer_Jessica_Emily_dacade[Mary_Lisa_Jennifer_Jessica_Emily_dacade['Name']=='Jennifer'].plot(x='dacade', y='dacade_total',kind = 'bar',color = 'blue',ax=ax3,label='Jennifer',xlim=None, ylim=[0,800000])

ax4 = fig.add_subplot(1,5,4)

Mary_Lisa_Jennifer_Jessica_Emily_dacade[Mary_Lisa_Jennifer_Jessica_Emily_dacade['Name']=='Jessica'].plot(x='dacade', y='dacade_total', kind='bar', color ='orange',ax=ax4,label = 'Jessica',xlim=None, ylim=[0,800000])

ax5 = fig.add_subplot(1,5,5)

Mary_Lisa_Jennifer_Jessica_Emily_dacade[Mary_Lisa_Jennifer_Jessica_Emily_dacade['Name']=='Emily'].plot(x='dacade', y='dacade_total',kind = 'bar',color = 'black',ax=ax5,label='Emily',xlim=None, ylim=[0,800000])

fig.suptitle('Counts of names in each dacade',fontsize = 12)