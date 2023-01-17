# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import bq_helper

from bq_helper import BigQueryHelper



london = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', dataset_name = 'london_crime')
london.list_tables()
london.head('crime_by_lsoa', num_rows = 20)
london.table_schema('crime_by_lsoa')
query1 = """

SELECT DISTINCT borough

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

"""



boroughs = london.query_to_pandas_safe(query1)

boroughs

query2 = """

SELECT borough, COUNT(DISTINCT lsoa_code) AS codes

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

GROUP BY borough

"""

codes_per_borough = london.query_to_pandas_safe(query2)

codes_per_borough
codes_sorted = codes_per_borough.sort_values('codes', ascending = False )

codes_sorted
codes_sorted.set_index('borough', inplace = True)

codes_sorted.head(10).plot(kind = 'bar', color = 'r', figsize = (12,6))

plt.ylabel('No. of codes')
query3 = """

SELECT DISTINCT major_category

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

"""

major_category = london.query_to_pandas_safe(query3)

major_category
query4 = """

SELECT major_category, SUM(value) AS instances

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

GROUP BY major_category

ORDER BY instances DESC

"""



major_category_crime = london. query_to_pandas_safe(query4)

major_category_crime
major_category_crime.set_index('major_category', inplace = True)
major_category_crime. plot(kind = 'barh', figsize = (12,6), color = 'm', title = 'Crime by major categories')

plt.xlabel('No. of instances')
query4 = """

SELECT DISTINCT minor_category, SUM(value) AS instances

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

GROUP BY minor_category

"""

minor_category = london.query_to_pandas_safe(query4)

minor_category
query5 = """

SELECT DISTINCT borough,SUM(value) AS instances,year, month

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

GROUP BY borough, year, month

ORDER BY borough DESC

"""

crime_by_borough = london.query_to_pandas_safe(query5)

crime_by_borough
crime_by_borough.groupby(['borough'])['instances'].sum().sort_values(ascending = False).head()
# Query adapted from previous to also group by major_category of crime

query6 = """

SELECT year, month, major_category, sum(value) AS `crime`

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

WHERE borough = 'Westminster'

GROUP BY year, month, major_category

ORDER BY year, month, crime DESC;

"""

# Perform and store the query results

wes_major_crime = london.query_to_pandas_safe(query6)

wes_major_crime
#crime by major category 2008

wes_major_crime_2008 = wes_major_crime.loc[wes_major_crime.year == 2008,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

wes_major_crime_2008
wes_major_crime_2009 = wes_major_crime.loc[wes_major_crime.year == 2009,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

wes_major_crime_2010 = wes_major_crime.loc[wes_major_crime.year == 2010,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

wes_major_crime_2011 = wes_major_crime.loc[wes_major_crime.year == 2011,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

wes_major_crime_2012 = wes_major_crime.loc[wes_major_crime.year == 2012,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

wes_major_crime_2013 = wes_major_crime.loc[wes_major_crime.year == 2013,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

wes_major_crime_2014 = wes_major_crime.loc[wes_major_crime.year == 2014,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

wes_major_crime_2015 = wes_major_crime.loc[wes_major_crime.year == 2015,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

wes_major_crime_2016 = wes_major_crime.loc[wes_major_crime.year == 2016,['major_category',

                                                                         'crime']].groupby(['major_category'])['crime'].sum()

pd.DataFrame({'crime_maj_2008': wes_major_crime_2008,

             'crime_maj_2009': wes_major_crime_2009,

             'crime_maj_2010': wes_major_crime_2010,

             'crime_maj_2011': wes_major_crime_2011,

             'crime_maj_2012': wes_major_crime_2012,

             'crime_maj_2013': wes_major_crime_2013,

             'crime_maj_2014': wes_major_crime_2014,

             'crime_maj_2015': wes_major_crime_2015,

             'crime_maj_2016': wes_major_crime_2016}).plot(kind = 'bar',

                                                           figsize = (20,12),

                                                           title = 'Crime by major category through (2008-2016)')

wes_major_crime.loc[wes_major_crime.year == 2008,

                    ['month','crime']].groupby(['month'])['crime'].sum().plot(figsize = (12,6), 

                                                                              title = 'Criminal activities throughout the year(2008)')

plt.ylabel('Total no. of criminal activities')
wes_major_crime.loc[wes_major_crime.year == 2009,

                    ['month','crime']].groupby(['month'])['crime'].sum().plot(figsize = (12,6),

                                                                              color = 'r',

                                                                              title = 'Criminal activities throughout the year(2009)')

plt.ylabel('Total no. of criminal activities')
wes_major_crime.loc[wes_major_crime.year == 2016,

                    ['month','crime']].groupby(['month'])['crime'].sum().plot(figsize = (12,6),

                                                                              color = 'm',

                                                                              title = 'Criminal activities throughout the year(2016)')

plt.ylabel('Total no. of criminal activities')
wes_major_crime.groupby(['year'])['crime'].sum().plot(figsize = (12,6),

                                                      title = 'Crime trajectory(2008-2016) in Westminster')

plt.ylabel('Total crimes in a year')
# Query adapted from previous to also group by major_category of crime

query7 = """

SELECT year, month, minor_category, sum(value) AS `crime`

FROM `bigquery-public-data.london_crime.crime_by_lsoa`

WHERE borough = 'Camden'

GROUP BY year, month, minor_category

ORDER BY year, month, crime DESC;

"""

# Perform and store the query results

cam_minor_crime = london.query_to_pandas_safe(query7)

cam_minor_crime
cam_minor_crime['minor_category'].describe()
cam_min_10 = cam_minor_crime.groupby(['minor_category'])['crime'].sum().sort_values(ascending = False).head(10)

cam_min_10
cam_min_10.plot(kind = 'bar',

                figsize = (12,6),

                title = '10 minor crime categories with the most crimes in Camden borough')
cam_minor_crime_2008 = cam_minor_crime.loc[cam_minor_crime.year == 2008,:]

cam_minor_crime_2008
cam_minor_crime_2008.groupby(['month'])['crime'].sum().sort_values(ascending = False)
cam_minor_crime_2008.groupby(['month'])['crime'].sum().plot(figsize = (12,6),

                                                            title = 'Crime level by months(2008)', color = 'g')
A = cam_minor_crime_2008.groupby(['minor_category'])['crime'].sum().sort_values(ascending = False).head(10)

A
cam_minor_crime_2008.groupby(['minor_category'])['crime'].sum().sort_values(ascending = False).head(10).plot(kind = 'bar',

                                                                                                             figsize = (12,6),

                                                                                                             title = 'Categories with most no. of crimes (2008)'),
cam_minor_crime_2009 = cam_minor_crime.loc[cam_minor_crime.year == 2009,:]

cam_minor_crime_2010 = cam_minor_crime.loc[cam_minor_crime.year == 2010,:]

cam_minor_crime_2011 = cam_minor_crime.loc[cam_minor_crime.year == 2011,:]
B = cam_minor_crime_2009.groupby(['minor_category'])['crime'].sum().sort_values(ascending = False).head(10)

C = cam_minor_crime_2010.groupby(['minor_category'])['crime'].sum().sort_values(ascending = False).head(10)

D = cam_minor_crime_2011.groupby(['minor_category'])['crime'].sum().sort_values(ascending = False).head(10)
pd.DataFrame({'2008':A,

             '2009':B,

             '2010':C,

             '2011':D}).plot(kind = 'bar',

                             figsize = (20,10),

                             title = 'Categories with most crimes in Camden(2008-2011)')