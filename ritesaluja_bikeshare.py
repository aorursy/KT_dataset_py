# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/san-francisco"))



from google.cloud import bigquery

from bq_helper import BigQueryHelper

# Any results you write to the current directory are saved as output.



!pip install pygsheets

import pygsheets
#get the required data from Big_Query

QUERY = """

        SELECT AVG(duration_sec) as Avg_Duration_Sec, EXTRACT(DATE FROM start_date) AS StartingDate 

        FROM `bigquery-public-data.san_francisco.bikeshare_trips`

        WHERE EXTRACT(DATE FROM start_date) BETWEEN DATE('2016-03-01') AND DATE('2016-03-31')

        GROUP BY StartingDate

        ORDER BY StartingDate

        """



bq_assistant = BigQueryHelper('bigquery-public-data', 'san_francisco')

df = bq_assistant.query_to_pandas(QUERY)

df.head(3)
#After fetching credentials from Google API

#Connect with the client

gc = pygsheets.authorize(service_file='../input/san-francisco/credentials.json')



#create sheet

gc.create('BikeshareTrips')

#open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)

sh = gc.open('BikeshareTrips')



#select the first sheet 

wks = sh[0]



#update the first sheet with req data, starting at cell B2. 

wks.set_dataframe(df,(1,1))



#send email as update

sh.share("salujaritesh@gmail.com")