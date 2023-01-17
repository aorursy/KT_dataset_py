# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly.express as px

import plotly.io as pio



pio.templates.default = 'plotly_dark'





from sqlalchemy import create_engine
filename_train = '/kaggle/input/covid19-global-forecasting-week-3/train.csv'

filename_test = '/kaggle/input/covid19-global-forecasting-week-3/test.csv'



train_df = pd.read_csv(filename_train)

test_df = pd.read_csv(filename_test)





engine = create_engine('sqlite://', echo=False)

train_df.to_sql('train', con=engine)

test_df.to_sql('test', con=engine)
table = 'train'



query = """

    WITH non_zero_confirm AS (

        SELECT *, SUM(ConfirmedCases) AS confirmed

        FROM {0}

        GROUP BY Country_Region, Date HAVING confirmed > 0

    )

    

    SELECT MIN(Date) AS first_date, Country_Region AS country, confirmed

    FROM non_zero_confirm

    GROUP BY Country_Region

    ORDER BY first_date

""".format(table)



print(query)



first_confirmed_case = pd.read_sql(query, engine)



first_confirmed_case
table = 'train'



query = """

    WITH non_zero AS (

        SELECT Date, Country_Region, SUM(ConfirmedCases) as confirmed

        FROM {0}

        GROUP BY Date, Country_Region HAVING confirmed > 0

    )

        

    

    SELECT sub.*, (julianday(sub.Date) -  julianday(sub.first_date)) AS since_first_confirmed_case

    FROM (

        SELECT *,

                MIN(Date) OVER (PARTITION BY Country_Region ORDER BY Date) AS first_date

        FROM non_zero

    ) sub

    

    

""".format(table)





first_confirmed_date_df = pd.read_sql(query, engine)





first_confirmed_date_df[first_confirmed_date_df['Country_Region'] == 'US'][:5]


fig = px.line(first_confirmed_date_df,

              x='since_first_confirmed_case', y='confirmed',

             color='Country_Region')



fig.show()
table = 'train'



query = """

    WITH non_zero AS (

        SELECT Date, Country_Region, SUM(Fatalities) as death

        FROM {0}

        GROUP BY Date, Country_Region HAVING death > 0

    )

        

    

    SELECT sub.*, (julianday(sub.Date) -  julianday(sub.first_date)) AS since_first_death_case

    FROM (

        SELECT *,

                MIN(Date) OVER (PARTITION BY Country_Region ORDER BY Date) AS first_date

        FROM non_zero

    ) sub

    

    

""".format(table)





first_death_date_df = pd.read_sql(query, engine)





first_death_date_df[first_death_date_df['Country_Region'] == 'US'][:5]


fig = px.line(first_death_date_df,

              x='since_first_death_case', y='death',

             color='Country_Region')



fig.show()
table = 'train'



query = """

    WITH non_zero AS (

        SELECT t.*,

                SUM(t.ConfirmedCases) AS confirmed

        FROM {0} AS t

        GROUP BY t.Date, t.Country_Region HAVING confirmed > 0

    )

    

    

    SELECT f.Country_Region AS country, 

            f.Date, 

            f.first_date,

            (julianday(f.Date) - julianday(f.first_date)) AS since_first_confirmed,

            f.confirmed, 

            f.new_confirmed_cases

    FROM (

        SELECT n.*,

                n.confirmed - LAG(n.confirmed) OVER (PARTITION BY n.Country_Region ORDER BY n.Date) AS new_confirmed_cases,

                MIN(n.Date) OVER (PARTITION BY n.Country_Region ORDER BY n.Date) As first_date

        FROM non_zero AS n

    ) AS f



""".format(table)



print(query)



daily_confirmed_df = pd.read_sql(query, engine)



country = 'US'

daily_confirmed_df[daily_confirmed_df['country'] == country].tail()
fig = px.line(daily_confirmed_df,

             x='since_first_confirmed',

             y='new_confirmed_cases',

             color='country')



fig.show()
table = 'train'



query = """

    WITH non_zero AS (

        SELECT t.*,

                SUM(t.Fatalities) AS death

        FROM {0} AS t

        GROUP BY t.Date, t.Country_Region HAVING death > 0

    )

    

    SELECT f.Country_Region AS country,

            f.Date,

            f.first_date,

            (julianday(f.Date) - julianday(f.first_date)) AS since_first_death,

            f.death,

            f.new_death_case

    FROM (

        SELECT n.*,

                n.death - LAG(n.death) OVER (PARTITION BY n.Country_Region ORDER BY n.Date) AS new_death_case,

                MIN(n.Date) OVER (PARTITION BY n.Country_Region ORDER BY n.Date) AS first_date

        FROM non_zero AS n

    ) AS f

""".format(table)



print(query)



daily_death_df = pd.read_sql(query, engine)





country = 'US'

daily_death_df[daily_death_df['country'] == country].tail()
fig = px.line(daily_death_df,

             x='since_first_death', y='new_death_case',

             color='country')



fig.show()