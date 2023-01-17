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
!pip install bar_chart_race



import bar_chart_race as bcr
India_df = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

India_df.head()
India_df.info()
India_df.Date = pd.to_datetime(India_df['Date'], dayfirst=True)

India_df.drop(India_df.index[5091:5126], axis=0, inplace=True)

India_df
India_df['State/UnionTerritory'].unique()
India_df['State/UnionTerritory'].replace({"Telengana" : "Telangana", "Telengana***" : "Telangana",

                                          "Telangana***" : "Telangana"}, inplace = True)



India_df['State/UnionTerritory'].replace({"Daman & Diu" : "Dadra and Nagar Haveli and Daman and Diu",

                                          "Dadar Nagar Haveli" : "Dadra and Nagar Haveli and Daman and Diu"},

                                         inplace = True)
India_df = India_df[(India_df['State/UnionTerritory'] != 'Unassigned') &

                    (India_df['State/UnionTerritory'] != 'Cases being reassigned to states')]
India_df = India_df[['Date', 'State/UnionTerritory', 'Confirmed']]

India = pd.pivot_table(India_df, values = 'Confirmed', index = 'Date', columns = 'State/UnionTerritory')
India.fillna(0, inplace = True)

India = India[India.index >= '2020-03-01']

India
# use filename = 'covid-19.mp4' if you want to download the video.



bcr.bar_chart_race(df = India, title = 'COVID-19 CASES ACROSS INDIA', figsize=(6,4), steps_per_period=10,

                  period_summary_func = lambda v, r: {'x': .98, 'y': .18, 

                                      's': f'Total Cases: {v.sum():,.0f}',

                                      'ha': 'right', 'size': 12, 'family': 'Courier New'})