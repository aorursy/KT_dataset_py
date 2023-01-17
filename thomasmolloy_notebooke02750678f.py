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
df_2009 = pd.read_csv('/kaggle/input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2009.csv')

df_2019 = pd.read_csv('/kaggle/input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2019.csv')

df_2014 = pd.read_csv('/kaggle/input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2014.csv')

df_2004 = pd.read_csv('/kaggle/input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2004.csv')
df_2004.head()
df_2009.head()
df_2014.head()
df_2019.head()
def sort_df_by_criminal_cases(df):

    return df.sort_values(by='Criminal Cases', ascending=False)
df_2004 = sort_df_by_criminal_cases(df_2004)

df_2009 = sort_df_by_criminal_cases(df_2009)

df_2014 = sort_df_by_criminal_cases(df_2014)

df_2019 = sort_df_by_criminal_cases(df_2019)
def find_highest_crime_city(df, citylist):

    max_city = ''

    max_crime = 0

    for city in citylist:

        total_crime = df.loc[df.City == city, 'Criminal Cases'].sum()

        if total_crime > max_crime:

            max_city = city

            max_crime = total_crime

        

    return max_city, max_crime

        
def top_five_crime_candidates(df):

    return df[:5][['Candidate','Criminal Cases']]
city_list_2004 = df_2004['City'].unique()

max_city_2004, max_crime_2004 = find_highest_crime_city(df_2004, city_list_2004)

print(max_city_2004, max_crime_2004)
city_list_2009 = df_2009['City'].unique()

max_city_2009, max_crime_2009 = find_highest_crime_city(df_2009, city_list_2009)

print(max_city_2009, max_crime_2009)
city_list_2014 = df_2014['City'].unique()

max_city_2014, max_crime_2014 = find_highest_crime_city(df_2014, city_list_2014)

print(max_city_2014, max_crime_2014)
city_list_2019 = df_2019['City'].unique()

max_city_2019, max_crime_2019 = find_highest_crime_city(df_2019, city_list_2019)

print(max_city_2019, max_crime_2019)
top_five_crime_candidates(df_2004)
top_five_crime_candidates(df_2009)
top_five_crime_candidates(df_2014)
top_five_crime_candidates(df_2019)
import plotly.express as px
fig_2004 = px.pie(df_2004, names='Education', title='2004 Election -- Percent Distribution of Education of Candidates')

fig_2004.show()
fig_2009 = px.pie(df_2009, names='Education', title='2009 Election -- Percent Distribution of Education of Candidates')

fig_2009.show()
fig_2014 = px.pie(df_2014, names='Education', title='2014 Election -- Percent Distribution of Education of Candidates')

fig_2014.show()
fig_2019 = px.pie(df_2019, names='Education', title='2019 Election -- Percent Distribution of Education of Candidates')

fig_2019.show()