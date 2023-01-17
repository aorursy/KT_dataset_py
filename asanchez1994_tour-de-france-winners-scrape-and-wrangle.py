# Import modules

import numpy as np

import pandas as pd

import re

import requests

import csv

from datetime import datetime, timedelta

from bs4 import BeautifulSoup
"""

# Get html from Wikipedia

url = 'https://en.wikipedia.org/wiki/List_of_Tour_de_France_general_classification_winners'

r = requests.get(url).text



# Instantiate Soup Object

soup = BeautifulSoup(r, features='lxml')

x = soup.find_all('table', class_="wikitable")



# Isolate desired table and get column headings

my_table = x[1]

row_list = my_table.find_all('tr')



# Loop over table and collect rows in list

final_list = []

no_of_rows = len(row_list)

for i in range(no_of_rows):

    # Parse strings for better formatting 

    lists = row_list[i].text.split('\n')

    b = [l for l in lists if l != '']

    c = list(map(lambda x: x.replace('\xa0',''),b))

    final_list.append(c)

    

# Convert scraped data to pandas DataFrame

# df = pd.DataFrame(final_list)

# df.columns = df.iloc[0]

# df.drop(0, inplace=True)



# Uncomment to save unprocessed data to csv 

# with open("tdf_scraped_data.csv", "w", newline="") as f:

#    writer = csv.writer(f)

#    writer.writerows(final_list)

"""
path = '../input/tdf-winners-unprocessed/tdf_scraped_data.csv'

df = pd.read_csv('../input/tdf-winners-unprocessed/tdf_scraped_data.csv', encoding='latin')
# Remove years cancelled due to war 

df = df.dropna()



# Reformat column names for readability/typability

df.columns = map(str.lower, df.columns)

df.columns = [s.replace(' ', '_') for s in df.columns]

df.set_index('year', inplace=True)



# Add a column for Lance Armstrong controversies

df['controversial'] = df.index.map(lambda x: x.endswith('[B]'))
# Fix year formatting

def removeB(x):

    if x.endswith('[B]'):

        x = x[:4]

    return x

        

df.index = df.index.map(removeB)

df.index = df.index.astype(int)
# Cast distance, stage_wins, and stages_in_lead to int 

df.rename(columns={'distance': 'distance_km'}, inplace=True)

df['distance_km'] = df['distance_km'].map(lambda x : x[:5])

df['distance_km'] = df['distance_km'].map(lambda x : x.replace(',', ''))

df = df.astype({'distance_km': 'int','stage_wins': 'int',

           'stages_in_lead': 'int'})
# Cast margin column to timedeltas

helper = list(df['margin'])

helper[22] = '+ 44\' 23"'

df['margin'] = helper

df['margin'].unique()



# Render points years as nan

def margin_strip(x):

    if 'h' in x:

        fin = datetime.strptime(x, '+ %Hh %M\' %S\"')

    elif ' \'' in x:

        fin = datetime.strptime(x, '+ %M \'%S\"')

    elif '\'' in x:

        fin = datetime.strptime(x, '+ %M\' %S\"')

    elif '\"' in x:

        fin = datetime.strptime(x, '+ %S\"')

    else: 

        return np.nan

    return timedelta(hours=fin.hour, minutes=fin.minute, 

                     seconds=fin.second)



df['margin'] = df['margin'].map(margin_strip)
# Rename and cast time/points column to timedelta

df.rename(columns={'time/points':'overall_time'}, inplace=True)



# Render points years as nan

def time_points_parse(x):

    try:

        pattern_text = r'(?P<hour>\w+)h\s+(?P<minute>\w+)\'\s(?P<second>\w+)\"'

        pattern = re.compile(pattern_text)

        match = pattern.match(x)

        h, m, s = map(int, match.groups())

        return timedelta(hours=h, minutes=m, seconds=s)

    except:

        return np.nan

    

df['overall_time'] = df['overall_time'].map(time_points_parse)
# Add columns for overall_time and margin in seconds

def to_seconds(x):

    return x.total_seconds()

df[['overall_time_sec','margin_sec']] = df[['overall_time','margin']].applymap(to_seconds)
# Uncomment below to save as csv



#df.to_csv('tdf_winners.csv')