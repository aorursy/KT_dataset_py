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



athlete_events_filepath = '/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv'

noc_regions_filepath = '/kaggle/input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv'

        

# Any results you write to the current directory are saved as output.
athlete_events = pd.read_csv(athlete_events_filepath)

noc_regions = pd.read_csv(noc_regions_filepath)
athlete_events.columns.tolist()
athlete_events.sample(10)
sport_names = athlete_events['Sport'].unique().tolist()
dir(athlete_events.groupby('Sport'))
athlete_events.groupby('Sport').get_group('Athletics')
athletes_by_sport = athlete_events.groupby('Sport')

sport_to_events = {}

for (sport_name,athletes_data) in athletes_by_sport:

    sport_to_events[sport_name] = athletes_data['Event'].unique().tolist()

sport_to_events
sport_names
india_data = athlete_events[athlete_events['Team'] == 'India']
country_names = athlete_events['Team'].unique().tolist()
country_data = {}

country_stats = {}

for country_name in country_names:

    country_data[country_name] = athlete_events[athlete_events['Team'] == country_name]
india_data = country_data['India']

india_data
india_data['Medal']

india_data['Medal'].unique()

india_gold_data = india_data[india_data['Medal'] == 'Gold']

india_gold_data