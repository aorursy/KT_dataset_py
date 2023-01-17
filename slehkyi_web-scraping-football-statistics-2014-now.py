import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

from bs4 import BeautifulSoup
# create urls for all seasons of all leagues

base_url = 'https://understat.com/league'

leagues = ['La_liga', 'EPL', 'Bundesliga', 'Serie_A', 'Ligue_1', 'RFPL']

seasons = ['2014', '2015', '2016', '2017', '2018', '2019']
# Starting with latest data for Spanish league, because I'm a Barcelona fan

url = base_url+'/'+leagues[0]+'/'+seasons[4]

res = requests.get(url)

soup = BeautifulSoup(res.content, "lxml")



# Based on the structure of the webpage, I found that data is in the JSON variable, under <script> tags

scripts = soup.find_all('script')



# Check our <script> tags

# for el in scripts:

#   print('*'*50)

#   print(el.text)
import json



string_with_json_obj = ''



# Find data for teams

for el in scripts:

    if 'teamsData' in el.text:

      string_with_json_obj = el.text.strip()

      

# print(string_with_json_obj)



# strip unnecessary symbols and get only JSON data

ind_start = string_with_json_obj.index("('")+2

ind_end = string_with_json_obj.index("')")

json_data = string_with_json_obj[ind_start:ind_end]



json_data = json_data.encode('utf8').decode('unicode_escape')
# convert JSON data into Python dictionary

data = json.loads(json_data)

print(data.keys())

print('='*50)

print(data['138'].keys())

print('='*50)

print(data['138']['id'])

print('='*50)

print(data['138']['title'])

print('='*50)

print(data['138']['history'][0])



# Print pretty JSON data to check out what we have there

# s = json.dumps(data, indent=4, sort_keys=True)

# print(s)
# Get teams and their relevant ids and put them into separate dictionary

teams = {}

for id in data.keys():

  teams[id] = data[id]['title']
# EDA to get a feeling of how the JSON is structured

# Column names are all the same, so we just use first element

columns = []

# Check the sample of values per each column

values = []

for id in data.keys():

  columns = list(data[id]['history'][0].keys())

  values = list(data[id]['history'][0].values())

  break



print(columns)

print(values)
sevilla_data = []

for row in data['138']['history']:

  sevilla_data.append(list(row.values()))

  

df = pd.DataFrame(sevilla_data, columns=columns)

df.head(2)
# Getting data for all teams

dataframes = {}

for id, team in teams.items():

  teams_data = []

  for row in data[id]['history']:

    teams_data.append(list(row.values()))

    

  df = pd.DataFrame(teams_data, columns=columns)

  dataframes[team] = df

  print('Added data for {}.'.format(team))

  
# Sample check of our newly created DataFrame

dataframes['Barcelona'].head(2)
for team, df in dataframes.items():

  dataframes[team]['ppda_coef'] = dataframes[team]['ppda'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)

  dataframes[team]['oppda_coef'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)

  

# And check how our new dataframes look based on Sevilla dataframe

dataframes['Sevilla'].head(2)
cols_to_sum = ['xG', 'xGA', 'npxG', 'npxGA', 'deep', 'deep_allowed', 'scored', 'missed', 'xpts', 'wins', 'draws', 'loses', 'pts', 'npxGD']

cols_to_mean = ['ppda_coef', 'oppda_coef']
frames = []

for team, df in dataframes.items():

  sum_data = pd.DataFrame(df[cols_to_sum].sum()).transpose()

  mean_data = pd.DataFrame(df[cols_to_mean].mean()).transpose()

  final_df = sum_data.join(mean_data)

  final_df['team'] = team

  final_df['matches'] = len(df)

  frames.append(final_df)

  

full_stat = pd.concat(frames)
full_stat = full_stat[['team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'xG', 'npxG', 'xGA', 'npxGA', 'npxGD', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'xpts']]

full_stat.sort_values('pts', ascending=False, inplace=True)

full_stat.reset_index(inplace=True, drop=True)

full_stat['position'] = range(1,len(full_stat)+1)
full_stat['xG_diff'] = full_stat['xG'] - full_stat['scored']

full_stat['xGA_diff'] = full_stat['xGA'] - full_stat['missed']

full_stat['xpts_diff'] = full_stat['xpts'] - full_stat['pts']
cols_to_int = ['wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'deep', 'deep_allowed']

full_stat[cols_to_int] = full_stat[cols_to_int].astype(int)
col_order = ['position','team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'xG', 'xG_diff', 'npxG', 'xGA', 'xGA_diff', 'npxGA', 'npxGD', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'xpts', 'xpts_diff']

full_stat = full_stat[col_order]

pd.options.display.float_format = '{:,.2f}'.format

full_stat.head(10)
season_data = dict()

season_data[seasons[4]] = full_stat

print(season_data)

full_data = dict()

full_data[leagues[0]] = season_data

print(full_data)
full_data = dict()

for league in leagues:

  

  season_data = dict()

  for season in seasons:    

    url = base_url+'/'+league+'/'+season

    res = requests.get(url)

    soup = BeautifulSoup(res.content, "lxml")



    # Based on the structure of the webpage, I found that data is in the JSON variable, under <script> tags

    scripts = soup.find_all('script')

    

    string_with_json_obj = ''



    # Find data for teams

    for el in scripts:

        if 'teamsData' in el.text:

          string_with_json_obj = el.text.strip()



    # print(string_with_json_obj)



    # strip unnecessary symbols and get only JSON data

    ind_start = string_with_json_obj.index("('")+2

    ind_end = string_with_json_obj.index("')")

    json_data = string_with_json_obj[ind_start:ind_end]

    json_data = json_data.encode('utf8').decode('unicode_escape')

    

    

    # convert JSON data into Python dictionary

    data = json.loads(json_data)

    

    # Get teams and their relevant ids and put them into separate dictionary

    teams = {}

    for id in data.keys():

      teams[id] = data[id]['title']

      

    # EDA to get a feeling of how the JSON is structured

    # Column names are all the same, so we just use first element

    columns = []

    # Check the sample of values per each column

    values = []

    for id in data.keys():

      columns = list(data[id]['history'][0].keys())

      values = list(data[id]['history'][0].values())

      break

      

    # Getting data for all teams

    dataframes = {}

    for id, team in teams.items():

      teams_data = []

      for row in data[id]['history']:

        teams_data.append(list(row.values()))



      df = pd.DataFrame(teams_data, columns=columns)

      dataframes[team] = df

      # print('Added data for {}.'.format(team))

      

    

    for team, df in dataframes.items():

      dataframes[team]['ppda_coef'] = dataframes[team]['ppda'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)

      dataframes[team]['oppda_coef'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)

      

    cols_to_sum = ['xG', 'xGA', 'npxG', 'npxGA', 'deep', 'deep_allowed', 'scored', 'missed', 'xpts', 'wins', 'draws', 'loses', 'pts', 'npxGD']

    cols_to_mean = ['ppda_coef', 'oppda_coef']

    

    frames = []

    for team, df in dataframes.items():

      sum_data = pd.DataFrame(df[cols_to_sum].sum()).transpose()

      mean_data = pd.DataFrame(df[cols_to_mean].mean()).transpose()

      final_df = sum_data.join(mean_data)

      final_df['team'] = team

      final_df['matches'] = len(df)

      frames.append(final_df)



    full_stat = pd.concat(frames)

    

    full_stat = full_stat[['team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'xG', 'npxG', 'xGA', 'npxGA', 'npxGD', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'xpts']]

    full_stat.sort_values('pts', ascending=False, inplace=True)

    full_stat.reset_index(inplace=True, drop=True)

    full_stat['position'] = range(1,len(full_stat)+1)

    

    full_stat['xG_diff'] = full_stat['xG'] - full_stat['scored']

    full_stat['xGA_diff'] = full_stat['xGA'] - full_stat['missed']

    full_stat['xpts_diff'] = full_stat['xpts'] - full_stat['pts']

    

    cols_to_int = ['wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'deep', 'deep_allowed']

    full_stat[cols_to_int] = full_stat[cols_to_int].astype(int)

    

    col_order = ['position', 'team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts', 'xG', 'xG_diff', 'npxG', 'xGA', 'xGA_diff', 'npxGA', 'npxGD', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'xpts', 'xpts_diff']

    full_stat = full_stat[col_order]

    full_stat = full_stat.set_index('position')

    # print(full_stat.head(20))

    

    season_data[season] = full_stat

  

  df_season = pd.concat(season_data)

  full_data[league] = df_season

  

data = pd.concat(full_data)

data.head()

  
data.to_csv('understat.com.csv')