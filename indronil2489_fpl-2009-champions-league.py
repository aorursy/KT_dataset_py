# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import requests
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
url = 'https://fantasy.premierleague.com/api/leagues-classic/447290/standings/'
r = requests.get(url)
json = r.json()
json.keys()
teams = pd.DataFrame(json.get('standings'))['results']
teams
league_dict={'SK Jim':['Premier League','Liverpool'],
        'Rufait Nahin':['Bundesliga','Leverkusen'],
        'Fahim Shahriar Swapnil':['Bundesliga','Schalke'],
        'Rakib Fardin':['Serie A','Fiorentina'],
        'Nazmuz Sakib Adnan':['Premier League','Arsenal'],
        'Muhtasim Shobael Aninda':['Serie A','Napoli'],
        'Mushfiqur Rahman':['Serie A','Lazio'],
        'Nafis Al Asad':['Premier League','Manchester City'],
        'Bidyut Saha':['Serie A','Juventus'],
        'Md Mukarrom Hossain Rayat':['Bundesliga','Leipzig'],
        'Mohaimin Rafi':['La Liga','Real Madrid'],
        'Sharaf Juhayr':['La Liga','Sevilla'],
        'Tausif Mashroor':['Bundesliga','Bayern Munich'],
        'Readul Alam Shuvo':['Serie A','AC Milan'],
        'Mashrur Koushik':['Serie A','Roma'],
        'Niaz Mahmood':['Premier League','Leicester City'],
        'Indronil Bhattacharjee Prince':['Bundesliga','Hoffenheim'],
        'Shojib Shouvo':['La Liga','Real Sociedad'],
        'Tawhid Sultan':['Premier League','Manchester United'],
        'Sakib Jr.':['Bundesliga','Frankfurt'],
        'Kaiser Mehedi':['La Liga','Barcelona'],
        'Nasim Mahmud Akash':['La Liga','Valencia'],
        'Saif Rubab':['La Liga','Villareal'],
        'Sazzad Zahid Sajol':['Premier League','Everton'],
        'shofiqur rahman':['Premier League','Tottenham'],
        'Ahrab Akash':['Serie A','Inter Milan'],
        'Karim Xbcc':['Bundesliga','Dortmund'],
        'Tanvir Sabbir':['Premier League','Chelsea'],
        'Homayed Naser':['Bundesliga','Wolfsburg'],
        'Sarwar Sabbir':['La Liga','Granada'],
        'Riamul Islam':['La Liga','Atletico Madrid'],
        'Rifat Morshed':['Serie A','Atalanta']}
owner_name=[]
team_name=[]
point=[]
league=[]
league_team=[]
for i in range(0,32):
    owner_name.append(teams[i]['player_name'])
    league.append(league_dict.get(teams[i]['player_name'])[0])
    league_team.append(league_dict.get(teams[i]['player_name'])[1])
    team_name.append(teams[i]['entry_name'])
    point.append(teams[i]['event_total'])
df = pd.DataFrame()
df['league'] = league
df['league_team'] = league_team
df['team_name'] = team_name
df['owner_name'] = owner_name
df['gw_point'] = point
df.loc[df['league'] == 'Premier League'].sort_values(by=['gw_point'], ascending=False)
df.loc[df['league'] == 'La Liga'].sort_values(by=['gw_point'], ascending=False)
df.loc[df['league'] == 'Serie A'].sort_values(by=['gw_point'], ascending=False)
df.loc[df['league'] == 'Bundesliga'].sort_values(by=['gw_point'], ascending=False)