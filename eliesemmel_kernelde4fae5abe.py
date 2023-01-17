import re
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

pd.set_option('notebook_repr_html', True)
import requests
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
league = ['GB1','FR1','L1','IT1','ES1']
league_page = "https://www.transfermarkt.com/jumplist/startseite/wettbewerb/"
def get_club_details(tr_tag):
    club = tr_tag.find_all('a',class_="vereinprofil_tooltip")[1]
    club_link = club['href']
    club_name = club.get_text()
    club_value = tr_tag.find_all('td',class_="rechts show-for-small show-for-pad nowrap")[0].get_text()
    return tuple((club_link,club_name,club_value))
clubs_list = []
for league_id in league:
    page = requests.get(league_page + league_id,headers = headers)
    soup = bs(page.content, 'html.parser')
    tbody_container = soup.find_all('tbody')[1]
    tr_container = tbody_container.find_all('tr')
    for tr_tag in tr_container :
        clubs_list.append(get_club_details(tr_tag))
print('All the club were uploaded')
def get_players_club(player):
    player_id = player['id']
    player_link = player['href']
    player_name = player.get_text()
    return tuple((player_id,player_link,player_name,club_name,club_value))
url_site = "https://www.transfermarkt.com"
player_list = []
for club_link,club_name,club_value in clubs_list:
    page = requests.get(url_site + club_link,headers = headers)
    soup = bs(page.content, 'html.parser')
    tbody_container = soup.find_all('tbody')[1]
    players_details = tbody_container.find_all('a',class_="spielprofil_tooltip")
    for player in players_details[::2] :
        player_list.append(get_players_club(player))
print('All the players were uploaded')
def get_profil_detail(soup):
    table_container = soup.find_all('table', class_="auflistung")[0]
    td_container = table_container.find_all('td')
    if td_container[1].find('a') == None:
        birth = td_container[0].find('a')['href'].split("/")[-1]# take it in format YYYY-MM-DD for datetime later
        height = td_container[3].get_text().split("m")[0]# I remove the m of meter 
        country = td_container[4].find('img')['title']
        role = td_container[5].get_text().strip()
        foot = td_container[6].get_text()
    else :
        birth = td_container[1].find('a')['href'].split("/")[-1]
        height = td_container[4].get_text().split("m")[0]# I remove the m of meter 
        country = td_container[5].find('img')['title']
        role = td_container[6].get_text().strip()
        foot = td_container[7].get_text()
    tbody_container = soup.find_all('tbody')[0]
    tr_transfer_container = tbody_container.find_all('tr',class_="zeile-transfer")
    transfer_list = []
    for tr_transfer_tag in tr_transfer_container:
        td_transfer_container = tr_transfer_tag.find_all("td")
        tranfer_from = td_transfer_container[5].get_text()
        transfer_to = td_transfer_container[9].get_text()
        transfer_season = td_transfer_container[0].get_text()
        transfer_date = td_transfer_container[1].get_text()
        transfer_list.append(tuple((tranfer_from,transfer_to,transfer_season,transfer_date)))
    return tuple((Id,name,club,club_value,birth,height,country,role,foot,link.split("/")[1],transfer_list))
        
player_details = []
i=0
for Id,link,name,club,club_value in player_list:
    i=i+1
    if i%500 == 0:
        print("new league upload")
    try:
        page = requests.get(url_site + link,headers = headers)
        soup = bs(page.content, 'html.parser')
        player_details.append(get_profil_detail(soup))
    except Exception as e:
        player_details.append(tuple((Id,name,club,club_value,None,None,None,None,None,link.split("/")[1],[])))
        continue
print("all player details uploaded")
def get_injuries_details(soup):
    tbody_container = soup.find_all('tbody')[0]
    tr_container = tbody_container.find_all('tr')
    injuries_list = []
    for tr_tag in tr_container:
        season = tr_tag.find_all('td')[0]
        injury = tr_tag.find_all('td')[1]
        time_out = tr_tag.find_all('td')[4]
        injuries_list.append(tuple((season.get_text(),injury.get_text(),time_out.get_text().split()[0])))
    return injuries_list
player_list = []
for Id,name,club,club_value,birth,height,country,role,foot,name_link,transfer_list in player_details:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
        page = requests.get("https://www.transfermarkt.com/{}/verletzungen/spieler/{}".format(name_link,Id),headers=headers)
        soup = bs(page.content, 'html.parser')
        player_list.append(tuple((Id,name,club,club_value,birth,height,country,role,foot,transfer_list,get_injuries_details(soup))))
    except Exception as e:
        player_list.append(tuple((Id,name,club,club_value,birth,height,country,role,foot,transfer_list,[])))
        continue
print("all player injuries details uploaded")
print("End of uploading from Transfermarkt")
def get_list_club_lequipe(h2_tag,league_id):
    if league_id == 'EQ_D1' :
        club = h2_tag.find_all('a')[1]
        club_link = club['href']
        club_name = club['title']
        return tuple((club_link,club_name))
    else:
        club = h2_tag.find('a')
        club_link = club['href']
        club_name = club['title']
        return tuple((club_link,club_name))
league = ['EQ_ANG','EQ_D1','EQ_ALL','EQ_ITA','EQ_ESP']
league_page = "https://www.lequipe.fr/Football/"
clubs_list_lequipe = []
for league_id in league:
    print(league_id)
    page = requests.get(league_page + league_id + ".html",headers = headers)
    soup = bs(page.content, 'html.parser')
    tbody_container = soup.find_all('div',class_="listeclubs")[0]
    h2_container = tbody_container.find_all('h2')
    for h2_tag in h2_container :
        clubs_list_lequipe.append(get_list_club_lequipe(h2_tag,league_id))
print('All the club were uploaded')
url_site = "https://www.lequipe.fr"
player_list_lequipe = []
for club_link,club_name in clubs_list_lequipe:
    page = requests.get(url_site + club_link,headers = headers)
    soup = bs(page.content, 'html.parser')
    tbody_container = soup.find_all('table')[4]
    players_details = tbody_container.find_all('tr')
    for player in players_details[1:] :
        player_details = player.find_all('a')[0]
        weight = player.find_all('td')[6]
        player_name = player_details.get_text().strip()
        player_list_lequipe.append(tuple((player_name,club_name,weight.get_text())))
print("all player from lequipe were uploaded")
club_to_player_map = {}
for player in player_list_lequipe:
    if player[1] in club_to_player_map.keys():
        club_to_player_map[player[1]][player[0]] = player[2]
    else:
        club_to_player_map[player[1]] = {}
        club_to_player_map[player[1]][player[0]] = player[2]
#Just to verify the name
lequipe_clubs = club_to_player_map.keys()
print(lequipe_clubs)
import difflib
df_players = pd.DataFrame(player_list)
df_players.head()
Final_player = []
for player_transfertmarkt in player_list:
    try:
        list_of_players_lequipe = club_to_player_map[difflib.get_close_matches(player_transfertmarkt[2], lequipe_clubs,cutoff=0.1)[0]].keys()
        weight_player = club_to_player_map[difflib.get_close_matches(player_transfertmarkt[2], lequipe_clubs,cutoff=0.1)[0]][difflib.get_close_matches(player_transfertmarkt[1][:15], list_of_players_lequipe,cutoff=0.1)[0]]
        Final_player.append(tuple((player_transfertmarkt[0],player_transfertmarkt[1],player_transfertmarkt[2],player_transfertmarkt[3],player_transfertmarkt[4],weight_player,player_transfertmarkt[5],player_transfertmarkt[6],player_transfertmarkt[7],player_transfertmarkt[8],player_transfertmarkt[9],player_transfertmarkt[10])))
    except Exception as e:
        print(player_transfertmarkt[2])
        Final_player.append(tuple((player_transfertmarkt[0],player_transfertmarkt[1],player_transfertmarkt[2],player_transfertmarkt[3],player_transfertmarkt[4],None,player_transfertmarkt[5],player_transfertmarkt[6],player_transfertmarkt[7],player_transfertmarkt[8],player_transfertmarkt[9],player_transfertmarkt[10])))
print("END")
print(Final_player)
                                                                                                                                        
df_player = pd.DataFrame(Final_player)
df_player.columns = ["id", "name", "club","club_value","birth","height","weight","country","role","foot","transfers","injuries"]
df_player.head()
df_player.shape
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline

pd.set_option('notebook_repr_html', True)
tuples = []
i=0
for t in open('../input/Final-player.txt'):
    i=i+1
    if i == 2580:
        tuples.append(eval(t[1:-1]))
    else:
        tuples.append(eval(t[1:-2]))
df = pd.DataFrame(tuples)
df.columns = ["id", "name", "club","club_value","birth","weight","height","country","role","foot","transfers","injuries"]
number_of_players = df.shape[0]
df.shape
df.head()
df = df[df['birth'].notnull()]
number_of_new_players = df.shape[0]
df.shape
total_player = number_of_players-number_of_new_players
print(str(total_player) + " players removed")
injuries_list = [injurie[1] for injurie in df['injuries'].sum()]
s = pd.Series(injuries_list).value_counts()
for injur in (s[s > 75].index):
    print(injur)
s[:5]
s[:20].plot(kind='pie')
df.head()
s = df.apply(lambda x: pd.Series(x[11]),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'injurie'
df = df.drop(['injuries'], axis=1).join(s)
df.reset_index()
df.head()


df.shape
df = df.reset_index(drop=True)
df.head()
df = df[df['injurie'].notnull()]
df.shape
df_injurie = pd.DataFrame(df['injurie'].tolist(), index=df.index)
df_injurie.head()
df = df.drop('injurie', axis=1).join(df_injurie)
df.head()
df = df.rename(index=str, columns={0: "season", 1: "type" , 2 : "days"})
df.head()
import datetime
df.season = df.season.apply(lambda x : '20' + x[3:] + '-01-01')
df.head()
def injurie_age(season,birth):
    return datetime.datetime.strptime(season, '%Y-%m-%d').year - datetime.datetime.strptime(birth, '%Y-%m-%d').year

df['age'] = df.apply(lambda x : injurie_age(x['season'], x['birth']), axis=1)
df.head()
df[(df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize='true').sort_index().plot(kind='bar',style=None)
muscle_injurie = ['Hamstring Injury','Muscular problems','Muscle Injury','Torn Muscle Fibre',
                  'Adductor problems','Thigh Muscle Strain','Groin Injury','Muscle Fatigue',
                  'Achilles tendon problems','Torn muscle bundle','Biceps femoris muscle injury']
df[(df['type'].isin(muscle_injurie)) & (df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize='true').sort_index().plot(kind='bar',style=None)
fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
ax.set_title('   General injuries RED against muscle injuries BLUE')
width = 0.4

df[(df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize='true').sort_index().plot(kind='bar', color='red', ax=ax, width=width, position=1)
df[(df['type'].isin(muscle_injurie))
   & (df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize='true').sort_index().plot(kind='bar',color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('general inj')
ax2.set_ylabel('muscle inj')

plt.show()

df[(df['type'] == 'Ill') & (df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize='true').sort_index().plot(kind='bar',style=None,title="Ill injurie according to age")
df.head()
df = df[df['days'] != '?']
df["days"] = pd.to_numeric(df["days"])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,10))
fig.suptitle('Injury recovery time according to age', fontsize=20)
df.groupby('type').get_group('Ankle Injury').groupby('age').filter(lambda x: len(x) > 6).groupby('age')['days'].mean().plot(ax=axes[0,0],title="Ankle Injury",color='blue')
df.groupby('type').get_group('Hamstring Injury').groupby('age').filter(lambda x: len(x) > 0).groupby('age')['days'].mean().plot(ax=axes[0,1],title="Hamstring Injury",color='blue')
df.groupby('type').get_group('Cruciate Ligament Rupture').groupby('age').filter(lambda x: len(x) > 5).groupby('age')['days'].mean().plot(ax=axes[1,0],title="Cruciate Ligament Rupture",color='blue')
df.groupby('age')['days'].mean().plot(ax=axes[1,1],title="All Injuries",color='blue')
df[(df['type'] == 'Cruciate Ligament Rupture')
   & (df['age'] > 19) & (df['age'] < 36)]['age'].value_counts(normalize='true').sort_index().plot(kind='bar',style=None)
df.groupby('type')['days'].mean().sort_values(ascending=False)[:5].plot(kind='bar')
df.head()
df['height'] = df['height'].str.replace(",",".")
df['height'] = df['height'].str.replace('\\xa0',"").astype(float)
df = df[df['weight'] != '-']
df['weight'] = df['weight'].astype(float)
def define_bmi(height,weight):
    if (weight)/(height*height) > 28:
        return 1
    return 0
df['bmi'] = df.apply(lambda x : define_bmi(x['height'], x['weight']), axis=1)
df.head()
players_high_bmi = set(df[df['bmi'] == 1].groupby('name').groups.keys())
len(players_high_bmi)
players_high_bmi
df['club_value'] = df['club_value'].apply(lambda x : x.split(",")[0]).astype(int)
df['club_value'] = df['club_value'].apply(lambda x : 1000 if x == 1 else x).astype(int)
df.head()
df['club_value'].plot(kind='hist',bins=25)
fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111)
df['club_value'].plot(kind='hist',bins=40,cumulative=True,ax=ax)
plt.plot([0,1000], [2000,2000], 'r-', lw=2)
plt.plot([0,1000], [4000,4000], 'r-', lw=2)
plt.plot([0,1000], [6000,6000], 'r-', lw=2)
plt.plot([0,1000], [8000,8000], 'r-', lw=2)
plt.plot([0,1000], [10000,10000], 'r-', lw=2)

def club_to_category(value):
    if 0 < value <=  75 :
        return 1
    if 75 < value <= 125 :
        return 2
    if 125 < value <= 225 :
        return 3
    if 225 < value <= 275 :
        return 4
    if 275 < value <= 600 :
        return 5
    else :
        return 6  
df['club_value'] = df['club_value'].apply(club_to_category)
df.head()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16,10))
fig.suptitle('Injury recovery time according to budget club', fontsize=20)
df.groupby('type').get_group('Hamstring Injury').groupby('club_value').mean()['days'].plot(kind='bar',ax=axes[0,0],title="Hamstring Injury")
df.groupby('type').get_group('Ankle Injury').groupby('club_value').mean()['days'].plot(kind='bar',ax=axes[0,1],title="Ankle Injury")
df.groupby('type').get_group('Muscular problems').groupby('club_value').mean()['days'].plot(kind='bar',ax=axes[0,2],title="Muscular problems")
df.groupby('type').get_group('Muscle Injury').groupby('club_value').mean()['days'].plot(kind='bar',ax=axes[1,0],title="Muscle Injury")
df.groupby('type').get_group('Unknown Injury').groupby('club_value').mean()['days'].plot(kind='bar',ax=axes[1,1],title="Unknown Injury")
df.groupby('club_value').mean()['days'].plot(kind='bar',ax=axes[1,2],title="Total Injury")