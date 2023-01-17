import requests
import pickle    

game_data = []
year = '2015'
season_type = '02' # 02 is the indicator for regular season
max_game_ID = 1290 # A season can have around 1270 games so we go a bit overboard
for i in range(0,max_game_ID):
    r = requests.get(url='http://statsapi.web.nhl.com/api/v1/game/' + year + season_type +str(i).zfill(4)+'/feed/live')
    data = r.json()
    game_data.append(data)
    
with open('./'+year+'FullDataset.pkl', 'wb') as f:
    pickle.dump(game_data, f, pickle.HIGHEST_PROTOCOL)

## Other useful links!
#https://webcache.googleusercontent.com/search?q=cache:ILYmzDuxW70J:https://github.com/dword4/nhlapi+&cd=1&hl=en&ct=clnk&gl=ca
#https://statsapi.web.nhl.com/api/v1/game/2019020202/feed/live