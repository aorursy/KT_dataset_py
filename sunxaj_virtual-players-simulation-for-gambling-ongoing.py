import pandas as pd
import random
import math
from datetime import datetime, timedelta
from random import randint
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
%matplotlib inline
players = []

def get_skillup_coef(x):
    return round(1000000000 / ((9-x) ** 4))

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10
def generate_record(_id):
    host_skill = 10
    wins = 0
    loss = 0
    rounds = 0
    checkin_time = None
    checkout_time = None
    start_time = None
    
    if not any(player['id'] == _id for player in players):
        start_time = datetime.now()+ timedelta(minutes=randint(1,600))
        checkin_time = start_time.strftime('%Y-%m-%d %H:%M')
        init_token = round(random.uniform(1,10)) * 100
        player_skill = random.uniform(6,9)
        player = {'id':_id, 'skill':player_skill, 'wealth' : init_token, 'last_visit': start_time} 
        players.append(player)

    else:
        init_token = roundup(next(player['wealth'] for player in players if player.get("id") == _id) * random.uniform(0.7, 1.3)) 
        player_skill = next(player['skill'] for player in players if player.get("id") == _id)
        start_time = next(player['last_visit'] for player in players if player.get("id") == _id) + timedelta(days=randint(3,7))
        checkin_time = start_time.strftime('%Y-%m-%d %H:%M')
        player = next(player for player in players if player.get("id") == _id)
        player['last_visit'] = start_time
        
    token = init_token
    
    while token > 0 and rounds <= random.uniform(20,50):
        host_luck = random.uniform(10, 20)
        player_luck = randint(8, 28)
        if host_skill * host_luck > player_skill * player_luck:
            bet = round(init_token * random.uniform(0.05,0.1))
            if token >= bet:
                token -= bet   
            else: 
                token = 0
                checkout_time = (datetime.now()+ timedelta(hours=rounds/5)).strftime('%Y-%m-%d %H:%M')
            loss += 1 
            
        elif host_skill * host_luck < player_skill * player_luck:
            token += round(init_token * random.uniform(0.05,0.1)) #win x% token in 1 round
            wins += 1

        skillup_coef = get_skillup_coef(player_skill)
        player_skill += math.sqrt(rounds/skillup_coef)
        rounds += 1
        
    checkout_time = (start_time + timedelta(hours=rounds/5)).strftime('%Y-%m-%d %H:%M')
    player = next(player for player in players if player.get("id") == _id)
    player['skill'] = player_skill
    
    row = pd.Series({'player_id': _id, 'checkin time': checkin_time, 'checkout time': checkout_time,
                     'played rounds': rounds, 'initial token': init_token, 'withdraw':token, 'profit': token - init_token,
                     'player_skill': player_skill, 'wins': wins, 'loss': loss})
    return row
def generator(init_id, num):
    records = pd.DataFrame(columns = ["player_id",'checkin time', 'checkout time', 'initial token',
                                      'withdraw','played rounds','profit', 'player_skill', 'wins', 'loss'])
    for i in range(num):
        records = records.append(generate_record(i+init_id) , ignore_index= True)
        if i % 39 == 1:
            for j in range (1, randint(4, 10)):
                records = records.append(generate_record(i+init_id) , ignore_index= True)
        elif i % 13 == 1:
            for j in range (1, randint(2, 6)):
                records = records.append(generate_record(i+init_id) , ignore_index= True)
        elif i % 7 == 1:
            for j in range (1, 2):
                records = records.append(generate_record(i+init_id) , ignore_index= True)
    df = records.copy()
    return records.sort_values(by=['player_id'])        
df = generator(1000, 500)
df['checkin time'] =  pd.to_datetime(df['checkin time'], format='%Y-%m-%d %H:%M')
df['checkout time'] =  pd.to_datetime(df['checkout time'], format='%Y-%m-%d %H:%M')
df
df[df['player_id'] == 1001]
df.info()
df.player_id.value_counts()
data = [go.Scatter( x=df['checkin time'], y=df['profit'])]

iplot(data, filename='pandas-time-series')
