import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import glob
import json
import re
import requests
from urllib.parse import quote

main_dir = '../input/magic-the-gathering-top8-some-decks-and-events'
#main_dir = '../input/magic-the-gathering-top8-some-decks-and-events/events'
df_all_events = pd.read_csv(os.path.join(main_dir,"df_events_v2.csv"))
df_all_events.head()
df_all_events.info()
error_date_index = []
for ind in tqdm(df_all_events.index):
    date = df_all_events['event_date'][ind]
    if date != '00/00/00':
        datetime.strptime(date, '%d/%m/%y')
    else:
        error_date_index.append(ind)
df_all_events = df_all_events.drop(error_date_index)
df_all_events['event_date'] = df_all_events['event_date'].astype('datetime64[ns]') 
df_all_events.info()
mask = (df_all_events['event_date'] > '01-01-2010') & (df_all_events['event_date'] <= '01-01-2020') & (df_all_events['event_format'] == 'ST')
some_events = df_all_events.loc[mask]
print(some_events.head())
print(some_events.info())
def walker_players_4_event(id_evt):
    try:
        path_players_info = os.path.join(main_dir, 'events', id_evt, 'players_info.csv')
        path_players_decks = os.path.join(main_dir, 'events', id_evt, 'players_decks')

        df_info_p = pd.read_csv(path_players_info).drop(columns='Unnamed: 0')

        decks_players = glob.glob(os.path.join(path_players_decks, "*.json"))
        decks = []
        for dck in decks_players:
            id_player = re.findall(r"player_([0-9]{3,7})_deck.json", dck)[0]
            with open(dck) as f: 
                dict_p = json.load(f)
                dict_p['id_player'] = id_player
                decks.append(dict_p)

        return df_info_p, decks
    except:
        return df_info_p, decks
samples_df_all = []
samples_decks_all = []

for ind in tqdm(some_events.index):
    event_id = some_events['event__id'][ind]
    try:
        df, deck = walker_players_4_event(str(event_id))
        samples_df_all.append(df)
        samples_decks_all.append(deck)
    except:
        print('error in ', ind)
samples_df = pd.concat(samples_df_all, ignore_index=True).drop(columns='player_player')
samples_df.info()
mask = (samples_df['player_title'].str.contains("Control", na=False)) & (samples_df['player_result'] == '1')
tops_1 = samples_df.loc[mask]
print(tops_1.info())
print(tops_1.head())
tops_1_decks = []
for deck_event in samples_decks_all:
    for deck in deck_event:
        if int(deck['id_player']) in list(tops_1['player__id']):
            tops_1_decks.append(deck)
r = requests.get('https://api.scryfall.com/cards/search?q='+quote('Island'))
print(r.status_code)
print(r.json()['data'][0])
tops_1_decks_texts = []
for deck in tqdm(tops_1_decks):
    d_d = deck['main_deck']
    #d_d_text = []
    for card in d_d:
        r = requests.get('https://api.scryfall.com/cards/search?q='+quote(card[0]))
        if r.status_code == 200:
            ft_pkg = r.json()['data'][0]
            if 'oracle_text' in ft_pkg and 'type_line' in ft_pkg:
                if ft_pkg['type_line'] == 'Sorcery':
                    text = ft_pkg['oracle_text']
                    tops_1_decks_texts.append(text)
            else:
                pass
                #print('[+]',card[0],'- This card doesn`t include a oracle_text tag')
        else:
            pass
            #print('[+]',card[0],'- Error in the status_code of the request')

tops_1_decks_texts = set(tops_1_decks_texts)
tops_1_decks_texts