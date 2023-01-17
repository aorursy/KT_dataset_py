from pandas.io.json import json_normalize

import requests

import pandas as pd

import numpy as np

import time
# champion info load

req2 = requests.get('http://ddragon.leagueoflegends.com/cdn/10.6.1/data/en_US/champion.json')



champ_ls = list(req2.json()['data'].keys())



champ_df = pd.DataFrame()

for i in range(len(champ_ls)):

    pre_df = json_normalize(req2.json()['data'][champ_ls[i]])

    champ_df = champ_df.append(pre_df)
champ_df.head()
req = requests.get('http://ddragon.leagueoflegends.com/cdn/10.6.1/data/en_US/item.json')



item_ls = []

for i in list(range(0,10000)):

    try:

        a = req.json()['data'][str(i)]

        item_ls.append(str(i))

    except:

        pass
item_table = pd.DataFrame()

for i in item_ls:

    item_id = i

    try:

        name = req.json()['data'][i]['name']

    except:

        name = np.nan

        

    try:

        upper_item = req.json()['data'][i]['into']

    except:

        upper_item = np.nan

    

    try:

        explain = req.json()['data'][i]['plaintext']

    except:

        explain = np.nan

    

    try:

        buy_price = req.json()['data'][i]['gold']['base']

    except:

        buy_price = np.nan

    

    try:

        sell_price = req.json()['data'][i]['gold']['sell']

    except:

        sell_price = np.nan

        

    try:

        tag = req.json()['data'][i]['tags'][0]

    except:

        tag = np.nan

    

    pre_df = pd.DataFrame({'item_id' : [item_id],

                           'name' : [name],

                           'upper_item' : [upper_item],

                           'explain' : [explain],

                           'buy_price' : [buy_price],

                           'sell_price' : [sell_price],

                           'tag' : [tag]

                          })

    

    item_table = item_table.append(pre_df)

    
item_table.head()