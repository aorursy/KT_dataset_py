import re

import time

import json

import random



import pandas as pd

import requests

from pandas.io.json import json_normalize



pd.set_option('display.max_columns', 100)
MIN_SLEEP_TIME = 60

AMOUNT_PER_PACKET = 100

TOTAL_NOTEBOOK_AMOUNT = 5000
start_url = 'https://www.kaggle.com/kernels.json?sortBy=voteCount&group=everyone&pageSize=' + str(AMOUNT_PER_PACKET)



headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/"

                         "58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0"}



proxies = {"http": "http://123.207.96.189:80"}
def get_table_from_url(url):

    

    if 'after' not in url:

        response = requests.get(start_url, proxies=proxies, headers=headers)

    else:

        response = requests.get(url, proxies=proxies, headers=headers)



    text = response.text

    nested_text = json.loads(text)

    df_packet = pd.io.json.json_normalize(nested_text)



    connect_id = df_packet.id[AMOUNT_PER_PACKET-1]

    new_url = 'https://www.kaggle.com/kernels.json?sortBy=voteCount&group=everyone&pageSize=' + str(AMOUNT_PER_PACKET) + '&after=' + str(connect_id)

    

    return df_packet, new_url
full_table, next_url = get_table_from_url(start_url)



while full_table.shape[0] < TOTAL_NOTEBOOK_AMOUNT:

    

    print('Top ' + str(full_table.shape[0]) + ' notebooks information collected')

    table_to_add, next_url = get_table_from_url(next_url)

    full_table = pd.concat([full_table, table_to_add])

    

    sleep_time = MIN_SLEEP_TIME + random.random()*10

    print('Let\'s sleep for ' + str(sleep_time) + ' seconds!')

    time.sleep(sleep_time)
first_columns = ['id','title','totalVotes', 'author.id', 'author.displayName', 'scriptUrl','totalComments', 'scriptVersionDateCreated', 'isFork', 'languageName', 'isNotebook', 'isGpuEnabled', 'bestPublicScore', 'lastRunExecutionTimeSeconds','medal',]

column_names_sorted = first_columns + list(set(full_table.columns) - set(first_columns))

table_sorted = full_table[column_names_sorted].reset_index(drop=True)



output_name = 'top_' + str(TOTAL_NOTEBOOK_AMOUNT) + '_voted_kaggle_notebooks.csv'

table_sorted.to_csv(output_name)
table_sorted.head()