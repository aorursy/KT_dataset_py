import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt



import json, time, requests, os

from requests.exceptions import Timeout

from datetime import datetime, timedelta

from multiprocessing import Pool

import seaborn as sns



def get_text(url):

    try:

        return eval(requests.get(url, timeout = 2).text)

    except:

        return eval(requests.get(url, timeout = 4).text)

    

null = 'nothing'

all_chara = get_text('https://tinygrail.com/api/chara/mrc/1/5000')['Value']

valhalla = get_text("https://tinygrail.com/api/chara/user/chara/tinygrail/1/4000")["Value"]["Items"]

auction_state = {i['Id']: [i['State'],i['Price']] for i in valhalla}
def depth(chara_index):

    chara_id = all_chara[chara_index]['Id']

    url = 'https://tinygrail.com/api/chara/depth/{0}/'.format(chara_id)

    result = get_text(url)

    if result['State'] == 0:

        return [chara_id, result['Value']]

    else:

        return [chara_id, '0']
def auction_price(chara_index):

    chara_id = all_chara[chara_index]['Id']

    url = f'https://tinygrail.com/api/chara/{chara_id}'

    result = get_text(url)

    if result['State'] == 0:

        if chara_id in auction_state:

            return [chara_id] + auction_state[chara_id]

        else:

            return [chara_id, 0, result['Value']['Price']]

    else:

        return [chara_id, 0, 0]
def user_activedate(chara_id):

    url = 'https://tinygrail.com/api/chara/users/{0}/1/1000'.format(chara_id)

    result = get_text(url)

    output = []

    if result['State'] == 0:

        for i in result['Value']['Items']:

            output.append([i['Id'], i['Name'], i['LastActiveDate']])

    return output
def find_you(chara_uid):

    ### input is a list of the form [chard_id, uid]

    ### find all stocks that a user has (if possible)

    chara_id = all_chara[chara_uid[0]]['Id']

    chara_info = all_chara[chara_uid[0]]

    data = get_text('https://tinygrail.com/api/chara/users/'+str(chara_id)+'/1/1000')['Value']['Items']

    flows = 0

    uid = chara_uid[1]

    if uid == 701:

        for i in data:

            #if i['Id'] == uid:

            if i["LastActiveDate"] == "1970-01-01T00:00:00":

                flows = i['Balance']

                break

    else:

        for i in data:

            if i['Id'] == uid:

                flows = i['Balance']

                break

    return [chara_id, uid, flows]
def deal_history(chara_index):

    chara_id = all_chara[chara_index]['Id']

    chara_info = all_chara[chara_index]

    data = get_text('https://tinygrail.com/api/chara/charts/'+str(chara_id)+

                             '/2019-07-24')['Value']#[1:]

    deals = []

    ico_flag = True

    for i in data:

        if ico_flag != True:

            deals.append([i['Amount'],i['End'],i['Time'], 'normal'])

        else:

            deals.append([i['Amount'],i['End'],i['Time'], 'ico'])

            ico_flag = False

    return [chara_id,chara_info, deals]
todo_list = list(range(len(all_chara)))

if __name__ == '__main__':

    with Pool(4) as p:

        ss5 = p.map(depth, todo_list)

    with Pool(4) as p:

        ss4 = p.map(auction_price, todo_list)

    with Pool(4) as p:

        ss = p.map(deal_history, todo_list)

#     with Pool(4) as p:

#         ss3 = p.map(user_activedate, todo_list)

        

#     todo_list = [[i, 701] for i in todo_list]

#     with Pool(4) as p:

#         ss2 = p.map(find_you, todo_list)

# ss33 = []

# for i in ss3:

#     for j in i:

#         ss33.append(j)
df = pd.DataFrame(ss, columns = ['id','info','deals']).set_index('id')

#df2 = pd.DataFrame(ss2, columns = ['id','uid','flows']).set_index('id')

#df3 = pd.DataFrame(ss33, columns = ['id','name','last_active']).set_index('id')

df4 = pd.DataFrame(ss4, columns = ['id','amount','auction_price']).set_index('id')

df5 = pd.DataFrame(ss5, columns = ['id','depth']).set_index('id')

df.to_csv('tinygrail_reload.csv')

#df2.to_csv('tinygrail_stk.csv')

#df3.to_csv('tinygrail_active.csv')

df4.to_csv('tinygrail_auction_price.csv')

df5.to_csv('tinygrail_depth.csv')