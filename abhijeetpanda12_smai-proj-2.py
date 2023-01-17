import pandas as pd

import numpy as np

import random

from datetime import datetime, date
buys = pd.read_csv('../input/yoochoose-buys.dat', header=None)

# clicks_data = pd.read_csv('yoochoose-data/yoochoose-clicks.dat', header=None)
buys.columns = ['session', 'timestamp', 'item', 'price', 'quantity']

# clicks_data.columns = ['session', 'timestamp', 'item', 'category']
click_bought = pd.read_csv('../input/bought.csv', dtype = { 'session':int,

                                                'timestamp':str,

                                                'item':int,

                                                'category':str}

                )
time_format = '%H:%M:%S'
def read_time_stamp(time):

    date,time = time.split('T')

    yy, mm, dd = date.split('-')

    time = time[:-1]

    h, m, s = time.split(':')

    return {'dd':dd,

            'mm':mm,

            'yy':yy,

            'h':h,

            'm':m,

            's':s.split('.')[0],

            'date':date,

            'time':time.split('.')[0]

           }
def time_diff_secs(start, end):

    tdelta = datetime.strptime(end, time_format) - datetime.strptime(start, time_format)

    return tdelta.seconds
b_sessions = sorted(list(buys['session'].unique())) # all bought sessions

items = buys['item']

click_items = click_bought['item']

q_vals = buys[['item','quantity']].values

s_items = buys[['session','item']].values
clicks = {}

quantity = {}

buy_count = {}

popularity = {}

session_item = {}
for session in b_sessions:

    session_item[session] = []
for si in s_items:

    session_item[si[0]].append(si[1])
for item in items:

    clicks[item]=0

    quantity[item]=0

    buy_count[item]=0
for item in click_items:

    try:

        clicks[item]+=1

    except:

        pass
for ll in q_vals:

    quantity[ll[0]]+=ll[1]

    buy_count[ll[0]]+=1
for item in items:

    try:

        popularity[item]=float("{0:.2f}".format(buy_count[item]/clicks[item],2))

    except:

        popularity[item]=0.00
item_data = []
for item in items:

    total_clicks = clicks[item]

    quant = quantity[item]

    buy = buy_count[item]

    pop = popularity[item]

    item_data.append([item, total_clicks, quant, buy, pop])
item_df = pd.DataFrame(item_data)
item_df.columns = ['item', 'total_clicks', 'quantity_sold', 'buys', 'popularity']
session_data = []

time_bias = 2

for bb in b_sessions:

    bs = buys[buys['session']==bb] # buys DF in a particular session

    cbs = click_bought[click_bought['session']==bb] # clicks DF in a particular session

    its = set(cbs['item']) # items clicked in a particular session

    b_its = set(bs['item']) # items bought in a particular sessions



    # ==========overall session feature extraction==========

    # number of clicks

    n_clicks = len(cbs)

    time = []

    date_var = []

    # time spent on each item in this session

    it_time = {}

    for it in its:

        it_time[it] = 0

    it_time[cbs.iloc[0]['item']]=time_bias

    prev = read_time_stamp(cbs.iloc[0]['timestamp'])['time']

    max_time = 0

    td = 0

    for ind in range(1,len(cbs)):

        now = read_time_stamp(cbs.iloc[ind]['timestamp'])['time']

        tdd = time_diff_secs(prev, now)

        it_time[cbs.iloc[ind]['item']]+= tdd

        td += tdd

        if tdd > max_time:

        # max time spent in this session

            max_time = tdd

        prev = now

    #average time

    avg = float("{0:.2f}".format(td/n_clicks,2))



    # number of unique categories in session

    noc = len(cbs['category'].unique())

    # avg. pop score

    sum_pop = 0

    for it in its:

        # sum of pop_scores

        try:

            sum_pop += popularity[it]

        except:

            pass

    avg_pop_score = sum_pop/len(its)

    # number of unique items in this session

    n_unique_items = len(its)

    # ==========session-item feature extraction==========

    for it in its:

        it_cbs = cbs[cbs['item']==it]

        # day of week of first click in this session

        ts = it_cbs.iloc[0]['timestamp']

        rdt = read_time_stamp(ts)

        f_c_time = rdt['time'] # time of first click in this session

        f_dw = date(int(rdt['yy']),int(rdt['mm']),int(rdt['dd'])).weekday()

        # day of week of last click in this session

        ts = it_cbs.iloc[-1]['timestamp']

        rdt = read_time_stamp(ts)

        l_c_time = rdt['time'] # time of last click in this session

        l_dw = date(int(rdt['yy']),int(rdt['mm']),int(rdt['dd'])).weekday()

        # Number of clicks in this item

        n_clicks_item = len(it_cbs)

        # duration between first click and last click of this item in this session

        dur_f_l = time_diff_secs(f_c_time, l_c_time)

        if dur_f_l == 0:

            dur_f_l = 2

        # pop_score

        try:

            it_pop = popularity[it]

        except:

            it_pop = 0

        # if this item is the first click

        if cbs.iloc[0]['item'] == it:

            f_click = 1

        else:

            f_click = 0

        # if this item is the last click

        if cbs.iloc[-1]['item'] == it:

            l_click = 1

        else:

            l_click = 0

        # purchased or not

        if it in b_its:

            purchased = 1

        else:

            purchased = 0

        session_data.append([bb,it, td, avg, max_time, n_clicks, avg_pop_score, noc,

              it_time[it], f_dw, l_dw, n_clicks_item, dur_f_l, f_click, l_click, it_pop,

              purchased])



session_df = pd.DataFrame(session_data)

session_df.columns = ['session', 'item', 'total_time', 'avg.time_clicks', 'max_time',

                      'n_clicks', 'avg_pop_score', 'no_of_categories',

                      'item_time', 'dow_first', 'dow_last','item_clicks', 'duration_f_l',

                      'f_click', 'l_click', 'item_pop',

                      'purchased']
session_df.to_csv('final_bought_sessions.csv', index=False)