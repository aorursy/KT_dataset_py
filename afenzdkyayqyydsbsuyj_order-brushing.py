import pandas as pd

import numpy as np

import csv

from numba import njit

from numba.typed import Dict



input_filename  = "/kaggle/input/open-2-shopee-code-league-order-brushing/order_brush_order.csv"

output_filename = "submission.csv"



DEFAULT_USERID = '0'

CONCENTRATE_THRESHOLD = 3

ONE_HOUR = np.timedelta64(1, 'h')
def format_users(userids):

    ''' Returns formatted userids as string '''

    return '&'.join([str(userid) for userid in sorted(userids)])



def get_top_suspicious_users(tally):

    ''' Extracts top userids from dictionary tally and returns formatted string '''

    if not tally: return DEFAULT_USERID

    m = max(tally.values())

    return format_users(filter(lambda k: tally[k]==m, tally))



@njit

def flag_suspicious(flag, shopid, userid, event_time):

    ''' Returns a column of whether the given order is in an order brush period '''

    curr_shopid = shopid[0]

    prev_time = event_time[0] - ONE_HOUR

    for i in range(len(flag)):

        if shopid[i] != curr_shopid:

            curr_shopid = shopid[i]

            prev_time = event_time[i] - ONE_HOUR

        curr_time = event_time[i]

        j = i+1

        users = {userid[i]: 1}

        while j < len(flag) and shopid[j] == curr_shopid and event_time[j] - curr_time <= ONE_HOUR:

            if userid[j] not in users: users[userid[j]] = 0

            users[userid[j]] += 1

            j += 1

        while j-i < CONCENTRATE_THRESHOLD*len(users) and event_time[j-1] - prev_time > ONE_HOUR:

            j -= 1

            users[userid[j]] -= 1

            if users[userid[j]] == 0: users.pop(userid[j])

        if j-i >= CONCENTRATE_THRESHOLD*len(users):

            flag[i:j] = 1

        prev_time = curr_time

    return flag

@njit

def tally_suspicious(flag, shopid, userid):

    ''' Returns a dictionary containing shopid as key and userid value_counts as value '''

    result = Dict.empty(key_type=0,value_type={0:0})

    curr_shopid = shopid[0]

    shop_tally = Dict.empty(key_type=0,value_type=0)

    for i in range(len(shopid)):

        if shopid[i] != curr_shopid:

            result[curr_shopid] = shop_tally.copy()

            curr_shopid = shopid[i]

            shop_tally.clear()

        if flag[i]:

            if userid[i] not in shop_tally:

                shop_tally[userid[i]] = 0

            shop_tally[userid[i]] += 1

    result[curr_shopid] = shop_tally

    return result



def write_csv(filename, header, data):

    with open(filename, 'w') as f:

        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(data)
%%time



# 1. Read in the data

df = pd.read_csv(input_filename, parse_dates=['event_time'], usecols = ['shopid','userid','event_time'])



# 2. Sort data by shopid, event_time

df.sort_values(by=['shopid', 'event_time'], inplace=True)



shopid = df['shopid'].values

userid = df['userid'].values

event_time = df['event_time'].values



# 3. Flag all orders that are found within valid-order brushed periods

flag = flag_suspicious(np.zeros(shopid.shape, dtype=int), shopid, userid, event_time)



# 4. Get the userid frequency counts for each shopid

all_shopid_tally = tally_suspicious(flag, shopid, userid)



# 5. For each shopid, get the top suspicious userid(s) if any

data = ([shopid_, get_top_suspicious_users(tally)] for shopid_, tally in all_shopid_tally.items())



# 6. Write the result to file

write_csv(filename=output_filename, header=['shopid', 'userid'], data=data)