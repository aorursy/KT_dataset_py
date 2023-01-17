import pandas as pd

import numpy as np

import time



start_time = time.time()



df = pd.read_csv("../input/logistics-shopee-code-league/delivery_orders_march.csv")

df
def replaceaddress(add):

    area = add.split()[-1].lower()

    return area



df['buyeraddress'] = df['buyeraddress'].apply(replaceaddress)

df['selleraddress'] = df['selleraddress'].apply(replaceaddress)

df['combined'] = df['buyeraddress'] + "-" + df['selleraddress']



def get_SLA_days(d):

    s, e = d.split("-")

    if s == "manila":

        if e == "manila":

            return 3

        elif e == "luzon":

            return 5

    if s == "luzon":

        if e == "manila" or "luzon":

            return 5

    return 7



df['sla_days'] = df['combined'].apply(get_SLA_days)
from datetime import datetime



gmt_offset = 8 * 60 * 60



for col in ["pick", "1st_deliver_attempt", "2nd_deliver_attempt"]:

    df[col] = df[col] + gmt_offset

    df[col] = pd.to_datetime(df[col], unit='s')

    df[col] = df[col].values.astype('datetime64[D]')

    

df.head()
from datetime import date



algo_time = time.time()



# PHs and Sundays

hol_list = [date(2020, 3, 25), date(2020, 3, 30), date(2020, 3, 31),

            date(2020, 3, 1), date(2020, 3, 8), date(2020, 3, 15), 

            date(2020, 3, 22), date(2020, 3, 29), date(2020, 4, 5)]



def is_late(row):

    first = row['1st_deliver_attempt']

    second = row['2nd_deliver_attempt']

    pick = row['pick']

    

    # second delivery exists

    if pd.notnull(second):

        num_days = (second - first).days

        

        if num_days > 3: #earlier stop

            for hol in hol_list:

                if first <= hol <= second:

                    num_days -= 1

            if num_days > 3:

                return 1

    

    # check first delivery is ok

    num_days = (first - pick).days

    

    if num_days > row['sla_days']: #earlier stop

        for hol in hol_list:

            if pick <= hol <= first:

                num_days -= 1

        if num_days > row['sla_days']:

            return 1

    

    return 0

    

df['is_late'] = df.apply(is_late, axis=1)



print("Algo time elapsed:", time.time() - algo_time, "seconds")
submission = df.copy()

submission['is_late'] = submission['is_late'].fillna(0)

submission = submission[['orderid', 'is_late']]

submission['is_late'] = submission['is_late'].astype(int)



submission['is_late'].value_counts()
submission.to_csv('submission.csv', index=False)
print("Overall time elapsed:", time.time() - start_time, "seconds")