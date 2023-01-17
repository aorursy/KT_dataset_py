import pandas as pd

import re

from datetime import datetime

import numpy as np

pd.options.mode.chained_assignment = None  # |default='warn'
df_sla = pd.read_excel("../input/open-shopee-code-league-logistic/SLA_matrix.xlsx")
## SLA Matrix

df_sla
## Delivery data

df_delivery = pd.read_csv("../input/open-shopee-code-league-logistic/delivery_orders_march.csv")
df_delivery.shape
df_delivery.head()
## SLA matrix

matrix = df_sla.iloc[1:5, 2:].values

city_order = ["metro manila", "luzon", "visayas", "mindanao"]
matrix
## SLA 2nd attempt

print(df_sla.iloc[8:,:1].values[0][0])
## Sample orders

df_sample = df_delivery.sample(n=1000)

df_sample.head()
def find_city(x):

    for city in city_order:

        if city.lower() in x:

            return city.lower()
## Get Origin

def get_origin(df):

    origin = find_city(df['selleraddress'].lower())

    return origin
## Get Destination

def get_destination(df):

    destination = find_city(df['buyeraddress'].lower())

    return destination
df_sample['origin'] = df_sample.apply(get_origin, axis=1)

df_sample['destination'] = df_sample.apply(get_destination, axis=1)
df_sample.head()
def get_sla(df):

    sla = matrix[city_order.index(df['origin']), city_order.index(df['destination'])]

    days = int(re.search(r'\d', sla).group(0))

    return days
df_sample['sla'] = df_sample.apply(get_sla, axis=1)
df_sample.head()
## Convert all date column to date

pick = pd.to_datetime(df_sample['pick'], unit='s').dt.date

first_deliver = pd.to_datetime(df_sample['1st_deliver_attempt'], unit='s').dt.date

second_deliver = pd.to_datetime(df_sample['2nd_deliver_attempt'], unit='s').dt.date
df_sample['pick'] = pick

df_sample['1st_deliver_attempt'] = first_deliver

df_sample['2nd_deliver_attempt'] = second_deliver
df_sample.head()
## Initiate public holidays

public_holidays = ["2020-03-08", "2020-03-25", "2020-03-30", "2020-03-31"]
def get_busday_first(df):

    create_date = str(df['pick'])

    resolve_date = str(df['1st_deliver_attempt'])



    create_datetime = datetime.strptime(create_date, '%Y-%m-%d')

    resolve_datetime = datetime.strptime(resolve_date, '%Y-%m-%d')



    busday = np.busday_count(create_date, resolve_date, holidays=public_holidays, weekmask=[1,1,1,1,1,1,0])



    return busday    
df_sample['1st_deliver_days'] = df_sample.apply(get_busday_first, axis=1)
df_sample.head()
## Saving checkpoint

# import pickle

# pickle.dump(df_sample, open("1st_fullfilment.pickle", "wb"))
def get_busday_second(df):

    create_date = str(df['1st_deliver_attempt'])

    resolve_date = str(df['2nd_deliver_attempt'])

    

    if resolve_date == 'NaT':

        resolve_date = create_date



    create_datetime = datetime.strptime(create_date, '%Y-%m-%d')

    resolve_datetime = datetime.strptime(resolve_date, '%Y-%m-%d')



    busday = np.busday_count(create_date, resolve_date, holidays=public_holidays, weekmask=[1,1,1,1,1,1,0])



    return busday   
df_sample['2nd_deliver_days'] = df_sample.apply(get_busday_second, axis=1)
df_sample.head()
## Saving checkpoint

# import pickle

# pickle.dump(df_sample, open("2nd_fullfilment.pickle", "wb"))
not_late_df = df_sample[(df_sample['1st_deliver_days'] <= df_sample['sla']) & (df_sample['2nd_deliver_days'] <= 3)][['orderid']]

not_late_df['is_late'] = 0

not_late_df.head()
late_df = df_sample[~df_sample['orderid'].isin(not_late_df['orderid'])][['orderid']]

late_df['is_late'] = 1

late_df.head()
not_late_df.orderid.nunique() + late_df.orderid.nunique()
res_df = pd.concat([late_df, not_late_df], axis=0)

res_df.head()