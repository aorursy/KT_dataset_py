#importing required libraries

import pandas as pd

import numpy as np

import sklearn as skl



from pandas import Series,DataFrame



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



from datetime import datetime, timedelta

from collections import defaultdict
df = pd.read_csv("../input/shopee-code-league-20/_DA_Order_Brushing/order_brush_order.csv")

df.head()
df_subset = df.head(100500)
shop = defaultdict(list)
for i in range(df_subset.shape[0]):

    orderid, sid, userid, time = df.iloc[i]

    #print(et)

    et = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

    shop[sid].append((et, userid, orderid))
#Converting  to a normal dict.

data = dict(shop)
shops = df_subset['shopid'].unique()
shops_dict = defaultdict(list)

for shop in shops:

    #print(shop)

    shops_dict[shop].append(data[shop])

        

shops_dict = dict(shops_dict)
""" mul_shops gives a list of shopids with more than 1 purchase"""

mul_shops = []

for shop in shops_dict:

    if (len(shops_dict[shop][0])>1):

        mul_shops.append(shop)
""" user_dict gives a list of users for each shop"""

user_dict = defaultdict(list)

for shop in mul_shops:

    for i in range(len(shops_dict[shop][0])):

        user_dict[shop].append(shops_dict[shop][0][i][1])



user_dict = dict(user_dict)
order=[]

shopid=[]

for shop in mul_shops:

    """Obtaining the min and max time of purchase of a shop with multiple purchases will give a time-span of buying"""

    max_time = max(shops_dict[shop][0])

    min_time = min(shops_dict[shop][0])



    time_delta = max_time[0]-min_time[0] # time_delta is the time-span of buying



    """ Order-concentration can be interpreted as orders per each hr of purchase"""

    order_concentration = (time_delta/len(shops_dict[shop][0])).seconds/3600

    

    """ For each hour, we are looking at how much of the order quantity belonged to each one of unique customer"""

    order_per_cust_con = order_concentration/len(set(user_dict[shop]))  #unique user found out using set-function

    #this pretty much gives the idea of how much order is concentrated to each customer.

                                                                      

    if (order_per_cust_con > 3): #Considering shops with order concentration is greater than 3 per hr per customer.

        order.append(order_per_cust_con)

        shopid.append(shop)
suspicious_order_brushing = DataFrame({'ShopID':shopid,'Order per UniqueCust':order})

suspicious_order_brushing