#Importing Relevant Libraries

import datetime

import pandas as pd

import math

import numpy as np
#Mounting Kaggle Drive

import os

for dirname, _, filenames in os.walk('/kaggle/input'): 

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_orders = pd.read_csv("/kaggle/input/challenge04-dataset/delivery_orders_march.csv")

df_sla = pd.read_csv("/kaggle/input/shopeechallenge04dataset/SLA.csv")
df_orders.head()
#Converting the datetime 

df_orders["pick"] = df_orders["pick"].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))

df_orders["1st_deliver_attempt"] = df_orders["1st_deliver_attempt"].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))



def datetime_converter(x):

    if math.isnan(x):

        return math.nan

    else:

        return datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d')



df_orders["2nd_deliver_attempt"] = df_orders["2nd_deliver_attempt"].apply(datetime_converter)
#Getting Locations

def get_location(x):

    x = x.lower()

    if "metro manila" in x:

        return "Metro Manila"

    elif "luzon" in x:

        return "Luzon"

    elif "visayas" in x:

        return "Visayas"

    elif "mindanao" in x:

        return "Mindanao"

    else:

        return math.nan

    

df_orders["buyeraddress"] = df_orders["buyeraddress"].apply(get_location)

df_orders["selleraddress"] = df_orders["selleraddress"].apply(get_location)
df_orders.columns
#For First Period

holidays_list = ["2020-03-25", "2020-03-30", "2020-03-31"]

week_mask_str = "1111110"



def f(x):

    return [x["pick"], x["1st_deliver_attempt"]]



df_orders['first_period_combine'] = df_orders.apply(f, axis=1)



df_orders["1st Attempt Period"] = df_orders["first_period_combine"].apply(lambda x: np.busday_count(x[0], x[1], weekmask=week_mask_str, holidays = holidays_list))

#df_orders["1st Attempt Period"] = df_orders["1st Attempt Period"].apply(lambda x: x + 1)
#For Second Period

def f(x):

    return [x["1st_deliver_attempt"], x["2nd_deliver_attempt"]]



df_orders['second_period_combine'] = df_orders.apply(f, axis=1)



def period_calculator(x):

    if math.nan in x:

        return -1

    else:

        return np.busday_count(x[0], x[1], weekmask=week_mask_str, holidays = holidays_list)



df_orders["2nd Attempt Period"] = df_orders["second_period_combine"].apply(period_calculator)

#df_orders["2nd Attempt Period"] = df_orders["2nd Attempt Period"].apply(lambda x: x + 1)
df_orders
df_orders = df_orders.drop(columns=["first_period_combine","second_period_combine"])
#Combining Address

def destination_combine(x):

    return [x["buyeraddress"], x["selleraddress"]]

df_orders["Address Combined"] = df_orders.apply(destination_combine, axis=1)





def destination_period(x):

    buyeradd = x[0]

    selleradd= x[1]

    #Origin Metro Manila

    if buyeradd == "Metro Manila" and selleradd == "Metro Manila":

        return 3

    elif buyeradd == "Metro Manila" and selleradd == "Luzon":

        return 5

    elif buyeradd == "Metro Manila" and selleradd == "Visayas":

        return 7

    elif buyeradd == "Metro Manila" and selleradd == "Mindanao":

        return 7

    #Origin Luzon

    elif buyeradd == "Luzon" and selleradd == "Metro Manila":

        return 5

    elif buyeradd == "Luzon" and selleradd == "Luzon":

        return 5

    elif buyeradd == "Luzon" and selleradd == "Visayas":

        return 7

    elif buyeradd == "Luzon" and selleradd == "Mindanao":

        return 7

    #Origin Visayas

    elif buyeradd == "Visayas" and selleradd == "Metro Manila":

        return 7

    elif buyeradd == "Visayas" and selleradd == "Luzon":

        return 7

    elif buyeradd == "Visayas" and selleradd == "Visayas":

        return 7

    elif buyeradd == "Visayas" and selleradd == "Mindanao":

        return 7

    #Origin Mindanao

    elif buyeradd == "Mindanao" and selleradd == "Metro Manila":

        return 7

    elif buyeradd == "Mindanao" and selleradd == "Luzon":

        return 7

    elif buyeradd == "Mindanao" and selleradd == "Visayas":

        return 7

    elif buyeradd == "Mindanao" and selleradd == "Mindanao":

        return 7



df_orders["SLA Working Days"] = df_orders["Address Combined"].apply(destination_period)

df_orders = df_orders.drop(columns=["Address Combined"])
def period_combine(x):

    return [x["1st Attempt Period"], x["2nd Attempt Period"], x["SLA Working Days"]]



df_orders["Period Combined"] = df_orders.apply(period_combine, axis=1)
df_orders
def punctuality_status(x):

    first = x[0]

    second = x[1]

    period = x[2]

    

    #First attempt sucessful

    if second == -1:

        if first <= period:

            return 0

        else:

            return 1

    #Second Attempt

    else:

        if second <= 3:

            return 0

        else:

            return 1

    

df_orders["is_late"] = df_orders["Period Combined"].apply(punctuality_status)
#Removing all other columns excepts for orderid and is_late

df_orders.drop(df_orders.columns.difference(['orderid','is_late']), 1, inplace=True)
df_orders[df_orders["is_late"]==1].count()
df_orders.to_csv("potato_challenge_04_results.csv", index=False)