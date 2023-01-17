# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/students-2-shopee-code-league-order-brushing/order_brush_order.csv")
df.head()
df.describe()
df.dtypes
df.shape
unique_shops_and_buyers = df.drop_duplicates(subset=['shopid', 'userid']).drop(columns=['orderid', 'event_time'])

unique_shops_and_buyers = unique_shops_and_buyers.sort_values(by=['shopid', 'userid'])

unique_shops_and_buyers.head()
df['event_time'] = pd.to_datetime(df.event_time)
df.sort_values(by=['shopid', 'event_time', 'userid'], inplace=True)

df.head(16)
unique_shops = unique_shops_and_buyers.drop(columns=['userid']).drop_duplicates(subset=['shopid'])

unique_shops.shape
df.dtypes
df = df.set_index('event_time').sort_index()
df.head()
def keep(window, windows):

    windows.append(window.copy().unique().tolist())

    return window[-1]
#iterate through

ret = []

for i, row in unique_shops.iterrows():

    shopid = row['shopid'] 

    total_transactions = df.loc[df['shopid'] == shopid]

    total, col = total_transactions.shape

    if (total < 3):

        ret.append([shopid, 0])

    else:

        suspicious = []

        total_transactions = total_transactions.sort_index()

        total_transactions['seen'] = 1

        # The rolling sum gives us the moving sum in 1 hour periods for each shop. This gives us the total

        # number of transactions in any given hour based on datetime index.

        total_transactions['hourly'] = total_transactions['seen'].rolling('3600s',closed='both').sum()

        total_transactions = total_transactions.drop(columns=['shopid']).sort_values(by=['userid']).sort_index()

        total_transactions = total_transactions.drop(columns=['orderid'])

        if (total_transactions['hourly'].max() < 3):

            ret.append([shopid, 0])

            continue

        # https://stackoverflow.com/questions/45254174/how-do-pandas-rolling-objects-work the ans to accumulating over a window

        # The following stack overflow method uses another list to store the unique userid which contributes to the hourly transactions

        total_transactions['strid'] = total_transactions['userid'].apply(lambda val: str(val))

        windows = list()

        total_transactions['strid'].rolling('3600s',closed='both').apply(keep, args=(windows,))

        total_transactions['window'] = windows

        total_transactions['count'] = total_transactions['window'].apply(lambda val: len(val))

        total_transactions['ratio'] = total_transactions['hourly'] / total_transactions['count']

                

        total_transactions = total_transactions.loc[total_transactions['ratio'] >= 3.0]



        

        totals, col = total_transactions.shape

        

        if totals == 0:

            ret.append([shopid, 0])

            continue

        else:

            # Formating suspicious userid

            suspicious = total_transactions['strid'].unique().tolist()

            temp = ""

            for val in suspicious:

                temp = temp + '&' + str(val)

            #

            temp = temp[1:]

            ret.append([shopid, temp])



final_df = pd.DataFrame(ret, columns=['shopid', 'userid'])
final_df.to_csv("/kaggle/working/submission.csv", index=False)