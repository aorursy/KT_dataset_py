# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, timedelta





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
headers = ['orderid', 'shopid', 'userid', 'event_time']

dtypes = {'orderid': 'int', 'shopid': 'int', 'userid': 'int', 'event_time': 'float'}

parse_dates = ['event_time']

df = pd.read_csv('../input/orderbrushing/order_brush_order.csv', dtype=dtypes, parse_dates=parse_dates)

df['event_time'] = pd.to_datetime(df['event_time'],format ='%Y-%m-%d %H:%M:%S')

df = df.sort_values(by='shopid',ascending =True)
shopidlist = []

useridlist = []

dict = {}

grouped_df = df.groupby('shopid')

for index, rows in grouped_df:  # Grouped transactions by distinct shop ids



    if len(rows) <= 2:

        if rows['shopid'].iloc[0] not in shopidlist:

            shopidlist.append(

                rows['shopid'].iloc[0])  # Each shop with less than 3 distinct transactions cannot be suspicious

            useridlist.append('0')

    else:



        df_timesorted = rows.sort_values(by='event_time')

        for index, timesortedrows in df_timesorted.iterrows():

            etime = timesortedrows['event_time']



            timeblock = df_timesorted.loc[

                (df_timesorted['event_time'] >= etime) & (df_timesorted['event_time'] < etime + timedelta(hours=1))]



            sus_transaction = []  # list of sus transactions

            if len(timeblock) > 2:

                freq = timeblock['userid'].value_counts()

                con_rate = len(timeblock) / len(freq)



                if con_rate >= 3:

                    print(etime)

                    print(timeblock)

                    print(freq.values[0])

                    maxcount = freq.max()  # Gets data on each order brushing period

                    for i in range(0, len(freq)):

                        if freq.values[i] == maxcount:

                            if [freq.index.values[i], maxcount] not in sus_transaction:

                                sus_transaction.append([freq.index.values[i],

                                                        maxcount])  # Store userids with highest transaction count per brushing period

                    if timesortedrows['shopid'] not in dict:

                        dict[timesortedrows['shopid']] = sus_transaction

                    else:

                        for i in range(0, len(sus_transaction)):

                            for j in range(0, len(dict[timesortedrows['shopid']])):

                                if sus_transaction[i][0] == dict[timesortedrows['shopid']][j][0]:

                                    if dict[timesortedrows['shopid']][j][1] < sus_transaction[i][1]:

                                        dict[timesortedrows['shopid']][j][1] = sus_transaction[i][1]

                else:

                    if timeblock['shopid'].iloc[0] not in shopidlist:

                        shopidlist.append(rows['shopid'].iloc[

                                              0])  # Ensuring all shops are added into the shopidlist. Suspicious users will replace the 0s in useridlist

                        useridlist.append('0')

            else:

                if timeblock['shopid'].iloc[0] not in shopidlist:

                    shopidlist.append(rows['shopid'].iloc[0])

                    useridlist.append('0')



                    

                        



                                    

                    

                

            

                    


shopidlisttemp = shopidlist

useridlisttemp = useridlist

for key, value in dict.items():

    # shopid, [userid,items bought]

    if len(value) == 1:

        for n, j in enumerate(shopidlisttemp):

            if j == key:

                useridlisttemp[n] = value[0][0]

    else:

        totalcount = 0

        highest = 0

        highestuserid = ""

        for i in range(0, len(value)):

            totalcount += value[i][1]

        for i in range(0, len(value)):

            prop = value[i][1] / totalcount  # proportion for each userid

            if prop > highest:

                highest = prop

                highestuserid = str(value[i][0])

            elif prop == highest:

                highestuserid = highestuserid + "&" + str(value[i][0])

        useridlisttemp[n] = highestuserid



newdf = {'shopid': shopidlisttemp, 'userid': useridlisttemp}

newdf = pd.DataFrame(newdf)

print(newdf)

newdf.to_csv("order_brushing_output.csv",index=False)