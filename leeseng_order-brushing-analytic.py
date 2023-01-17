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
import pandas as pd

import datetime

import numpy as np



# 0.96719, but late submission. LOL
df = pd.read_csv('/kaggle/input/order_brush_order.csv')

base27=datetime.datetime.strptime('2019-12-27 00:00:00', '%Y-%m-%d %H:%M:%S')

df['seconds'] = df['event_time'].apply(lambda x : (datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')-base27).days*86400+(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')-base27).seconds)

df=df.sort_values(by='seconds')
df.describe(include='all')
df['event_time'].sort_values()
#only analyze transactions that's more than 3 for any shop and userid pairs.



g = df.groupby(['shopid','userid'])

sdf = g.filter(lambda x: len(x['orderid'])>=3)

print('filtered.')





shopids = sdf['shopid'].unique()

result_df = pd.DataFrame(columns=['shopid','userid'])



i=0



for si in shopids:

    i=i+1

    if i%10==0:

        print('shop #: ',i)



    shop_df = sdf[sdf['shopid']==si]

    seconds = shop_df['seconds'].unique()

    seconds.sort()





    

    # print('result:',result_df)

    last_s=-1

    for s in seconds:

        if s>=last_s:

            last_s=last_s+30 # flusher takes about 30 seconds to create 1 order. optimization.

        else:

            continue

#         if s%1000==0:

#             print('s:',s)



        hour_df = shop_df[(shop_df['seconds'] >= s) & (shop_df['seconds']<=s+3600)]

        a =hour_df.groupby(['shopid'], as_index=False)['orderid'].count().rename(columns={'orderid':'count'}) # transaction count

        b = hour_df.groupby(['shopid', 'userid'], as_index=False).size().reset_index().rename(columns={0:'count'}) # users count

    #     print('b:', b)

        b = b[['shopid', 'count']].groupby('shopid', as_index=False).count() # users count



    #     print('a:',a)



    #     print('b:',b)

    #     print(a)

    #     print(b)

        shopids =a[((a/b>=3)['count'])]['shopid'].unique()

    #     print('shopids:',shopids)



        for c in shopids: # null check for concentration rate



            for u in hour_df['userid'].unique():

                if sum(hour_df['userid']==u)>=3:

    #                 print(u)

                    rdf=pd.DataFrame({'shopid':[si], 'userid':[u]})

                    result_df = pd.concat([result_df,rdf])
suspicious_df = result_df[~result_df.duplicated()]

suspicious_df = suspicious_df.sort_values(by=['shopid','userid'])



data = pd.DataFrame(df['shopid'].unique(), columns = ['shopid'])

data = data.sort_values(by=['shopid'])

data['userid'] = '0'



for i in range(len(suspicious_df)):

    if data[data['shopid']==suspicious_df.iloc[i]['shopid']]['userid'].values=='0':

        data['userid'][data['shopid']==suspicious_df.iloc[i]['shopid']] = str(suspicious_df.iloc[i]['userid'])

#         print(str(suspicious_df.iloc[i]['userid']))

        print(data[data['shopid']==suspicious_df.iloc[i]['shopid']]['userid'])

        

    else:

        data['userid'][data['shopid']==suspicious_df.iloc[i]['shopid']] = data[data['shopid']==suspicious_df.iloc[i]['shopid']]['userid']+'&'+str(suspicious_df.iloc[i]['userid'])

        



data.to_csv('submission.csv', index=False)
result_df