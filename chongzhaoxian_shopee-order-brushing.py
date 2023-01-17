import pandas as pd

import numpy as np

from pandas.tseries.offsets import DateOffset

pd.options.mode.chained_assignment = None
df = pd.read_csv("../input/open-2-shopee-code-league-order-brushing/order_brush_order.csv")

df_raw = df
df['event_time'] = pd.to_datetime(df.event_time)

df = df.set_index('event_time').sort_index()

df = df.reset_index().set_index('orderid')

shop_list = list(df.shopid.unique())
shop = shop_list[0]

shop_df = df[df['shopid']==shop]

shop_df['rolling_H'] = shop_df[::-1][['event_time','userid']].rolling('H', on='event_time').count().userid

shop_df.reset_index(inplace=True)

index_no = list(shop_df[shop_df['rolling_H']>=3].index)

rolling_no = list(shop_df.rolling_H.loc[index_no])

k = 0

shop_df['orderBrush'] =''

for i in index_no:

  temp_df = shop_df.iloc[i: int((i + rolling_no[k])) ,:]

  no_user = len(temp_df.userid.unique())

  conc = rolling_no[k]/no_user

  if conc >= 3:  

    shop_df.iloc[i: (i + int(rolling_no[k]))  ,5] = 1

  else:

    c = rolling_no[k]-1

    if i != 0:

      time = shop_df.event_time.iloc[i-1]

    else:

      time = shop_df.event_time.iloc[i] - DateOffset(hours=1)

    while (c>=3) & (conc<=3):

      if temp_df.event_time.loc[int(i + c)] -DateOffset(hours=1) > time:

        temp_df = shop_df.iloc[i: int((i + c)) ,:]

        no_user = len(temp_df.userid.unique())

        conc = c/no_user

      else:

        break

      if conc >= 3:  

        shop_df.iloc[i: int(i+c) ,5] = 1

        break

      else:

        pass

      c-=1

  k+=1

complete_df = shop_df

del shop_list[0]
#d = 1

for shop in shop_list:

  shop_df = df[df['shopid']==shop]

  shop_df['rolling_H'] = shop_df[::-1][['event_time','userid']].rolling('H', on='event_time').count().userid

  shop_df.reset_index(inplace=True)

  index_no = list(shop_df[shop_df['rolling_H']>=3].index)

  rolling_no = list(shop_df.rolling_H.loc[index_no])

  k = 0

  shop_df['orderBrush'] =''

  for i in index_no:

    temp_df = shop_df.iloc[i: int((i + rolling_no[k])) ,1:4:2]

    no_user = len(temp_df.userid.unique())

    conc = rolling_no[k]/no_user

    if conc >= 3:  

      shop_df.iloc[i: (i + int(rolling_no[k]))  ,5] = 1

    else:

      c = rolling_no[k]-1

      if i != 0:

        time = shop_df.event_time.iloc[i-1]

      else:

        time = shop_df.event_time.iloc[i] - DateOffset(hours=1)

      for p in range(int(c)-2):

      #while (c>=3) & (conc<=3):

        if temp_df.event_time.loc[int(i + c)] -DateOffset(hours=1) > time:

          temp_df = shop_df.iloc[i: int((i + c)) ,1:4:2]

          no_user = len(temp_df.userid.unique())

          conc = c/no_user

        else:

          break

        if conc >= 3:  

          shop_df.iloc[i: int(i+c) ,5] = 1

          break

        else:

          pass

        c-=1

    k+=1

  complete_df = pd.concat([complete_df,shop_df]) 

  #print(d)

  #d+=1
complete = complete_df[complete_df['orderBrush']==1]

s_shop = list(complete.shopid)
shop = s_shop[0]

temp = complete[complete['shopid']==shop]

df_counts = temp['userid'].value_counts()

max_count = df_counts.max()

s_user = list(df_counts[df_counts==max_count].index)

for user in s_user:

  temp.loc[temp['userid']==user,'S_User'] = user

complete2_df = temp

del s_shop[0]



for shop in s_shop:

  temp = complete[complete['shopid']==shop]

  df_counts = temp['userid'].value_counts()

  max_count = df_counts.max()

  s_user = list(df_counts[df_counts==max_count].index)

  for user in s_user:

    temp.loc[temp['userid']==user,'S_User'] = user

  complete2_df = pd.concat([complete2_df,temp]) 
complete2 = complete2_df.dropna()

complete2 = complete2[['shopid','userid']]

complete2.drop_duplicates(inplace=True)

first = complete2[complete2.duplicated(subset = 'shopid',keep='first')]

last = complete2[complete2.duplicated(subset = 'shopid',keep='last')]

dupli = pd.merge(first,last, on='shopid')

for i in range(len(dupli)): 

  if dupli.loc[i,'userid_x'] < dupli.loc[i,'userid_y']:

    dupli.loc[i,['userid_x','userid_y']] = dupli.loc[i,['userid_x','userid_y']].astype(str)

    dupli.loc[i,'userid'] = dupli.loc[i,'userid_x'] + '&'+ dupli.loc[i,'userid_y']

  elif dupli.loc[i,'userid_x'] > dupli.loc[i,'userid_y']:

    dupli.loc[i,['userid_x','userid_y']] = dupli.loc[i,['userid_y','userid_x']].astype(str)

    dupli.loc[i,'userid'] = dupli.loc[i,'userid_y'] + '&'+ dupli.loc[i,'userid_x']

  else:

    pass

dupli = dupli[['shopid','userid']]

merging1 = pd.concat([complete2,dupli])

merging1.drop_duplicates(subset='shopid',keep='last',inplace=True)
df_raw = df_raw[['shopid']]

df_raw.drop_duplicates(inplace=True)

merging = pd.merge(df_raw,merging1,how='left',on='shopid')

merging.fillna(0,inplace=True)
merging.to_csv('/kaggle/working/result.csv',index=False)