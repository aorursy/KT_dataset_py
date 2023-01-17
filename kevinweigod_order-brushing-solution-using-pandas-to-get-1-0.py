import pandas as pd

input_path = '/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv'

df = pd.read_csv(input_path)

df.head()
df.groupby('shopid').size().describe()
df.groupby('shopid')['userid'].count().sort_values(ascending=False).head(10)
unique_shopid = df['shopid'].unique()

print(f'{len(unique_shopid)} shops unique')



shopid_over3_df = pd.DataFrame(df.groupby('shopid').size()[df.groupby('shopid').size() >= 3]).to_dict()

shopid_over3 = shopid_over3_df[0].keys()

print(f'{len(shopid_over3)} shops more than 3 orders')
df['event_time'] = pd.to_datetime(df['event_time'])

df['ts'] = df[['event_time']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)

df.head()
df = df.sort_values(by='ts', ascending=True)

df
from datetime import datetime

import time



count = 0

ans_dict = dict()



for shop_id in shopid_over3:

    time_record = time.time()

    cheat = 0

    # record the cheating row index in shop_df

    cheat_order_list = set() 

    

    shop_df = df[df['shopid'] == shop_id]

    len_shop_data = len(shop_df)

    

    # initial previous_order_time which is 1 hour earlier than the start_order_time, here we set 9999 secs which is greater than 3600 secs (aka 1 hour)

    previous_order_time = shop_df.iloc[0]['ts'] - 9999

    # We don't need to check for the last 2 orders since it need more than 3 orders to fit the condition of brushing order

    for start_order in range(len_shop_data - 2): 

        start_order_time = shop_df.iloc[start_order]['ts']

        # same as the reason above, end_order is start from start_order+2

        for end_order in range(start_order + 2, len_shop_data): 

            

            # start_order_time and end_order_time need to within 1 hour

            end_order_time = shop_df.iloc[end_order]['ts']

            if end_order_time > start_order_time + 3600:

                break

            

            # Find the event time of next_order

            # edge condition which end_order is the last order of this shop

            if end_order == len_shop_data - 1: 

                # similar to how we did for previous_order_time

                next_order_time = shop_df.iloc[end_order]['ts'] + 9999 

            else:

                next_order_time = shop_df.iloc[end_order + 1]['ts']

            

            # check whether next_order_time - previous_order_time > 1 hour

            if next_order_time > previous_order_time + 3601:

                # we can not divide orders which have same timestamp into different time interval

                if next_order_time == end_order_time: 

                    continue



                tmp_df = shop_df.iloc[start_order: end_order + 1]

                order_num = tmp_df.orderid.nunique()

                user_num = tmp_df.userid.nunique()



                # check whether order brushing

                if order_num / user_num >= 3:

                    cheat = 1

                    cheat_order_list.update(list(range(start_order, end_order + 1)))

        # refresh the previous order time

        previous_order_time = start_order_time 

        

    count += 1 



    if cheat == 1:

        # find the brushing user

        tmp_df = shop_df.iloc[list(cheat_order_list)]

        tmp_dict = tmp_df.groupby('userid').size().to_dict()

        max_time = max(list(tmp_dict.values()))

        

        ans_dict[shop_id] = []

        for user in tmp_dict:

            if tmp_dict[user] == max_time:

                ans_dict[shop_id].append(user)

                        

        print(f'{count:4d}, shop {shop_id} used {time.time() - time_record:4.2f} sec(s), cheat!!!')
# count how many shop is brushing (tips: the true num is 315 shops)

len(ans_dict)
ans_shop = []

ans_user = []

for shop in ans_dict.keys():

    if len(ans_dict[shop]) > 1:

        user_cheat = set(ans_dict[shop])

        user_cheat = sorted(list(user_cheat), reverse=False)

        ans_user.append("&".join([str(i) for i in user_cheat]))

    else:

        ans_user.append(str(ans_dict[shop][0]))

    ans_shop.append(shop)



for shop in unique_shopid:

    if shop not in ans_dict.keys():

        ans_shop.append(shop)

        ans_user.append("0")



print(len(ans_shop))

print(len(ans_user))



df_ans = pd.DataFrame({'shopid': ans_shop, 'userid': ans_user})

df_ans.to_csv('/kaggle/working/prediction.csv',index=False)