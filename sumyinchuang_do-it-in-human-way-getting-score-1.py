import pandas as pd

import datetime

import numpy as np
input_path = '/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv'

df = pd.read_csv(input_path)

df = df.drop(columns = ["orderid"])

time = pd.to_datetime(df.event_time)

df['event_time'] = time

df = df.sort_values(by="event_time").reset_index(drop = True)

grouped = df.groupby('shopid')

delta = pd.Timedelta(hours=1)

sec = pd.Timedelta(seconds = 1)
userdict = {}

for name,group in grouped:

    

    #step 1 look for repeated buyers

    #generate possible_idxs

    possible_idxs = []

    userlist = []

    dups_user = group.pivot_table(index=['userid'], aggfunc='size')

    if dups_user.max() >=3:

        max_user = (dups_user[dups_user.values >=3 ])

        userlist = (max_user.index.values)

    if len(userlist):

        for user in userlist:

            suspected_df = group[group['userid'] == user]

            #step 2 only pick 3 purchases first

            for start_idx in range(len(suspected_df)-2):

                start_time = suspected_df.iloc[start_idx].event_time

                end_idx = start_idx + 2

                end_time = suspected_df.iloc[end_idx].event_time

                #this if condition is to make sure the three purchases are made within 1 hour (this is step 1 actually 

                #but this is how programming works)

                if end_time <= start_time + delta:

                    possible_idxs.append([suspected_df.iloc[start_idx].name, suspected_df.iloc[end_idx].name])       

    

    #so now we have all the possible time period order brushing likely to happen

    #we need to expand them into 1 hour and see whether they still fulfil the requirement

    #step 3

    confirmed_idxs = []

    for idx in possible_idxs:

        # here we define the starting, ending, previous and next time event to perform calculation

        start_loc = idx[0]

        start_iloc = np.flatnonzero(group.index==start_loc)[0]

        current_user = group.iloc[start_iloc].userid

        start_time = group.iloc[start_iloc].event_time

        end_loc = idx[1]

        end_iloc = np.flatnonzero(group.index==end_loc)[0]

        end_time = group.iloc[end_iloc].event_time

        #if the period happpens at the boundary, we assign the time before/after for it to be large so we do not

        #need to include more purchase in

        pre_loc = start_iloc - 1 if start_iloc >0 else None

        nex_loc = end_iloc + 1 if end_iloc < (len(group)-1) else None

        pre_loc_time = group.iloc[pre_loc].event_time + sec if pre_loc else start_time - 1000*delta

        nex_loc_time = group.iloc[nex_loc].event_time - sec if nex_loc else end_time + 1000*delta

        pre_user = group.iloc[pre_loc].userid if pre_loc else 0

        nex_user = group.iloc[nex_loc].userid if nex_loc else 0    

        # first we check is it fulfil 1 hour time already

        while nex_loc_time - pre_loc_time < delta:

            pre_pre = pre_loc -1 if pre_loc else None

            nex_nex = nex_loc +1 if nex_loc < (len(group)-1) else None

            pre_pre_time = group.iloc[pre_pre].event_time + sec if pre_pre else pre_loc_time - 1000*delta

            nex_nex_time = group.iloc[nex_nex].event_time - sec if nex_nex else nex_loc_time + 1000*delta

            td1= pre_loc_time - pre_pre_time

            td2= nex_nex_time - nex_loc_time

            #if not fulfil 1 hour time

            #we first choose the neighbour point by the userid

            #we want to include the same buyer aka consecutive purchase which will increate the concentration rate

            if pre_user == current_user:

                start_iloc = pre_loc

                pre_loc = pre_pre

                pre_loc_time = pre_pre_time

                pre_user = group.iloc[pre_loc].userid if pre_loc else 0

                continue

            elif nex_user == current_user:

                end_iloc = nex_loc

                nex_loc = nex_nex

                nex_loc_time = nex_nex_time

                nex_user = group.iloc[nex_loc].userid if nex_loc else 0

                continue

            #next we pick the one with larger space to include minimum number of purchase from other buyers

            #this is a little bit tricky but you can think of it

            if td1> td2:

                start_iloc = pre_loc

                pre_loc = pre_pre

                pre_loc_time = pre_pre_time

                pre_user = group.iloc[pre_loc].userid if pre_loc else 0

            else:

                end_iloc = nex_loc

                nex_loc = nex_nex

                nex_loc_time = nex_nex_time

                nex_user = group.iloc[nex_loc].userid if nex_loc else 0

        #now we check the concentration rate

        #calculate concentration_rate

        pur_cnt = end_iloc - start_iloc +1

        user_cnt = group.iloc[start_iloc:end_iloc+1].userid.nunique()

        c_rate = pur_cnt/user_cnt

        if c_rate >= 3:

            current_list = [x for x in range(start_iloc,end_iloc+1)]

            confirmed_idxs += current_list



    #here we might include some duplicated value

    #remove duplicate value



    confirmed_idxs = list(set(confirmed_idxs))

    confirmed_idxs.sort()

    

    #now we have all the timeframe with order brushing

    #we just need to find the user with the largest number of purchase here

    #I used pivot table there to calculate

    answer = 0

    if confirmed_idxs:

        confirmed_df = group.iloc[confirmed_idxs]

        confirmed_dups_user = confirmed_df.pivot_table(index=['userid'], aggfunc='size')

        max_user = (confirmed_dups_user[confirmed_dups_user == confirmed_dups_user.max()])

        answer = '&'.join(str(x) for x in max_user.index)



    userdict[name]= answer

    

    

#now we have the answer!!

print(userdict)

print(len(userdict))
output = pd.DataFrame()

output['shopid'] = userdict.keys()

output['userid'] = userdict.values()

output.to_csv('/kaggle/working/prediction.csv',index=False)