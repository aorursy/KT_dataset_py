# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import math

import os

for dirname, _, filenames in os.walk('/kaggle/input'):     

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class StringConverter(dict):

    def __contains__(self, item):

        return True



    def __getitem__(self, item):

        return str



    def get(self, default=None):

        return str
credit_df = pd.read_csv("/kaggle/input/ungrd-rd2-auo/credit_cards.csv", converters=StringConverter())

device_df = pd.read_csv("/kaggle/input/ungrd-rd2-auo/devices.csv", converters=StringConverter())

account_df = pd.read_csv("/kaggle/input/ungrd-rd2-auo/bank_accounts.csv", converters=StringConverter())

order_df = pd.read_csv("/kaggle/input/ungrd-rd2-auo/orders.csv", converters=StringConverter())
## master user dataframe

## get unique user df

user_df = pd.DataFrame({'userid':pd.concat([order_df.buyer_userid,order_df.seller_userid]).unique()})

user_df.tail()

print(user_df)
user_df = user_df.merge(credit_df, on='userid', how='left')

user_df = user_df.merge(device_df, on='userid', how='left')

user_df = user_df.merge(account_df, on='userid', how='left')

user_df.head(5)
#master_user_df = user_df.groupby(['userid']).agg(lambda x: set(x)).applymap(list).reset_index()

master_user_df = user_df.groupby(['userid']).agg(lambda x: set(x)).reset_index()

master_user_df.head()
df_temp = master_user_df[:]
df_temp.set_index('userid', inplace=True)

df_temp.head()
device_df.head()
## dict of unique credit_card, device, bank    

master_device_df = device_df.groupby(['device']).agg(lambda x: set(x)).reset_index()

master_device_df.set_index('device', inplace=True)

master_account_df = account_df.groupby(['bank_account']).agg(lambda x: set(x)).reset_index()

master_account_df.set_index('bank_account', inplace=True)

master_credit_df = credit_df.groupby(['credit_card']).agg(lambda x: set(x)).reset_index()

master_credit_df.set_index('credit_card', inplace=True)
master_device_dict = master_device_df.to_dict('index')

master_account_dict = master_account_df.to_dict('index')

master_credit_dict = master_credit_df.to_dict('index')

master_df_dict = df_temp.to_dict('index')

### new main function



# buyer_id = '223406364'

# seller_id = '193350172'

# depth_search = 10

def Find_fruad_userV2(buyer_id ,seller_id,depth_search=3):



    Fruad = 0

    related_user_set = set()

    for infor in master_df_dict[buyer_id].keys():

        for sub_infor in master_df_dict[buyer_id][infor]:

            try:

                if math.isnan(sub_infor):

                    continue

            except:

                #print(sub_infor)

                if infor == 'device':

                    if len(master_device_dict[sub_infor]['userid'])>=2:  ## Found related device user

                        related_user_set.update(master_device_dict[sub_infor]['userid'] - {buyer_id})

                if infor == 'credit_card':

                    if len(master_credit_dict[sub_infor]['userid'])>=2:  ## Found related credit card user

                        related_user_set.update(master_credit_dict[sub_infor]['userid'] - {buyer_id})

                if infor == 'bank_account':

                    if len(master_account_dict[sub_infor]['userid'])>=2:  ## Found related account user

                        related_user_set.update(master_account_dict[sub_infor]['userid'] - {buyer_id})

                ## Found related account user

                ## Found related credit_card user

    #print(f'releated user is {related_user_set}')



    continue_search = True



    for i in range(depth_search - 1):

        if i == 0: 

            next_related_user_set = related_user_set.copy()

            new_found = set()



        else:

            ### if continue search

            if continue_search:

                next_related_user_set = new_found.copy()

                new_found= set()



        #print(f'depth: {i+2}')

        if continue_search:

            for user_id in next_related_user_set:

                #print(f'------------User id: {user_id}------')

                for infor in master_df_dict[user_id].keys():

                    for sub_infor in master_df_dict[user_id][infor]:

            

                        try:

                            if math.isnan(sub_infor):

                                continue

                        except:

         

                            if infor == 'device':

                                if len(master_device_dict[sub_infor]['userid'] - {buyer_id} - related_user_set)>=1:  ## Found related user

                                   new_found.update(master_device_dict[sub_infor]['userid'] - {buyer_id} - related_user_set)

                                   #print(master_device_dict[sub_infor]['userid'] - {buyer_id} - related_device_user_set)

                            if infor == 'credit_card':

                                if len(master_credit_dict[sub_infor]['userid'] - {buyer_id} - related_user_set)>=1:  ## Found related user

                                   new_found.update(master_credit_dict[sub_infor]['userid'] - {buyer_id} - related_user_set)

                            if infor == 'bank_account':

                                if len(master_account_dict[sub_infor]['userid'] - {buyer_id} - related_user_set)>=1:  ## Found related user

                                   new_found.update(master_account_dict[sub_infor]['userid'] - {buyer_id} - related_user_set)

                

            if len(new_found) == 0: ##device related user is not found then end search

                continue_search = False

            else:

                related_user_set.update(new_found)

                #print(f'Found device related: {new_found_device}')

                #print(related_device_user_set)

        

        if continue_search == False :

            #print('Stop searching process')

            break  



#         print('-------------------------------')



    ## if seller in related user Then buyer is fruad (buyer and seller is the same person)

    if seller_id in related_user_set:

        #print('Yes')

        Fruad = 1

        return Fruad

    return Fruad
# ### new main function



# # buyer_id = '223406364'

# # seller_id = '193350172'

# # depth_search = 10

# def Find_fruad_user(buyer_id ,seller_id,depth_search=3):



#     Fruad = 0

#     related_device_user_set = set()

#     related_credit_user_set = set()

#     related_account_user_set = set()

#     for infor in master_df_dict[buyer_id].keys():

#         for sub_infor in master_df_dict[buyer_id][infor]:

#             try:

#                 if math.isnan(sub_infor):

#                     continue

#             except:

#                 #print(sub_infor)

#                 if infor == 'device':

#                     if len(master_device_dict[sub_infor]['userid'])>=2:  ## Found related device user

#                         related_device_user_set.update(master_device_dict[sub_infor]['userid'] - {buyer_id})

#                 if infor == 'credit_card':

#                     if len(master_credit_dict[sub_infor]['userid'])>=2:  ## Found related credit card user

#                         related_credit_user_set.update(master_credit_dict[sub_infor]['userid'] - {buyer_id})

#                 if infor == 'bank_account':

#                     if len(master_account_dict[sub_infor]['userid'])>=2:  ## Found related account user

#                         related_account_user_set.update(master_account_dict[sub_infor]['userid'] - {buyer_id})

#                 ## Found related account user

#                 ## Found related credit_card user

#     print(f'releated device user is {related_device_user_set}')

#     print(f'releated credit card user is {related_credit_user_set}')

#     print(f'releated account user is {related_account_user_set}')

#     continue_device_search = True

#     continue_credit_search = True

#     continue_account_search = True

#     for i in range(depth_search - 1):

#         if i == 0: 

#             next_device_related_user_set = related_device_user_set.copy()

#             next_credit_related_user_set = related_credit_user_set.copy()

#             next_account_related_user_set = related_account_user_set.copy()

#             new_found_device = set()

#             new_found_credit = set()

#             new_found_account = set()

#         else:

#             ### if continue search

#             if continue_device_search:

#                 next_device_related_user_set = new_found_device.copy()

#                 new_found_device = set()

#             if continue_credit_search:  

#                 next_credit_related_user_set = new_found_credit.copy()

#                 new_found_credit = set()

#             if continue_account_search:

#                 next_account_related_user_set = new_found_account.copy()

#                 new_found_account = set()

#         print(f'depth: {i+2}')

#         if continue_device_search:

#             for user_id in next_device_related_user_set:

#                 print(f'------------User id: {user_id}------')

#                 for infor in master_df_dict[user_id]['device']:

            

#                     try:

#                         if math.isnan(infor):

#                             continue

#                     except:

#                         ### Problemmmmmmmmmmmmmmm

#                         print(infor)

#                         if len(master_device_dict[infor]['userid'] - {buyer_id} - related_device_user_set)>=1:  ## Found related user

#                            new_found_device.update(master_device_dict[infor]['userid'] - {buyer_id} - related_device_user_set)

#                            print(master_device_dict[infor]['userid'] - {buyer_id} - related_device_user_set)

                

#             if len(new_found_device) == 0: ##device related user is not found then end search

#                 continue_device_search = False

#                 print('Stop searching on Device')

#             else:

#                 related_device_user_set.update(new_found_device)

#                 #print(f'Found device related: {new_found_device}')

#                 #print(related_device_user_set)

#         if continue_credit_search:  

#             for user_id in next_credit_related_user_set:

#                 for infor in master_df_dict[user_id]['credit_card']:

#                     try:

#                         if math.isnan(infor):

#                             continue

#                     except:

#                         if len(master_credit_dict[infor]['userid'] - {buyer_id} - related_credit_user_set)>=1:  ## Found related user

#                            new_found_credit.update(master_credit_dict[infor]['userid'] - {buyer_id}- related_credit_user_set)

#             if len(new_found_credit) == 0: ##credit related user is not found then end search

#                 continue_credit_search = False

#                 print('Stop searching on Credit card')

#             else:

#                 related_credit_user_set.update(new_found_credit)

#                 #print(f'Found credit related: {new_found_credit}')

#                 #print(related_credit_user_set)           

#         if continue_account_search:   

#             for user_id in next_account_related_user_set:

#                 for infor in master_df_dict[user_id]['bank_account']:

#                     try:

#                         if math.isnan(sub_infor):

#                             continue

#                     except:

#                         if len(master_account_dict[infor]['userid'] - {buyer_id} - related_account_user_set)>=1:  ## Found related user

#                            new_found_account.update(master_account_dict[infor]['userid'] - {buyer_id}- related_account_user_set)

#             if len(new_found_account) == 0: ##account related user is not found then end search

#                 continue_account_search = False

#                 print('Stop searching on Bank account')

#             else:

#                 related_account_user_set.update(new_found_account)

#                 #print(f'Found account related: {new_found_account}')

#                 #print(related_credit_user_set)



#         if continue_device_search == False and continue_credit_search == False and continue_account_search == False :

#             print('Stop searching process')

#             break  



# #         print('-------------------------------')



# #     print('-----------------Final-----------------')

# #     print(related_device_user_set)

# #     print(related_credit_user_set)

# #     print(related_account_user_set)



#     ## if seller in related user Then buyer is fruad (buyer and seller is the same person)

#     if seller_id in related_device_user_set:

#         Fruad = 1

#         return Fruad

#     if seller_id in related_credit_user_set:

#         Fruad = 1

#         return Fruad

#     if seller_id in related_account_user_set:

#         Fruad = 1

#         return Fruad

#     return Fruad
# ### Dict search (Can handle only direct user)

# def check_same_devicev2(buyer_id,seller_id):

#     Fruad = False

#     buyer_value_set = master_df_dict[buyer_id]['device']

#     if len(buyer_value_set) == 1 and \

#         np.nan in buyer_value_set:

#         return Fruad

#     seller_value_set = master_df_dict[seller_id]['device']

#     if len(seller_value_set) == 1 and \

#         np.nan in seller_value_set:

#         return Fruad

#     for value in buyer_value_set:

#         if value in seller_value_set:

#             Fruad = True

#             return Fruad

#     return Fruad

# def check_same_accountv2(buyer_id,seller_id):

#     Fruad = False

#     buyer_value_set = master_df_dict[buyer_id]['credit_card']

#     if len(buyer_value_set) == 1 and \

#         np.nan in buyer_value_set:

#         return Fruad

#     seller_value_set = master_df_dict[seller_id]['credit_card']

#     if len(seller_value_set) == 1 and \

#         np.nan in seller_value_set:

#         return Fruad

#     for value in buyer_value_set:

#         if value in seller_value_set:

#             Fruad = True

#             return Fruad

#     return Fruad

# def check_same_credit_cardv2(buyer_id,seller_id): 

#     Fruad = False

#     buyer_value_set = master_df_dict[buyer_id]['bank_account']

#     if len(buyer_value_set) == 1 and \

#         np.nan in buyer_value_set:

#         return Fruad

#     seller_value_set = master_df_dict[seller_id]['bank_account']

#     if len(seller_value_set) == 1 and \

#         np.nan in seller_value_set:

#         return Fruad

#     for value in buyer_value_set:

#         if value in seller_value_set:

#             Fruad = True

#             return Fruad

#     return Fruad

# def check_same_device(buyer_id,seller_id):

#     Fruad = False

#     buyer_value_list = master_user_df[master_user_df.userid ==buyer_id].device.values[0]

#     if len(buyer_value_list) == 1 and \

#         str(buyer_value_list[0]) == 'nan':

#         return Fruad

#     seller_value_list = master_user_df[master_user_df.userid ==seller_id].device.values[0]

#     if len(seller_value_list) == 1 and \

#         str(seller_value_list[0]) == 'nan':

#         return Fruad

    

#     for value in buyer_value_list:

#         if value in seller_value_list:

#             Fruad = True

#     return Fruad

# def check_same_account(buyer_id,seller_id):

#     Fruad = False

#     buyer_value_list = master_user_df[master_user_df.userid ==buyer_id].bank_account.values[0]

#     if len(buyer_value_list) == 1 and \

#         str(buyer_value_list[0]) == 'nan':

#         return Fruad

#     seller_value_list = master_user_df[master_user_df.userid ==seller_id].bank_account.values[0]

#     if len(seller_value_list) == 1 and \

#         str(seller_value_list[0]) == 'nan':

#         return Fruad

    

#     for value in buyer_value_list:

#         if value in seller_value_list:

#             Fruad = True

#     return Fruad

# def check_same_credit_card(buyer_id,seller_id): 

#     Fruad = False

#     buyer_value_list = master_user_df[master_user_df.userid ==buyer_id].credit_card.values[0]

#     if len(buyer_value_list) == 1 and \

#         str(buyer_value_list[0]) == 'nan':

#         return Fruad

#     seller_value_list = master_user_df[master_user_df.userid ==seller_id].credit_card.values[0]

#     if len(seller_value_list) == 1 and \

#         str(seller_value_list[0]) == 'nan':

#         return Fruad

    

#     for value in buyer_value_list:

#         if value in seller_value_list:

#             Fruad = True

#     return Fruad

# def Check_fruad(buyer_id,seller_id):

#     Fruad = 0

#     device_check = check_same_device(buyer_id,seller_id)

#     account_check = check_same_account(buyer_id,seller_id)

#     credit_check = check_same_credit_card(buyer_id,seller_id)

#     if device_check or account_check or credit_check:

#         Fruad = 1

#     return Fruad

        

    
# def Check_fruadv2(buyer_id,seller_id):

#     Fruad = 0

#     device_check = check_same_devicev2(buyer_id,seller_id)

#     account_check = check_same_accountv2(buyer_id,seller_id)

#     credit_check = check_same_credit_cardv2(buyer_id,seller_id)

#     if device_check or account_check or credit_check:

#         Fruad = 1

#     return Fruad   

        
# start_time = time.time()

# print(Check_fruad('223406364','227839480'))

# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()

# print(Check_fruadv2('223406364','227839480'))

# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()

# print(Find_fruad_user('223406364','227839480',depth_search=5))

# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()

# print(Find_fruad_user('223406364','193350172'))

# print("--- %s seconds ---" % (time.time() - start_time))
order_df['is_fraud'] = order_df.apply(lambda x: Find_fruad_userV2(x['buyer_userid'],x['seller_userid'],depth_search=100), axis=1)  
df = pd.DataFrame({"orderid": order_df.orderid,"is_fraud": order_df.is_fraud}) 

df.head()    
df.to_csv('submission.csv',index=False)    