import re

from pprint import pprint

from copy import deepcopy

from itertools import product

import pandas as pd

%matplotlib inline



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



pd.set_option('display.max_colwidth', None)
orders = pd.read_csv('../input/opn-rd2-acv/orders.csv')
# Specify dtypes prevents mixed types of int,str after reading. Makes merging accurate

bank = pd.read_csv('../input/opn-rd2-acv/bank_accounts.csv',dtype={'bank_account':str})  
cc = pd.read_csv('../input/opn-rd2-acv/credit_cards.csv')
devices = pd.read_csv('../input/opn-rd2-acv/devices.csv')
sum(orders.duplicated())

sum(bank.duplicated())

sum(cc.duplicated())

sum(devices.duplicated())
bank = bank.drop_duplicates()

sum(bank.duplicated())
orders.head()

orders.info()

orders.describe()
bank.head()

bank.info()
cc.head()

cc.info()
devices.head()

devices.info()
#buyer_userid seller_userid

device_merge = (orders.merge(devices,left_on='buyer_userid',right_on='userid').rename(columns={'device':'buyer_device'})

                     .merge(devices,left_on='seller_userid',right_on='userid').rename(columns={'device':'seller_device'})

               )

device_merge
device_fraud_mask = device_merge['buyer_device'] == device_merge['seller_device']

# 355 will become 291 later due to each orderid having multiple methods to fraud with different devices)

sum(device_fraud_mask)   
device_fraud = device_merge[device_fraud_mask]

device_fraud.head()
device_fraud['d_chain'] = (device_fraud['buyer_userid'].astype(str) 

                                  + '-"'

                                  +'device:'

                                  + device_fraud['buyer_device']

                                  + '"->'

                                  + device_fraud['seller_userid'].astype(str)

                                 )
device_fraud.head()
bank_merge = (orders.merge(bank,left_on='buyer_userid',right_on='userid').rename(columns={'bank_account':'buyer_bank'})

                     .merge(bank,left_on='seller_userid',right_on='userid').rename(columns={'bank_account':'seller_bank'})

               )

bank_merge
bank_fraud_mask = bank_merge['buyer_bank'] == bank_merge['seller_bank']
# bank_fraud_mask is 34 only if not converting to object (dtype={'bank_account':str}) when reading bank_accounts.csv

# 56 will become 54 later due to each orderid having multiple methods to fraud with different devices)

sum(bank_fraud_mask)     
bank_fraud = bank_merge[bank_fraud_mask]

bank_fraud.head()
bank_fraud['b_chain'] = (bank_fraud['buyer_userid'].astype(str) 

                                  + '-"'

                                  +'bank_account:'

                                  + bank_fraud['buyer_bank']

                                  + '"->'

                                  + bank_fraud['seller_userid'].astype(str)

                                 )
bank_fraud.head()
# Both bank and device fraud in same orderid 

len(set(bank_fraud.orderid) & set(device_fraud.orderid))
cc_merge = (orders.merge(cc,left_on='buyer_userid',right_on='userid').rename(columns={'credit_card':'buyer_cc'})

                .merge(cc,left_on='seller_userid',right_on='userid').rename(columns={'credit_card':'seller_cc'})

               )

cc_merge
cc_fraud_mask = cc_merge['buyer_cc'] == cc_merge['seller_cc']

sum(cc_fraud_mask)
is_fraud = ['not fraud']*len(orders)

result = dict(zip(orders.orderid,is_fraud))
len(device_fraud['d_chain'].values)
device_repeats = device_fraud.groupby('orderid').filter(lambda group:len(group)>1)

device_repeats
device_repeats.orderid.nunique()

device_repeats.groupby('orderid').size().sort_values(ascending=False)
# view some samples of top 2 orderid with most number of ways to device fraud

device_repeats.query('orderid==1955166134')  

device_repeats.query('orderid==1954135512')  
# sort descending so smallest alphetic string appears last

# smallest string will overwrite previously larger strings assigned to same orderid  

device_fraud_sorted = device_fraud.sort_values(by='d_chain',ascending=False)

d_chain_dic = dict(zip(device_fraud_sorted['orderid'].values,device_fraud_sorted['d_chain'].values))
len(d_chain_dic)
# update device first so later updating bank direct frauds on same orderid can overwrite direct device fraud chain

result.update(d_chain_dic)
len(bank_fraud['b_chain'].values)
bank_repeats = bank_fraud.groupby('orderid').filter(lambda group:len(group)>1)

bank_repeats



# Two order ids have 2 rows with same orderid with multiple bank acc --> get the smaller alphabetical bank
bank_repeats.groupby('orderid').size()
bank_fraud_sorted = bank_fraud.sort_values(by='b_chain',ascending=False)

b_chain_dic = dict(zip(bank_fraud_sorted['orderid'].values,bank_fraud_sorted['b_chain'].values))
len(b_chain_dic)
result.update(b_chain_dic)
buyer_cc_linkable = orders.buyer_userid.isin(cc.userid)

sum(buyer_cc_linkable)



buyer_bank_linkable = orders.buyer_userid.isin(bank.userid)

sum(buyer_bank_linkable)



buyer_device_linkable = orders.buyer_userid.isin(devices.userid)

sum(buyer_device_linkable)
seller_cc_linkable = orders.seller_userid.isin(cc.userid)

sum(seller_cc_linkable)



seller_bank_linkable = orders.seller_userid.isin(bank.userid)

sum(seller_bank_linkable)



seller_device_linkable = orders.seller_userid.isin(devices.userid)

sum(seller_device_linkable)
sum(~orders.buyer_userid.isin(cc.userid) &

    ~orders.buyer_userid.isin(bank.userid) &

    ~orders.buyer_userid.isin(devices.userid)

    )

sum(~orders.seller_userid.isin(cc.userid) &

    ~orders.seller_userid.isin(bank.userid) &

    ~orders.seller_userid.isin(devices.userid)

    )



# these number of people confirmed no fraud no matter link length since there is no way to start linking them  
# 48 unique buyers and 48 unique sellers among 56 direct bank fraud



len(bank_fraud.groupby('buyer_userid'))

bank_fraud.groupby('buyer_userid').size().sort_values(ascending=False)

len(bank_fraud.groupby('seller_userid'))

bank_fraud.groupby('seller_userid').size().sort_values(ascending=False)
bank_fraud.query('buyer_userid == 100918044')
# 48 unique buyers and 48 unique sellers among 56 direct device fraud



len(device_fraud.groupby('buyer_userid'))

device_fraud.groupby('buyer_userid').size().sort_values(ascending=False)

len(device_fraud.groupby('seller_userid'))

device_fraud.groupby('seller_userid').size().sort_values(ascending=False)
cc.groupby('userid').size().sort_values(ascending=False)

bank.groupby('userid').size().sort_values(ascending=False)

devices.groupby('userid').size().sort_values(ascending=False)
user_cc = cc.groupby('userid')['credit_card'].apply(list)

user_cc
user_bank = bank.groupby('userid')['bank_account'].apply(list)

user_bank
user_device = devices.groupby('userid')['device'].apply(list)

user_device
cc_user = cc.groupby('credit_card')['userid'].apply(list)

cc_user
cc_user.apply(len).value_counts().sort_index()
bank_user = bank.groupby('bank_account')['userid'].apply(list)

bank_user
bank_user.apply(len).value_counts().sort_index()
device_user = devices.groupby('device')['userid'].apply(list)

device_user
device_user.apply(len).value_counts().sort_index()
import networkx as nx
# basic undirected unweighted graph

# adding edges automatically adds nodes

# edges once added will be automatically deduplicated if the same/reverse edge tuple is added

g = nx.Graph()
for user,cc in user_cc.items():

    g.add_edges_from(product([user],['credit_card:'+item for item in cc]))
for user,bank in user_bank.items():

    g.add_edges_from(product([user],['bank_account:'+item for item in bank]))
for user,device in user_device.items():

    g.add_edges_from(product([user],['device:'+item for item in device]))
len(g.nodes)
order_paths = {}

for orderid,source,target in zip(orders.orderid,orders.buyer_userid,orders.seller_userid):

    try:

        path_list = list(nx.all_shortest_paths(g, source,target))

    # NetworkXNoPath    

    except:

        path_list = []

        

    order_paths[orderid] = path_list
path_df = pd.DataFrame({'orderid':list(order_paths.keys()),'paths':list(order_paths.values())})

path_df

path_df[path_df.paths.str.len()!=0] # path_df.paths.apply(len)!=0 or .astype(bool) works too
path_df_fraud = path_df[path_df.paths.astype(bool)] 

path_df_fraud
cc_mask = path_df_fraud.paths.apply(lambda outer_list:any(['credit_card' in str(item) for inner_list in outer_list for item in inner_list]))

path_df_fraud[cc_mask]
bank_mask = path_df_fraud.paths.apply(lambda outer_list:any(['bank_account' in str(item) for inner_list in outer_list for item in inner_list]))

path_df_fraud[bank_mask]
length_of_paths = path_df_fraud.paths.apply(lambda x:len(x[0]))

length_of_paths.value_counts().sort_index()
number_of_paths = path_df_fraud.paths.apply(lambda x:len(x))

number_of_paths.value_counts().sort_index()
def format_paths(paths):

    formatted_paths = []

    for path in paths:

        new_path = deepcopy(path)

        new_path[1::2] = ['-"'+item+'"->' for item in new_path[1::2]]

        formatted_paths.append(''.join(list(str(item) for item in new_path)))

    return formatted_paths



formatted_paths = path_df_fraud.paths.apply(format_paths)

formatted_paths
# for copying the df scheme and orderid column only, not the paths column (which will be overwritten)

first_path_df = path_df_fraud.copy()
first_path_df.paths = formatted_paths.str[0]

first_path_df
mask_bank_first = first_path_df['orderid'].isin(b_chain_dic.keys())
bank_nx = first_path_df[mask_bank_first ]
b_chain_series = pd.Series(index=list(b_chain_dic.keys()),data=list(b_chain_dic.values()))

b_chain_series
bank_nx_dic = dict(zip(bank_nx.orderid,bank_nx.paths))
bank_nx_series = pd.Series(index=list(bank_nx_dic.keys()),data=list(bank_nx_dic.values()))
bank_fraud_methods = pd.concat([b_chain_series,bank_nx_series],axis=1)

#bank_fraud_methods

bank_fraud_methods[bank_fraud_methods[0] != bank_fraud_methods[1]]
min_path_df = path_df_fraud.copy()
min_path_df.paths = formatted_paths.apply(min)
mask_bank_min = min_path_df['orderid'].isin(b_chain_dic.keys())
bank_min_nx = min_path_df[mask_bank_min]
bank_min_nx_dic = dict(zip(bank_min_nx.orderid,bank_min_nx.paths))
bank_min_nx_series = pd.Series(index=list(bank_min_nx_dic.keys()),data=list(bank_min_nx_dic.values()))
bank_fraud_methods = pd.concat([b_chain_series,bank_min_nx_series],axis=1)

bank_fraud_methods[bank_fraud_methods[0] != bank_fraud_methods[1]]   # everything is equal
min_path_dic = dict(zip(min_path_df.orderid,min_path_df.paths))
path_df_fraud[bank_mask & (length_of_paths==5) & (number_of_paths==4)] 
def sort_func(string):

    cc_links = string.count('credit_card')

    bank_links = string.count('bank_account')

    # device not needed because there's only 2 degrees of freedom, if #cc,#bank tie, #device must tie

    # device value will not affect sort result

    return (cc_links,bank_links)





test_formatted_paths = format_paths(path_df_fraud.query(f'orderid==1953272960').iloc[0,1])

print('original')

pprint(test_formatted_paths)

s = sorted(test_formatted_paths)

print('secondary sorted')

pprint(s)

s_func = sorted(s,key=sort_func,reverse=True)

print('primary sorted')

pprint(s_func)
path_df_fraud[bank_mask & (length_of_paths==7) & (number_of_paths==4)] 
# https://www.rexegg.com/regex-quantifiers.html#helpful

def sort_regex(string):

    cc_pattern = 'credit_card:(.*?)"'

    bank_pattern = 'bank_account:(.*?)"'

    device_pattern = 'device:(.*?)"'

    

    cc_match = ''.join(re.findall(cc_pattern, string))

    bank_match = ''.join(re.findall(bank_pattern, string))

    device_match = ''.join(re.findall(device_pattern, string))

    

    return (cc_match,bank_match,device_match)



test_path = format_paths(path_df_fraud.query('orderid == 1953361436').iloc[0,1])

test_path



sort_regex(test_path[0])
formatted_paths_sorted = formatted_paths.apply(sorted)

formatted_paths_sorted = formatted_paths_sorted.apply(sorted,key=sort_func,reverse=True)
# ensure sorting made a difference

sum(formatted_paths_sorted != formatted_paths)
sorted_first_path_df = path_df_fraud.copy()

sorted_first_path_df.paths = formatted_paths_sorted.str[0]

sorted_first_path_df



sorted_first_path_dic = dict(zip(sorted_first_path_df.orderid,sorted_first_path_df.paths))

len(sorted_first_path_dic)
min_path_dic == sorted_first_path_dic
formatted_paths_sorted_alpha = formatted_paths.apply(sorted,key=sort_regex)

formatted_paths_sorted_alpha = formatted_paths_sorted_alpha.apply(sorted,key=sort_func,reverse=True)
sum(formatted_paths_sorted!=formatted_paths_sorted_alpha)
alpha_sorted_first_path_df = path_df_fraud.copy()

alpha_sorted_first_path_df.paths = formatted_paths_sorted_alpha.str[0]

alpha_sorted_first_path_df



alpha_sorted_first_path_dic = dict(zip(alpha_sorted_first_path_df.orderid,alpha_sorted_first_path_df.paths))

len(alpha_sorted_first_path_dic)
sorted_first_path_dic == alpha_sorted_first_path_dic
is_fraud = ['not fraud']*len(orders)

result = dict(zip(orders.orderid,is_fraud))
result.update(sorted_first_path_dic)
result_df = pd.DataFrame({'orderid':list(result.keys()),'is_fraud':list(result.values())})
result_df.to_csv('all_fraud_2step_sorted.csv',index=False)
result_df.head()
sum(result_df.is_fraud != 'not fraud')
result_df[result_df.is_fraud != 'not fraud']
# https://www.rexegg.com/regex-quantifiers.html#helpful

def sort_regex(string):

    cc_pattern = 'credit_card:(.*?)"'

    bank_pattern = 'bank_account:(.*?)"'

    device_pattern = 'device:(.*?)"'

    

    cc_match = ''.join(re.findall(cc_pattern, string))

    bank_match = ''.join(re.findall(bank_pattern, string))

    device_match = ''.join(re.findall(device_pattern, string))

    print(cc_match,bank_match,device_match)

    

    return (cc_match,bank_match,device_match)
paths = [

    '1-"device:100"->4-"bank_account:100"->3',

    '1-"bank_account:200"->6-"device:100"->3',

    '1-"bank_account:10"->2-"bank_account:10"->3',  

]

print('original')

pprint(paths)

s = sorted(paths)

print('secondary sorted (no regex)')

pprint(s)



s_func = sorted(s,key=sort_func,reverse=True)

print('primary sorted (no regex)')

pprint(s_func)



s = sorted(paths,key=sort_regex)

print('secondary sorted (with regex)')

pprint(s)



s_func = sorted(s,key=sort_func,reverse=True)

print('primary sorted (with regex)')

pprint(s_func)