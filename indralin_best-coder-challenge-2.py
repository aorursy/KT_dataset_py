import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
devices = pd.read_csv('/kaggle/input/ptr-rd2-ahy/devices.csv', low_memory=False)
bank_acc = pd.read_csv('/kaggle/input/ptr-rd2-ahy/bank_accounts.csv', low_memory=False)
credit_cards = pd.read_csv('/kaggle/input/ptr-rd2-ahy/credit_cards.csv', low_memory=False)
orders = pd.read_csv('/kaggle/input/ptr-rd2-ahy/orders.csv', low_memory=False)

print('devices shape:', devices.shape)
print('bank accounts shape:', bank_acc.shape)
print('credit cards shape:', credit_cards.shape)
print('orders shape:', orders.shape)
devices.head()
bank_acc.head()
credit_cards.head()
orders.head()
final_df = devices.merge(bank_acc, on='userid', how='left').merge(credit_cards, on='userid', how='left')

final_df.shape
final_df.head(10)
print('device unique values:', final_df['device'].nunique())
print('bank account unique values:', final_df['bank_account'].nunique())
print('credit card unique values:', final_df['credit_card'].nunique())
def fraud_detection(user_id_a, user_id_b):
    # buyer user
    storage_a = []

    for device in final_df[final_df['userid'] == user_id_a].device:
        if device not in storage_a:
            storage_a.append(device)
        
    for account in final_df[final_df['userid'] == user_id_a].bank_account:
        if (account not in storage_a) and (pd.notnull(account) == True):
            storage_a.append(account)
        
    for cc in final_df[final_df['userid'] == user_id_a].credit_card:
        if (cc not in storage_a) and (pd.notnull(cc) == True):
            storage_a.append(cc)
    
    # seller user
    storage_b = []

    for device in final_df[final_df['userid'] == user_id_b].device:
        if device not in storage_b:
            storage_b.append(device)
        
    for account in final_df[final_df['userid'] == user_id_b].bank_account:
        if (account not in storage_b) and (pd.notnull(account) == True):
            storage_b.append(account)
        
    for cc in final_df[final_df['userid'] == user_id_b].credit_card:
        if (cc not in storage_b) and (pd.notnull(cc) == True):
            storage_b.append(cc)
    
    # convert to pandas series
    storage_a_series = pd.Series(storage_a)
    storage_b_series = pd.Series(storage_b)
    
    result = storage_a_series.isin(storage_b_series).sum()
    
    if result > 0:
        return 1
    
    elif result == 0:
        return 0
orders['buyer_seller'] = list(zip(orders['buyer_userid'], orders['seller_userid']))
orders.head()
fraud_detection(26855196, 16416890)
is_fraud = []

for buyer, seller in tqdm(orders['buyer_seller']):
    result = fraud_detection(buyer, seller)
    is_fraud.append(result)
orders['is_fraud'] = is_fraud
orders['is_fraud'].value_counts()
orders[['orderid', 'is_fraud']]
orders[['orderid', 'is_fraud']].to_csv('submission2.csv', index=False)  # remember to remove index in submission..