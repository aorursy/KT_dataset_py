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
#CLEANING THE DATA
bank_accounts = pd.read_csv("../input/ptr-rd2-ahy/bank_accounts.csv")
bank_accounts['bank_account'] = bank_accounts.apply(lambda x:str(x['bank_account']), axis = 1)
credit_cards = pd.read_csv("../input/ptr-rd2-ahy/credit_cards.csv")
devices = pd.read_csv("../input/ptr-rd2-ahy/devices.csv")
orders = pd.read_csv("../input/ptr-rd2-ahy/orders.csv")
for x in [bank_accounts, credit_cards, devices, orders]:
    x.drop_duplicates(keep = 'first', inplace = True)
    x.reset_index(inplace = True)
is_fraud = [0]*len(orders)

#FINDING FRAUD ORDERS BASED ON BANK ACCOUNT
dup_bank = bank_accounts[bank_accounts.duplicated(subset='bank_account', keep=False)]
for x in dup_bank['bank_account'].unique():
    a = dup_bank[dup_bank['bank_account']==x]['userid']
    b = orders[orders['buyer_userid'].isin(a) & orders['seller_userid'].isin(a)]
    for y in range(0,len(b.index)):
        is_fraud[b.index[y]] = 1
        
#FINDING FRAUD ORDERS BASED ON CREDIT CARD
dup_credit = credit_cards[credit_cards.duplicated(subset='credit_card', keep=False)]
for x in dup_credit['credit_card'].unique():
    a = dup_credit[dup_credit['credit_card']==x]['userid']
    b = orders[orders['buyer_userid'].isin(a) & orders['seller_userid'].isin(a)]
    for y in range(0,len(b.index)):
        is_fraud[b.index[y]] = 1
        
#FINDING FRAUD ORDERS BASED ON DEVICE
dup_device = devices[devices.duplicated(subset='device', keep=False)]
for x in dup_device['device'].unique():
    a = dup_device[dup_device['device']==x]['userid']
    b = orders[orders['buyer_userid'].isin(a) & orders['seller_userid'].isin(a)]
    for y in range(0,len(b.index)):
        is_fraud[b.index[y]] = 1
        
#CREATING SUBMISSION FILE
submission = pd.DataFrame({'orderid':orders['orderid'], 'is_fraud':is_fraud}, columns = ['orderid', 'is_fraud'])
submission.to_csv("submission.csv", index = False)