import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

historical_transactions = pd.read_csv('../input/historical_transactions.csv',
                                      parse_dates=['purchase_date'])

new_transactions = pd.read_csv('../input/new_merchant_transactions.csv',
                               parse_dates=['purchase_date'])

train = pd.read_csv('../input/train.csv')
def view_purchase_graph(card_id):
    purchases = historical_transactions[historical_transactions['card_id'] == card_id].sort_values('purchase_date')
    plt.figure(figsize=(8,6))
    plt.scatter(pd.to_datetime(purchases['purchase_date']).astype(np.int),
                               purchases['purchase_amount'],
                               c=['red' if a == 'N' else 'blue' for a in purchases['authorized_flag']])
    plt.xlabel('time', fontsize=12)
    plt.ylabel('Historical Purchase Amount', fontsize=12)
    plt.show()

def view_new_purchase_graph(card_id):
    purchases = new_transactions[new_transactions['card_id'] == card_id].sort_values('purchase_date')
    plt.figure(figsize=(8,6))
    plt.scatter(pd.to_datetime(purchases['purchase_date']).astype(np.int), purchases['purchase_amount'])
    plt.xlabel('time', fontsize=12)
    plt.ylabel('New Purchase Amount', fontsize=12)
    plt.show()
select = train[train['target'] > 0][train['target'] < 1]
select.head()
select.shape
historical_transactions.merge(select).sort_values(['card_id', 'purchase_date']).head()
new_transactions.merge(select).sort_values(['card_id', 'purchase_date']).head()
for card in select['card_id'][:20]:
    print('-')
    print(train[train['card_id'] == card]['target'])
    view_purchase_graph(card)
    view_new_purchase_graph(card)
select = train[train['target'] > 10]
select.head()
select.shape
historical_transactions.merge(select).sort_values(['card_id', 'purchase_date']).head()
new_transactions.merge(select).sort_values(['card_id', 'purchase_date']).head()
for card in select['card_id'][:20]:
    print('-')
    print(train[train['card_id'] == card]['target'])
    view_purchase_graph(card)
    view_new_purchase_graph(card)
select = train[train['target'] < -10][train['target'] > -30]
select.head()
select.shape
historical_transactions.merge(select).sort_values(['card_id', 'purchase_date']).head()
new_transactions.merge(select).sort_values(['card_id', 'purchase_date']).head()
for card in select['card_id'][:20]:
    print('-')
    print(train[train['card_id'] == card]['target'])
    view_purchase_graph(card)
    view_new_purchase_graph(card)
select = train[train['target'] < -30]
select.head()
select.shape
historical_transactions.merge(select).sort_values(['card_id', 'purchase_date']).head()
new_transactions.merge(select).sort_values(['card_id', 'purchase_date']).head()
for card in select['card_id'][:20]:
    print('-')
    print(train[train['card_id'] == card]['target'])
    view_purchase_graph(card)
    view_new_purchase_graph(card)