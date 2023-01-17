import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

train.head()
print(test.shape)

test.head()
stores = pd.read_csv('./../input/stores.csv')

print(stores.shape)

stores.head()
items = pd.read_csv('./../input/items.csv')

print(items.shape)

items.head()
oil = pd.read_csv('./../input/oil.csv')

print(oil.shape)

oil.head()
transactions = pd.read_csv('./../input/transactions.csv')

print(transactions.shape)

transactions.head()