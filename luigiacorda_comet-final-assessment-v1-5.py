# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
aisles = pd.read_csv('../input/aisles.csv')

aisles = aisles.sort_values('aisle_id')

aisles.head(5)
orders = pd.read_csv('../input/orders.csv')



orders = orders[(orders.eval_set == 'prior')]
products = pd.read_csv('../input/products.csv')

products = products.sort_values('aisle_id')

products.head(5)
product_with_aisle = pd.merge(products, aisles, on='aisle_id')

product_with_aisle.head(5)
order_products_prior = pd.read_csv('../input/order_products__prior.csv')

order_products_prior = order_products_prior.sort_values('order_id')

order_products_prior.head(5)
product_aisle_order = pd.merge(product_with_aisle, order_products_prior, on='product_id')

product_aisle_order.head(5)
aisle_table = product_aisle_order[['aisle_id', 'aisle']]

aisle_table = aisle_table.groupby('aisle')[['aisle']].count().sort_values(['aisle'], ascending=False)

aisle_table.head(10)
aisle_table.head(10).plot(kind = 'barh').invert_yaxis()
aisle_table.tail(10).plot(kind = 'barh').invert_yaxis()
ohod = pd.DataFrame(orders.groupby('order_hour_of_day')['order_hour_of_day'].count())

ohod = ohod.sort_values(['order_hour_of_day'], ascending = False)

ohod
ohod.sort_index().plot(legend = None)
day = pd.DataFrame(orders[(orders.order_hour_of_day == 10)].groupby('order_dow')['order_dow'].count())

day = day.sort_values(['order_dow'], ascending = False)

day
day.sort_index().plot(legend = None)
oppdow = orders.groupby('order_dow')['order_dow'].count()

oppdowarr = []

oppdowarr.append({'order_dow': 0, 'Count': oppdow[0]})

oppdowarr.append({'order_dow': 1, 'Count': oppdow[1]})

oppdowarr.append({'order_dow': 2, 'Count': oppdow[2]})

oppdowarr.append({'order_dow': 3, 'Count': oppdow[3]})

oppdowarr.append({'order_dow': 4, 'Count': oppdow[4]})

oppdowarr.append({'order_dow': 5, 'Count': oppdow[5]})

oppdowarr.append({'order_dow': 6, 'Count': oppdow[6]})



oppdowdf = pd.DataFrame(oppdowarr)

oppdowdf = oppdowdf[['order_dow', 'Count']]

oppdowdf.set_index('order_dow', inplace = True)

oppdowdf
oppdowdf.sort_index().plot(legend = None)