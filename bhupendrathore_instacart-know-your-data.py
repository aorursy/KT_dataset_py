ins# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd
order_prior = pd.read_csv('../input/order_products__prior.csv')

order_train = pd.read_csv('../input/order_products__train.csv')

order_all = pd.concat([order_prior,order_train],axis = 0)



order_all.head(5)
products  = pd.read_csv('../input/products.csv')
grouped = order_all.groupby("product_id")["reordered"].aggregate({'Total_reorders': 'count'}).reset_index()

grouped = pd.merge(grouped, products[['product_id', 'product_name']], how='left', on=['product_id'])

grouped = grouped.sort_values(by='Total_reorders', ascending=False)

grouped.head(10)
orders = pd.read_csv('../input/orders.csv')

orders.head(10)

all_data = pd.merge