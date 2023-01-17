# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



departments = pd.read_csv("../input/departments.csv")

aisles = pd.read_csv("../input/aisles.csv")

order_products_prior = pd.read_csv("../input/order_products__prior.csv")

order_products_train = pd.read_csv("../input/order_products__train.csv")

orders = pd.read_csv("../input/orders.csv")

products = pd.read_csv("../input/products.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")



# Any results you write to the current directory are saved as output.
ord_prior = orders[orders['eval_set'] == "prior"]

ord_train = orders[orders['eval_set'] == "train"]

result = pd.merge(ord_prior[['order_id','order_dow','order_hour_of_day']], order_products_prior, how='inner', on=['order_id'])
rnew = pd.merge(products[['product_id','product_name']],result,how="inner",on=['product_id'])

#rnew.groupby('product_name')
highest_ord = rnew['product_name'].value_counts()[:5]

highest_ord

highest_ord.plot(kind='barh')
highest_day = rnew['order_dow'].value_counts().plot(kind='bar')

