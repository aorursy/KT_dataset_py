# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
orders = pd.read_csv("../input/orders.csv")

products = pd.read_csv("../input/products.csv")

aisles = pd.read_csv("../input/aisles.csv")

departments= pd.read_csv("../input/departments.csv")

train = pd.read_csv("../input/order_products__train.csv")

prior = pd.read_csv("../input/order_products__prior.csv")
orders.head(5)
train.head(5)
prior.head(5)
orders[orders.user_id==1]
train[train.order_id.isin(orders.order_id[orders.user_id==1])]
len(prior.order_id[prior.order_id.isin(orders.order_id[orders.user_id==1])].drop_duplicates())