# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as py
import pandas as pd
import matplotlib.pyplot as plot
import plotly.plotly as py
import plotly.graph_objs as go
import random as rand
import math as math
import pandas_profiling as pp
departments = pd.read_csv("../input/departments.csv")
aisles = pd.read_csv("../input/aisles.csv")
order_products_train = pd.read_csv("../input/order_products__train.csv")
products = pd.read_csv("../input/products.csv")
orders = pd.read_csv("../input/orders.csv")
order_products_prior =  pd.read_csv("../input/order_products__prior.csv")

if (departments is None) or (aisles is None) or (order_products_train is None) or (products is None) or (orders is None) or (order_products_prior is None):
    print("Error reading files.")
else:
    print("Successfully read all files.")
print("Display random data from orders, aisles and products DataFrames", end="\n\n")

print("orders")
for i in range(0, 4):
    print(orders.loc[math.ceil(rand.random()*100)], '\n')
    i+=1

print("aisles")
for i in range(0, 4):
    print(aisles.loc[math.ceil(rand.random()*100)], '\n')
    i+=1
    
print("products")
for i in range(0, 4):
    print(products.loc[math.ceil(rand.random()*100)], '\n')
    i+=1    
print("Merge aisles and products by product_id")

aisle_with_products = pd.merge(aisles, products, on='aisle_id')
aisle_with_products
frames = [order_products_prior]
merged_order_products = pd.concat(frames)
temp = merged_order_products.groupby(['order_id']).agg('count')
print("products per aisle")
temp = aisle_with_products.groupby(['aisle_id', 'aisle']).agg('count')
temp_df = temp['product_id']
print

aisle_name = []
values = []

for i in range(0, len(aisles)):
    aisle_name.append(temp.index[i][0])
    values.append(temp.iloc[i,1])
    
plot.barh(np.arange(len(values)), values)
plot.xticks(np.arange(len(values)), values)
plot.ylabel('number of products')
frames = [order_products_prior, order_products_train]
merged_order_products = pd.concat(frames)
temp = merged_order_products.groupby(['order_id']).agg('count')
print("number of products per order", temp['product_id'].median())
merged_df = pd.merge(aisle_with_products, merged_order_products, on='product_id')
merged_df
pp.ProfileReport(merged_df)