import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.cubehelix_palette(10, start=2, rot=0, dark=0.2, light=0.8, reverse=True)

%matplotlib inline





def read_data(val):

    return pd.read_csv("../input/{0}".format(val))



aisles = read_data('aisles.csv')

depts = read_data('departments.csv')

orders = read_data('orders.csv')

products = read_data('products.csv')

op_prior = read_data('order_products__prior.csv')

op_train = read_data('order_products__train.csv')
combined_data = pd.concat([op_prior, op_train])[['product_id']].sample(int(1.5e7)) # sample 15 million for memory

combined_data = combined_data.merge(products,  how="left", on="product_id")

combined_data = combined_data.merge(aisles,    how="left", on="aisle_id")

combined_data = combined_data.merge(depts,     how="left",  on="department_id")

combined_data.tail()
def get_value_count_and_plot(data, title, limit=10):

    value_count = data.product_id.value_counts()[:limit]

    prod_ids = value_count.index.values

    prod_names = products[products.product_id.isin(prod_ids)].product_name

    prod_names = [name[:25]+'...' if len(name)>25 else "{0:<28}".format(name) for name in prod_names] # truncate names for presentation



    sns.set(font_scale=2)

    sns.barplot(y=prod_names, x=value_count.values, orient='h', palette=color)

    sns.plt.suptitle(title)

    plt.show()



    

title =" Top Products in Overall"

get_value_count_and_plot(combined_data, title)



for d in combined_data.department_id.unique():

    print("-"*23)

    title =" Top Products in {}".format(depts.ix[d-1].department)

    data = combined_data[combined_data.department_id==d]

    get_value_count_and_plot(data, title)