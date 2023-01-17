%matplotlib inline 

import numpy as np # imports a fast numerical programming library

import scipy as sp #imports stats functions, amongst other things

import matplotlib as mpl # this actually imports matplotlib

import matplotlib.cm as cm #allows us easy access to colormaps

import matplotlib.pyplot as plt #sets up plotting under plt

import pandas as pd #lets us handle data as dataframes

#sets up pandas table display

pd.set_option('display.width', 500)

pd.set_option('display.max_columns', 100)

pd.set_option('display.notebook_repr_html', True)

import seaborn as sns #sets up styles and gives us more plotting options

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#loading all data files

# Aisles is a kind of lane that describe food items/prodcuts in that lane

aisles = pd.read_csv("../input/aisles.csv")

# it could be super class for aisles

departments = pd.read_csv("../input/departments.csv")



order_products_prior = pd.read_csv("../input/order_products__prior.csv")

order_products_train = pd.read_csv("../input/order_products__train.csv")

orders = pd.read_csv("../input/orders.csv")

products = pd.read_csv("../input/products.csv")
## Printing small info about each table

print("aisles", aisles.shape, "aisles columns", aisles.columns)

print()

print("departments", departments.shape, "departments columns", departments.columns)

print()

print("order_products_prior", order_products_prior.shape, "order_products_prior columns", order_products_prior.columns)

print()

print("order_products_train", order_products_train.shape, "order_products_train columns", order_products_train.columns)

print()

print("orders", orders.shape, "orders columns", orders.columns)

print()

print("products", products.shape, "products columns", products.columns)
order_products_all = pd.concat([order_products_train, order_products_prior], axis=0)
order_products_all_merged = order_products_all.merge(products, left_on = 'product_id', right_on = 'product_id', how = 'left').merge()
order_products_all_merged2 = order_products_all_merged.merge(departments, left_on = 'department_id', right_on = 'department_id', how = 'left' )
order_products_all_merged2.head()