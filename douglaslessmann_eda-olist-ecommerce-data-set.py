import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os #using operating system dependent functionality
import datetime #datetime module supplies classes for manipulating dates and times.
import math # provides access to the mathematical functions
from IPython.display import display, HTML

#For Plotting
# Using plotly + cufflinks in offline mode
import plotly.express as px
import plotly.graph_objs as go
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True)
init_notebook_mode(connected=True)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Pandas option
pd.options.display.float_format = '{:.2f}'.format
# Input data files are available in the "../input/" directory.
# Listing the available files 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# sales data set for train
olist_products_dataset = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv')
# calendar
olist_orders_dataset = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')
# sell price
olist_order_items_dataset = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')
olist_products_dataset.head()
olist_products_dataset.shape
print("Unique product_id: '{}'".format(olist_products_dataset.product_id.unique().shape[0]))

olist_orders_dataset.head()
olist_orders_dataset.shape
print("Unique order_id	: '{}'".format(olist_orders_dataset.order_id.unique().shape[0]))
olist_order_items_dataset.head()
olist_order_items_dataset.shape
print("Unique product_id: '{}'".format(olist_order_items_dataset.product_id.unique().shape[0]))
print("Unique order_id	: '{}'".format(olist_order_items_dataset.order_id.unique().shape[0]))
olist_order_items_dataset["SK_order_id_product_id"] = olist_order_items_dataset["product_id"] + olist_order_items_dataset["order_id"]
print("Unique SK: '{}'".format(olist_order_items_dataset.SK_order_id_product_id.unique().shape[0]))