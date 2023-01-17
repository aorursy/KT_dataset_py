import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns                  # sea born is library related to matplot to give attractiveness to graphics 

color = sns.color_palette()            # to choose color of matplot graphics 

sns.palplot(color)
aisles=pd.read_csv('aisles.csv') 

opp =pd.read_csv('order_products__prior.csv')

opt=pd.read_csv('order_products__train.csv')

orders= pd.read_csv('orders.csv')

products=pd.read_csv('products.csv')

departments=pd.read_csv('departments.csv')



sns.barplot(count.index, count.values, alpha=0.8)

plt.xlabel('eval set type')

plt.ylabel('number of occurrences')

plt.title('bar chart for eval set count')

plt.figure(figsize=(12,8))

plt.show()