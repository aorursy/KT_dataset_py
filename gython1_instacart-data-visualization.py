# import libraries



import os 

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from statsmodels.graphics.mosaicplot import mosaic

from pylab import rcParams



%matplotlib inline

sns.set(context="notebook", font_scale=1.2)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Set the working directory



os.chdir('../input/')
# Data import



aisles = pd.read_table('aisles.csv', sep=',')

departments = pd.read_table('departments.csv', sep=',')

order_products_train = pd.read_table('order_products__train.csv', sep=',')

orders = pd.read_table('orders.csv', sep=',')

products = pd.read_table('products.csv', sep=',')
# Some first basic checkups



print('Total number od aisles: {}'.format(aisles['aisle'].value_counts().sum()))

print('Total number od departments: {}'.format(departments['department'].value_counts().sum()))
# Replacement of strings over id's - aisles and departments in products



products = pd.merge(products, departments, on="department_id", how='left')

products = pd.merge(products, aisles, on="aisle_id", how='left')

products.drop(['aisle_id', 'department_id'], axis=1, inplace=True)
# We don't need aisles and departments anymore and I like to keep my memory clean :)



del aisles, departments
# PRODUCTS - DEPARTMENT COUNTPLOT



sns.set_style("whitegrid", {'axes.edgecolor': '.8','axes.linewidth': 0.1,'ytick.major.size': 0.1,'ytick.minor.size': 0.1})

plt.figure(figsize=(12,6))



ncount = products["department"].notnull().sum()

ax = sns.countplot(x="department", data=products, order=products["department"].value_counts().index, color="c")



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), ha='center', va='bottom', fontsize=12)

    

ax.xaxis.set_label_coords(0.5, -0.35)



ax = plt.xticks(rotation=70)



print("Total N =", products['department'].value_counts().sum())