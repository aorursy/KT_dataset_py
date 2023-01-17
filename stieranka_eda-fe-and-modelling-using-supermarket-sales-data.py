import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')


import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

import os
print(os.listdir('.'))

#import the dataset
train = pd.read_csv('../input/train.csv')

train.head().transpose()
#plot some heatmap to find correlation amoung the features
corrmat = train.corr()
f, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corrmat, square=True)
print(corrmat)
#get percentage of missing values
train_missing = (train.isnull().sum() / len(train)) * 100
train_missing = train_missing.drop(train_missing[train_missing == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage' : train_missing})
missing_data

cat_col = ['Product_Fat_Content', 'Product_Type', 'Supermarket_Location_Type', 'Supermarket_Type']
for col in cat_col:
    sns.set()
    cols = ['Product_Identifier', 'Supermarket_Identifier', 'Product_Fat_Content', 'Product_Shelf_Visibility', 'Product_Type', 'Product_Price', 'Supermarket_Opening_Year', 'Supermarket_Location_Type', 'Supermarket_Type', 'Product_Supermarket_Sales']
    plt.figure()
    sns.pairplot(train[cols], size = 3.0, hue=col)
    plt.show()
