import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
%matplotlib inline
train_sub = pd.read_csv('../input/train.csv', nrows=50000)
#test = pd.read_csv('test.csv') # Not using this right yet
categories = {c:i for i,c in enumerate(train_sub['Category'])}
# Get general distances based on coordinates (not used yet)
train_sub['Distance'] = [np.linalg.norm((x,y)) for x,y in zip(train_sub.X, train_sub.Y)]
# Add numerical category column (not used yet)
train_sub['Category_Num'] = [categories[t] for t in train_sub.Category]
# Add hour column
train_sub['Hour'] = list(map(lambda x: int(x.split(' ')[1].split(':')[0]), train_sub.Dates))
train_sub[:1]
# Plot 2D histogram of data with constraint
# Default constraint is category, which has 36 unique values (i.e. 12x3)
def plot_data(constraint='Category', rownum=12, colnum=3):
    _, ax = plt.subplots(nrows=rownum, ncols=colnum,figsize=(10,30))
    i = 0
    j = 0
    for cat in train_sub[constraint].unique():
        cat_sub = train_sub[train_sub[constraint] == cat]
        try:
            ax[j][i].hist2d(cat_sub.X, cat_sub.Y, bins=60, norm=LogNorm(), cmap=plt.cm.jet)
            ax[j][i].set_title('{0} {1}'.format(cat, len(cat_sub)))
        except(KeyError, IndexError):
            pass
        i+=1
        if i > colnum-1:
            i = 0
            j+=1
    plt.tight_layout()
    plt.show()
# Density of crime by Category
plot_data()
# Desnity of crime by hour
plot_data('Hour', 8, 3)
# Density of crime per district
plot_data('PdDistrict', 5, 2)
# Only works in jupyter w/ python 2.7.*
# Second plotting function for extra constraints
def plot_data_filtered(constraint, constraint2, rownum=12, colnum=3):
    _, ax = plt.subplots(nrows=rownum, ncols=colnum,figsize=(10,30))
    i = 0
    j = 0
    for cat in train_sub[constraint].unique():
        cat_sub = train_sub[(train_sub[constraint] == cat) & constraint2]
        try:
            ax[j][i].hist2d(cat_sub.X, cat_sub.Y, bins=60, norm=LogNorm(), cmap=plt.cm.jet)
            ax[j][i].set_title('{0} {1}'.format(cat, len(cat_sub)))
        except(KeyError, IndexError):
            pass
        i+=1
        if i > colnum-1:
            i = 0
            j+=1
    plt.tight_layout()
    plt.show()
# Density by hour with second constraint (first unique category value)
c2 = (train_sub['Category'] == train_sub['Category'].unique()[0])
print(train_sub['Category'].unique()[0])
constraint = 'Hour'
rownum,colnum = 8,3
plot_data_filtered(constraint,c2,rownum,colnum)
# Crime density by category after noon
c2 = (train_sub['Hour'] > 12)
constraint = 'Category'
rownum,colnum = 8,3
plot_data_filtered(constraint,c2,rownum,colnum)
