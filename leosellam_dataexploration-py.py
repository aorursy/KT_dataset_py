# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib
matplotlib.use('TkAgg')

from scipy import stats, integrate
import matplotlib.pyplot as plt
import pylab
import seaborn as sns # A Python visualization library based on matplotlib
sns.set(color_codes=True)


FIG_PATH = "../output"

"""
A method to plot the distribution
of a variable
@param : array of numerical variable
@param : fig_name. if false, plot won't be saved.
@return : plot interactive window
"""

def distribution_plot(x, fig_name):
    fig = sns.distplot(x)
    if fig_name:
        plt.savefig(FIG_PATH + fig_name + '.png')
        plt.close()
    else:
        sns.plt.show()

def save_and_close(fig_name):
    plt.savefig(FIG_PATH + fig_name + '.png')
    plt.close

def plot_xy_by_place_id(df, place_id):
    index = df["place_id"] ==  place_id;
    plt.scatter(df['x'][index], df['y'][index])

df_train = pd.read_csv("../input/train.csv")
nRows = len(df_train);



unique_place_id = set(df_train['place_id'])
list_unique_place_id = list(unique_place_id)
print(len(unique_place_id))

print('So we expect approximatively ' +  str(nRows/len(unique_place_id)) + ' row_id per place_id')

# -------   Define a smaller dataset -------
# It will allow to be more agile with
# the methods we'll test

boots = np.random.randint(1, nRows, int(0.0001*nRows))
df_small = df_train.iloc[boots, ];
plt.scatter(df_small['x'], df_small['y']) # random 2D-uniform distribution.


# ----- Location and place_id ------
# Let's take the first place_id of the dataset and check the (x, y)
# distribution for this place_id only.


ind = 0;
plot_xy_by_place_id(df_train, list_unique_place_id[ind])
title = 'x_y_' + str( list_unique_place_id[ind]) + '_place_id'
save_and_close(title)

ind = np.random.randint(1, len(list_unique_place_id));
plot_xy_by_place_id(df_train, list_unique_place_id[ind])
title = 'x_y_' + str( list_unique_place_id[ind]) + '_place_id'
save_and_close(title)

ind = np.random.randint(1, len(list_unique_place_id));
plot_xy_by_place_id(df_train, list_unique_place_id[ind])
title = 'x_y_' + str( list_unique_place_id[ind]) + '_place_id'
save_and_close(title)






