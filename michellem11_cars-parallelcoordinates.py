# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline



import pandas as pd

import numpy as np



df = pd.read_csv("../input/original-cars/original cars.csv")

df['Horsepower'] = pd.to_numeric(df['Horsepower'].replace('?', np.nan))

df['MPG'] = pd.cut(df['MPG'], [8, 16, 24, 32, 50])

pd.plotting.parallel_coordinates(

    df[['MPG', 'Displacement', 'Cylinders', 'Horsepower', 'Weight', 'Acceleration']], 

    'MPG')



plt.show()
from matplotlib import ticker



cols = ['Displacement', 'Cylinders', 'Horsepower', 'Weight', 'Acceleration']

x = [i for i, _ in enumerate(cols)]

colours = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']



# create dict of categories: colours

colours = {df['MPG'].cat.categories[i]: colours[i] for i, _ in enumerate(df['MPG'].cat.categories)}



# Create (X-1) sublots along x axis

fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))



# Get min, max and range for each column

# Normalize the data for each column

min_max_range = {}

for col in cols:

    min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]

    df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))



# Plot each row

for i, ax in enumerate(axes):

    for idx in df.index:

        mpg_category = df.loc[idx, 'MPG']

        ax.plot(x, df.loc[idx, cols], colours[mpg_category])

    ax.set_xlim([x[i], x[i+1]])

    

# Set the tick positions and labels on y axis for each plot

# Tick positions based on normalised data

# Tick labels are based on original data

def set_ticks_for_axis(dim, ax, ticks):

    min_val, max_val, val_range = min_max_range[cols[dim]]

    step = val_range / float(ticks-1)

    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]

    norm_min = df[cols[dim]].min()

    norm_range = np.ptp(df[cols[dim]])

    norm_step = norm_range / float(ticks-1)

    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]

    ax.yaxis.set_ticks(ticks)

    ax.set_yticklabels(tick_labels)



for dim, ax in enumerate(axes):

    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))

    set_ticks_for_axis(dim, ax, ticks=6)

    ax.set_xticklabels([cols[dim]])

    



# Move the final axis' ticks to the right-hand side

ax = plt.twinx(axes[-1])

dim = len(axes)

ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))

set_ticks_for_axis(dim, ax, ticks=6)

ax.set_xticklabels([cols[-2], cols[-1]])





# Remove space between subplots

plt.subplots_adjust(wspace=0)



# Add legend to plot

plt.legend(

    [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in df['MPG'].cat.categories],

    df['MPG'].cat.categories,

    bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)



plt.title("Values of car attributes by MPG category")



plt.show()