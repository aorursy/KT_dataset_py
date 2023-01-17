% matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import pylab

import matplotlib



np.random.seed(12345)



df = pd.DataFrame([np.random.normal(33500,150000,3650), 

                   np.random.normal(41000,90000,3650), 

                   np.random.normal(41000,120000,3650), 

                   np.random.normal(48000,55000,3650)], 

                  index=[1992,1993,1994,1995])

df = df.T

df.head()
import matplotlib as mpl

import numpy as np

from matplotlib import cm

from numpy.random import randn

from matplotlib.colors import LinearSegmentedColormap 



# Calculate means and stds

means = df.apply(np.mean, axis=0)

std = df.apply(np.std, axis=0)



# Establish a 95% Confidence Interval for the point estimates in df.

CI = []

for i in std:

    CI.append(1.96*i/df.count().values[1]**0.5)

CI = np.asarray(CI)

top = means.values + CI

bottom = means.values - CI



# Specify labels for the barchart.

labels = list(means.index.values.astype(str))



# Set the look of the error bars.

error_config = {'elinewidth': 2, 'ecolor': 'dimgray', 'capsize': 15, 'capthick': 2}



# Calculate the probability that the constant value v is lower than 

# the mean of each distribution in df.

def prob_lower(value, distribution, ssize, n):

    '''Determines the simulated probability that

    a given 'value' is strictly lower than the mean

    of a given 'distribution', by drawing n random

    samples of size 'ssize' '''

    count = 0

    for i in range(n):

        x = np.mean(np.random.choice(distribution, size=ssize))

        if value < x:

            count +=1

    return float(count/n)





# Plot bar chart

fig, ax = plt.subplots(1,1)

x = range(len(labels))

y = means.values

ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)

ax.yaxis.set_ticks_position('left')

ax.xaxis.set_ticks_position('bottom')

ax.set_xticks(range(len(x)+1))

ax.set_xticklabels(labels)

ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

barchart = ax.bar(x, y, width=1, align = 'center', color = 'lightcoral', alpha=0.5, yerr = CI, error_kw = error_config)



# Create colorbar

cmap = LinearSegmentedColormap.from_list(name='color_map', colors =['blue', 'white', 'red'], N=10)

# http://stackoverflow.com/questions/8342549/

sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0.,vmax=1.))

sm._A = []

cbar = plt.colorbar(sm, alpha=0.5, aspect=16, shrink=1, orientation = 'horizontal')



# Specify an initial value for constant value along y-axis and draw horizontal line.

v = 40000

lhor = ax.axhline(v)



# Set figure title including value of interest.

ax.set_title("Value of interest = {}".format(format(int(v), ',')))



# Apply conditional colouring based on the probability of

# the constant value v to be lower than the mean of each distribution

# in df.

for column, bar in zip(df.columns.values, barchart):

    prob = prob_lower(v, df[column], 1000, 1000)

    bar.set_color(cmap(prob)[:3])



plt.tight_layout()

plt.show()



#Add interactivity

def callback(event):

    v = event.ydata

    lhor.set_ydata(v)

    ax.set_title("Value of interest = {}".format(format(int(v), ',')))

    for column, bar in zip(df.columns.values, barchart):

        prob = prob_lower(v, df[column], 1000, 1000)

        bar.set_color(cmap(prob)[:3])



# tell mpl_connect we want to pass a 'pick_event' into onpick when the event is detected

fig.canvas.callbacks.connect('button_press_event', callback)