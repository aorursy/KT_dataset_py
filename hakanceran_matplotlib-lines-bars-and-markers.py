# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import library

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
labels = ['G1', 'G2', 'G3', 'G4', 'G5']



men_means = [20, 35, 30, 35, 27]

women_means = [25, 32, 34, 20, 25]

men_std = [2, 3, 4, 1, 2]

women_std = [3, 5, 2, 3, 3]



# the width of the bars: can also be len(x) sequence

width = 0.35



fig, ax = plt.subplots()



ax.bar(labels, men_means, width, yerr=men_std, label='Men')

ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means, label='Women')



ax.set_ylabel('Scores')

ax.set_title('Scores by group and gender')



ax.legend()



plt.show()
labels = ['G1', 'G2', 'G3', 'G4', 'G5']

men_means = [20, 34, 30, 35, 27]

women_means = [25, 32, 34, 20, 25]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, men_means, width, label='Men')

rects2 = ax.bar(x + width/2, women_means, width, label='Women')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Scores')

ax.set_title('Scores by group and gender')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)



fig.tight_layout()



plt.show()
# Fixing random state for reproducibility

np.random.seed(19680801)





plt.rcdefaults()

fig, ax = plt.subplots()



# Example data

people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')

y_pos = np.arange(len(people))

performance = 3 + 10 * np.random.rand(len(people))

error = np.random.rand(len(people))



ax.barh(y_pos, performance, xerr=error, align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(people)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Performance')

ax.set_title('How fast do you want to go today?')



plt.show()
fig, ax = plt.subplots()

ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue')

ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),

               facecolors=('tab:orange', 'tab:green', 'tab:red'))

ax.set_ylim(5, 35)

ax.set_xlim(0, 200)

ax.set_xlabel('seconds since start')

ax.set_yticks([15, 25])

ax.set_yticklabels(['Bill', 'Jim'])

ax.grid(True)

ax.annotate('race interrupted', (61, 25),

            xytext=(0.8, 0.9), textcoords='axes fraction',

            arrowprops=dict(facecolor='black', shrink=0.05),

            fontsize=16,

            horizontalalignment='right', verticalalignment='top')



plt.show()
data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}

names = list(data.keys())

values = list(data.values())



fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

axs[0].bar(names, values)

axs[1].scatter(names, values)

axs[2].plot(names, values)

fig.suptitle('Categorical Plotting')
cat = ["bored", "happy", "bored", "bored", "happy", "bored"]

dog = ["happy", "happy", "happy", "happy", "bored", "bored"]

activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]



fig, ax = plt.subplots()

ax.plot(activity, dog, label="dog")

ax.plot(activity, cat, label="cat")

ax.legend()



plt.show()
# Fixing random state for reproducibility

np.random.seed(19680801)



dt = 0.01

t = np.arange(0, 30, dt)

nse1 = np.random.randn(len(t))                 # white noise 1

nse2 = np.random.randn(len(t))                 # white noise 2



# Two signals with a coherent part at 10Hz and a random part

s1 = np.sin(2 * np.pi * 10 * t) + nse1

s2 = np.sin(2 * np.pi * 10 * t) + nse2



fig, axs = plt.subplots(2, 1)

axs[0].plot(t, s1, t, s2)

axs[0].set_xlim(0, 2)

axs[0].set_xlabel('time')

axs[0].set_ylabel('s1 and s2')

axs[0].grid(True)



cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)

axs[1].set_ylabel('coherence')



fig.tight_layout()

plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1)

# make a little extra space between the subplots

fig.subplots_adjust(hspace=0.5)



dt = 0.01

t = np.arange(0, 30, dt)



# Fixing random state for reproducibility

np.random.seed(19680801)





nse1 = np.random.randn(len(t))                 # white noise 1

nse2 = np.random.randn(len(t))                 # white noise 2

r = np.exp(-t / 0.05)



cnse1 = np.convolve(nse1, r, mode='same') * dt   # colored noise 1

cnse2 = np.convolve(nse2, r, mode='same') * dt   # colored noise 2



# two signals with a coherent part and a random part

s1 = 0.01 * np.sin(2 * np.pi * 10 * t) + cnse1

s2 = 0.01 * np.sin(2 * np.pi * 10 * t) + cnse2



ax1.plot(t, s1, t, s2)

ax1.set_xlim(0, 5)

ax1.set_xlabel('time')

ax1.set_ylabel('s1 and s2')

ax1.grid(True)



cxy, f = ax2.csd(s1, s2, 256, 1. / dt)

ax2.set_ylabel('CSD (db)')

plt.show()
fig = plt.figure()

x = np.arange(10)

y = 2.5 * np.sin(x / 20 * np.pi)

yerr = np.linspace(0.05, 0.2, 10)



plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')



plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')



plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,

             label='uplims=True, lolims=True')



upperlimits = [True, False] * 5

lowerlimits = [False, True] * 5

plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,

             label='subsets of uplims and lolims')



plt.legend(loc='lower right')
fig = plt.figure()

x = np.arange(10) / 10

y = (x + 0.1)**2



plt.errorbar(x, y, xerr=0.1, xlolims=True, label='xlolims=True')

y = (x + 0.1)**3



plt.errorbar(x + 0.6, y, xerr=0.1, xuplims=upperlimits, xlolims=lowerlimits,

             label='subsets of xuplims and xlolims')



y = (x + 0.1)**4

plt.errorbar(x + 1.2, y, xerr=0.1, xuplims=True, label='xuplims=True')



plt.legend()

plt.show()