# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
frame = pd.read_csv('../input/Pokemon.csv')
frame.head()
vals1 = [frame['Type 1'].value_counts()[key] for key in frame['Type 1'].value_counts().index]

vals2 = [frame['Type 2'].value_counts()[key] for key in frame['Type 1'].value_counts().index]

inds = np.arange(len(frame['Type 1'].value_counts().index))

width = .45

color1 = np.random.rand(3)

color2 = np.random.rand(3)

handles = [patches.Patch(color=color1, label='Type 1'), patches.Patch(color=color2, label='Type 2')]

plt.bar(inds, vals1, width, color=color1)

plt.bar(inds+width, vals2, width, color=color2)

plt.gca().set_xticklabels(frame['Type 1'].value_counts().index)

plt.gca().set_xticks(inds+width)

plt.xticks(rotation=90)

plt.legend(handles=handles)
stats = frame.columns[5:11]

plt.figure(figsize=(25, 20))



for ii, stat in enumerate(stats):

    title = "Distributions of {stat}".format(

        stat = stat

    )

    plt.subplot(3, 3, ii+1)

    plt.title(title)

    sns.distplot(frame[stat])

    x = plt.gca().get_xlim()[1] * .6

    y = plt.gca().get_ylim()[1] * .9

    plt.text(x, y, '$\mu: {mu: .2f}, \sigma: {sigma: .2f}$'.format(mu = frame[stat].mean(), sigma=frame[stat].std()))

    

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(25,20))



for ii, stat in enumerate(stats):

    title = "{stat} Distributions of First Types".format(

        stat = stat

    )

    plt.subplot(3, 3, ii+1)

    plt.title(title)

    plt.xticks(rotation=90)

    sns.boxplot(x='Type 1', y=stat, data = frame)

    plt.axhline(frame[stat].median(), color=np.random.rand(3))

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(25,20))

for ii, stat in enumerate(stats):

    title = "{stat} Distributions of Second Types".format(

        stat = stat

    )

    plt.subplot(3, 3, ii+1)

    plt.title(title)

    plt.xticks(rotation=90)

    sns.boxplot(x='Type 2', y=stat, data = frame)

    plt.axhline(frame[stat].median())

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(25,20))

for ii, stat in enumerate(stats):

    title = "{stat} Distributions of Second Types".format(

        stat = stat

    )

    plt.subplot(3, 3, ii+1)

    plt.title(title)

    plt.xticks(rotation=90)

    sns.boxplot(x='Type 2', y=stat, data = frame)

    plt.axhline(frame[stat].median())

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(25,20))

for ii, stat in enumerate(stats):

    title = "{stat} Distributions of Second Types".format(

        stat = stat

    )

    plt.subplot(3, 3, ii+1)

    plt.title(title)

    plt.xticks(rotation=90)

    sns.boxplot(x='Legendary', y=stat, data = frame)

    plt.axhline(frame[stat].median())

    

plt.tight_layout()

plt.show()