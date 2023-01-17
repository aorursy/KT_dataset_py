import kagglegym

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import matplotlib.cm as cm

env = kagglegym.make()

observation = env.reset()

train = observation.train
id_groups = train.groupby(['id'])

products = train['id'].unique()

subset = train[train['id'].isin(products[0:500])]
subset2 = subset[['timestamp','id','y']]

yvalues = subset2.pivot_table(values='y', index='timestamp', columns='id')

yvalues = yvalues[0:400]

fig = plt.figure()

plt.imshow(yvalues, cmap=cm.seismic)

plt.title('y values for several products over time')

plt.xlabel('id')

plt.ylabel('timestamp')
plt.rcParams['figure.figsize'] = (8, 4)

interesting_factors = ['technical_43', 'technical_19', 'technical_29', 'technical_21']

for f in interesting_factors:

    subset3 = subset[['timestamp', 'id', f]]

    fvalues = subset3.pivot_table(values=f, index='timestamp', columns='id')

    fvalues = fvalues[0:400]

    fig = plt.figure()

    plt.title("y vs {}".format(f))

    plt.axis('off')

    ax = fig.add_subplot(1, 2, 1)

    ax.imshow(yvalues, cmap=cm.seismic)

    ax = fig.add_subplot(1, 2, 2)

    ax.imshow(fvalues, cmap=cm.seismic)
