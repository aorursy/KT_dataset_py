import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE



import seaborn as sns

sns.set(style="whitegrid", palette="muted")

current_palette = sns.color_palette()
dataset = pd.read_csv('../input/2015.csv')



X = dataset.ix[:, 2:].values

X = TSNE(n_components = 2).fit_transform(X)

Y = dataset.ix[:, 0].values



fig = plt.figure(figsize=(40,40))



for i, label in enumerate(Y):

    x, y = X[i, :]

    plt.scatter(x, y, color = current_palette[2])

    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    

plt.show()

    
dataset = pd.read_csv('../input/2016.csv')



X = dataset.ix[:, 2:].values

X = TSNE(n_components = 2).fit_transform(X)

Y = dataset.ix[:, 0].values



fig = plt.figure(figsize=(40,40))



for i, label in enumerate(Y):

    x, y = X[i, :]

    plt.scatter(x, y, color = current_palette[2])

    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    

plt.show()
data_2016 = pd.read_csv('../input/2016.csv')

data_2015 = pd.read_csv('../input/2015.csv')



data_2016 = data_2016.loc[data_2016['Country'] == 'Malaysia']

data_2015 = data_2015.loc[data_2015['Country'] == 'Malaysia']



data_2016 = data_2016.ix[:, 6:]

data_2015 = data_2015.ix[:, 5:]



columns = tuple(data_2016.dtypes.index)



data_2015 = data_2015.ix[:, :].values

data_2016 = data_2016.ix[:, :].values
N = data_2015[0, :].shape[0]



from pylab import rcParams

rcParams['figure.figsize'] = 14, 7



ind = np.arange(N)

width = 0.4

fig, ax = plt.subplots()



bar1 = ax.bar(ind, data_2015[0, :].tolist(), width, color = current_palette[1])

bar2 = ax.bar(ind + width, data_2016[0, :].tolist(), width, color = current_palette[2])



ax.set_ylabel('Scores')

ax.set_title('Scores by Country Attributes')

ax.set_xticks(ind + width / 2)

ax.set_xticklabels(columns)



ax.legend((bar1[0], bar2[0]), ('2015', '2016'), loc='upper center')



plt.show()