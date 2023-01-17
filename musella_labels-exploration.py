import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline
# load labels dataframe

labels = pd.read_csv('../input/train.csv')
# get list of tags 

from functools import reduce

tags = np.unique(reduce(lambda x,y: x+y, map(lambda z: z.split(" "), np.unique(labels.tags))))



print(tags)
# add columns to encode each of the tags

for tag in tags:

    func = lambda x: 1 if (tag in x["tags"].split(" ")) else 0

    labels[tag] = labels.apply(func,axis=1,raw=False)
# some plotting function helpers



plt.rcParams['image.cmap'] = 'bwr'



def corr_matrix(df,figsize=(6,6),fignum=0):

        corr = df.corr()

        plt.figure(figsize=figsize)

        plt.matshow(corr,fignum=fignum, vmin=-1, vmax=1)

        plt.xticks(np.arange(len(corr.columns)),corr.columns,rotation='vertical')

        plt.yticks(np.arange(len(corr.columns)),corr.columns)

        # plt.clims(-1.,1.)                                                                                                                                                                   

        plt.colorbar()



def cooccurence_matrix(df,figsize=(6,6),fignum=0):

    c_matrix = df.T.dot(df) # / df.index.size

    #print(type(c_matrix))

    cols = c_matrix.sum(axis=0)

    for col in c_matrix.columns: c_matrix[col] /= cols[col]

    #print(c_matrix.sum(axis=0))

    plt.figure(figsize=figsize)

    plt.matshow(c_matrix,fignum=fignum, vmin=0, vmax=1,cmap=plt.get_cmap('Blues'))

    plt.xticks(np.arange(len(c_matrix.columns)),c_matrix.columns,rotation='vertical')

    plt.yticks(np.arange(len(c_matrix.columns)),c_matrix.columns)

    #return c_matrix

    plt.colorbar()

        
# Histogram of label instances

frequencies = (labels[tags].sum()/labels.index.size).sort_values(ascending=False) 
# sort tags by sequence

sorted_tags = frequencies.index.tolist()



# put wheather tags in front

wheather_tags = ['clear','partly_cloudy','haze','cloudy']



other_tags = list(filter(lambda x: not x in wheather_tags, sorted_tags))

sorted_tags = wheather_tags + other_tags 
# plot sorted tags

plt.figure(figsize=(10,6))

frequencies[sorted_tags].plot.bar()
# sanity check

frequencies[wheather_tags].sum()
# zoom-in on the wheather tags

plt.figure()#figsize=(10,6))

frequencies[wheather_tags].plot.bar()
# plot correlation matrix between non-wheather tags

corr_matrix(labels[other_tags])

plt.show()



# correlation matrixes conditional on the wheather tags

for condition in wheather_tags:

    print(condition)

    corr_matrix(labels[labels[condition] == 1][other_tags])

    plt.show()
# same as above but for the co-concurrence matrix

cooccurence_matrix(labels[other_tags])

plt.show()

for condition in wheather_tags:

    print(condition)

    cooccurence_matrix(labels[labels[condition] == 1][other_tags])

    plt.show()
# how many (non-wheather) tags do the images have?

density = labels[other_tags].sum(axis=1)



plt.figure(figsize=(10,6))

density.hist(bins=np.linspace(-0.5,9.5,11))



density.describe()
# stratify density by wheather condition

labels['density'] = density

labels['wheather'] = labels.apply(lambda x: sum(il*x[lab] for il,lab in enumerate(wheather_tags)), axis=1  )



labels.groupby('wheather').density.hist(bins=np.linspace(-0.5,9.5,11))
# plot tags frequency conditionally on the density 

labels_by_density = labels.groupby('density')[other_tags].sum()



for den in np.unique(density):

    print(den)

    labels_by_density.loc[den].plot.bar()

    plt.show()
# and the opposite: density distribution conditionally on the tag

for tag in other_tags:

    plt.figure()# figsize=(6,6))

    print(tag)

    labels[labels[tag] == 1]['density'].hist(bins=np.linspace(-0.5,9.5,11))

    plt.show()