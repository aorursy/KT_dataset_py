# module imports

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# ggplot style visuals

plt.style.use('ggplot')



# set input directory

in_dir = '../input/'
pd.read_csv(in_dir+'train.csv', nrows=10)
# what is type?

pdtrain = pd.read_csv(in_dir+'train.csv')

# we will need this one later

TYPES = list(pdtrain["type"].unique())

print(TYPES)

del pdtrain
# is test.csv different?

pd.read_csv(in_dir+'test.csv', nrows=10)
pd.read_csv(in_dir+'structures.csv', nrows=10)
dftrain = (pd.read_csv(in_dir+'train.csv')

           .drop(columns=['id','molecule_name','atom_index_0','atom_index_1']))

dist_table = dftrain.pivot_table(values='scalar_coupling_constant'

                                ,columns='type'

                                ,aggfunc=[np.mean, np.std])
# reformat the pivot table

dist_table = pd.DataFrame({"MEAN":dist_table['mean'].loc['scalar_coupling_constant']

                        ,"STDEV":dist_table['std'].loc['scalar_coupling_constant']})
dist_table
plt.figure(figsize=(10,18))

for T,splot in zip(TYPES, range(1,1+len(TYPES))):

    plt.subplot(len(TYPES),1,splot,sharex=plt.gca())

    plt.hist(dftrain

             ['scalar_coupling_constant']

             [dftrain['type']==T], bins=50,rwidth=0.9,density=True)

    plt.title(T,loc='right')

    plt.xticks([])

_=plt.xticks(np.arange(-50,151,50))
dftrain = pd.read_csv(in_dir+'train.csv').drop(columns=['id','molecule_name','atom_index_0','atom_index_1'])
# line 1 draw from normal dist, line 2 predict mean #

# uncomment whichever you need                      #

#---------------------------------------------------#



# random_pred = np.random.normal(loc=dist_table['MEAN'].loc[dftrain['type']], scale=dist_table['STDEV'].loc[dftrain['type']]*0.5,)

random_pred = dist_table['MEAN'].loc[dftrain['type']]
plt.figure(figsize=(10,10))

plt.subplot(111)

plt.title('Randomly guessed value according to type vs Real')

for T in TYPES:

    bool_only_T = (dftrain['type']==T).values

    # do not plot all data (too big)

    plt.scatter(random_pred[bool_only_T][::150]

                ,dftrain['scalar_coupling_constant'][bool_only_T][::150]

                ,label=T,s=4)



plt.ylabel('Real')

plt.xlabel('Guess')



plt.xlim(plt.ylim()[0], plt.ylim()[1]) # set limits to be same

minv, maxv = plt.xlim()

straight_line, = plt.plot([minv,maxv],[minv,maxv], c='green', alpha=0.5, label='Ideal case')

_ = plt.legend()
def eval_model(preds, real, ts):

    score=0

    for T in TYPES:

        bool_t = ts==T

        nt = len(preds[bool_t])

        temp = np.sum(np.abs(preds[bool_t]-real[bool_t]) / nt)

        score += np.log(temp)

    return score/(len(TYPES))

eval_model(random_pred.values,dftrain['scalar_coupling_constant'].values,dftrain['type'].values)