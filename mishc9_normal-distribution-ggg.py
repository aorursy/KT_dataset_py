import pandas as pd

from pandas import DataFrame

import math

import scipy.stats as st

from functools import reduce  # use reduce of course



'''This solution usese only properties of normal distribution and only features whish are normaly distributed.

Actually there's no train.'''



def isTypeProb(data, predict, name_list, param_list):

    ''' data: training dataset, 

    predict: test dataset, 

    name_list: list of creature's names, 

    param_list: list of used creature's features'''

    local_df = DataFrame()  # dataframe to fill with z_scored features

    for name in name_list:  # do it for each ccreature

        # here is the dict. {name: nameMean} with means. Add new entry for each creature

        nameMean = {p: data[(data.type == name)].__getitem__(p).mean() for p in param_list}

        # another dict. {name: nameStd} with std's. Add new entry for each creature

        nameStd = {p: data[(data.type == name)].__getitem__(p).std(ddof=0) for p in param_list}

        # calculate prob's of different features (using normal distribution)

        probByName_list = [2 * st.norm.cdf(-abs((predict[p] - nameMean[p])/nameStd[p])) for p in param_list]

        # calculate feature's product with reduce

        probByName = reduce(lambda res, x: res*x, probByName_list, 1)

        # add to resulting dataframe

        local_df[name] = probByName

        # return DataFrame with probabilities for each entry as creature

    return local_df





# read train and test data

data = pd.read_csv('train.csv')

test= pd.read_csv('test.csv')

# create sample submission file for predicted results

sample = DataFrame()

# add id's column

sample['id'] = test['id']

# some lists with features and creatures

paramList = ['hair_length', 'has_soul', 'rotting_flesh', 'bone_length']

typeList = ['Ghoul', 'Ghost', 'Goblin']

# predict probabilities!

probs = isTypeProb(data, test, typeList, paramList)

# and most probably creature for each entry

sample['type'] = probs.idxmax(axis=1)

# DO NOT FORGET TO REMOVE FIRST COLUMN!