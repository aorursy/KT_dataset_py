#imports 

import math

from random import shuffle

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load data

train = pd.read_csv('../input/train.csv')

train[['id','cast','popularity','budget','revenue']].head()
# budget / revenue hist

sns.distplot(train['revenue'])
# getNameSets : function to transform the serialized python dictionaries in cast/crew

# into a table of unique names

jobs = ['Writer','Director']

def getNameSets(dataset):

    idnameset = {}

    namecountset = {}

    for tup in dataset.itertuples():

        # eval python code in cast/crew columns into filtered name lists

        shortlist = []

        if isinstance(tup.cast, "".__class__):

            evaled_cast = eval(tup.cast)

            shortlist = shortlist + evaled_cast[0:5] # first 5 (presorted by 'order')

        if isinstance(tup.crew, "".__class__):

            evaled_crew = eval(tup.crew)

            crewlist = [x for x in evaled_crew if x['job'] in jobs] # match jobs

            shortlist = shortlist + crewlist

        for obj in shortlist:

            if not (tup.id in idnameset):

                idnameset[tup.id] = {}

            idnameset[tup.id][obj['name']] = True 

            if not (obj['name'] in namecountset):

                namecountset[obj['name']] = 0

            namecountset[obj['name']] += 1

    return (idnameset, namecountset)

(idnameset, namecountset) = getNameSets(train)

print(f'unique names count {len(namecountset.keys())}, records count {len(idnameset.keys())}')
# names that appear in more than one record

repeat_names = [{'name': name, 'records': count} for name,count in namecountset.items() if count > 1]

# print(len(repeat_names))

distdf = pd.DataFrame(repeat_names)['records']

sns.distplot(distdf)
# top 10

top_10_names = sorted(repeat_names, key=lambda x: -x['records'])[0:10]

sns.barplot(y="name",x="records", data=pd.DataFrame(top_10_names))
# load test data

test = pd.read_csv('../input/test.csv')

len(test)
# build names and y dfs for test data, only using names used from training set

(test_idnameset) = getNameSets(test)

test_list = []

relevant_ids = {}

def defaultNameObj():

    obj = {}

    for name in namecountset.keys():

        obj[name] = 0.0

    return obj

# make train rows from known names

for tup in test.itertuples():

    robj = defaultNameObj()

    if tup.id in test_idnameset:

        for name in test_idnameset[tup.id].keys():

            if name in namecountset:

                relevant_ids[tup.id] = True

                robj[name] = 1.0

    test_list.append(robj)

len(relevant_ids.keys())