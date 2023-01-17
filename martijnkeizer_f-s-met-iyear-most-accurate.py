# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# this will remove warnings messages

import warnings

warnings.filterwarnings('ignore')



# import library for plotting

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])

data.head()
data = data[pd.notnull(data['nkill'])]



print(data.nkill.count())

#data.dropna(subset=['nkill'])

total_nkill = data.nkill.sum()

avg_kill = total_nkill/data.nkill.count()

print(avg_kill)



data["avg_nkill"] = avg_kill

data["boolean_nkill"] = data["nkill"] > data["avg_nkill"]

col_names = data.columns.tolist()

print("Column names: ")

print(col_names)
#data['nkill'] = data['nkill'].replace(np.nan, 0)

data = data[pd.notnull(data['iyear'])]

data = data[pd.notnull(data['imonth'])]

data = data[pd.notnull(data['country'])]

data = data[pd.notnull(data['region'])]

data = data[pd.notnull(data['vicinity'])]

data = data[pd.notnull(data['success'])]

data = data[pd.notnull(data['suicide'])]

data = data[pd.notnull(data['attacktype1'])]

data = data[pd.notnull(data['targtype1'])]

data = data[pd.notnull(data['individual'])]

data = data[pd.notnull(data['nperps'])]

data = data[pd.notnull(data['claimed'])]

data = data[pd.notnull(data['weaptype1'])]

data = data[pd.notnull(data['nkillter'])]

data = data[pd.notnull(data['nwound'])]

#data = data[pd.notnull(data['boolean_nkill'])]



data.index = pd.RangeIndex(len(data.index))

#isolate target data

y = data['boolean_nkill']

y = y*1



# We don't need these columns

to_drop = ['nkill','boolean_nkill', 'gname', 'nwoundte','eventid','iday', 'extended', 'country_txt', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'specificity', 'location', 'summary', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'multiple', 'attacktype2', 'attacktype3', 'targsubtype1', 'corp1', 'target1', 'natlty1', 'targtype2', 'targsubtype2', 'corp2', 'target2', 'natlty2', 'targtype3', 'targsubtype3', 'corp3', 'target3', 'natlty3', 'gsubname', 'gname2', 'gsubname2', 'gname3', 'gsubname3', 'motive', 'guncertain1', 'guncertain2', 'guncertain3', 'nperpcap', 'claimmode', 'compclaim', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4', 'weapdetail', 'nwoundus', 'property', 'propextent', 'propvalue', 'propcomment', 'ishostkid', 'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'hostkidoutcome', 'nreleased', 'addnotes', 'scite1', 'scite2', 'scite3', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related', 'avg_nkill']

data_feat_space = data.drop(to_drop,axis=1)







# Pull out features for future use

features = data_feat_space.columns

X = data_feat_space.as_matrix().astype(np.float)



print(features)
# this will remove warnings messages

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesClassifier


# Build a forest and compute the feature importances

forest = ExtraTreesClassifier(n_estimators=250,random_state=0)



forest.fit(X, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure(figsize=(20,10))

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()