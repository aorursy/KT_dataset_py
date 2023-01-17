# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import urllib

import io

import zipfile

import surprise



# Download zip file

# tmpFile = urllib.request.urlopen('https://librec.net/datasets/fimltrust.zip')

# Unzip file

# tmpFile = zipfile.zipFile(io.BytesIO(tmpFile.read()))

# Open desired data file as pandas dataframe, close zipfile

# dataset = pd.read_table(io.BytesIO(tmpFile.read('ratings.txt')), sep = ' ', names = ['uid', 'iid', 'rating'])

dataset = pd.read_csv('/kaggle/input/ratingstxt/ratings.txt', sep = ' ', names = ['uid', 'iid', 'rating'])

# tmpFile.close()



dataset.head()
lower_rating = dataset['rating'].min()

upper_rating = dataset['rating'].max()

print('Review range: {0} to {1}'.format(lower_rating, upper_rating))
reader = surprise.Reader(rating_scale = (0.5,4))

data = surprise.Dataset.load_from_df(dataset, reader)
alg = surprise.SVDpp()

output = alg.fit(data.build_full_trainset())
# the uids and iids should be set as strings

pred = alg.predict(uid = '50', iid = '52')

score = pred.est

print(score)
# Get a list of all movie ids

iids = dataset['iid'].unique()

# Get a list of iids that uid 50 has rated

iids50 = dataset.loc[dataset['uid'] == 50, 'iid']

# Remove the iids that uid 50 has rated from the list of all movie ids

iids_to_pred = np.setdiff1d(iids, iids50)

# np.setdiff1d -> Return the unique values in ar1 that are not in ar2
testset = [[50, iid, 4.] for iid in iids_to_pred]

predictions = alg.test(testset)

predictions[0]
pred_ratings = np.array([pred.est for pred in predictions])

# Find the indec of the maximum predicted rating

i_max = pred_ratings.argmax()

# Use this to find the corresponding iid to recommend

iid = iids_to_pred[i_max]

print('Top item for user 50 has iid {0} with predicted rating {1}'.format(iid, pred_ratings[i_max]))
pred_ratings = np.array([pred.est for pred in predictions])

# Find the indec of the maximum predicted rating

i_max = pred_ratings.argpartition(pred_ratings, 0, axis = None)[0:]

# Use this to find the corresponding iid to recommend

iid = iids_to_pred[i_max]

print('Top item for user 50 has iid {0} with predicted rating {1}'.format(iid, pred_ratings[i_max]))
param_grid = {'lr_all' : [.001, .01], 'reg_all' : [.1, .5]}

gs = surprise.model_selection.GridSearchCV(surprise.SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# Print combination of parameters that gave best RMSE score

print(gs.best_params['rmse'])
alg = surprise.SVDpp(lr_all = 0.01) # parameter choices can be added here

output = surprise.model_selection.cross_validate(alg, data, verbose = True)