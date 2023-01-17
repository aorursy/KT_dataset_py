# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai.tabular import *
df = pd.read_csv('../input/train_V2.csv')
path = Path('../')
df_test = pd.read_csv('../input/test_V2.csv')
df_output = pd.read_csv('../input/sample_submission_V2.csv')
df[df.winPlacePerc.isnull()]
df = df.drop([2744604], axis=0)
dep_var = 'winPlacePerc'

cat_names = ['Id', 'groupId', 'matchId', 'matchType']

cont_names = ['assists', 'boosts', 'damageDealt', 'DBNOs',

              'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill', 'matchDuration', 'maxPlace', 'numGroups',

              'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',

              'winPoints']

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(df_test, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(df.loc[:500000], path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_rand_pct(valid_pct=0.2)

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())
learn = tabular_learner(data, layers=[200,100], metrics=mse)
learn.fit(1)
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(3e-3))
# l = []

# for x in range(df_test.Id.count()):

#     df_output.loc[x, 'winPlacePerc'] = str(float(preds[x]))

#     if x%100 == 0:

#         print(x)

# print("done")
# data_to_submit.to_csv('csv_to_submit.csv', index = False)
preds,y = learn.get_preds(ds_type=DatasetType.Test)
flist = []

for x in range(df_test.Id.count()):

        flist.append(float(preds[x][0]))

    
df_output['winPlacePerc'] = flist
data_to_submit = pd.DataFrame({

    'Id':df_output['Id'],

    'winPlacePerc':df_output['winPlacePerc']

})
data_to_submit.to_csv('csv_to_submit.csv', index = False)