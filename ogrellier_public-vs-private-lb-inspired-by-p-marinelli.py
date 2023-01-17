# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
train_df = pd.read_csv('../input/train.csv').set_index('ID')

train_target = train_df.y.copy()

train_target[train_target>260] = 175

train_df.drop(['y'], axis=1, inplace=True)
bin_f = [f for f in train_df.columns

         if (len(train_df[f].value_counts()) == 2)]



hcc_f = [f for f in train_df.columns

         if (len(train_df[f].value_counts()) > 2)]

data = train_df[bin_f].copy()

for f in hcc_f:

    data = pd.concat([data, pd.get_dummies(train_df[f], prefix=f)], axis=1)
indexes = np.arange(len(data))

train_index = data.index

np.random.seed(1234)

ratio = .81

n_rounds = 50



publ_la_r2_scores = np.zeros(n_rounds)

priv_la_r2_scores = np.zeros(n_rounds)

publ_rf_r2_scores = np.zeros(n_rounds)

priv_rf_r2_scores = np.zeros(n_rounds)

publ_all_r2_scores = np.zeros(n_rounds)

priv_all_r2_scores = np.zeros(n_rounds)



# Train a lasso

# Fit a Lasso

la = Lasso(alpha=.01, normalize=False, max_iter=3000)

rfr = RandomForestRegressor(

    n_estimators=100,

    max_features=.25,

    min_samples_split=100,

    random_state=1,

    n_jobs=-1)



for i in range(n_rounds):

    # shuffle the indices

    np.random.shuffle(indexes)

    

    # Split the indices to get a train and submission dataset

    trn_idx = indexes[:int(len(indexes) * .5)]

    sub_idx = indexes[int(len(indexes) * .5):]

    

    # Split the data

    trn_df, trn_y = data.loc[train_index[trn_idx]], train_target.loc[train_index[trn_idx]]

    sub_df, sub_y = data.loc[train_index[sub_idx]], train_target.loc[train_index[sub_idx]]

    

    # Fit the lasso

    la.fit(trn_df.values, trn_y.values)

    rfr.fit(trn_df.values, trn_y.values)

    # print(r2_score(trn_y, rfr.predict(trn_df.values)), 

    #       r2_score(sub_y, rfr.predict(sub_df.values)))

    

    # Get submission predictions

    la_sub_preds = la.predict(sub_df.values)

    rf_sub_preds = rfr.predict(sub_df.values)

    

    # Now split the submission fold in public and private LB

    priv_la_r2_scores[i] = r2_score(sub_y.values[:int(len(sub_y) * ratio)],

                             la_sub_preds[:int(len(sub_y) * ratio)])

    publ_la_r2_scores[i] = r2_score(sub_y.values[int(len(sub_y) * ratio):],

                             la_sub_preds[int(len(sub_y) * ratio):])

    priv_rf_r2_scores[i] = r2_score(sub_y.values[:int(len(sub_y) * ratio)],

                             rf_sub_preds[:int(len(sub_y) * ratio)])

    publ_rf_r2_scores[i] = r2_score(sub_y.values[int(len(sub_y) * ratio):],

                             rf_sub_preds[int(len(sub_y) * ratio):])

    priv_all_r2_scores[i] = r2_score(

        sub_y.values[:int(len(sub_y) * ratio)],

        (rf_sub_preds[:int(len(sub_y) * ratio)] + la_sub_preds[:int(len(sub_y) * ratio)]) / 2)

    publ_all_r2_scores[i] = r2_score(

        sub_y.values[int(len(sub_y) * ratio):],

        (rf_sub_preds[int(len(sub_y) * ratio):] + la_sub_preds[int(len(sub_y) * ratio):]) / 2)

    
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(2, 2)

ax = plt.subplot(gs[0, 0])

sns.regplot(publ_la_r2_scores, priv_la_r2_scores, ax=ax)

ax.set_title('Lasso')

ax = plt.subplot(gs[0, 1])

sns.regplot(publ_rf_r2_scores, priv_rf_r2_scores, ax=ax)

ax.set_title('RandomForest')

ax = plt.subplot(gs[1, 0])

sns.regplot(publ_all_r2_scores, priv_all_r2_scores, ax=ax)

ax.set_title('Lasso + RandomForest')

plt.tight_layout()