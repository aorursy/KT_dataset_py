import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

from pandas.io.common import EmptyDataError

plt.rcParams['figure.figsize'] = (10, 6)

style.use('ggplot')



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, StratifiedKFold

from scipy.stats import skew, kurtosis

from sklearn.preprocessing import StandardScaler

from sklearn.base import clone, BaseEstimator, TransformerMixin



import os

from functools import partial

import re

from collections import Counter



import warnings

warnings.filterwarnings('ignore')
import umap
user_root = "../input/Archived-users/Archived users/"

user_fn_list = os.listdir(user_root)
def read_one_file(fn, root):

    out = dict()

    with open(root + fn) as f:

        for line in f.readlines():

            k, v = line.split(": ")

            out[k] = v.strip()

            out['ID'] = re.findall(r'_(\w+)\.', fn)[0]

    return out
users_list = list(map(partial(read_one_file, root=user_root), user_fn_list))
users = pd.DataFrame(users_list)

users.replace('------', np.nan, inplace=True)

users.replace('', np.nan, inplace=True)

users['Levadopa'] = users['Levadopa'] == 'True'

users['MAOB'] = users['MAOB'] == 'True'

users['Parkinsons'] = users['Parkinsons'] == 'True'

users['Tremors'] = users['Tremors'] == 'True'

users['Other'] = users['Other'] == 'True'
users.head()
keys_root = "../input/Archived-Data/Tappy Data/"

keys_fn_list = os.listdir(keys_root)
sample = pd.read_csv(keys_root + keys_fn_list[0], delimiter='\t', header=None, usecols=range(8))

sample.columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime']

sample.head()
def read_one_key_file(fn, root):

    try:

        df = pd.read_csv(root + fn, delimiter='\t', header=None, error_bad_lines=False,

                         usecols=range(8), low_memory=False,

                        dtype={0:'str', 1:'str', 2:'str', 3:'str', 4:'float', 5:'str', 6:'float', 7:'float'})

        df.columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime']

    except ValueError:

        # should try to remove the bad lines and return

#         df = pd.DataFrame(columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime'])

        try:

            df = pd.read_csv(root + fn, delimiter='\t', header=None, error_bad_lines=False,

                             usecols=range(8), low_memory=False)

            df.columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime']

            df = df[df['ID'].apply(lambda x: len(str(x)) == 10)

                   & df['Date'].apply(lambda x: len(str(x)) == 6)

                   & df['TS'].apply(lambda x: len(str(x)) == 12)

                   & np.in1d(df['Hand'], ["L", "R", "S"])

                   & df['HoldTime'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)

                   & np.in1d(df['Direction'], ['LL', 'LR', 'RL', 'RR', 'LS', 'SL', 'RS', 'SR', 'RR'])

                   & df['LatencyTime'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)

                   & df['FlightTime'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)]

            df['HoldTime'] = df['HoldTime'].astype(np.float)

            df['LatencyTime'] = df['HoldTime'].astype(np.float)

            df['FlightTime'] = df['HoldTime'].astype(np.float)

        except EmptyDataError:

            df =  pd.DataFrame(columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime'])

    except EmptyDataError:

        df =  pd.DataFrame(columns = ['ID', 'Date', 'TS', 'Hand', 'HoldTime', 'Direction', 'LatencyTime', 'FlightTime'])

    return df
keys_list = list(map(partial(read_one_key_file, root=keys_root), keys_fn_list))
keys = pd.concat(keys_list, ignore_index=True, axis=0)
keys.head()
keys.shape
key_id_set = set(keys['ID'].unique())

print("total user ids in key logs: {0}".format(len(key_id_set)))

user_id_set = set(users['ID'].unique())

print("total user ids in user info: {0}".format(len(user_id_set)))

overlap_id_set = key_id_set.intersection(user_id_set)

print("overlapping ids: {0}".format(len(overlap_id_set)))

diff_id_set = key_id_set.symmetric_difference(user_id_set)

print("non-matching ids: {0}".format(len(diff_id_set)))
sufficient_keys = keys.groupby('ID').size() >= 2000

user_w_sufficient_data = set(sufficient_keys.index[sufficient_keys])

user_eligible = set(users[((users['Parkinsons']) & (users['Impact'] == 'Mild') 

                       | (~users['Parkinsons']))

                      & (~users['Levadopa'])]['ID'])

user_valid = user_w_sufficient_data.intersection(user_eligible)
len(user_valid)
users.query('ID in @user_valid').groupby('Parkinsons').size()
# valid_keys = keys[(keys['HoldTime'] > 0)

#                    & (keys['LatencyTime'] > 0)

#                    & (keys['HoldTime'] < 2000)

#                    & (keys['LatencyTime'] < 2000)

#                    & np.isin(keys['ID'], list(user_valid))]



valid_keys = keys[np.isin(keys['ID'], list(user_valid))]



valid_keys.shape
hold_by_user =  valid_keys[valid_keys['Hand'] != 'S'].groupby(['ID', 'Hand'])['HoldTime'].agg([np.mean, np.std, skew, kurtosis])
hold_by_user.head(10)
latency_by_user = valid_keys[np.isin(valid_keys['Direction'], ['LL', 'LR', 'RL', 'RR'])].groupby(['ID', 'Direction'])['LatencyTime'].agg([np.mean, np.std, skew, kurtosis])
latency_by_user.head(10)
hold_by_user_flat = hold_by_user.unstack()

hold_by_user_flat.columns = ['_'.join(col).strip() for col in hold_by_user_flat.columns.values]

hold_by_user_flat['mean_hold_diff'] = hold_by_user_flat['mean_L'] - hold_by_user_flat['mean_R']

hold_by_user_flat.head()
latency_by_user_flat = latency_by_user.unstack()

latency_by_user_flat.columns = ['_'.join(col).strip() for col in latency_by_user_flat.columns.values]

latency_by_user_flat['mean_LR_RL_diff'] = latency_by_user_flat['mean_LR'] - latency_by_user_flat['mean_RL']

latency_by_user_flat['mean_LL_RR_diff'] = latency_by_user_flat['mean_LL'] - latency_by_user_flat['mean_RR']

latency_by_user_flat.head()
combined = pd.concat([hold_by_user_flat, latency_by_user_flat], axis=1)
combined.shape
combined.head()
full_set = pd.merge(combined.reset_index(), users[['ID', 'Parkinsons']], on='ID')

full_set.set_index('ID', inplace=True)

# full_set.dropna(inplace=True)  # should investigate why there are NAs despite choosing sequence length >= 2000

full_set.shape
full_set.head()
umapModel = umap.UMAP()

embed = umapModel.fit_transform(full_set.iloc[:, :-1], full_set.iloc[:, -1])
notPak = full_set.Parkinsons.map(lambda x: not x)

plt.scatter(embed[full_set.Parkinsons, 0], embed[full_set.Parkinsons, 1], color="red")

plt.scatter(embed[notPak, 0], embed[notPak, 1], color="blue")
import seaborn as sns

plt.figure(figsize=(10, 18))



for i, c in enumerate(full_set.columns[:-1]):

    plt.subplot(9, 3, i + 1)

    sns.distplot(full_set[c][full_set.Parkinsons], color="red", label="Patients")

    sns.distplot(full_set[c][notPak], color="blue", label="not Patients")
full_set.shape, embed.shape, train_X.shape
from sklearn.svm import LinearSVC

model = LinearSVC()

rs = 42



train_X = np.concatenate([full_set.iloc[:, :-1], pd.DataFrame(embed)], axis=1)

train_y = full_set.iloc[:, -1]



scoring = ['accuracy', 'f1', 'roc_auc']

scores = cross_validate(

    model,

    train_X,

    train_y,

    cv=StratifiedKFold(n_splits=5, random_state=rs),

    scoring=scoring,

    return_train_score=True

)
result = pd.DataFrame(scores)

result.index.name = "fold"

result
# class SubsetTransformer(TransformerMixin):

#     def __init__(self, start=0, end=None):

#         self.start = start

#         self.end = end

        

#     def fit(self, *_):

#         return self

    

#     def transform(self, X, *_):

#         if self.end is None:

#             return X.iloc[:, self.start:]

#         else:

#             return X.iloc[:, self.start: self.end]
# rs = 61
# select_left = SubsetTransformer(0, 9)

# select_right = SubsetTransformer(9, -1)

# scale = StandardScaler()

# lda = LinearDiscriminantAnalysis()

# gb = GradientBoostingClassifier(max_depth=7, n_estimators=127, random_state=rs)

# # ensemble here

# pl_left = Pipeline([('select_left', select_left), 

#                     ('normalise', scale), 

#                     ('LDA', lda), 

#                     ('classify', gb)])

# pl_right = Pipeline([('select_left', select_right), 

#                     ('normalise', clone(scale)), 

#                     ('LDA', clone(lda)), 

#                     ('classify', clone(gb))])

# vote = VotingClassifier([('left', pl_left), ('right', pl_right)], weights=[1, 1.2], voting="soft")
# train_X, test_X, train_y, test_y = train_test_split(full_set.iloc[:, :-1], full_set.iloc[:, -1], test_size=0.35, stratify=full_set.iloc[:, -1], random_state=rs)



# scoring = ['accuracy', 'precision', 'recall', 'f1']

# scores = cross_validate(vote, train_X, train_y, cv=StratifiedKFold(n_splits=10, random_state=rs), scoring=scoring, return_train_score=True)
# vote.fit(train_X, train_y)

# (vote.predict(test_X) == test_y).sum(), test_y.shape[0]
# pd.DataFrame(scores).mean()