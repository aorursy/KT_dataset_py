import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_palette("bright")

%matplotlib inline



print("Data & File Sizes")

data_dir = '../input/cat-in-the-dat/'

for f in os.listdir(data_dir):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize(f'{data_dir}{f}') / 1000000, 2)) + 'MB')
df_train = pd.read_csv(f'{data_dir}train.csv')

pd.set_option('display.max_columns', 200) # show all columns

df_train.head()
df_test = pd.read_csv(f'{data_dir}test.csv')

df_test.head(1)
sns.countplot(df_train['target']).set_title('train')
bin_columns = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

nom_columns = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

ord_columns = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

cyc_columns = ['day', 'month']
df_train[nom_columns+ord_columns].nunique()
lc_nom_columns = nom_columns[0:5]

hc_nom_columns = nom_columns[5:10]

lc_ord_columns = ord_columns[0:5]

hc_ord_columns = ord_columns[5:6]
fig, ax = plt.subplots(2, 3, figsize=(16,8))

fig.suptitle('Binary Distribution vs. Distribution of Target Variable')



for ax, name in zip(ax.flatten(), list(df_train[bin_columns].columns)):

    sns.countplot(x=df_train[name], ax=ax, hue=df_train['target'], saturation=1)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



df_train['bin_3'] = le.fit_transform(df_train['bin_3'])

df_train['bin_4'] = le.fit_transform(df_train['bin_4'])

df_test['bin_3'] = le.fit_transform(df_test['bin_3'])

df_test['bin_4'] = le.fit_transform(df_test['bin_4'])



df_train[bin_columns].head()
fig, ax = plt.subplots(5, 1, figsize=(18,10))

fig.suptitle('Distribution of Target Variable Ratio = 1 \n (lowest → highest ratio)')



ordinal_ordering = {}



for ax, name in zip(ax.flatten(), list(df_train[lc_ord_columns].columns)):

    # calculate the ratio of target counts

    ct = pd.crosstab(df_train[name], df_train['target']).apply(lambda r: r/r.sum(), axis = 1)

    # unstack the cross-tabulated df

    stacked = ct.stack().reset_index().rename(columns = {0: 'ratio'}) 

    # sort by target ratio

    stacked = stacked.sort_values(['target', 'ratio'], ascending = [False, True]) 

    sns.barplot(x = stacked[name], y = stacked['ratio'], ax = ax, hue = stacked['target'])

    

    # create mapping for encoding

    ordinal_ordering[name] = stacked[name].unique()
# show the order of encoding for each ordinal column

ordinal_ordering
# loop through low-cardinality ordinal columns and encode them

for col in lc_ord_columns:

    nbr_to_replace = len(ordinal_ordering[col])

    # print(nbr_to_replace) # quality control

    df_train[col].replace(to_replace = ordinal_ordering[col], 

                          # had to drop a pythonic line ¯\_(ツ)_/¯

                          value = [x for x in range(0, len(ordinal_ordering[col]))], 

                          inplace = True)

    df_test[col].replace(to_replace = ordinal_ordering[col], 

                          # had to drop a pythonic line ¯\_(ツ)_/¯

                          value = [x for x in range(0, len(ordinal_ordering[col]))], 

                          inplace = True)

    

#df_train[lc_ord_columns].nunique() # quality control - should match nbr_to_replace



df_train[lc_ord_columns].head()
ord_5_num_unique = len(df_train['ord_5'].unique().tolist())

print(f'unique values in ord_5: {ord_5_num_unique}')



sample = list(df_train['ord_5'].sample(10))

print(f'ex of ord_5 values: {sample}')



str_lengths = df_train['ord_5'].str.len().nunique()

print(f'different string lengths in ord_5: {str_lengths}')
fig, ax = plt.subplots(figsize=(16,6))



ordinal_ordering = {}





fig.suptitle('Distribution Target Variable ratio \n (lowest → highest ratio)')



# calculate the ratio of target counts

ct = pd.crosstab(df_train['ord_5'], df_train['target']).apply(lambda r: r/r.sum(), axis = 1)

stacked = ct.stack().reset_index().rename(columns = {0: 'ratio'})

stacked = stacked.sort_values(['target', 'ratio'], ascending = [False, True])



ordinal_ordering['ord_5'] = stacked['ord_5'].unique() # for encoding



sns.barplot(x = stacked['ord_5'], y = stacked['ratio'], hue = stacked['target'])



# show less x-ticks

every_nth = 10

for n, label in enumerate(ax.xaxis.get_ticklabels()):

    if n % every_nth != 0:

        label.set_visible(False)
nbr_to_replace = len(ordinal_ordering['ord_5'])



df_train['ord_5'].replace(to_replace = ordinal_ordering['ord_5'], 

                          value = [x for x in range(0, len(ordinal_ordering['ord_5']))],

                          inplace = True)

df_test['ord_5'].replace(to_replace = ordinal_ordering['ord_5'], 

                          value = [x for x in range(0, len(ordinal_ordering['ord_5']))],

                          inplace = True)

    

df_train['ord_5'].head()
# one hot encode low-cardinality nominal variables

df_train = pd.get_dummies(df_train, columns = lc_nom_columns)

df_test = pd.get_dummies(df_test, columns = lc_nom_columns)

df_train.filter(regex='nom_[0-4]_').head()
def freq_encoding(df, cols):

    for col in cols:

        # get variable frequencies

        frequencies = (df.groupby(col).size()) / len(df) 

        # encode frequencies

        df[f'{col}_freq'] = df[col].apply(lambda x : frequencies[x]) 

    return df
df_train = freq_encoding(df_train, hc_nom_columns)

df_test = freq_encoding(df_test, hc_nom_columns)

df_train.filter(regex='nom_[5-9]_freq').head()
def feature_hashing(df, cols):

    for col in cols:

        df[f'{col}_hashed'] = df[col].apply(lambda x: hash(str(x)) % 5000)

    return df
df_train = feature_hashing(df_train, hc_nom_columns)

df_test = feature_hashing(df_test, hc_nom_columns)

df_train.filter(regex='nom_[5-9]_hashed').head()
def encode_target_smooth(data, target, categ_variables, smooth):

    """    

    Apply target encoding with smoothing.

    

    Parameters

    ----------

    data: pd.DataFrame

    target: str, dependent variable

    categ_variables: list of str, variables to encode

    smooth: int, number of observations to weigh global average with

    

    Returns

    --------

    encoded_dataset: pd.DataFrame

    code_map: dict, mapping to be used on validation/test datasets 

    defaul_map: dict, mapping to replace previously unseen values with

    """

    train_target = data.copy()

    code_map = dict()    # stores mapping between original and encoded values

    default_map = dict() # stores global average of each variable

    

    for col in categ_variables:

        prior = data[target].mean()

        n = data.groupby(col).size()

        mu = data.groupby(col)[target].mean()

        mu_smoothed = (n * mu + smooth + prior) / (n + smooth)

        

        train_target.loc[:, col] = train_target[col].map(mu_smoothed)

        code_map[col] = mu_smoothed

        default_map[col] = prior

    return train_target, code_map, default_map
# additive smoothing

train_target_smooth, target_map, default_map = encode_target_smooth(df_train, 'target', hc_nom_columns, 500)

test_target_smooth = df_train.copy()

for col in hc_nom_columns:

    encoded_col = test_target_smooth[col].map(target_map[col])

    mean_encoded = pd.DataFrame({f'{col}_mean_enc': encoded_col})

    df_train = pd.concat([df_train, mean_encoded], axis=1)

    

df_train.filter(regex='nom_[5-9]_mean_enc').head()
def impact_coding_leak(data, feature, target, n_folds=20, n_inner_folds=10):

    from sklearn.model_selection import StratifiedKFold

    '''

    ! Using oof_default_mean for encoding inner folds introduces leak.

    

    Source: https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features

    

    Changelog:    

    a) Replaced KFold with StratifiedFold due to class imbalance

    b) Rewrote .apply() with .map() for readability

    c) Removed redundant apply in the inner loop

    '''

    impact_coded = pd.Series()

    

    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True) # KFold in the original

    oof_mean_cv = pd.DataFrame()

    split = 0

    for infold, oof in kf.split(data[feature], data[target]):



        kf_inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True)

        inner_split = 0

        inner_oof_mean_cv = pd.DataFrame()

        oof_default_inner_mean = data.iloc[infold][target].mean()

        

        for infold_inner, oof_inner in kf_inner.split(data.iloc[infold], data.loc[infold, target]):

            # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)

            oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()



            # Also populate mapping (this has all group -> mean for all inner CV folds)

            inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')

            inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)

            inner_split += 1



        # compute mean for each value of categorical value across oof iterations

        inner_oof_mean_cv_map = inner_oof_mean_cv.mean(axis=1)



        # Also populate mapping

        oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')

        oof_mean_cv.fillna(value=oof_default_mean, inplace=True)

        split += 1



        feature_mean = data.loc[oof, feature].map(inner_oof_mean_cv_map).fillna(oof_default_mean)

        impact_coded = impact_coded.append(feature_mean)

            

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean



def impact_coding(data, feature, target, n_folds=20, n_inner_folds=10):

    from sklearn.model_selection import StratifiedKFold

    '''

    ! Using oof_default_mean for encoding inner folds introduces leak.

    

    Source: https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features

    

    Changelog:    

    a) Replaced KFold with StratifiedFold due to class imbalance

    b) Rewrote .apply() with .map() for readability

    c) Removed redundant apply in the inner loop

    d) Removed global average; use local mean to fill NaN values in out-of-fold set

    '''

    impact_coded = pd.Series()

        

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True) # KFold in the original

    oof_mean_cv = pd.DataFrame()

    split = 0

    for infold, oof in kf.split(data[feature], data[target]):



        kf_inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True)

        inner_split = 0

        inner_oof_mean_cv = pd.DataFrame()

        oof_default_inner_mean = data.iloc[infold][target].mean()

        

        for infold_inner, oof_inner in kf_inner.split(data.iloc[infold], data.loc[infold, target]):

                    

            # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)

            oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()

            

            # Also populate mapping (this has all group -> mean for all inner CV folds)

            inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')

            inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)

            inner_split += 1



        # compute mean for each value of categorical value across oof iterations

        inner_oof_mean_cv_map = inner_oof_mean_cv.mean(axis=1)



        # Also populate mapping

        oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')

        oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True) # <- local mean as default

        split += 1



        feature_mean = data.loc[oof, feature].map(inner_oof_mean_cv_map).fillna(oof_default_inner_mean)

        impact_coded = impact_coded.append(feature_mean)

    

    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean



def encode_target_cv(data, target, categ_variables, impact_coder=impact_coding):

    """Apply original function for each <categ_variables> in  <data>

    Reduced number of validation folds

    """

    train_target = data.copy() 

    

    code_map = dict()

    default_map = dict()

    for f in categ_variables:

        print(f'cv impact encoding {f}')

        train_target.loc[:, f], code_map[f], default_map[f] = impact_coder(train_target, f, target)

        

    return train_target, code_map, default_map

train_target_cv, code_map, default_map = encode_target_cv(df_train[hc_nom_columns+['target']], 

                                                          'target', hc_nom_columns, 

                                                          impact_coder=impact_coding)



train_target_cv = train_target_cv.drop('target', axis=1)
for col in train_target_cv.columns:

    train_target_cv = train_target_cv.rename(columns={col: f'{col}_cvmean_enc'})

train_target_cv.head()
df_train = pd.concat([df_train, train_target_cv], axis=1)
day_values = sorted(df_train['day'].unique().tolist())

print(f'day values: {day_values}')

plt.plot(day_values)
def sin_cos_encode(df, cols):

    for col in cols:

        col_max_val = max(df[col])

        df[f'{col}_sin'] = np.sin(2*np.pi * df[col]/ col_max_val) # sin transform

        df[f'{col}_cos'] = np.cos(2*np.pi * df[col]/ col_max_val) # cos transform

    return df
df_train = sin_cos_encode(df_train, cyc_columns)

df_train.filter(regex='_(sin|cos)').head()
sample = df_train[['month_sin', 'month_cos']].sample(100)

sample.plot.scatter('month_sin', 'month_cos').set_aspect('equal')
# drop hexadecimal nominal columns

X_train = df_train.drop(columns=['target', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1)

y_train = df_train['target']
from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=31429)
import xgboost as xgb



# set parameters for xgboost

params = {

    'objective': 'binary:logistic',

    'eval_metric': 'auc',

    'eta': 0.02,

    'max_depth': 4

}



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watch_list = [(d_train, 'train'), (d_valid, 'valid')]



model = xgb.train(params, d_train, 400, watch_list, early_stopping_rounds=50, verbose_eval=25)
#model.get_score(importance_type='gain')

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()