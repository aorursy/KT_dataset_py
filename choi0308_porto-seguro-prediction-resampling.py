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
import pandas as pd

sample_submission = pd.read_csv("../input/porto-seguro-safe-driver-prediction/sample_submission.csv")

test = pd.read_csv("../input/porto-seguro-safe-driver-prediction/test.csv")

train = pd.read_csv("../input/porto-seguro-safe-driver-prediction/train.csv")
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel

from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
train.head()
train.info()
train.drop_duplicates()

print(train.shape)

print(test.shape)

data = []

for col in train.columns:

    if col == 'target':

        role = 'target'

    elif col == 'id':

        role = 'id'

    else:

        role = 'input'

        

    if 'bin' in col or col == 'target':

        level = 'binary'

    elif 'cat' in col or col == 'id':

        level ='nominal'

    elif train[col].dtype == 'float':

        level = 'interval'

    elif train[col].dtype == 'int':

        level = 'ordinal'

        

    keep = True

    if col == 'id':

        keep = False

        

    dtype = train[col].dtype

        

    col_dict = {

        'varnames' : col,

        'role' : role,

        'level' : level,

        'keep' : keep,

        'dtype' : dtype

    }

    data.append(col_dict)

    

data_dic = pd.DataFrame(data, columns = ['varnames', 'role', 'level', 'keep', 'dtype'])

data_dic.set_index('varnames', inplace=True)
data_dic
data_dic[(data_dic.level == 'nominal') & (data_dic.keep)].index
pd.DataFrame({'count' : data_dic.groupby(['role', 'level'])['role'].size()}).reset_index()
var = data_dic[(data_dic.level == 'interval') & (data_dic.keep)].index

train[var].describe()
var = data_dic[(data_dic.level == 'ordinal') & (data_dic.keep)].index

train[var].describe()
var = data_dic[(data_dic.level == 'binary') & (data_dic.keep)].index

train[var].describe()
apriori = 0.10



idx_0 = train[train.target==0].index

idx_1 = train[train.target==1].index

print(idx_0)

print(idx_1)





nb_0 = len(train.loc[idx_0])

nb_1 = len(train.loc[idx_1])

print(nb_0)

print(nb_1)



undersampling_rate = ((1-apriori) * nb_1) / (nb_0 * apriori)

undersampling_nb_0 = int(undersampling_rate*nb_0)



print('Rate to undersample records with target=0 : {}'.format(undersampling_rate))

print('Number of records with target=0 after undersampling: {}'.format(undersampling_nb_0))



# Randomly select records with target

undersampled_idx = shuffle(idx_0, random_state=42, n_samples=undersampling_nb_0)

# print(undersampled_idx)



idx_list = list(undersampled_idx) + list(idx_1)

# print(idx_list)



#return undersample to dataframe

train = train.loc[idx_list].reset_index(drop=True)
vars_with_missing = []



for col in train.columns:

    missings = train[train[col] == -1][col].count()

    if missings > 0:

        vars_with_missing.append(col)

        missing_per = missings / train.shape[0]

        

        print('Variable {} has {} records ({:.2f}) with missing values'.format(col, missings, missing_per))

        

print('Total there are {} variables with missing values'.format(len(vars_with_missing)))
print('ps_car_03 mean : {}'.format(train['ps_car_03_cat'].mean()))

print('ps_car_03 median : {}'.format(train['ps_car_03_cat'].median()))

print('ps_car_03 mode : {}'.format(train['ps_car_03_cat'].mode()))



print('ps_car_03 mean : {}'.format(train['ps_car_05_cat'].mean()))

print('ps_car_03 median : {}'.format(train['ps_car_05_cat'].median()))

print('ps_car_03 mode : {}'.format(train['ps_car_05_cat'].mode()))



print('ps_car_11 mean : {}'.format(train['ps_car_11'].mean()))

print('ps_car_11 median : {}'.format(train['ps_car_11'].median()))

print('ps_car_11 mode : {}'.format(train['ps_car_11'].mode()))

## Imputing



mean_imp = SimpleImputer(missing_values=-1, strategy='mean')

mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')



train['ps_car_03_cat'] = mode_imp.fit_transform(train[['ps_car_03_cat']]).ravel()

train['ps_car_05_cat'] = mode_imp.fit_transform(train[['ps_car_05_cat']]).ravel()

train['ps_car_07_cat'] = mode_imp.fit_transform(train[['ps_car_07_cat']]).ravel()

train['ps_car_09_cat'] = mode_imp.fit_transform(train[['ps_car_09_cat']]).ravel()

train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()

train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()

var = data_dic[(data_dic.level == 'nominal') & (data_dic.keep)].index



for col in var:

    dist_values = train[col].value_counts().shape[0]

    print('Variable {} has {} distinct values'.format(col, dist_values))
def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(trn_series=None, 

                  tst_series=None, 

                  target=None, 

                  min_samples_leaf=1, 

                  smoothing=1,

                  noise_level=0):



    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data

    prior = target.mean()

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_trn_series.index = trn_series.index 

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
train_encoded, test_encoded = target_encode(train['ps_car_11_cat'],

                                           test['ps_car_11_cat'],

                                           target=train.target,

                                           min_samples_leaf=100,

                                           smoothing=10,

                                           noise_level=0.01)



train['ps_car_11_cat_te'] = train_encoded

train.drop('ps_car_11_cat', axis=1, inplace=True)

test['ps_car_11_cat_te'] = train_encoded

test.drop('ps_car_11_cat', axis=1, inplace=True)
var = data_dic[(data_dic.level == 'nominal') & (data_dic.keep)].index



for f in var:

    plt.figure()

    fig, ax = plt.subplots(figsize=(20,10))

    # Calculate the percentage of target=1 per category value

    cat_perc = train[[f, 'target']].groupby([f],as_index=False).mean()

    cat_perc.sort_values(by='target', ascending=False, inplace=True)

    # Bar plot

    # Order the bars descending on target mean

    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])

    plt.ylabel('% target', fontsize=18)

    plt.xlabel(f, fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show();
# corr map



def corr_heatmap(var):

    correlations = train[var].corr()

    

    # create color map

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    

    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0,

               fmt='.2f', square=True, linewidth=.5, annot=True, cbar_kws={'shrink':.75}

               )

    plt.show()

    

var = data_dic[(data_dic.level=='interval') & (data_dic.keep)].index

corr_heatmap(var)

    
s = train.sample(frac=0.1)
#ps_reg_02 and ps_reg_03

sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
#ps_car_12 and ps_car_13

sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
#ps_car_12 and ps_car_14

sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
#ps_car_13 and ps_car_15

sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
var = data_dic[(data_dic.level=='ordinal') & (data_dic.keep)].index

corr_heatmap(var)
data_dic.drop('ps_car_11_cat', axis=0, inplace=True)
var = data_dic[(data_dic.level=='nominal') & (data_dic.keep)].index



print('before get dummies ', train.shape)



train = pd.get_dummies(train, columns=var, drop_first=True)

print('after get dummies ', train.shape)
var = data_dic[(data_dic.level=='interval') & (data_dic.keep)].index



poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

interactions = pd.DataFrame(data=poly.fit_transform(train[var]), columns=poly.get_feature_names(var))

interactions.drop(var, axis=1, inplace=True)



print('before creating interaction variables', train.shape[1])

train = pd.concat([train, interactions], axis=1)

print('after creating interaction variables', train.shape[1])

selector = VarianceThreshold(threshold=0.01)



selector.fit(train.drop(['id', 'target'], axis=1))



f = np.vectorize(lambda x : not x)



v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]



print('{} variables have too low variance '.format(len(v)))

print('these vars are  {} '.format(list(v)))
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']



feat_labels = X_train.columns



rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)



rf.fit(X_train, y_train)

importances = rf.feature_importances_



indices = np.argsort(rf.feature_importances_)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f " % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
sfm = SelectFromModel(rf, threshold='median')