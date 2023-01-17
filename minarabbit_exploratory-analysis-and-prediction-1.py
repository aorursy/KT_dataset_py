import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression



pd.set_option('display.max_columns', 100)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
trainset = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')

testset = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
trainset.head()
print('Train dataset (rows, cols):', trainset.shape, '\nTest dataset (rows, cols):', testset.shape)
print('Columns in train and not in test dataset:', set(trainset.columns)-set(testset.columns))
# uses code from https://www.kaggle.com/bertcarremans/data-preparation-exploration (see references)



data=[]

for feature in trainset.columns:

    # Defining the role

    if feature == 'target':

        use = 'target'

    elif feature == 'id':

        use = 'id'

    else:

        use = 'input'

        

    # Defining the type

    if 'bin' in feature or feature == 'target':

        type = 'binary'

    elif 'cat' in feature or feature == 'id':

        type = 'categorical'

    elif trainset[feature].dtype == float or isinstance(trainset[feature].dtype, float):

        type = 'real'

    elif trainset[feature].dtype == int:

        type = 'integer'

        

    # Initialize preserve to True for all variables except for id

    preserve = True

    if feature == 'id':

        preserve = False

        

    # Defining the data type

    dtype = trainset[feature].dtype

    

    category='none'

    # Defining the category

    if 'ind' in feature:

        category='individual'

    elif 'reg' in feature:

        category='registration'

    elif 'car' in feature:

        category='car'

    elif 'calc' in feature:

        category = 'calculated'

        

    # Creating a Dict that contains all the metadata for the variable

    feature_dictionary = {

        'varname': feature,

        'use':use,

        'type':type,

        'preserve':preserve,

        'dtype':dtype,

        'category':category

    }

    data.append(feature_dictionary)

    

metadata = pd.DataFrame(data, columns=['varname', 'use', 'type', 'preserve', 'dtype', 'category'])

metadata.set_index('varname', inplace=True)

metadata
metadata[(metadata.type=='categorical') & (metadata.preserve)].index
pd.DataFrame({'count':metadata.groupby(['category'])['category'].size()}).reset_index()
pd.DataFrame({'count':metadata.groupby(['use', 'type'])['use'].size()}).reset_index()
plt.figure()

fig, ax = plt.subplots(figsize=(6,6))

x = trainset['target'].value_counts().index.values

y = trainset['target'].value_counts().values



# Bar plot

# Order the bars descending on target mean

sns.barplot(ax=ax, x=x, y=y)

plt.ylabel('Number of values', fontsize=12)

plt.xlabel('Target value', fontsize=12)

plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
variable =  metadata[(metadata.type=='real') & (metadata.preserve)].index

trainset[variable].describe()
(pow(trainset['ps_car_12']*10, 2)).head(10)
(pow(trainset['ps_car_15'], 2)).head(10)
sample = trainset.sample(frac=0.05)

var = ['ps_car_12', 'ps_car_15', 'target']

sample = sample[var]

sns.pairplot(sample, hue='target', palette='Set1', diag_kind='kde')

plt.show()
var = metadata[(metadata.type=='real') & (metadata.preserve)].index

i=0

t1=trainset.loc[trainset['target'] != 0]

t0=trainset.loc[trainset['target'] == 0]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(3,4,figsize=(16,12))



for feature in var:

    i += 1

    plt.subplot(3,4,i)

    sns.kdeplot(t1[feature], bw=0.5, label='target=1')

    sns.kdeplot(t0[feature], bw=0.5, label='target=0')

    plt.ylabel('Density plot', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()

def corr_heatmap(var):

    correlations = trainset[var].corr()

    

    # Create color map ranging between two colors

    cmap = sns.diverging_palette(50,10, as_cmap=True)

    

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=.5, annot=True, cbar_kws={'shrink':.75})

    plt.show()

    

var = metadata[(metadata.type=='real')&(metadata.preserve)].index

corr_heatmap(var)
sample = trainset.sample(frac=0.05)

var = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_15', 'target']

sample = sample[var]

sns.pairplot(sample, hue='target', palette='Set1', diag_kind='kde')

plt.show()
v = metadata[(metadata.type=='binary') & (metadata.preserve)].index

trainset[v].describe()
bin_col = [col for col in trainset.columns if '_bin' in col]

zero_list=[]

one_list=[]

for col in bin_col:

    zero_list.append((trainset[col]==0).sum()/trainset.shape[0]*100)

    one_list.append((trainset[col]==1).sum()/trainset.shape[0]*100)

plt.figure()

fig, ax = plt.subplots(figsize=(6,6))

# Bar plot

p1 = sns.barplot(ax=ax, x=bin_col, y=zero_list, color='blue')

p2 = sns.barplot(ax=ax, x=bin_col, y=one_list, bottom=zero_list, color='red')

plt.ylabel('Percent of zero/one [%]', fontsize=12)

plt.xlabel('Binary features', fontsize=12)

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

plt.tick_params(axis='both', which='major', labelsize=12)

plt.legend((p1, p2), ('Zero', 'One'))

plt.show()
var = metadata[(metadata.type=='binary') & (metadata.preserve)].index

i=0

var = [col for col in trainset.columns if '_bin' in col]

t1=trainset.loc[trainset['target'] != 0]

t0=trainset.loc[trainset['target'] == 0]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(6,3,figsize=(12,24))



for feature in var:

    i += 1

    plt.subplot(6,3,i)

    sns.kdeplot(t1[feature], bw=0.5, label='target=1')

    sns.kdeplot(t0[feature], bw=0.5, label='target=0')

    plt.ylabel('Density plot', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
var = metadata[(metadata.type=='categorical') & (metadata.preserve)].index



for feature in var:

    fig, ax = plt.subplots(figsize=(6,6))

    # Calculate the percentage of target=1 per category value

    cat_perc = trainset[[feature, 'target']].groupby([feature], as_index=False).mean()

    cat_perc.sort_values(by='target', ascending=False, inplace=True)

    # Bar plot

    # Order the bars descending on target mean

    sns.barplot(ax=ax, x=feature, y='target', data=cat_perc, order=cat_perc[feature])

    plt.ylabel('Percent of target with value 1 [%]', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.show()
var = metadata[(metadata.type=='categorical') & (metadata.preserve)].index

i=0

t1=trainset.loc[trainset['target'] != 0]

t0=trainset.loc[trainset['target'] == 0]



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(4,4,figsize=(16,16))



for feature in var:

    i += 1

    plt.subplot(4,4,i)

    sns.kdeplot(t1[feature], bw=0.5, label='target=1')

    sns.kdeplot(t0[feature], bw=0.5, label='target=0')

    plt.ylabel('Density plot', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
var = metadata[(metadata.category == 'registration') & (metadata.preserve)].index



# Bar Plot

sns.set_style('whitegrid')



plt.figure()

fig, ax = plt.subplots(1,3,figsize=(12,4))

i = 0

for feature in var:

    i = i + 1

    plt.subplot(1,3,i)

    sns.kdeplot(trainset[feature], bw=0.5, label='train')

    sns.kdeplot(testset[feature], bw=0.5, label='test')

    plt.ylabel('Distribution', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    # plt.setp(labels, rotation=90)

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
var = metadata[(metadata.category == 'car') & (metadata.preserve)].index



# Bar plot

sns.set_style('whitegrid')



plt.figure()

fig, ax = plt.subplots(4,4,figsize=(20,16))

i = 0

for feature in var:

    i = i+1

    plt.subplot(4,4,i)

    sns.kdeplot(trainset[feature], bw=0.5, label='train')

    sns.kdeplot(testset[feature], bw=0.5, label='test')

    plt.ylabel('Distribution', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    # plt.setp(labels, rotation=90)

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show()
var = metadata[(metadata.category == 'individual') & (metadata.preserve)].index



# Bar plot

sns.set_style('whitegrid')



plt.figure()

fig, ax = plt.subplots(5,4,figsize=(20,16))

i = 0

for feature in var:

    i = i + 1

    plt.subplot(5,4,i)

    sns.kdeplot(trainset[feature], bw=0.5, label="train")

    sns.kdeplot(testset[feature], bw=0.5, label="test")

    plt.ylabel('Distribution', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    #plt.setp(labels, rotation=90)

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();
var = metadata[(metadata.category == 'calculated') & (metadata.preserve)].index



# Bar plot

sns.set_style('whitegrid')



plt.figure()

fig, ax = plt.subplots(5,4,figsize=(20,16))

i = 0

for feature in var:

    i = i + 1

    plt.subplot(5,4,i)

    sns.kdeplot(trainset[feature], bw=0.5, label="train")

    sns.kdeplot(testset[feature], bw=0.5, label="test")

    plt.ylabel('Distribution', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    #plt.setp(labels, rotation=90)

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();
vars_with_missing = []



for feature in trainset.columns:

    missings = trainset[trainset[feature] == -1][feature].count()

    if missings >0 :

        vars_with_missing.append(feature)

        missings_perc = missings / trainset.shape[0]

        

        print('Variable {} has {} records ({:.2%}) with missing values.'.format(feature, missings, missings_perc))

print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
col_to_drop = trainset.columns[trainset.columns.str.startswith('ps_calc_')]

trainset = trainset.drop(col_to_drop, axis=1)

testset = testset.drop(col_to_drop, axis=1)
# Dropping the variables with too many missing values

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

trainset.drop(vars_to_drop, inplace=True, axis=1)

testset.drop(vars_to_drop, inplace=True, axis=1)

metadata.loc[(vars_to_drop), 'keep'] = False # Updating the meta
# Script by https://www.kaggle.com/ogrellier

# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features



def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(trn_series=None, 

                  tst_series=None, 

                  target=None, 

                  min_samples_leaf=1, 

                  smoothing=1,

                  noise_level=0):

    """

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior  

    """ 

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

    
train_encoded, test_encoded = target_encode(trainset['ps_car_11_cat'],

                                           testset['ps_car_11_cat'],

                                           target=trainset.target,

                                           min_samples_leaf=100,

                                           smoothing=10,

                                           noise_level=0.01)



trainset['ps_car_11_cat_te'] = train_encoded

trainset.drop('ps_car_11_cat', axis=1, inplace=True)

metadata.loc['ps_car_11_cat', 'keep'] = False # Updating the metadata

testset['ps_car_11_cat_te'] = test_encoded

testset.drop('ps_car_11_cat', axis=1, inplace=True)
desired_apriori = 0.10



# Get the indices per target value

idx_0 = trainset[trainset.target == 0].index

idx_1 = trainset[trainset.target == 1].index



# Get original number of records per target value

nb_0 = len(trainset.loc[idx_0])

nb_1 = len(trainset.loc[idx_1])



# Calculate the undersampling rate and resulting number of records with target=0

undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)

undersampled_nb_0 = int(undersampling_rate*nb_0)

print('Rate to undersample records with target=0: {}'.format(undersampling_rate))

print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))



# Randomly select records with target=0 to get at the desired a priori

undersampled_idx = shuffle(idx_0, random_state=314, n_samples=undersampled_nb_0)



# Construct list with remaining indices

idx_list = list(undersampled_idx) + list(idx_1)



# Return undersample data fram

trainset=trainset.loc[idx_list].reset_index(drop=True)
trainset = trainset.replace(-1, np.nan)

testset = testset.replace(-1, np.nan)
cat_features = [ a for a in trainset.columns if a.endswith('cat')]



for column in cat_features:

    temp = pd.get_dummies(pd.Series(trainset[column]))

    trainset = pd.concat([trainset,temp], axis=1)

    trainset = trainset.drop([column], axis=1)

    

for column in cat_features:

    temp = pd.get_dummies(pd.Series(testset[column]))

    testset = pd.concat([testset, temp], axis=1)

    testset = testset.drop([column], axis=1)
id_test = testset['id'].values

target_train = trainset['target'].values



trainset = trainset.drop(['target', 'id'], axis=1)

testset = testset.drop(['id'], axis=1)
print('Train dataset (rows, cols):', trainset.values.shape, '\nTest dataset (rows, cols):', testset.values.shape)
class Ensemble (object):

    def __init__(self, n_splits, stacker, base_models):

        self.n_splits = n_splits

        self.stacker = stacker

        self.base_models = base_models

        

    def fit_predict(self, X, y, T):

        X = np.array(X)

        y = np.array(y)

        T = np.array(T)

        

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=314).split(X, y))

        

        S_train = np.zeros((X.shape[0], len(self.base_models)))

        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            

            for j, (train_idx, test_idx) in enumerate(folds):

                X_train = X[train_idx]

                y_train = y[train_idx]

                X_holdout = X[test_idx]

                

                print('Base model %d: fit %s model | fold %d' % (i+1, str(clf).split('(')[0], j+1))

                clf.fit(X_train, y_train)

                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')

                print('cross_score [roc-auc]: %.5f [gini]: %.5f' % (cross_score.mean(), 2*cross_score.mean()-1))

                y_pred = clf.predict_proba(X_holdout)[:,1]

                

                S_train[test_idx, i] = y_pred

                S_test_i[:, j] = clf.predict_proba(T)[:, 1]

                

            S_test[:,i] = S_test_i.mean(axis=1)

            

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')

        # Calculate gini factor as 2 * AUC -1

        print("Stacker score [gini]: %.5f" % (2*results.mean()-1))

        

        self.stacker.fit(S_train, y)

        res = self.stacker.predict_proba(S_test)[:, 1]

        return res
# LightGBM params

# lgb_1

lgb_params1 = {}

lgb_params1['learning_rate'] = 0.02

lgb_params1['n_estimators'] = 650

lgb_params1['max_bin'] = 10

lgb_params1['subsample'] = 0.8

lgb_params1['subsmaple_freq'] = 10

lgb_params1['colsample_bytree'] = 0.8

lgb_params1['min_child_samples'] = 500

lgb_params1['seed'] = 314

lgb_params1['num_threads'] = 4



# lgb_2

lgb_params2 = {}

lgb_params2['n_estimators'] = 1090

lgb_params2['learning_rate'] = 0.02

lgb_params2['colsample_bytree'] = 0.3

lgb_params2['subsample'] = 0.7

lgb_params2['subsmaple_freq'] = 2

lgb_params2['num_leaves'] = 16

lgb_params2['seed'] = 314

lgb_params2['num_threads'] = 4



# lgb_3

lgb_params3 = {}

lgb_params2['n_estimators'] = 1100

lgb_params2['max_depth'] = 4

lgb_params2['learning_rate'] = 0.02

lgb_params2['seed'] = 314

lgb_params2['num_threads'] = 4



# XGBoost params

xgb_params = {}

xgb_params['objective'] = 'binary:logistic'

xgb_params['learning_rate'] = 0.04

xgb_params['n_estimators'] = 490

xgb_params['max_depth'] = 4

xgb_params['subsample'] = 0.9

xgb_params['colsample_bytree'] = 0.9

xgb_params['min_child_samples'] = 10

xgb_params['num_threads'] = 4
# Base models

lgb_model1=LGBMClassifier(**lgb_params1)

lgb_model2=LGBMClassifier(**lgb_params2)

lgb_model3=LGBMClassifier(**lgb_params3)

xgb_model = XGBClassifier(**xgb_params)



# Stacking model

log_model = LogisticRegression()
stack = Ensemble(n_splits=3, stacker=log_model, base_models = (lgb_model1, lgb_model2, lgb_model3, xgb_model))
y_prediction = stack.fit_predict(trainset, target_train, testset)
submission = pd.DataFrame()

submission['id'] = id_test

submission['target'] = y_prediction

submission.to_csv('stacked.csv', index=False)