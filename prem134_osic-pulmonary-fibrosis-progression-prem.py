import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns
# Load Data

df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

df_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
df_train.head()
# Training Set FVC Measurements Per Patient

training_sample_counts = df_train.rename(columns={'Weeks': 'Samples'}).groupby('Patient').agg('count')['Samples']

#print(f'Training Set FVC Measurements Per Patient \n{("-") * 41}\n{training_sample_counts}')
df_submission = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )

df_submission.head()
print(f'FVC Statistical Summary\n{"-" * 23}')



print(f'Mean: {df_train["FVC"].mean():.6}  -  Median: {df_train["FVC"].median():.6}  -  Std: {df_train["FVC"].std():.6}')

print(f'Min: {df_train["FVC"].min()}  -  25%: {df_train["FVC"].quantile(0.25)}  -  50%: {df_train["FVC"].quantile(0.5)}  -  75%: {df_train["FVC"].quantile(0.75)}  -  Max: {df_train["FVC"].max()}')

print(f'Skew: {df_train["FVC"].skew():.6}  -  Kurtosis: {df_train["FVC"].kurtosis():.6}')

missing_values_count = df_train[df_train["FVC"].isnull()].shape[0]

training_samples_count = df_train.shape[0]

print(f'Missing Values: {missing_values_count}/{training_samples_count} ({missing_values_count * 100 / training_samples_count:.4}%)')

fig, axes = plt.subplots(ncols=2, figsize=(18, 6), dpi=150)

sns.distplot(df_train['FVC'], label='FVC', ax=axes[0])

stats.probplot(df_train['FVC'], plot=axes[1])

axes[0].set_title(f'FVC Distribution in Training Set', size=15, pad=15)

axes[1].set_title(f'FVC Probability Plot', size=15, pad=15)
g = sns.pairplot(df_train[['FVC', 'Weeks', 'Percent', 'Age', 'Sex']], aspect=1.4, hue='Sex', height=5, diag_kind='kde', kind='reg')



g.axes[3, 0].set_xlabel('FVC', fontsize=20)

g.axes[3, 1].set_xlabel('Weeks', fontsize=20)

g.axes[3, 2].set_xlabel('Percent', fontsize=20)

g.axes[3, 3].set_xlabel('Age', fontsize=20)

g.axes[0, 0].set_ylabel('FVC', fontsize=20)

g.axes[1, 0].set_ylabel('Weeks', fontsize=20)

g.axes[2, 0].set_ylabel('Percent', fontsize=20)

g.axes[3, 0].set_ylabel('Age', fontsize=20)



g.axes[3, 0].tick_params(axis='x', labelsize=15)

g.axes[3, 1].tick_params(axis='x', labelsize=15)

g.axes[3, 2].tick_params(axis='x', labelsize=15)

g.axes[3, 3].tick_params(axis='x', labelsize=15)

g.axes[0, 0].tick_params(axis='y', labelsize=15)

g.axes[1, 0].tick_params(axis='y', labelsize=15)

g.axes[2, 0].tick_params(axis='y', labelsize=15)

g.axes[3, 0].tick_params(axis='y', labelsize=15)



g.fig.suptitle('Tabular Data Feature Distributions and Interactions', fontsize=25, y=1.08)



plt.show()
# As seen from the plots above, the only strong correlation is between FVC and Percent. The other features' correlations are between -0.1 and 0.1.

fig = plt.figure(figsize=(10, 10), dpi=100)



sns.heatmap(df_train.corr(), annot=True, square=True, cmap='coolwarm', annot_kws={'size': 15},  fmt='.2f')   



plt.tick_params(axis='x', labelsize=18, rotation=75)

plt.tick_params(axis='y', labelsize=18, rotation=0)

plt.title('Tabular Data Feature Correlations', size=20, pad=20)



plt.show()
import os

import numpy as np

import pandas as pd

import random

import math



from tqdm.notebook import tqdm



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.metrics import mean_squared_error

import category_encoders as ce



from sklearn.linear_model import Ridge, ElasticNet

from functools import partial

import scipy as sp



import warnings

warnings.filterwarnings("ignore")
def seed_everything(seed=777):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
OUTPUT_DICT = './'



ID = 'Patient_Week'

TARGET = 'FVC'

SEED = 777

seed_everything(seed=SEED)



N_FOLD = 7
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
otest
# construct train input

train = pd.concat([train,otest])

output = pd.DataFrame()

gb = train.groupby('Patient')

tk0 = tqdm(gb, total=len(gb))

for _, usr_df in tk0:

    usr_output = pd.DataFrame()

    for week, tmp in usr_df.groupby('Weeks'):

        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Age': 'base_Age'}

        tmp = tmp.rename(columns=rename_cols)

        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent']

        _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')

        _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']

        usr_output = pd.concat([usr_output, _usr_output])

        print(usr_output)

    output = pd.concat([output, usr_output])

    

train = output[output['Week_passed']!=0].reset_index(drop=True)
train
# construct test input

test = otest.rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Age': 'base_Age'})

submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])

submission['predict_Week'] = submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

test = submission.drop(columns=['FVC', 'Confidence']).merge(test, on='Patient')

test['Week_passed'] = test['predict_Week'] - test['base_Week']

test.set_index('Patient_Week', inplace=True)
OUTPUT_DICT = './'



ID = 'Patient_Week'

TARGET = 'FVC'

SEED = 777

seed_everything(seed=SEED)



N_FOLD = 7
folds = train[['Patient', TARGET]].copy()

Fold = GroupKFold(n_splits=N_FOLD)

groups = folds['Patient'].values

for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):

    folds.loc[val_index, 'fold'] = int(n)

folds['fold'] = folds['fold'].astype(int)
#===========================================================

# model

#===========================================================

def run_single_model(clf, train_df, test_df, folds, features, target, fold_num=0):

    

    trn_idx = folds[folds.fold!=fold_num].index

    val_idx = folds[folds.fold==fold_num].index

    

    y_tr = target.iloc[trn_idx].values

    X_tr = train_df.iloc[trn_idx][features].values

    y_val = target.iloc[val_idx].values

    X_val = train_df.iloc[val_idx][features].values

    

    oof = np.zeros(len(train_df))

    predictions = np.zeros(len(test_df))

    clf.fit(X_tr, y_tr)

    

    oof[val_idx] = clf.predict(X_val)

    predictions += clf.predict(test_df[features])

    return oof, predictions





def run_kfold_model(clf, train, test, folds, features, target, n_fold=7):

    

    oof = np.zeros(len(train))

    predictions = np.zeros(len(test))

    feature_importance_df = pd.DataFrame()



    for fold_ in range(n_fold):



        _oof, _predictions = run_single_model(clf,

                                              train, 

                                              test,

                                              folds,  

                                              features,

                                              target, 

                                              fold_num=fold_)

        oof += _oof

        predictions += _predictions/n_fold

    return oof, predictions
#Predict Dataset
target = train[TARGET]

test[TARGET] = np.nan



# features

cat_features = ['Sex', 'SmokingStatus']

num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]

features = num_features + cat_features

drop_features = [TARGET, 'predict_Week', 'Percent', 'base_Week']

features = [c for c in features if c not in drop_features]



if cat_features:

    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')

    ce_oe.fit(train)

    train = ce_oe.transform(train)

    test = ce_oe.transform(test)
scorelist = []

for alpha1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

    for l1s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

        

        print(" For alpha:",alpha1,"& l1_ratio:",l1s)

        clf = ElasticNet(alpha=alpha1, l1_ratio = l1s)

        oof, predictions = run_kfold_model(clf, train, test, folds, features, target, n_fold=N_FOLD)



        train['FVC_pred'] = oof

        test['FVC_pred'] = predictions



        # baseline score

        train['Confidence'] = 100

        train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

        train['diff'] = abs(train['FVC'] - train['FVC_pred'])

        train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

        train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])

        score = train['score'].mean()

        print(score)



        def loss_func(weight, row):

            confidence = weight

            sigma_clipped = max(confidence, 70)

            diff = abs(row['FVC'] - row['FVC_pred'])

            delta = min(diff, 1000)

            score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)

            return -score



        results = []

        tk0 = tqdm(train.iterrows(), total=len(train))

        for _, row in tk0:

            loss_partial = partial(loss_func, row=row)

            weight = [100]

            result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')

            x = result['x']

            results.append(x[0])



        # optimized score

        train['Confidence'] = results

        train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

        train['diff'] = abs(train['FVC'] - train['FVC_pred'])

        train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

        train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])

        score = train['score'].mean()

        scorelist.append(score)

        print(score)
scorelist
TARGET = 'Confidence'



target = train[TARGET]

test[TARGET] = np.nan



# features

cat_features = ['Sex', 'SmokingStatus']

num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]

features = num_features + cat_features

drop_features = [ID, TARGET, 'predict_Week', 'base_Week', 'FVC', 'FVC_pred']

features = [c for c in features if c not in drop_features]



oof, predictions = run_kfold_model(clf, train, test, folds, features, target, n_fold=N_FOLD)
train['Confidence'] = oof

train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))

train['diff'] = abs(train['FVC'] - train['FVC_pred'])

train['delta'] = train['diff'].apply(lambda x: min(x, 1000))

train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])

score = train['score'].mean()

print(score)
test['Confidence'] = predictions

test = test.reset_index()
sub = submission[['Patient_Week']].merge(test[['Patient_Week', 'FVC_pred', 'Confidence']], on='Patient_Week')

sub = sub.rename(columns={'FVC_pred': 'FVC'})



for i in range(len(otest)):

    sub.loc[sub['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    sub.loc[sub['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

    

sub[sub.Confidence<1]



sub.to_csv('submission.csv', index=False, float_format='%.1f')