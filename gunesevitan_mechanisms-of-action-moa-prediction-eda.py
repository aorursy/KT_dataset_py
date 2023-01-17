import warnings

warnings.simplefilter(action='ignore', category=UserWarning)



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

from scipy.stats import skew, kurtosis



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



import lightgbm as lgb



SEED = 721991
df_train = pd.read_csv('../input/lish-moa/train_features.csv')

df_test = pd.read_csv('../input/lish-moa/test_features.csv')



df_train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

df_train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

df_test_targets = pd.read_csv('../input/lish-moa/sample_submission.csv')



target_features_scored = list(df_train_targets_scored.columns[1:])

target_features_nonscored = list(df_train_targets_nonscored.columns[1:])

df_train_targets_scored[target_features_scored] = df_train_targets_scored[target_features_scored].astype(np.uint8)

df_train_targets_nonscored[target_features_nonscored] = df_train_targets_nonscored[target_features_nonscored].astype(np.uint8)

df_test_targets[target_features_scored] = df_test_targets[target_features_scored].astype(np.float32)



df_train = df_train.merge(df_train_targets_scored, on='sig_id', how='left')

df_train = df_train.merge(df_train_targets_nonscored, on='sig_id', how='left')

df_test = df_test.merge(df_test_targets, on='sig_id', how='left')



del df_train_targets_scored, df_train_targets_nonscored, df_test_targets



print(f'Training Set Shape = {df_train.shape}')

print(f'Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

print(f'Test Set Shape = {df_test.shape}')

print(f'Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')
g_features = [feature for feature in df_train.columns if feature.startswith('g-')]

c_features = [feature for feature in df_train.columns if feature.startswith('c-')]

other_features = [feature for feature in df_train.columns if feature not in g_features and 

                                                             feature not in c_features and 

                                                             feature not in target_features_scored and

                                                             feature not in target_features_nonscored]



print(f'Number of g- Features: {len(g_features)}')

print(f'Number of c- Features: {len(c_features)}')

print(f'Number of Other Features: {len(other_features)} ({other_features})')
print(f'Number of Scored Target Features: {len(target_features_scored)}')

print(f'Number of Non-scored Target Features: {len(target_features_nonscored)}')
def mean_columnwise_logloss(y_true, y_pred):        

    y_pred = np.clip(y_pred, 1e-15, (1 - 1e-15))

    score = - np.mean(np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))

    return score
scored_targets_classified = df_train[target_features_scored].sum(axis=1)

nonscored_targets_classified = df_train[target_features_nonscored].sum(axis=1)



fig, axes = plt.subplots(figsize=(32, 8), ncols=2)



sns.countplot(scored_targets_classified, ax=axes[0])

sns.countplot(nonscored_targets_classified, ax=axes[1])



for i in range(2):

    axes[i].tick_params(axis='x', labelsize=20)

    axes[i].tick_params(axis='y', labelsize=20)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    

axes[0].set_title(f'Training Set Unique Scored Targets per Sample', size=22, pad=22)

axes[1].set_title(f'Training Set Unique Non-scored Targets per Sample', size=22, pad=22)



plt.show()
fig = plt.figure(figsize=(12, 60))



sns.barplot(x=df_train[target_features_scored].sum(axis=0).sort_values(ascending=False).values,

            y=df_train[target_features_scored].sum(axis=0).sort_values(ascending=False).index)



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)

plt.xlabel('')

plt.ylabel('')

plt.title('Training Set Scored Targets Classification Counts', size=18, pad=18)



plt.show()
fig = plt.figure(figsize=(12, 110))



sns.barplot(x=df_train[target_features_nonscored].sum(axis=0).sort_values(ascending=False).values,

            y=df_train[target_features_nonscored].sum(axis=0).sort_values(ascending=False).index)



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)

plt.xlabel('')

plt.ylabel('')

plt.title('Training Set Non-scored Targets Classification Counts', size=18, pad=18)



plt.show()
fig, axes = plt.subplots(figsize=(32, 15), ncols=2, dpi=150)



sns.heatmap(df_train[target_features_scored].corr(),

            annot=False,

            square=True,

            cmap='coolwarm',

            cbar=False,

            yticklabels=False,

            xticklabels=False,

            ax=axes[0])  



sns.heatmap(df_train[target_features_nonscored].corr(),

            annot=False,

            square=True,

            cmap='coolwarm',

            cbar=False,

            yticklabels=False,

            xticklabels=False,

            ax=axes[1])   



axes[0].set_title('Training Set Scored Target Correlations', size=25, pad=25)

axes[1].set_title('Training Set Non-scored Target Correlations', size=25, pad=25)



plt.show()
fig, axes = plt.subplots(figsize=(24, 24), nrows=3, ncols=2)



sns.countplot(df_train['cp_type'], ax=axes[0][0])

sns.countplot(df_test['cp_type'], ax=axes[0][1])



sns.countplot(df_train['cp_time'], ax=axes[1][0])

sns.countplot(df_test['cp_time'], ax=axes[1][1])



sns.countplot(df_train['cp_dose'], ax=axes[2][0])

sns.countplot(df_test['cp_dose'], ax=axes[2][1])



for i in range(3):

    for j in range(2):

        axes[i][j].tick_params(axis='x', labelsize=15)

        axes[i][j].tick_params(axis='y', labelsize=15)

        axes[i][j].set_xlabel('')

        axes[i][j].set_ylabel('')

        

for i, feature in enumerate(['cp_type', 'cp_time', 'cp_dose']):

    for j, dataset in enumerate(['Training', 'Test']):

        axes[i][j].set_title(f'{dataset} Set {feature} Distribution', size=18, pad=18)



plt.show()
df_control = df_train[df_train['cp_type'] == 'ctl_vehicle']

df_compound = df_train[df_train['cp_type'] == 'trt_cp']



print(f'{len(df_control)}/{len(df_train)} samples are treated with a control perturbation and {len(df_control[df_control[target_features_scored].sum(axis=1) == 0])}/{len(df_control)} of those samples have all zero targets')

print(f'{len(df_compound)}/{len(df_train)} samples are treated with a compound and {len(df_compound[df_compound[target_features_scored].sum(axis=1) == 0])}/{len(df_compound)} of those samples have all zero targets')
df_target_counts_by_cp_time = pd.DataFrame(columns=['target', 'cp_time', 'count'])



for target_feature in target_features_scored:    

    for cp_time in [24, 48, 72]:

        count = len(df_train[(df_train['cp_time'] == cp_time) & (df_train[target_feature] == 1)])

        df_target_counts_by_cp_time = df_target_counts_by_cp_time.append({'target': target_feature, 'cp_time': cp_time, 'count': count}, ignore_index=True)

        

df_target_counts_by_cp_time['total_count'] = df_target_counts_by_cp_time.groupby('target')['count'].transform('sum')

df_target_counts_by_cp_time.sort_values(by=['total_count', 'target'],ascending=False, inplace=True)



fig = plt.figure(figsize=(15, 75), dpi=100)



sns.barplot(x=df_target_counts_by_cp_time['count'],

            y=df_target_counts_by_cp_time['target'],

            hue=df_target_counts_by_cp_time['cp_time'])



plt.xlabel('')

plt.ylabel('')

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, prop={'size': 20})

plt.title('Training Set cp_time Distribution in Scored Targets', size=18, pad=18)



plt.show()



del df_target_counts_by_cp_time
df_target_counts_by_cp_dose = pd.DataFrame(columns=['target', 'cp_dose', 'count'])



for target_feature in target_features_scored:    

    for cp_dose in ['D1', 'D2']:

        count = len(df_train[(df_train['cp_dose'] == cp_dose) & (df_train[target_feature] == 1)])

        df_target_counts_by_cp_dose = df_target_counts_by_cp_dose.append({'target': target_feature, 'cp_dose': cp_dose, 'count': count}, ignore_index=True)

        

df_target_counts_by_cp_dose['total_count'] = df_target_counts_by_cp_dose.groupby('target')['count'].transform('sum')

df_target_counts_by_cp_dose.sort_values(by=['total_count', 'target'],ascending=False, inplace=True)



fig = plt.figure(figsize=(15, 75), dpi=100)



sns.barplot(x=df_target_counts_by_cp_dose['count'],

            y=df_target_counts_by_cp_dose['target'],

            hue=df_target_counts_by_cp_dose['cp_dose'])



plt.xlabel('')

plt.ylabel('')

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, prop={'size': 20})

plt.title('Training Set cp_dose Distribution in Scored Targets', size=18, pad=18)



plt.show()



del df_target_counts_by_cp_dose
df_train[(['sig_id'] + c_features)]
def scatterplot_cfeature(feature_group, seeds):

        

    fig, axes = plt.subplots(ncols=5, figsize=(36, 5), dpi=100, constrained_layout=True)

    title_size = 25

    label_size = 25



    for i, feature in enumerate(feature_group):

        

        np.random.seed(seeds[i])

        target = np.random.choice(target_features_scored)

        if len(target) > 25:

            target_title = target[:25]

        else:

            target_title = target

            

        sns.scatterplot(df_train[feature], df_train[target], s=100, ax=axes[i])

        axes[i].set_xlabel('')

        axes[i].set_ylabel('')

        axes[i].tick_params(axis='x', labelsize=label_size)

        axes[i].tick_params(axis='y', labelsize=label_size)

        

        for label in axes[i].get_yticklabels():

            if i % 5 == 0:

                label.set_visible(True)

            else:

                label.set_visible(False)

                

        axes[i].set_title(f'{feature} vs {target_title}', size=title_size, pad=title_size)

    

    plt.show()

    

for i, feature_group in enumerate(np.array_split(c_features, len(c_features) // 5), 2):

    scatterplot_cfeature(feature_group, seeds=np.arange(1, 6) * i)
def distplot_cfeature(feature_group):

        

    fig, axes = plt.subplots(ncols=5, figsize=(36, 5), dpi=100, constrained_layout=True)

    title_size = 25

    label_size = 25



    for i, feature in enumerate(feature_group):

        sns.distplot(df_train[feature], label='Training', ax=axes[i], hist_kws={'alpha': 0.25})

        sns.distplot(df_test[feature], label='Test', ax=axes[i], hist_kws={'alpha': 0.25})

        axes[i].set_xlabel('')

        axes[i].tick_params(axis='x', labelsize=label_size)

        axes[i].tick_params(axis='y', labelsize=label_size)

        

        if i % 5 == 0:

            axes[i].legend(prop={'size': 25})

            

        axes[i].set_title(f'{feature} Distribution', size=title_size, pad=title_size)

    

    plt.show()

    

for feature_group in np.array_split(c_features, len(c_features) // 5):

    

    for c_feature in feature_group: 

        train_mean = df_train[c_feature].mean()

        train_median = df_train[c_feature].median()

        train_std = df_train[c_feature].std()

        train_min = df_train[c_feature].min()

        train_max = df_train[c_feature].max()

        train_skew = skew(df_train[c_feature])

        train_kurt = kurtosis(df_train[c_feature])

        train_var = np.var(df_train[c_feature])



        test_mean = df_test[c_feature].mean()

        test_median = df_test[c_feature].median()

        test_std = df_test[c_feature].std()

        test_min = df_test[c_feature].min()

        test_max = df_test[c_feature].max()

        test_skew = skew(df_test[c_feature])

        test_kurt = kurtosis(df_test[c_feature])

        test_var = np.var(df_test[c_feature])



        print(f'{c_feature} Train - Mean: {train_mean:.4} - Median: {train_median:.4} - Std: {train_std:.4} - Min: {train_min:.4} - Max: {train_max:.4} - Skew {train_skew:.4} - Kurt {train_kurt:.4} - Var {train_var:.4}')

        print(f'{c_feature} Test - Mean: {test_mean:.4} - Median: {test_median:.4} - Std: {test_std:.4} - Min: {test_min:.4} - Max: {test_max:.4} - Skew {test_skew:.4} - Kurt {test_kurt:.4} - Var {test_var:.4}\n')



    distplot_cfeature(feature_group)

fig = plt.figure(figsize=(20, 20))



ax = sns.heatmap(df_train[c_features].corr(),

                 annot=False,

                 square=True)



ax.tick_params(axis='x', labelsize=20, rotation=0, pad=20)

ax.tick_params(axis='y', labelsize=20, rotation=0, pad=20)



for idx, label in enumerate(ax.get_xticklabels()):

    if idx % 5 == 0:

        label.set_visible(True)

    else:

        label.set_visible(False)

        

for idx, label in enumerate(ax.get_yticklabels()):

    if idx % 5 == 0:

        label.set_visible(True)

    else:

        label.set_visible(False)

        

cbar = ax.collections[0].colorbar

cbar.ax.tick_params(labelsize=30, pad=20)



plt.title('Cell Viability Features Correlations', size=30, pad=30)

plt.show()
df_train[(['sig_id'] + g_features)]
def scatterplot_gfeature(feature_group, seeds):

        

    fig, axes = plt.subplots(ncols=4, figsize=(36, 5), dpi=100, constrained_layout=True)

    title_size = 25

    label_size = 25



    for i, feature in enumerate(feature_group):

                

        np.random.seed(seeds[i])

        target = np.random.choice(target_features_scored)

        if len(target) > 25:

            target_title = target[:25]

        else:

            target_title = target

            

        sns.scatterplot(df_train[feature], df_train[target], s=100, ax=axes[i])

        axes[i].set_xlabel('')

        axes[i].set_ylabel('')

        axes[i].tick_params(axis='x', labelsize=label_size)

        axes[i].tick_params(axis='y', labelsize=label_size)

        

        for label in axes[i].get_yticklabels():

            if i % 5 == 0:

                label.set_visible(True)

            else:

                label.set_visible(False)

                

        axes[i].set_title(f'{feature} vs {target_title}', size=title_size, pad=title_size)

    

    plt.show()

    

shuffled_g_features = np.copy(g_features)

np.random.shuffle(shuffled_g_features)

for i, feature_group in enumerate(np.array_split(shuffled_g_features, len(shuffled_g_features) // 4)[:10], 1):

    scatterplot_gfeature(feature_group, seeds=np.arange(1, 6) * i)
def distplot_gfeature(feature_group):

        

    fig, axes = plt.subplots(ncols=4, figsize=(36, 5), dpi=100, constrained_layout=True)

    title_size = 25

    label_size = 25



    for i, feature in enumerate(feature_group):

        sns.distplot(df_train[feature], label='Training', ax=axes[i], hist_kws={'alpha': 0.25})

        sns.distplot(df_test[feature], label='Test', ax=axes[i], hist_kws={'alpha': 0.25})

        axes[i].set_xlabel('')

        axes[i].tick_params(axis='x', labelsize=label_size)

        axes[i].tick_params(axis='y', labelsize=label_size)

        

        if i % 5 == 0:

            axes[i].legend(prop={'size': 25})

            

        axes[i].set_title(f'{feature} Distribution', size=title_size, pad=title_size)

    

    plt.show()



shuffled_g_features = np.copy(g_features)

np.random.shuffle(shuffled_g_features)

for feature_group in np.array_split(shuffled_g_features, len(shuffled_g_features) // 4)[:10]:

    

    for c_feature in feature_group: 

        train_mean = df_train[c_feature].mean()

        train_median = df_train[c_feature].median()

        train_std = df_train[c_feature].std()

        train_min = df_train[c_feature].min()

        train_max = df_train[c_feature].max()

        train_skew = skew(df_train[c_feature])

        train_kurt = kurtosis(df_train[c_feature])

        train_var = np.var(df_train[c_feature])



        test_mean = df_test[c_feature].mean()

        test_median = df_test[c_feature].median()

        test_std = df_test[c_feature].std()

        test_min = df_test[c_feature].min()

        test_max = df_test[c_feature].max()

        test_skew = skew(df_test[c_feature])

        test_kurt = kurtosis(df_test[c_feature])

        test_var = np.var(df_test[c_feature])



        print(f'{c_feature} Train - Mean: {train_mean:.4} - Median: {train_median:.4} - Std: {train_std:.4} - Min: {train_min:.4} - Max: {train_max:.4} - Skew {train_skew:.4} - Kurt {train_kurt:.4} - Var {train_var:.4}')

        print(f'{c_feature} Test - Mean: {test_mean:.4} - Median: {test_median:.4} - Std: {test_std:.4} - Min: {test_min:.4} - Max: {test_max:.4} - Skew {test_skew:.4} - Kurt {test_kurt:.4} - Var {test_var:.4}\n')



    distplot_gfeature(feature_group)

fig = plt.figure(figsize=(20, 20))



ax = sns.heatmap(df_train[g_features].corr(),

                 annot=False,

                 square=True)



ax.tick_params(axis='x', labelsize=20, rotation=90, pad=20)

ax.tick_params(axis='y', labelsize=20, rotation=0, pad=20)



for idx, label in enumerate(ax.get_xticklabels()):

    if idx % 5 == 0:

        label.set_visible(True)

    else:

        label.set_visible(False)

        

for idx, label in enumerate(ax.get_yticklabels()):

    if idx % 5 == 0:

        label.set_visible(True)

    else:

        label.set_visible(False)

        

cbar = ax.collections[0].colorbar

cbar.ax.tick_params(labelsize=30, pad=20)



plt.title('Gene Expression Features Correlations', size=30, pad=30)

plt.show()
df_train['target'] = 0

df_test['target'] = 1



X = pd.concat([df_train.loc[:, g_features + c_features], df_test.loc[:, g_features + c_features]]).reset_index(drop=True)

y = pd.concat([df_train.loc[:, 'target'], df_test.loc[:, 'target']]).reset_index(drop=True)
K = 5

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)



scores = []

oof_predictions = pd.DataFrame(np.zeros((X.shape[0], 1)), columns=['target'])

feature_importance = pd.DataFrame(np.zeros((X.shape[1], K)), columns=[f'Fold_{i}_Importance' for i in range(1, K + 1)], index=X.columns)



parameters = {

    'num_iterations': 500,

    'early_stopping_round': 50,

    'num_leaves': 2 ** 5, 

    'learning_rate': 0.05,

    'bagging_fraction': 0.9,

    'bagging_freq': 1,

    'feature_fraction': 0.9,

    'feature_fraction_bynode': 0.9,

    'lambda_l1': 0,

    'lambda_l2': 0,

    'max_depth': -1,

    'objective': 'binary',

    'seed': SEED,

    'feature_fraction_seed': SEED,

    'bagging_seed': SEED,

    'drop_seed': SEED,

    'data_random_seed': SEED,

    'boosting_type': 'gbdt',

    'verbose': 1,

    'metric': 'auc',

    'n_jobs': -1,   

}



print('Running LightGBM Adversarial Validation Model\n' + ('-' * 45) + '\n')



for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):



    trn_data = lgb.Dataset(X.iloc[trn_idx, :], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[val_idx, :], label=y.iloc[val_idx])      

    model = lgb.train(parameters, trn_data, valid_sets=[trn_data, val_data], verbose_eval=50)

    feature_importance.iloc[:, fold - 1] = model.feature_importance(importance_type='gain')



    predictions = model.predict(X.iloc[val_idx, :], num_iteration=model.best_iteration)

    oof_predictions.loc[val_idx, 'target'] = predictions

    

    score = roc_auc_score(y.iloc[val_idx], predictions)

    scores.append(score)            

    print(f'\nFold {fold} - ROC AUC Score {score:.6}\n')

    

oof_score = roc_auc_score(y, oof_predictions)

print(f'\n{"-" * 30}\nLightGBM Adversarial Validation Model Mean ROC AUC Score {np.mean(scores):.6} [STD:{np.std(scores):.6}]')

print(f'LightGBM Adversarial Validation Model OOF ROC AUC Score: {oof_score:.6}\n{"-" * 30}')





plt.figure(figsize=(20, 20))

feature_importance['Mean_Importance'] = feature_importance.sum(axis=1) / K

feature_importance.sort_values(by='Mean_Importance', inplace=True, ascending=False)

sns.barplot(x='Mean_Importance', y=feature_importance.index[:50], data=feature_importance[:50])



plt.xlabel('')

plt.tick_params(axis='x', labelsize=18)

plt.tick_params(axis='y', labelsize=18)

plt.title('LightGBM Adversarial Validation Model Top 50 Feature Importance (Gain)', size=20, pad=20)



plt.show()



del X, y, oof_predictions, feature_importance, parameters, scores, oof_score