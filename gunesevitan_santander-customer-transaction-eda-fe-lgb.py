import numpy as np

import pandas as pd

pd.options.display.max_rows = 200

pd.options.display.max_columns = 200



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.cluster import KMeans

import lightgbm as lgb



import warnings

warnings.filterwarnings('ignore')



SEED = 42
df_train = pd.read_csv('../input/train.csv')

df_train.name = 'Training Set'

df_test = pd.read_csv('../input/test.csv')

df_test.name = 'Test Set'



print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Test Examples = {}'.format(df_test.shape[0]))

print('Training X Shape = {}'.format(df_train.shape))

print('Training y Shape = {}'.format(df_train['target'].shape[0]))

print('Test X Shape = {}'.format(df_test.shape))

print('Test y Shape = {}'.format(df_test.shape[0]))

print(df_train.columns)

print(df_test.columns)
print(df_train.info())

df_train.sample(5)
print(df_test.info())

df_test.sample(5)
ones = df_train['target'].value_counts()[1]

zeros = df_train['target'].value_counts()[0]

ones_per = ones / df_train.shape[0] * 100

zeros_per = zeros / df_train.shape[0] * 100



print('{} out of {} rows are Class 1 and it is the {:.2f}% of the dataset.'.format(ones, df_train.shape[0], ones_per))

print('{} out of {} rows are Class 0 and it is the {:.2f}% of the dataset.'.format(zeros, df_train.shape[0], zeros_per))



plt.figure(figsize=(8, 6))

sns.countplot(df_train['target'])



plt.xlabel('Target')

plt.xticks((0, 1), ['Class 0 ({0:.2f}%)'.format(ones_per), 'Class 1 ({0:.2f}%)'.format(zeros_per)])

plt.ylabel('Count')

plt.title('Training Set Target Distribution')



plt.show()
df_train_corr = df_train.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)

df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)



df_test_corr = df_test.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)

df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr['Correlation Coefficient'] == 1.0].index)
# Top 5 Highest Correlations in the Training Set

df_train_corr_nd.tail()
# Top 5 Highest Correlations between variables in the Training Set

df_train_corr_nd[np.logical_and(df_train_corr_nd['Feature 1'] != 'target', df_train_corr_nd['Feature 2'] != 'target')].tail()
# Top 5 Highest Correlations in the Test Set

df_test_corr_nd.tail()
df_train_unique = df_train.agg(['nunique']).transpose().sort_values(by='nunique')

df_test_unique = df_test.agg(['nunique']).transpose().sort_values(by='nunique')

df_uniques = df_train_unique.drop('target').reset_index().merge(df_test_unique.reset_index(), how='left', right_index=True, left_index=True)

df_uniques.drop(columns=['index_y'], inplace=True)

df_uniques.columns = ['Feature', 'Training Set Unique Count', 'Test Set Unique Count']
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(24, 12))



sns.barplot(x=df_train_unique.index[1:6], y="nunique", data=df_train_unique[1:].head(), ax=axs[0][0])

sns.barplot(x=df_test_unique.index[:5], y="nunique", data=df_test_unique.head(), ax=axs[0][1])

sns.barplot(x=df_train_unique.index[-6:-1], y="nunique", data=df_train_unique[-6:-1].tail(), ax=axs[1][0])

sns.barplot(x=df_test_unique.index[-6:-1], y="nunique", data=df_test_unique[-6:-1].tail(), ax=axs[1][1])



for i in range(2):

    for j in range(2):        

        axs[i][j].set(xlabel='Features', ylabel='Unique Count')

        

axs[0][0].set_title('Training Set Features with Least Unique Values')

axs[0][1].set_title('Test Set Features with Least Unique Values')

axs[1][0].set_title('Training Set Features with Most Unique Values')

axs[1][1].set_title('Test Set Features with Most Unique Values')



plt.show()
df_qdist = pd.DataFrame(np.zeros((200, 9)), columns=['Quartile 1 Positives', 'Quartile 2 Positives', 'Quartile 3 Positives', 'Quartile 4 Positives',

                                                     'Quartile 1 Positive Percentage', 'Quartile 2 Positive Percentage', 'Quartile 3 Positive Percentage', 'Quartile 4 Positive Percentage',

                                                     'Quartile Order'])

features = [col for col in df_train.columns.values.tolist() if col.startswith('var')]

quartiles = np.arange(0, 1, 0.25)

df_qdist.index = features



for i, feature in enumerate(features):

    for j, quartile in enumerate(quartiles):

        target_counts = df_train[np.logical_and(df_train[feature] >= df_train[feature].quantile(q=quartile), 

                                                df_train[feature] < df_train[feature].quantile(q=quartile + 0.25))].target.value_counts()

        

        ones_per = target_counts[1] / (target_counts[0] + target_counts[1]) * 100

        df_qdist.iloc[i, j] = target_counts[1]

        df_qdist.iloc[i, j + 4] = ones_per



pers = df_qdist.columns.tolist()[4:-1]         

        

for i, index in enumerate(df_qdist.index):

    order = df_qdist[pers].iloc[[i]].sort_values(by=index, ascending=False, axis=1).columns

    order_str = ''.join([col[9] for col in order])

    df_qdist.iloc[i, 8] = order_str        

                

df_qdist = df_qdist.round(2)

df_qdist.head(10)
# 5 features that doesn't have highest positive target percentage in 1st and 4th quartiles

df_qdist[np.logical_or(df_qdist['Quartile Order'].str.startswith('2'), df_qdist['Quartile Order'].str.startswith('3'))] 
for i, col in enumerate(pers):    

    print('There are {} features that have the highest positive target percentage in Quartile {}'.format(df_qdist[df_qdist['Quartile Order'].str.startswith(str(i + 1))].count()[0],

                                                                                                            i + 1))

    print('Quartile {} max positive target percentage = {}% ({})'.format(i + 1, df_qdist[col].max(), df_qdist[col].argmax()))

    print('Quartile {} min positive target percentage = {}% ({})\n'.format(i + 1, df_qdist[col].min(), df_qdist[col].argmin()))
features = [col for col in df_train.columns.tolist() if col.startswith('var')]



nrows = 50

fig, axs = plt.subplots(nrows=50, ncols=4, figsize=(24, nrows * 5))



for i, feature in enumerate(features, 1):

    plt.subplot(50, 4, i)

    sns.kdeplot(df_train[feature], bw='silverman', label='Training Set', shade=True)

    sns.kdeplot(df_test[feature], bw='silverman', label='Test Set', shade=True)

    

    plt.tick_params(axis='x', which='major', labelsize=8)

    plt.tick_params(axis='y', which='major', labelsize=8)

    

    plt.legend(loc='upper right')

    plt.title('Distribution of {} in Training and Test Set'.format(feature))

    

plt.show()
features = [col for col in df_train.columns.tolist() if col.startswith('var')]



nrows = 50

fig, axs = plt.subplots(nrows=50, ncols=4, figsize=(24, nrows * 5))



for i, feature in enumerate(features, 1):

    plt.subplot(50, 4, i)

    

    sns.distplot(StandardScaler().fit_transform(df_train[df_train['target'] == 0][feature].values.reshape(-1, 1)), label='Target=0', hist=True, color='#e74c3c')

    sns.distplot(StandardScaler().fit_transform(df_train[df_train['target'] == 1][feature].values.reshape(-1, 1)), label='Target=1', hist=True, color='#2ecc71')

    

    plt.tick_params(axis='x', which='major', labelsize=8)

    plt.tick_params(axis='y', which='major', labelsize=8)

    

    plt.legend(loc='upper right')

    plt.xlabel('')

    plt.title('Distribution of Target in {}'.format(feature))

    

plt.show()
test = df_test.drop(['ID_code'], axis=1).values



unique_count = np.zeros_like(test)



for feature in range(test.shape[1]):

    _, index, count = np.unique(test[:, feature], return_counts=True, return_index=True)

    unique_count[index[count == 1], feature] += 1

    

real_samples = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

synth_samples = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]



print('Number of real samples in test set is {}'.format(len(real_samples)))

print('Number of synthetic samples in test set is {}'.format(len(synth_samples)))
features = [col for col in df_train.columns if col.startswith('var')]

df_all = pd.concat([df_train, df_test.ix[real_samples]])



for feature in features:

    temp = df_all[feature].value_counts(dropna=True)



    df_train[feature + 'vc'] = df_train[feature].map(temp).map(lambda x: min(10, x)).astype(np.uint8)

    df_test[feature + 'vc'] = df_test[feature].map(temp).map(lambda x: min(10, x)).astype(np.uint8)



    df_train[feature + 'sum'] = ((df_train[feature] - df_all[feature].mean()) * df_train[feature + 'vc'].map(lambda x: int(x > 1))).astype(np.float32)

    df_test[feature + 'sum'] = ((df_test[feature] - df_all[feature].mean()) * df_test[feature + 'vc'].map(lambda x: int(x > 1))).astype(np.float32) 



    df_train[feature + 'sum2'] = ((df_train[feature]) * df_train[feature + 'vc'].map(lambda x: int(x > 2))).astype(np.float32)

    df_test[feature + 'sum2'] = ((df_test[feature]) * df_test[feature + 'vc'].map(lambda x: int(x > 2))).astype(np.float32)



    df_train[feature + 'sum3'] = ((df_train[feature]) * df_train[feature + 'vc'].map(lambda x: int(x > 4))).astype(np.float32) 

    df_test[feature + 'sum3'] = ((df_test[feature]) * df_test[feature + 'vc'].map(lambda x: int(x > 4))).astype(np.float32)

    

print('Training set shape after creating magic features: {}'.format(df_train.shape))

print('Test set shape after creating magic features: {}'.format(df_test.shape))
def augment(x, y, t=2):

    

    xs, xn = [], []

    

    for i in range(t // 2):

        mask = y == 0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        featnum = x1.shape[1] // 200 - 1



        for c in range(200):

            np.random.shuffle(ids)

            x1[:, [c] + [200 + featnum * c + idc for idc in range(featnum)]] = x1[ids][:, [c] + [200 + featnum * c + idc for idc in range(featnum)]]

        xn.append(x1)

    

    for i in range(t):

        mask = y > 0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        featnum = x1.shape[1] // 200 - 1

        

        for c in range(200):

            np.random.shuffle(ids)

            x1[:, [c] + [200 + featnum * c + idc for idc in range(1)]] = x1[ids][:, [c] + [200 + featnum * c + idc for idc in range(1)]]

        xs.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x, xs, xn])

    y = np.concatenate([y, ys, yn])

    

    return x, y
"""

def get_quartile_mask(df, feature, q):

    

    assert feature in df.columns

    

    # Returns a boolean mask of the given features' quartile

    if q==1:

        return np.logical_and(df[feature] >= df[feature].quantile(q=0), df[feature] < df[feature].quantile(q=0.25))

    elif q==2:

        return np.logical_and(df[feature] >= df[feature].quantile(q=0.25), df[feature] < df[feature].quantile(q=0.5))

    elif q==3:

        return np.logical_and(df[feature] >= df[feature].quantile(q=0.5), df[feature] < df[feature].quantile(q=0.75))

    elif q==4:

        return np.logical_and(df[feature] >= df[feature].quantile(q=0.75), df[feature] <= df[feature].quantile(q=1))

    else:

        return -1      

    

for df in [df_train, df_test]:

    df['quartile_rank'] = 0



# Ranking every cell by their quartile

for df in [df_train, df_test]:

    for col in variables:

        col_rank = df_qdist.loc[col, 'order']

        for i in range(1, 5):

            q_ind = df[get_quartile_mask(df, col, i)].index

            df.loc[q_ind, 'quartile_rank'] += col_rank[::-1].find(str(i)) + 1      

            

df_train['quartile_rank'] = MinMaxScaler().fit_transform(df_train['quartile_rank'].values.reshape(-1, 1))

df_test['quartile_rank'] = MinMaxScaler().fit_transform(df_test['quartile_rank'].values.reshape(-1, 1))

"""
def smooth_mean(df, by, on, weight):

    

    global_mean = df[on].mean()

    

    agg = df.groupby(by)[on].agg(['count', 'mean'])

    counts = agg['count']

    means = agg['mean']



    smooth = (counts * means + weight * global_mean) / (counts + weight)



    return df[by].map(smooth)



"""

for df in [df_train, df_test]:

    df['var_68'] = smooth_mean(df, 'var_68', 'var_81', 10)

"""
class KMeansFeaturizer:



    def __init__(self, k, target_scale=5.0, random_state=None):

        self.k = k

        self.target_scale = target_scale

        self.random_state = random_state

        self.encoder = OneHotEncoder(categories='auto').fit(np.array(range(k)).reshape(-1, 1))



    def fit(self, X, y=None):

        if y is None:

            kmeans = KMeans(n_clusters=self.k, n_init=20, random_state=self.random_state)

            kmeans.fit(X)

            self.kmeans = kmeans

            self.cluster_centers_ = kmeans.cluster_centers_

        else:

            Xy = np.hstack((X, y[:, np.newaxis] * self.target_scale))

            kmeans_pretrain = KMeans(n_clusters=self.k, n_init=20, random_state=self.random_state)

            kmeans_pretrain.fit(Xy)



            kmeans = KMeans(n_clusters=self.k, init=kmeans_pretrain.cluster_centers_[:, :2], n_init=1, max_iter=1)

            kmeans.fit(X)



            self.kmeans = kmeans

            self.cluster_centers_ = km_model.cluster_centers_

            

        return self



    def transform(self, X, y=None):

        clusters = self.kmeans.predict(X)

        return self.encoder.transform(clusters.reshape(-1, 1))



    def fit_transform(self, X, y=None):

        self.fit(X, y)

        return self.transform(X, y)

    

"""

train = df_train_orig.drop(columns=['ID_code', 'target'])

test = df_test_orig.drop(columns=['ID_code'])

all = pd.concat([train, test])



ks = [2, 5, 10, 25, 50]

for k in ks:

    kmf = KMeansFeaturizer(k=k, random_state=SEED)

    k_features = kmf.fit_transform(all)

    

    k_features_df = pd.DataFrame(k_features.toarray())

    k_features_df = k_features_df.add_prefix('k{}_group'.format(k))

    all = pd.concat([all, k_features_df], axis=1)

"""
def transform_feature(df, feature, transformation, **transform_params):

    

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 12))  

    plt.subplots_adjust(right=1.5)

       

    sns.distplot(df[df['target'] == 0][feature].values, label='Target=0', color='blue', ax=axs[0][0])

    sns.distplot(df[df['target'] == 1][feature].values, label='Target=1', color='red', ax=axs[0][0])

    axs[0][0].set_title('{} Target Distribution in Training Set'.format(feature))

    axs[0][0].legend()

    

    sns.distplot(transformation(df[df['target'] == 0][feature].values, **transform_params), label='Target=0', color='blue', ax=axs[0][1])

    sns.distplot(transformation(df[df['target'] == 1][feature].values, **transform_params), label='Target=1', color='red', ax=axs[0][1])

    axs[0][1].set_title('{} Target Distribution After Applying {} Function '.format(feature, transformation.__name__))

    axs[0][1].legend()

    

    sns.distplot(df[feature].values, label='Training Set', hist=False, color='grey', ax=axs[1][0])

    sns.distplot(df_test[feature].values, label='Test Set', hist=False, color='magenta', ax=axs[1][0])

    axs[1][0].set_title('{} Distribution in Training and Test Set'.format(feature))

    axs[1][0].legend()

    

    sns.distplot(transformation(df[feature].values, **transform_params), label='Training Set', hist=False, color='grey', ax=axs[1][1])

    sns.distplot(transformation(df_test[feature].values, **transform_params), label='Test Set', hist=False, color='magenta', ax=axs[1][1])

    axs[1][1].set_title('{} Distribution in Training and Test Set After Applying {} Function'.format(feature, transformation.__name__))

    axs[1][1].legend()

    

    plt.show()
transform_feature(df=df_train, feature='var_108', transformation=np.round, decimals=2)
gbdt_param = {

    # Core Parameters

    'objective': 'binary',

    'boosting': 'gbdt',

    'learning_rate': 0.01,

    'num_leaves': 15,

    'tree_learner': 'serial',

    'num_threads': 8,

    'seed': SEED,

    

    # Learning Control Parameters

    'max_depth': -1,

    'min_data_in_leaf': 50,

    'min_sum_hessian_in_leaf': 10,  

    'bagging_fraction': 0.6,

    'bagging_freq': 5,

    'feature_fraction': 0.05,

    'lambda_l1': 1.,

    'bagging_seed': SEED,

    

    # Others

    'verbosity ': 1,

    'boost_from_average': False,

    'metric': 'auc',

}
predictors = df_train.columns.tolist()[2:]

X_test = df_test[predictors]



n_splits = 5

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)



oof = df_train[['ID_code', 'target']]

oof['predict'] = 0

predictions = df_test[['ID_code']]

val_aucs = []

feature_importance_df = pd.DataFrame()
for fold, (train_ind, val_ind) in enumerate(skf.split(df_train, df_train.target.values)):

    

    X_train, y_train = df_train.iloc[train_ind][predictors], df_train.iloc[train_ind]['target']

    X_valid, y_valid = df_train.iloc[val_ind][predictors], df_train.iloc[val_ind]['target']



    N = 1

    p_valid, yp = 0, 0

        

    for i in range(N):

        print('\nFold {} - N {}'.format(fold + 1, i + 1))

        

        X_t, y_t = augment(X_train.values, y_train.values)

        weights = np.array([0.8] * X_t.shape[0])

        weights[:X_train.shape[0]] = 1.0

        print('Shape of X_train after augment: {}\nShape of y_train after augment: {}'.format(X_t.shape, y_t.shape))

        

        X_t = pd.DataFrame(X_t)

        X_t = X_t.add_prefix('var_')

    

        trn_data = lgb.Dataset(X_t, label=y_t, weight=weights)

        val_data = lgb.Dataset(X_valid, label=y_valid)

        evals_result = {}

        

        lgb_clf = lgb.train(gbdt_param, trn_data, 100000, valid_sets=[trn_data, val_data], early_stopping_rounds=5000, verbose_eval=1000, evals_result=evals_result)

        p_valid += lgb_clf.predict(X_valid)

        yp += lgb_clf.predict(X_test)

        

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = predictors

    fold_importance_df["importance"] = lgb_clf.feature_importance()

    fold_importance_df["fold"] = fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    oof['predict'][val_ind] = p_valid / N

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    

    predictions['fold{}'.format(fold + 1)] = yp / N

mean_auc = np.mean(val_aucs)

std_auc = np.std(val_aucs)

all_auc = roc_auc_score(oof['target'], oof['predict'])



print('Mean AUC: {}, std: {}.\nAll AUC: {}.'.format(mean_auc, std_auc, all_auc))
cols = (feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]



plt.figure(figsize=(15, 150))

sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

plt.title('LightGBM Features (Averaged over Folds)')

plt.show()
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)

predictions.to_csv('predictions.csv', index=None)

sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})

sub_df["target"] = predictions['target']

sub_df.to_csv("lgb_submission.csv", index=False)

oof.to_csv('lgb_oof.csv', index=False)