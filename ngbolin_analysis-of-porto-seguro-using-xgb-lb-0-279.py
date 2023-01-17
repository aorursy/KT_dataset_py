%matplotlib inline

import pandas as pd # Dataframe manipulation

import numpy as np 

import matplotlib.pyplot as plt # Base plotting

import seaborn as sns # Sophisticated plotting (?)

import warnings

# Ignore all warnings - users beware

warnings.filterwarnings("ignore")
# Read dataframe into Python

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# Combine the training and test dataset

df = pd.concat([df_train, df_test])
df.set_index('id', inplace = True)

df.head(5)
# print dimensions of dataframes

print(df.shape)

print(df_train.shape)

print(df_test.shape)
df.describe()
(pd.DataFrame(np.sum(df.apply(lambda x: x == -1))

              /len(df))[0][pd.DataFrame(np.sum(df.apply(lambda x: x == -1))/len(df))[0] != 0])
np.sum(pd.isnull(df))
categorical_features = df.columns[df.columns.str.endswith('cat')].tolist()

binary_features = df.columns[df.columns.str.endswith('bin')].tolist()

numeric_features = [feature for feature in df.columns.tolist()

                    if feature not in categorical_features and feature not in binary_features]
binary_numeric = binary_features + numeric_features
df[categorical_features].apply(set)
for feature in ['ps_car_02_cat', 'ps_car_03_cat', 'ps_car_05_cat', 

                'ps_car_07_cat', 'ps_car_08_cat', 'ps_ind_04_cat']:

    binary_numeric.append(feature)

    binary_features.append(feature)

    categorical_features.remove(feature)
categorical_features
df[df == -1] = np.nan
sns.set_style('white')

cmap = sns.diverging_palette(220, 10, as_cmap=True)



plt.figure(figsize=(20,15))



sns.heatmap(df[binary_numeric].corr(), vmin = -1, vmax = 1, cmap=cmap)



plt.show()
plt.figure(figsize=(20, 15))

(df.corr()

     .target

     .drop('target')

     .sort_values(ascending=False)

     .plot

     .barh())
print('No. of numeric features: %d' % len(numeric_features))

print('No. of binary features: %d' % len(binary_features))
plt.figure(figsize=(20,20))

for idx, num_feat in enumerate(numeric_features):

    plt.subplot(5, 6, idx+1)

    sns.distplot(df[num_feat].dropna(), kde = False, norm_hist=True)



plt.show()
plt.figure(figsize=(20,20))

for idx, bin_feat in enumerate(binary_features):

    plt.subplot(6, 4, idx+1)

    sns.distplot(df[bin_feat].dropna(), kde = False, norm_hist=True)



plt.show()
len(categorical_features)
plt.figure(figsize=(20,15))



for idx, cat_feat in enumerate(categorical_features):

    plt.subplot(4, 2, idx+1)

    sns.distplot(df[cat_feat].dropna(), kde=False, norm_hist=True)

    

plt.show()
plt.figure(figsize=(20,15))



for idx, cat_feat in enumerate(categorical_features):

    plt.subplot(4, 2, idx+1)

    sns.pointplot(x=cat_feat, y='target', data=df.iloc[:df_train.shape[0]])

    

plt.show()
fig, axs = plt.subplots(8, 1, figsize=(20, 25))



for ax, cat_feat in zip(axs, categorical_features):

    ax2 = ax.twinx()

    sns.distplot(df[cat_feat].dropna(), kde=False, norm_hist=True, ax = ax)

    sns.pointplot(x=cat_feat, y='target', data=df.iloc[:df_train.shape[0]], ax=ax2)

    

plt.show()
df[df == -1] = np.nan



# Binary and Numeric Features



no_of_features = sum(df[binary_numeric].corr()

                     .target

                     .abs()

                     .drop('target')

                     .sort_values(ascending=False) > 0.005)

no_of_features
bin_num_features = (df[binary_numeric].corr()

                    .target

                    .abs()

                    .drop('target')

                    .sort_values(ascending = False))[:no_of_features].index.tolist()
cat_features = [feature for feature in df.columns.tolist() 

                if (feature not in bin_num_features) and (feature.endswith('cat'))]
df_fs1 = df[bin_num_features + cat_features]



df_fs1['target'] = df.target

bin_num_feat = [column for column in df_fs1.columns 

                if column not in cat_features]
sns.set_style('white')

cmap = sns.diverging_palette(220, 10, as_cmap=True)



plt.figure(figsize=(20, 20))

sns.heatmap(df_fs1[bin_num_feat].iloc[:df_train.shape[0]].corr(), vmin = -1, vmax = 1, 

            annot = True, cmap = cmap)

plt.plot()
del df_fs1['ps_ind_14']
np.sum(df_fs1.isnull())
[feat for feat in df_fs1.columns.tolist() 

 if np.sum(pd.isnull(df_fs1[feat])) > (df_fs1.shape[0])*0.20]
del df_fs1['ps_car_03_cat']

del df_fs1['ps_car_05_cat']
[feat for feat in df_fs1.columns.tolist() 

 if (feat.endswith('cat'))  and ((np.sum(pd.isnull(df_fs1[feat]))) > 0)]
df_fs1.ps_car_02_cat.fillna('-1', inplace = True)

df_fs1.ps_car_07_cat.fillna('-1', inplace = True)

df_fs1.ps_ind_04_cat.fillna('-1', inplace = True)

df_fs1.ps_car_01_cat.fillna('-1', inplace = True)

df_fs1.ps_car_09_cat.fillna('-1', inplace = True)

df_fs1.ps_ind_02_cat.fillna('-1', inplace = True)

df_fs1.ps_ind_05_cat.fillna('-1', inplace = True)
[feat for feat in df_fs1.columns.tolist() 

 if np.sum(pd.isnull(df_fs1[feat])) > 0]
df_fs1['ps_car_12'].fillna(df_fs1['ps_car_12'].median(), inplace = True)

df_fs1['ps_reg_03'].fillna(df_fs1['ps_reg_03'].median(), inplace = True)

df_fs1['ps_car_14'].fillna(df_fs1['ps_car_14'].median(), inplace = True)
np.sum(df_fs1.isnull())
features = np.array([feature for feature in df_fs1.columns.tolist() 

                     if feature != 'target'])
random_state = 1212
idx = df_fs1[df_fs1.target.notnull()].index.tolist()



from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(50, random_state=random_state)

clf.fit(df_fs1[features].loc[idx], df_fs1.target.loc[idx])
importances = clf.feature_importances_

sorted_idx = np.argsort(importances)



plt.figure(figsize=(15, 10))



padding = np.arange(len(features)) + 0.5

plt.barh(padding, importances[sorted_idx], align='center')

plt.yticks(padding, features[sorted_idx])

plt.xlabel("Relative Importance")

plt.title("Variable Importance")



plt.show()
combined = df_fs1[features]

combined['target'] = df_train.set_index('id').target
plt.figure(figsize=(20,20))



plt.subplot(221)

sns.distplot(combined[combined.target == 0].ps_car_13.dropna(),

             bins = np.linspace(0, 4, 41), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_car_13.dropna(),

             bins = np.linspace(0, 4, 41), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_car_13 Distribution')



plt.subplot(222)

sns.distplot(combined[combined.target == 0].ps_reg_03,

             bins = np.linspace(0, 2, 11), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_reg_03,

             bins = np.linspace(0, 2, 11), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_reg_03 Distribution')



plt.subplot(223)

sns.distplot(combined[combined.target == 0].ps_car_14,

             bins = np.linspace(0.2, 0.6, 10), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_car_14, 

             bins = np.linspace(0.2, 0.6, 10), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_car_14 Distribution')



plt.subplot(224)

sns.distplot(combined[combined.target == 0].ps_ind_15.dropna(),

             bins = np.linspace(0, 15, 16), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_ind_15.dropna(),

             bins = np.linspace(0, 15, 16), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_ind_15 Distribution')
plt.figure(figsize=(20,15))



plt.subplot(221)

sns.distplot(combined[combined.target == 0].ps_ind_03.dropna(),

             bins = range(0, 8, 1), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_ind_03.dropna(),

             bins = range(0, 8, 1), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_ind_03 Distribution')



plt.subplot(222)

sns.distplot(combined[combined.target == 0].ps_reg_02.dropna(),

             bins = np.linspace(0, 2, 11), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_reg_02.dropna(),

             bins = np.linspace(0, 2, 11), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_reg_02 Distribution')



plt.subplot(223)

sns.distplot(combined[combined.target == 0].ps_car_11_cat.dropna(), 

             bins = range(0, 110, 5), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_car_11_cat.dropna(), 

             bins = range(0, 110, 5), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_car_11_cat Distribution')



plt.subplot(224)

sns.distplot(combined[combined.target == 0].ps_ind_01.dropna(),

             bins = range(0, 8, 1), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_ind_01.dropna(),

             bins = range(0, 8, 1), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_ind_01 Distribution')
plt.figure(figsize=(20,15))



plt.subplot(221)

sns.distplot(combined[combined.target == 0].ps_car_15.dropna(), 

             kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_car_15.dropna(), 

             kde = False, norm_hist = True, color = 'blue')

plt.title('ps_car_15 Distribution')



plt.subplot(222)

sns.distplot(combined[combined.target == 0].ps_reg_01.dropna().astype('float'),

             bins = range(0, 11, 1), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_reg_01.dropna().astype('float'),

             bins = range(0, 11, 1), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_reg_01 Distribution')



plt.subplot(223)

sns.distplot(combined[combined.target == 0].ps_car_01_cat.dropna().astype('float'), 

             bins = range(-1, 11, 1), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_car_01_cat.dropna().astype('float'), 

             bins = range(-1, 11, 1), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_car_01_cat Distribution')



plt.subplot(224)

sns.distplot(combined[combined.target == 0].ps_car_06_cat.dropna(), 

             bins = range(0, 17, 1), kde = False, norm_hist = True, color = 'red')

sns.distplot(combined[combined.target == 1].ps_car_06_cat.dropna(), 

             bins = range(0, 17, 1), kde = False, norm_hist = True, color = 'blue')

plt.title('ps_car_06_cat Distribution')
combined['target'] = df_train.set_index('id').target



plt.figure(figsize=(20, 15))

sns.heatmap(combined.corr(), annot = True, cmap = cmap)

plt.show()
ind_var = [feature for feature in combined.columns[sorted_idx][-10:] 

           if feature != 'target']

ind_var.reverse()
from sklearn.preprocessing import PolynomialFeatures



train = combined[pd.notnull(combined.target)][ind_var].reset_index(drop=True)



poly = PolynomialFeatures(interaction_only = True, include_bias = False)



train_interaction = pd.DataFrame(poly.fit_transform(train))

train_interaction['target'] = df_train.target
features = np.array([feature for feature in train_interaction.columns.tolist()

                     if feature != 'target'])



clf = RandomForestClassifier(50, random_state = random_state)

clf.fit(train_interaction.iloc[:df_train.shape[0]][features], 

        train_interaction.iloc[:df_train.shape[0]]['target'])
importances = clf.feature_importances_

sorted_idx = np.argsort(importances)



plt.figure(figsize=(15, 10))



plt.figure(figsize=(20, 20))

padding = np.arange(len(features)) + 0.5

plt.barh(padding, importances[sorted_idx], align='center')

plt.yticks(padding, features[sorted_idx])

plt.xlabel("Relative Importance")

plt.title("Variable Importance")



plt.show()
[feat for feat in ind_var]
combined['feature10'] = combined['ps_car_13'] * combined['ps_reg_03']
combined['target'] = df_train.set_index('id').target



plt.figure(figsize=(20, 20))

sns.heatmap(combined.corr(), annot = True)

plt.show()
features = np.array([feature for feature in combined.columns.tolist()

                     if feature != 'target'])



clf = RandomForestClassifier(50, random_state = random_state)

clf.fit(combined[pd.notnull(combined.target)][features], 

        combined[pd.notnull(combined.target)].target)
importances = clf.feature_importances_

sorted_idx = np.argsort(importances)



plt.figure(figsize=(15, 10))



plt.figure(figsize=(20, 20))

padding = np.arange(len(features)) + 0.5

plt.barh(padding, importances[sorted_idx], align='center')

plt.yticks(padding, features[sorted_idx])

plt.xlabel("Relative Importance")

plt.title("Variable Importance")



plt.show()
del combined['target']
X_train = combined.reset_index(drop = True).iloc[:df_train.shape[0], ]

X_test = combined.reset_index(drop = True).iloc[df_train.shape[0]:, ]
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score
features = X_train.columns.tolist



X = X_train.values; test = X_test.values



y = df_train.set_index('id').target.values
params = {

    'objective': 'binary:logistic',

    'min_child_weight': 12.0,

    'max_depth': 5,

    'colsample_bytree': 0.5,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.8,

    'max_delta_step': 1.5

}
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold



submission = pd.DataFrame()

submission['id'] = df_test['id'].values

submission['target'] = 0



nrounds=1000

folds = 5

skf = StratifiedKFold(n_splits=folds, random_state=random_state)



for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print('XGB KFold: %d: ' % int(i+1))

    

    X_subtrain, X_subtest = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    

    d_subtrain = xgb.DMatrix(X_subtrain, y_train) 

    d_subtest = xgb.DMatrix(X_subtest, y_valid) 

    d_test = xgb.DMatrix(test)

    

    watchlist = [(d_subtrain, 'subtrain'), (d_subtest, 'subtest')]

    

    mdl = xgb.train(params, d_subtrain, nrounds, watchlist, early_stopping_rounds=80, 

                    feval=gini_xgb, maximize=True, verbose_eval=50)

    

    # Predict test set based on the best_ntree_limit

    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)

    

    # Take the average of the prediction via 5 folds to predict for the test set

    submission['target'] += p_test/folds
submission.to_csv('submission.csv', index=False)