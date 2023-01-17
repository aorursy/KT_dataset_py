import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn')

%matplotlib inline

import seaborn as sns

import sklearn
pd.set_option('display.max_columns', 100)

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_train.columns
df_train.info()
df_train.describe()
train_miss = df_train.isnull().sum().sort_values(ascending=False)

train_miss_pct = df_train.isnull().sum().sort_values(ascending=False) / len(df_train)

df_train_miss = pd.concat([train_miss, train_miss_pct], axis=1, keys=['Total', 'Percent'])

df_train_miss = df_train_miss[df_train_miss['Total']>0]

df_train_miss
df_train = df_train.drop(df_train_miss.index[:6].tolist(), axis=1)
num_features = df_train.select_dtypes(['int64', 'float64']).columns

num_features
cat_features = df_train.select_dtypes(['object']).columns

cat_features
from IPython.display import display

for feat in num_features:

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 

                                        gridspec_kw={"height_ratios": (.15, .85),

                                                     'wspace': 0, 'hspace': 0})

    sns.boxplot(df_train[feat].dropna(),ax=ax_box)

    ax_box.set_xlabel('', visible=False)

    sns.distplot(df_train[feat].dropna(), kde=False, ax=ax_hist)

    ax_hist.set_yticks([])

    plt.tight_layout()

    display(df_train[[feat]].describe().T)

    plt.show()
for catg in list(cat_features) :

    print(df_train[catg].value_counts())

    print('='*50)
for cat_feat in cat_features:

    plt.figure(figsize=(13, 4))

    sns.countplot(df_train[cat_feat].dropna(), order = df_train[cat_feat].value_counts().index);
df_train[num_features].corr()['SalePrice'].sort_values(ascending=False)
corrmat = df_train[num_features].corr()

mask = np.zeros_like(corrmat)

mask[np.triu_indices_from(mask)] = True

mask[np.diag_indices_from(mask)] = False

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, mask=mask, linewidths=.5, vmax=0.9, 

            square=True, cmap="RdBu_r", center=0)
sns.clustermap(corrmat, linewidths=.5, vmax=0.9, 

               square=True, cmap="RdBu_r", center=0)
high_corr_feats = df_train.corr('spearman')['SalePrice'].abs().sort_values(ascending=False).index[1:11].tolist()

sns.pairplot(df_train[['SalePrice'] + high_corr_feats])
import statsmodels.api as sm

feat = high_corr_feats[-1]

ols_model = sm.OLS(np.log(df_train["SalePrice"]), sm.add_constant(df_train[feat]),                            

                   missing='drop')

ols_results = ols_model.fit()

print(ols_results.params)

slope, intercept = ols_results.params[1], ols_results.params[0]

print(intercept)

# print(slope, intercept)
import statsmodels.api as sm

from scipy.stats import pearsonr

for feat in high_corr_feats:

    # get coeffs of linear fit

    ols_model = sm.OLS(np.log(df_train["SalePrice"]), sm.add_constant(df_train[feat]),                            

                       missing='drop')

    ols_results = ols_model.fit()

    slope, intercept = ols_results.params[1], ols_results.params[0]

    r2 = ols_results.rsquared

    print(slope, intercept, r2)

    sns.jointplot(df_train[feat], np.log(df_train["SalePrice"]), kind="reg",

                  line_kws={'label': f"y={intercept:.2f}+{slope:.2f}x, r2: {r2:.2f}", 

                            'color': 'r'})

    plt.legend()

    plt.show()
## Robust Linear Regression with Huber Loss (robust to outliers)

import statsmodels.api as sm

from scipy.stats import pearsonr

for feat in high_corr_feats:

    # get coeffs of linear fit

    rlm_model = sm.RLM(np.log(df_train["SalePrice"]), sm.add_constant(df_train[feat]),                            

                       M=sm.robust.norms.HuberT(), missing='drop')

    rlm_results = rlm_model.fit()

    slope, intercept = rlm_results.params[1], rlm_results.params[0]

    r2 = pearsonr(rlm_results.fittedvalues, rlm_model.endog)[0] ** 2

    print(slope, intercept, r2)

    sns.jointplot(df_train[feat], np.log(df_train["SalePrice"]), kind="reg", robust=True,

                  line_kws={'label': f"y={intercept:.2f}+{slope:.2f}x, r2: {r2:.2f}"})

    plt.legend()

    plt.show()