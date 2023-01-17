# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import scipy.stats as stats

import matplotlib.ticker as mtick

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.info()
train.head()
train.describe()
train.describe(include=['object'])
null_count = train.isnull().sum()
target = 'SalePrice' # Let's store our predicted variable name

# We move to log space to remove the skew

price = pd.DataFrame({'price' : train[target], 

                      'log(price)': np.log(train[target]), 

                      'log(1+price)': np.log1p(train[target])})

price.hist(bins=20)

price.skew()
train[target] = np.log1p(train[target])



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                    test.loc[:,'MSSubClass':'SaleCondition']))

numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index


fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(9, 8))

for i in range(len(axs)):

    sns.distplot(train[numeric_feats[i]], ax=axs[i], hist=False)

    axs[i].set(ylabel='Density')

    axs[i].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
corr_matrix_pearson = train[numeric_feats].corr('pearson').abs()

corr_matrix_spearman = train[numeric_feats].corr('spearman').abs()
plt.subplots(figsize=(9, 8))



sns.heatmap(corr_matrix_pearson)

plt.subplots(figsize=(9, 8))



sns.heatmap(corr_matrix_spearman)
skewed_feats = train[numeric_feats].apply(lambda x: stats.skew(x.dropna()))
skew_list = []

skew_labels = []

for idx, val in skewed_feats.iteritems():

    skew_list.append(val)

    skew_labels.append(idx)

  

plt.figure(figsize=(9,8))

plt.xticks(range(len(skew_labels)), skew_labels, rotation=44, size='xx-small')

plt.plot(skew_list, 'bo-')

plt.plot([1 for s in skewed_feats], 'r--')

plt.show()
skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

all_data_skewed = all_data.copy() # We save the skewed features.

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
corr_matrix_pearson = train[numeric_feats].corr('pearson').abs()

corr_matrix_spearman = train[numeric_feats].corr('spearman').abs()
c = corr_matrix_pearson[corr_matrix_pearson != 1].unstack().dropna() # We filter out the diagonal

c = c.sort_values(kind='quicksort')

grid = np.arange(20)/20

idx = np.digitize(c,grid)



plt.plot(grid, c.values)
all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())



all_data_skewed = pd.get_dummies(all_data_skewed)

all_data_skewed = all_data_skewed.fillna(all_data_skewed.mean())
X_train = all_data[:train.shape[0]]

X_skewed = all_data_skewed[:train.shape[0]]

X_test = all_data[train.shape[0]:]



Y = train[target]
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model, X=X_train):

    rmse= np.sqrt(-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

cv_ridge_skewed = [rmse_cv(Ridge(alpha = alpha), X_skewed).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title='Validation')

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge_skewed = pd.Series(cv_ridge_skewed, index = alphas)

cv_ridge_skewed.plot(title='Validation')

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
cv_ridge_skewed.min()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, Y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

print("Lasso picked " + str(sum(coef[numeric_feats] != 0)) + " numerical variables and eliminated the other " +  str(sum(coef[numeric_feats] == 0)) + " numerical variables")

imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":Y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x="preds", y="residuals", kind="scatter", alpha=0.2)