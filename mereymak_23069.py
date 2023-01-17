# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import matplotlib.style as style
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import boxcox1p

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head(20)
train.info()
test_ID = test['Id']
del train['Id']
del test['Id']
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

varable = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[varable]], axis=1)
data.plot.scatter(x=varable, y='SalePrice');
train = train.drop(train[(train['TotalBsmtSF']>2800) & (train['SalePrice']<600000)].index)
fig, ax = plt.subplots()
ax.scatter(x = train['GarageArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)
plt.show()
train = train.drop(train[(train['GarageArea']>1200) & (train['SalePrice']<13.0)].index)
varable = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[varable]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=varable, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
train['SalePrice'] = np.log(train.SalePrice)
print ("Skew is:", train['SalePrice'].skew())
plt.hist(train['SalePrice'], color='blue')
plt.show()
all_data = pd.concat((train, test)).reset_index(drop=True)
corrmat = all_data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
corrmat = all_data.corr()
most_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(all_data[most_corr_features].corr(),annot=True,cmap="RdYlGn")
all_data.drop(columns={'PoolQC','MiscFeature','Alley','Fence','FireplaceQu'},inplace=True)
corr = all_data.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
