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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

train_df.head()
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

test_df.head()
train_df.describe()
print('Shapes: \nTrain:{} \nTest{}'.format(train_df.shape, test_df.shape))
df = pd.concat([train_df, test_df], sort=True)
df.head()
df.info()
train_df.plot('GrLivArea', 'SalePrice', kind='scatter')
train_df.plot('TotalBsmtSF', 'SalePrice', kind='scatter')
train_df.plot('GarageArea', 'SalePrice', kind='scatter')
df = df[np.logical_and(df.GrLivArea < 4500, df.TotalBsmtSF < 4500)]
def plot_distrib(s):

    sns.distplot(s , fit=norm);



    # Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(s)



    #Now plot the distribution

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

                loc='best')

    plt.ylabel('Frequency')

    plt.title('SalePrice distribution')



    #Get also the QQ-plot

    fig = plt.figure()

    res = stats.probplot(s, plot=plt)

    plt.show()
plot_distrib(train_df.SalePrice)
# Skewer

plot_distrib(np.log1p(train_df.SalePrice))
df.info()
df.isnull().sum().sort_values(ascending=False).loc[lambda x: x > 0]
def clean_df(df):

    columns_to_drop = ['Alley', 'MiscFeature', 'Utilities']

    df = df.drop(columns=columns_to_drop)



    object_cols = df.select_dtypes(include='object').columns

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    

    # print('object columns: {}\nint64 columns: {}\nfloat64 columns: {}'.format(object_cols, df.select_dtypes(include=['int64']).columns, df.select_dtypes(include=['float64']).columns))

    df.LotFrontage = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    df[object_cols] = df[object_cols].fillna('None').astype('category')

    df[numerical_cols] = df[numerical_cols].apply(lambda x: x.fillna(x.median()))

    

    return df
df = clean_df(df)



df.head()
df.SalePrice =  np.log1p(df.SalePrice)
plot_distrib(df.SalePrice)
df.info()
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_log_error, r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV, KFold

import lightgbm as lgb
res = pd.get_dummies(df)

threshold = len(train_df.index) + 1

X = res.loc[:threshold].drop(columns='SalePrice')

y = res.loc[:threshold].SalePrice



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('X_train: {}, y_train: {} \nX_test: {} y_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
light = lgb.LGBMRegressor(objective='regression',num_leaves=5, n_estimators=787,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

steps = [('scaler', MinMaxScaler()), ('reg', light)]

pipeline = Pipeline(steps)
def rmsle_cv(model):

    kf = KFold(10, shuffle=True, random_state=42).get_n_splits(X)

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))

    

    return rmse
# The root mean squared log error

print('RMSLE: %.7f' % rmsle_cv(pipeline).mean())
pipeline.fit(X, y)



y_pred_test = pipeline.predict(res.loc[threshold - 2:].drop(columns='SalePrice'))
submit_df = pd.DataFrame({'Id': res.loc[threshold - 2:].index, 'SalePrice': np.expm1(y_pred_test)})



submit_df.to_csv('submission.csv', index=False)

submit_df.shape
submit_df.head()