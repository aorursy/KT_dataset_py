# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

import seaborn as sns
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
!head ../input/sample_submission.csv
train_df.head()
cat_features = ['MSZoning','Street','Alley','LotShape','LandContour',

                'Utilities','LotConfig','LandSlope','Neighborhood','Condition1',

                'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

                'Exterior1st','Exterior2nd','MasVnrType','ExterQual',

                'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

                'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical',

                'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',

                'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
true_num_features = ['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',

                     'PoolArea','MiscVal','MoSold','LotArea','BsmtFinSF1','BsmtFinSF2',

                     'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                     'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',

                     'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','LotFrontage','MasVnrArea']
cat_num_features = ['MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd','YrSold']
for c in cat_features:

    print("{} {}".format(c,len(train_df[pd.isnull(train_df[c])])/len(train_df) ) )
train_df['SalePrice'] = train_df['SalePrice'].astype('float32')
plt.clf()

for i, feat_name in enumerate(['PoolQC','Fence','Alley','MiscFeature']):

    plt.figure(i,figsize=(10,5))

    plt.title(feat_name)    

    for group_name, group in train_df[~pd.isnull(train_df[feat_name])][[feat_name,'SalePrice']].groupby(feat_name):    

        if len(group) >1 :

            sns.distplot(group['SalePrice'],kde=False,label=group_name,norm_hist=True)

    plt.legend()

    plt.show()

plt.clf()

for i, feat_name in enumerate(cat_features):

    plt.figure(i,figsize=(10,5))

    plt.title(feat_name)    

    for group_name, group in train_df[~pd.isnull(train_df[feat_name])][[feat_name,'SalePrice']].groupby(feat_name):    

#         print(group)

        if len(group) >1 :

            sns.distplot(group['SalePrice'],kde=False,label=group_name,norm_hist=True)

    plt.legend()

    plt.show()

for feat_name in true_num_features:

    feat_non_null = train_df[~pd.isnull(train_df[feat_name])][[feat_name,'SalePrice']]

    print('Pearson Correlation coefficient between {} and {}: {}'.format('SalesPrice',feat_name,scipy.stats.pearsonr(feat_non_null[feat_name],feat_non_null['SalePrice'])))
filtered_cat_features = ['MSZoning','Street','LotShape','LandContour',

                'Utilities','LotConfig','LandSlope','Neighborhood','Condition1',

                'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

                'Exterior1st','Exterior2nd','MasVnrType','ExterQual',

                'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

                'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical',

                'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',

                'GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']







filtered_num_features = ['WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch',

                     'LotArea','BsmtFinSF1',

                     'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',

                     'BsmtFullBath','FullBath','HalfBath','BedroomAbvGr',

                     'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','LotFrontage','MasVnrArea']



filtered_cat_num_features = ['MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd']
filtered_train_df = train_df[filtered_cat_features+filtered_num_features+filtered_cat_num_features+['SalePrice','Id']]

filtered_test_df = test_df[filtered_cat_features+filtered_num_features+filtered_cat_num_features+['Id']]
filtered_train_df['SalePrice'] = np.log1p(filtered_train_df['SalePrice'])
for feat in filtered_cat_features:

    filtered_train_df[feat].fillna('nan',inplace=True)

    filtered_test_df[feat].fillna('nan',inplace=True)

    

for feat in filtered_num_features:

    feat_mean = filtered_train_df[feat].mean()

    filtered_train_df[feat].fillna(feat_mean,inplace=True)

    filtered_test_df[feat].fillna(feat_mean,inplace=True) # Yes, let's fill the test set with the mean of train set, otherwise it would be cheat.    

    

for feat in filtered_cat_num_features:

    feat_mean = str(int(filtered_train_df[feat].mean()))

    filtered_train_df[feat].fillna(feat_mean,inplace=True)

    filtered_test_df[feat].fillna(feat_mean,inplace=True)
for feat in filtered_num_features:

    feat_mean = filtered_train_df[feat].mean()

    feat_std = filtered_train_df[feat].std()

    filtered_train_df[feat] = filtered_train_df[feat].apply(lambda x: (x-feat_mean)/feat_std )

    filtered_test_df[feat] = filtered_test_df[feat].apply(lambda x: (x-feat_mean)/feat_std )
import tensorflow as tf
feat_column_cat = []



for feat_name in filtered_cat_features+filtered_cat_num_features:

    feat_column_cat.append(tf.feature_column.categorical_column_with_vocabulary_list(feat_name,list(filtered_train_df[feat_name].unique())))
feat_column_num = []



for feat_name in filtered_num_features:

    feat_column_num.append(tf.feature_column.numeric_column(feat_name))
len(feat_column_num)
train_input_fn = tf.estimator.inputs.pandas_input_fn(filtered_train_df.drop('SalePrice',axis=1),filtered_train_df['SalePrice'],num_epochs=300,batch_size=128,shuffle=True)

test_input_fn = tf.estimator.inputs.pandas_input_fn(filtered_test_df,batch_size=128,shuffle=True)
model = tf.estimator.DNNLinearCombinedRegressor('./wide_n_deep/',

                                               linear_feature_columns=feat_column_cat,

                                               dnn_feature_columns=feat_column_num,

                                               dnn_hidden_units=[23,16,4])
model = tf.contrib.estimator.forward_features(model,'Id')
!rm -rf wide_n_deep/
model.train(train_input_fn)
submission_dict = {'Id':[],'SalePrice':[]}

for prediction in model.predict(test_input_fn):    

    submission_dict['Id'].append(prediction['Id'])

    submission_dict['SalePrice'].append(np.expm1(prediction['predictions'][0]))

submission = pd.DataFrame(submission_dict)
submission.head()
submission.to_csv('submussion.csv',index=False)