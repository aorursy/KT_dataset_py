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
# トレーニングデータ、テストデータ、サンプルサブミットデータを読み込み

train = pd.read_csv('../input/allstate-claims-severity/train.csv')

test = pd.read_csv('../input/allstate-claims-severity/test.csv')

sample_submission = pd.read_csv('../input/allstate-claims-severity/sample_submission.csv')
# trainとtestを縦に連結

df_full = pd.concat([train, test], axis=0, sort=False)

print(df_full.shape) # df_fullの行数と列数を確認

df_full.describe() # df_fullの要約統計量
from sklearn import preprocessing

for column in ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19', 'cat20', 'cat21', 'cat22', 'cat23', 'cat24', 'cat25', 'cat26', 'cat27', 'cat28', 'cat29', 'cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat40', 'cat41', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46', 'cat47', 'cat48', 'cat49', 'cat50', 'cat51', 'cat52', 'cat53', 'cat54', 'cat55', 'cat56', 'cat57', 'cat58', 'cat59', 'cat60', 'cat61', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66', 'cat67', 'cat68', 'cat69', 'cat70', 'cat71', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76', 'cat77', 'cat78', 'cat79', 'cat80', 'cat81', 'cat82', 'cat83', 'cat84', 'cat85', 'cat86', 'cat87', 'cat88', 'cat89', 'cat90', 'cat91', 'cat92', 'cat93', 'cat94', 'cat95', 'cat96', 'cat97', 'cat98', 'cat99', 'cat100', 'cat101', 'cat102', 'cat103', 'cat104', 'cat105', 'cat106', 'cat107', 'cat108', 'cat109', 'cat110', 'cat111', 'cat112', 'cat113', 'cat114', 'cat115', 'cat116']:

    le = preprocessing.LabelEncoder()

    le.fit(df_full[column])

    train[column] = le.transform(train[column])

    test[column] = le.transform(test[column])
X_train = train.drop(['loss'], axis=1)

y_train = train['loss']
# machine learning

from sklearn import linear_model

from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import mean_absolute_error
# Ridge Regressor

ridge_reg= linear_model.Ridge(alpha=1.0)

ridge_reg.fit(X_train, y_train)

pred_train = ridge_reg.predict(X_train)

mae_ridge = mean_absolute_error(pred_train, y_train)



pred = ridge_reg.predict(test)
sample_submission['loss'] = pred

sample_submission.to_csv('ridge_reg.csv', index=False)
# Lasso Regressor

lasso_reg= linear_model.Lasso(alpha=1.0)

lasso_reg.fit(X_train, y_train)

pred_train = lasso_reg.predict(X_train)

mae_lasso = mean_absolute_error(pred_train, y_train)



pred = lasso_reg.predict(test)
sample_submission['loss'] = pred

sample_submission.to_csv('lasso_reg.csv', index=False)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LassoCV

from sklearn.metrics import mean_absolute_error



# 5-Cross-validation

train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=0)



lassocv_reg = LassoCV(alphas=10 ** np.arange(-6, 2, 0.1), cv=5)

lassocv_reg.fit(train_x, train_y)



pred = lassocv_reg.predict(train_x)

mae = mean_absolute_error(train_y, pred)
pred_train = lassocv_reg.predict(X_train)

mae_lassocv = mean_absolute_error(pred_train, y_train)



pred = lassocv_reg.predict(test)
sample_submission['loss'] = pred

sample_submission.to_csv('lassocv_reg.csv', index=False)
models = pd.DataFrame({

    'Model': ['Ridge Regression', 'Lasso Regression', 'Lasso Regression/CV'],

    'Score': [mae_ridge, mae_lasso, mae_lascv]})

models.sort_values(by='Score', ascending=False)