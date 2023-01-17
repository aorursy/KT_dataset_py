# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
PATH = "../input/"

train = pd.read_csv(PATH+'train.csv')
test = pd.read_csv(PATH+'test.csv')
df = train.copy()
df.head()
df.info()
df.shape
df = df[df.target >= -20]
df['first_active_month'] = pd.to_datetime(df['first_active_month'])
features = pd.get_dummies(df['first_active_month'].dt.month,
              prefix='month')
# features = pd.concat([features, 
#                 pd.get_dummies(df['first_active_month'].dt.year,
#               prefix='year')], 1)
for feature in ['feature_1',
               'feature_2',
               'feature_3']:
    features = pd.concat([features,
               pd.get_dummies(df[feature],
                  prefix=feature)], 1)
features.head()
sns.distplot(df['target'])

plt.show()
df.target.quantile(.9995)
sns.distplot(df.target)
X = normalize(features)
y = df['target']
pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)
plt.figure(dpi=100, figsize=(10, 8))
sns.heatmap(features.corr())
# combine feature_3_0 and feature_3_1
# features 2_1 and 2_2
# and features 1_2 and 1_3 with features 3_0 and 3_1
features['feature_30_31'] = features['feature_3_0'] + features['feature_3_1']
features['feature_21_22'] = features['feature_2_1'] + features['feature_2_2']
features['feature_12_13_21_22'] = features['feature_1_2']+features['feature_1_3']+features['feature_2_1']+features['feature_2_2']
features = features.drop(['feature_3_0', 
                          'feature_3_1',
                         'feature_2_1',
                          'feature_2_2'], 1)
features = features.drop('feature_30_31', 1)
plt.figure(dpi=100, figsize=(10, 8))
sns.heatmap(features.corr())
features['feature_21_22'].value_counts()
features['feature_2_3'].value_counts()
features['feature_21_22_2_3'] = np.where((features['feature_21_22']==1)|(features['feature_2_3']==1), 1, 0)
features = features.drop(['feature_21_22', 'feature_2_3'], 1)
plt.figure(dpi=100, figsize=(10, 8))
sns.heatmap(features.corr())
X = features
y = df['target']

lr = LinearRegression().fit(X,y)
y_pred = lr.predict(X)
mean_squared_error(y, y_pred)
# test ETL

df = test.copy()
df['first_active_month'] = pd.to_datetime(df['first_active_month'])

features = pd.get_dummies(df['first_active_month'].dt.month,
              prefix='month')

for feature in ['feature_1',
               'feature_2',
               'feature_3']:
    features = pd.concat([features,
               pd.get_dummies(df[feature],
                  prefix=feature)], 1)
    
features['feature_30_31'] = features['feature_3_0'] + features['feature_3_1']

features['feature_21_22'] = features['feature_2_1'] + features['feature_2_2']

features['feature_12_13_21_22'] = features['feature_1_2']+features['feature_1_3']+features['feature_2_1']+features['feature_2_2']

features = features.drop(['feature_3_0', 
                          'feature_3_1',
                         'feature_2_1',
                          'feature_2_2'], 1)
features = features.drop('feature_30_31', 1)

features['feature_21_22_2_3'] = np.where((features['feature_21_22']==1)|(features['feature_2_3']==1), 1, 0)
features = features.drop(['feature_21_22', 'feature_2_3'], 1)

X_test = features

y_pred = lr.predict(X_test)
preds = pd.DataFrame(test['card_id'], columns=['card_id'])

preds['target'] = y_pred
my_submission = preds.copy()
my_submission.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor().fit(X, y)

rfr.score(X, y)
y_pred = rfr.predict(X_test)

preds = pd.DataFrame(test['card_id'], columns=['card_id'])

preds['target'] = y_pred

my_submission = preds.copy()
my_submission.to_csv('rfr_submission.csv', index=False)
