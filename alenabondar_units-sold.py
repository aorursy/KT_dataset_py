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
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
df.info()
df = df.drop(columns=['currency_buyer', 'product_url', 'theme', 'merchant_profile_picture',
                     'merchant_has_profile_picture', ])
df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates()
df['markup'] = 100 - (100 * df['price'] / df['retail_price'])
plt.figure(figsize=(12,4))
sns.set(style='whitegrid')
ax = sns.boxplot(x='markup', data=df)
df.groupby('units_sold', as_index=False).aggregate({'markup': ['mean', 'count','std'],
                                                    'rating': ['mean', 'count','std'],
                                                    'merchant_rating': ['mean', 'count','std']})
%config InlineBackend.figure_format = 'png' 
sns.pairplot(df[['price', 'retail_price', 'units_sold', 'rating','rating_count', 
                 'merchant_rating_count', 'merchant_rating']]);
data = df[['price', 'retail_price', 'units_sold', 'rating','rating_count', 
                 'merchant_rating_count', 'merchant_rating']]
plt.subplots(figsize=(12,9))
ax = sns.countplot(x='units_sold', data=data)
corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
from sklearn.model_selection import train_test_split
X = data.drop(['units_sold'], axis=1)
y = data['units_sold']
y.shape, X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate
# from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
%pylab inline
cv = KFold(shuffle=True, random_state=42)
n_trees = range(100, 1000, 50)
rf_scoring = []
for n_tree in n_trees:
    rf_clf = RandomForestClassifier(n_estimators=n_tree, min_samples_split=5, random_state=42)
    score = cross_val_score(rf_clf, X_train.values, y_train.values, cv=cv, scoring='accuracy')    
    rf_scoring.append(score)
rf_scoring = np.asmatrix(rf_scoring)
pylab.plot(n_trees, rf_scoring.mean(axis = 1), marker='.', label='RandomForestClassifier')
pylab.grid(True)
pylab.xlabel('n_trees')
pylab.ylabel('score')
pylab.title('Accuracy score')
pylab.legend(loc='lower right')
import yellowbrick as yb
from yellowbrick.model_selection import FeatureImportances
fig, ах= plt.subplots(figsize=(6, 4))
fi_viz = FeatureImportances(rf_clf)
fi_viz.fit(X_train, y_train)
fi_viz.poof ()
gb_scoring = []
for n_tree in n_trees:
    estimator = GradientBoostingClassifier(n_estimators=n_tree, learning_rate=0.2)
    
    score = cross_val_score(estimator, X_train.values, y_train.values, cv=cv, 
                                             scoring='accuracy')    
    gb_scoring.append(score)
gb_scoring = np.asmatrix(gb_scoring)
pylab.plot(n_trees, rf_scoring.mean(axis = 1), marker='.', label='RandomForestClassifier')
pylab.plot(n_trees, gb_scoring.mean(axis = 1), marker='.', label='GradientBoostingClassifier')
pylab.grid(True)
pylab.xlabel('n_trees')
pylab.ylabel('score')
pylab.title('Accuracy score')
pylab.legend(loc='lower right')
xgb_scoring = []
for n_tree in n_trees:
    estimator = XGBClassifier(n_estimators=n_tree, learning_rate=0.2, max_depth=3,subsample=0.7, 
                              objective='binary:logistic', random_state=42)
    score = cross_val_score(estimator, X_train.values, y_train.values, cv=cv, 
                                             scoring='accuracy')    
    xgb_scoring.append(score)
xgb_scoring = np.asmatrix(xgb_scoring)
pylab.plot(n_trees, rf_scoring.mean(axis = 1), marker='.', label='RandomForestClassifier')
pylab.plot(n_trees, gb_scoring.mean(axis = 1), marker='.', label='GradientBoostingClassifier')
pylab.plot(n_trees, xgb_scoring.mean(axis = 1), marker='.', label='XGBoost')
pylab.grid(True)
pylab.xlabel('n_trees')
pylab.ylabel('score')
pylab.title('Accuracy score')
pylab.legend(loc='lower right')