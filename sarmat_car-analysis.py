# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
%pylab inline
pylab.rcParams['figure.figsize'] = (20, 10)
data = pd.read_csv('../input/car_ad.csv',encoding ='iso-8859-9')
data.head()
# Any results you write to the current directory are saved as output.
data.describe()
data = data[(data['year'] >= 1980) & (data['year'] < 2016)]
data = data[(data['price'] >= 500) & (data['price'] < 200000)]
data['price'].plot.kde()
data['drive'].value_counts().plot(kind='bar')
data['car'].value_counts().plot(kind='bar')
data['engType'].value_counts().plot(kind='bar')
data['registration'].value_counts().plot(kind = 'bar')
data.groupby("car")['model'].value_counts().plot(kind='bar')
feats = ['price', 'year', 'mileage']
fig, axes = plt.subplots(ncols=len(feats), nrows=1, figsize=(18,6))

for i, feat in enumerate(feats):
    sns.boxplot(data[feat], ax=axes[i], orient='v', width=0.5, color='g');
    axes[i].set_ylabel('')
    axes[i].set_title(feat)
from matplotlib.ticker import FuncFormatter
feats = ['price', 'year', 'mileage']

fig, axis = plt.subplots(ncols=3, figsize=(18, 6))
for i, feat in enumerate(feats):
    sns.boxplot(np.log(data[feat]), ax=axis[i], orient='v', width=0.5, color='g');
    y_formatter = FuncFormatter(lambda x, pos: ('%i')%(np.exp(x)))
    axis[i].yaxis.set_major_formatter(y_formatter)
data = data.fillna(0)
sns.heatmap(data.corr()\
            , annot=True, fmt='.2f')
fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sns.boxplot(x='engType', y='price', data=data, ax=axis[0]);
sns.boxplot(x='drive', y='price', data=data, ax=axis[1])
sns.boxplot(x='body', y='price', data=data)
fig, axis = plt.subplots(figsize=(20, 8), )
sns.boxplot(x='car', y='price', data=data);
axis.set_xticklabels(data['car'].unique(), rotation=80);
y = np.log1p(data['price'])
X = data.drop(['price'],axis=1)
for cat_feature in X.columns[X.dtypes == 'object']:
    X[cat_feature] = X[cat_feature].astype('category')  
    X[cat_feature].cat.set_categories(X[cat_feature].unique(), inplace=True)
X = pd.get_dummies(X, columns=X.columns[X.dtypes == 'category'])
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42)
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Start training... ")
gbm = lgb.LGBMRegressor(max_depth=-1, learning_rate=0.05, silent=False, metric='mean_absolute_error', 
                        n_jobs=10, n_estimators=1000,
                        verbose = 10)
gbm.fit(X_train,y_train)

y_tr = gbm.predict(X_test)
print( '\tMAE: ',  mean_absolute_error(y_tr, y_test )) 
print('Start predicting...')
y_preds = gbm.predict(X_test)
print('LGBM:')
print('\tMAE: ', mean_absolute_error(y_test, y_preds))
print('\tR2: ', r2_score(y_test, y_preds))
testing = X
testing = X.loc[(data['car'] == 'Audi') & (data['model'] == 'A4')]
pr = gbm.predict(testing)
ind = testing.index.values
show = data.loc[ind]
show['price'] = np.exp(pr)
show.head(20)