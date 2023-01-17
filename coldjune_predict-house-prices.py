# 导入相关数据包

import pandas as pd

import numpy as np

import seaborn as sbn

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data.head()
data = data.set_index(['Id'])

x_train, y_train = data.loc[:,:'SaleCondition'], data['SalePrice']
x_train.info()
x_train.describe()
fig = plt.figure(figsize=(24, 24))

count = 1

for x in x_train[x_train.columns[x_train.dtypes != 'object']]:

    ax = fig.add_subplot(8,5, count)

    ax.boxplot(x_train[x])

    ax.set_title(x)

    count += 1
x_train.hist(figsize=(24, 24))
plt.figure(figsize=(24, 24))

sbn.heatmap(x_train.corr(), linewidths=0.5)
fig = plt.figure(figsize=(24, 48))

count = 1

for x in x_train.columns[x_train.dtypes == 'object']:

    ax = fig.add_subplot(15, 3, count)

    temp_feature = x_train[x].value_counts()

    feature_bar = ax.bar(range(temp_feature.shape[0]), temp_feature.values,  align='center')

    ax.set_xticks(np.arange(temp_feature.shape[0]))

    if temp_feature.shape[0] > 10:

        indexs = [index[-2:] for index in temp_feature.index]

        ax.set_xticklabels(indexs)

    else:

        ax.set_xticklabels(temp_feature.index)

    for bar in feature_bar:

        height = bar.get_height()

        ax.text(bar.get_x()+bar.get_width()/2-0.1, 1.1*height, str(height))

    ax.set_ylim(0, 1.2 * temp_feature.values[0])

    ax.set_title(x+'('+str(np.sum(temp_feature))+')')

    count+=1

    

plt.subplots_adjust(hspace=0.9, bottom=0.1)



    
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureSelect(BaseEstimator, TransformerMixin):

    def __init__(self, obj=True):

        self.obj = obj

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return X[X.columns[X.dtypes == 'object']] if self.obj else X[X.columns[X.dtypes != 'object']] 

    

class StringImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                        index=X.columns)

        return self

    

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)

    

class DropFeature(BaseEstimator, TransformerMixin):

    def __init__(self, features):

        self.features = features

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return X.drop(self.features, axis=1)
from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

numeric_pipeline = Pipeline([

        ('drop', DropFeature(['Alley','PoolQC', 'Fence', 'MiscFeature'])),

        ('selector', FeatureSelect(False)),

        ('impute', SimpleImputer(strategy='median')),

        ('standard', StandardScaler())

])



cat_pipeline = Pipeline([

        ('drop', DropFeature(['Alley','PoolQC', 'Fence', 'MiscFeature'])),

        ('selector', FeatureSelect()),

        ('impute', StringImputer()),

        ('oneHot', OneHotEncoder())

])



full_pipeline = FeatureUnion([

        ('numeric_pipeline', numeric_pipeline),

        ('cat_pipeline', cat_pipeline)

])



x_train = full_pipeline.fit_transform(x_train)
from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

lr_pipeline = Pipeline([

    ('kpca', KernelPCA()),

    ('lr', LinearRegression())

])



lr_param = {

    'kpca__gamma': np.linspace(0.01, 0.1, 10),

    'kpca__kernel': ['rbf', 'sigmoid'],

    'lr__normalize': [False]

}



lr_grid_cv = GridSearchCV(lr_pipeline, param_grid=lr_param, cv=3, 

                          verbose=True, n_jobs=-1, iid=True)

lr_grid_cv.fit(x_train, y_train)
lr_grid_cv.best_params_
lr_grid_cv.best_score_
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import mean_squared_error

lr = lr_grid_cv.best_estimator_

lr_pred = cross_val_predict(lr, x_train, y_train, 

                            cv=3, verbose=True, n_jobs=-1)

mse = mean_squared_error(y_train, lr_pred)

np.sqrt(mse)
test = pd.read_csv('../input/test.csv')

index = np.array(test[['Id']])[:,0]

test = test.set_index(['Id'])

x_test = full_pipeline.transform(test)
pred =  lr.predict(x_test)

pred_df = pd.DataFrame({'Id':index,

                       'SalePrice':pred})

pred_df.to_csv('../input/prediction.csv', index='')