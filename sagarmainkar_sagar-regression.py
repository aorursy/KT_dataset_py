# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd


#Modeling Algorithms
from sklearn import linear_model, model_selection, ensemble, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor


from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler, LabelEncoder

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import missingno as msno
from yellowbrick.regressor import RegressionScoreVisualizer, PredictionError, ResidualsPlot

#Configure Visualizations
%matplotlib inline
mpl.style.use('ggplot')
plt.style.use('fivethirtyeight')
sns.set(context='notebook',palette='dark',style='whitegrid',color_codes=True)
params={
    'axes.labelsize':'large',
    'xtick.labelsize':'large',
    'legend.fontsize':20,
    'figure.dpi':150,
    'figure.figsize':[25,7]
}

plt.rcParams.update(params)

#Centre all plots
from IPython.core.display import HTML
HTML('''
    <style>
        .output.png{
            display:table-cell;
            text-align: centre;
            vertical-align:middle;
        }
    </style>
''')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv',index_col=0)
train_df.head()
train_df.info()
train_df.describe()
train_df.isnull().sum()
_ = msno.bar(train_df)
new_train_df = train_df.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1,inplace=False)
new_train_df.info()
fig ,ax = plt.subplots(figsize=(20,10))
_ = sns.boxplot(data=new_train_df,ax=ax)
corr = new_train_df.corr()
corr
fig, ax = plt.subplots(figsize=(20,10))
_= sns.heatmap(corr.iloc[:,1:10],annot=True,ax=ax)
fig, ax = plt.subplots(figsize=(20,10))
_= sns.heatmap(corr.iloc[:,11:20],annot=True,ax=ax)
fig, ax = plt.subplots(figsize=(20,10))
_= sns.heatmap(corr.iloc[:,21:30],annot=True,ax=ax)
fig, ax = plt.subplots(figsize=(20,10))
_= sns.heatmap(corr.iloc[:,30:36],annot=True,ax=ax)
X_train = train_df.drop(['SalePrice','Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1)
y_train = train_df['SalePrice']

test_df = pd.read_csv('../input/test.csv',index_col=0)
X_test  = test_df.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1,inplace=False)

def getItemsNotInTest(train,test):
    cat_vars = getColumns(train,False)
    trainCatValues = set()
    testCatValues  = set()
    for c in cat_vars:
        trainKeys= train[c].value_counts().to_dict().keys()
        [trainCatValues.add(v) for v in trainKeys ]

        testKeys = test[c].value_counts().to_dict().keys()
        [testCatValues.add(v) for v in testKeys]
    
    finalSet = set()
    for item in trainCatValues:
        if item not in testCatValues:
            finalSet.add(item)
    return finalSet
#Simple function to show descriptive stats on the categorical variables

def describe_categorical(X):
    '''
     Just like describe but returns the results for categorical variables only
    '''
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))
describe_categorical(X_train)

def getColumns(X,isNumeric=True):
    ''' 
        Return the Numeric or Categorical columns list
    '''
    if(isNumeric):
        return list(X_train.dtypes[X_train.dtypes != np.dtype('O')].index)
    else:
        return list(X_train.dtypes[X_train.dtypes == np.dtype('O')].index)

from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for variable in X.columns:
            #Fill missing data with the word "Missing"
            X[variable].fillna("Missing",inplace=True)
            #Create  array of dummies
            dummies = pd.get_dummies(X[variable],prefix=variable)
            #Update X to include dummies and drop the main variables
            X= pd.concat([X,dummies],axis=1)
            X.drop([variable],axis=1,inplace=True)
        return X

class ConvertToPandasDF(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return pd.DataFrame(X)
    
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('select_numeric',DataFrameSelector(getColumns(X_train,True))),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ('convertTo_DF',ConvertToPandasDF())
    ])

num_pipeline.fit_transform(X_train).info()
cat_pipeline = Pipeline([
        ('select_cat',DataFrameSelector(getColumns(X_train,False))),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder",CustomOneHotEncoder())
    ])

cat_pipeline.fit_transform(X_train)
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])

X_train_transformed = pd.DataFrame(preprocess_pipeline.fit_transform(X_train))
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(X_train_transformed,y_train)

lin_scores = cross_val_score(estimator=lr_model,
                            X=X_train_transformed,
                            y=y_train,
                            cv=5,
                            scoring="neg_mean_squared_error",
                            verbose=2)



lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train_transformed, y_train)

tree_scores = cross_val_score(estimator=lr_model,
                            X=X_train_transformed,
                            y=y_train,
                            cv=5,
                            scoring="neg_mean_squared_error",
                            verbose=2)



tree_rmse = np.sqrt(-tree_scores)
display_scores(tree_rmse)
num_pipeline_tree = Pipeline([
        ('select_numeric',DataFrameSelector(getColumns(X_train,True))),
        ('imputer', Imputer(strategy="median")),
        ('convertTo_DF',ConvertToPandasDF())
    ])
cat_pipeline_tree = Pipeline([
        ('select_cat',DataFrameSelector(getColumns(X_train,False))),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder",CustomOneHotEncoder())
    ])

preprocess_pipeline_tree = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline_tree),
        ("cat_pipeline", cat_pipeline_tree)
    ])

X_train_tree = pd.DataFrame(preprocess_pipeline_tree.fit_transform(X_train))


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train_tree, y_train)

tree_scores = cross_val_score(estimator=lr_model,
                            X=X_train_tree,
                            y=y_train,
                            cv=5,
                            scoring="neg_mean_squared_error",
                            verbose=2)



tree_rmse = np.sqrt(-tree_scores)
display_scores(tree_rmse)
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train_transformed, y_train)

forest_scores = cross_val_score(forest_reg, X_train_transformed, y_train,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [5,10, 30,50,70,74], 'max_features': [2, 4, 6, 8,12,15,18]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [5,10, 30,50,70], 'max_features': [2, 4, 6, 8,12,15,18]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train_transformed, y_train)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)


#Simple version that show all of the variables
feature_importances = pd.Series(grid_search.best_estimator_.feature_importances_[1:10],index=X_train.columns[1:10])
feature_importances.sort_values(ascending=True,inplace=True)
feature_importances[1:10].plot(kind="barh",figsize=(7,6));
X_test.info()
#X_train.info()
num_pipeline_test = Pipeline([
        ('select_numeric',DataFrameSelector(getColumns(X_test,True))),
        ('imputer', Imputer(strategy="median")),
        ('convertTo_DF',ConvertToPandasDF())
    ])
cat_pipeline_test = Pipeline([
        ('select_cat',DataFrameSelector(getColumns(X_test,False))),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder",CustomOneHotEncoder())
    ])

preprocess_pipeline_test = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline_test),
        ("cat_pipeline", cat_pipeline_test)
    ])

X_test_transformed = pd.DataFrame(preprocess_pipeline_test.fit_transform(X_test))

grid_search.best_estimator_.predict(X_test_transformed)
pd.DataFrame(pd.concat([pd.Series(X_train.columns),pd.Series(X_test.columns)],axis=1))
for c in getColumns(X_train,False):
    print(X_train[c].value_counts())
getItemsNotInTest(X_train,X_test)