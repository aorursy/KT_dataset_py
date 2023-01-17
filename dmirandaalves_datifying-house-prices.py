import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

from scipy.stats import norm, skew



sns.set(style="ticks", palette="pastel")
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train = train.drop('Id', axis = 1)



test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

ids_test = test['Id']



test = test.drop('Id', axis = 1)
correlation_target = train.corr()['SalePrice']

pd.DataFrame(correlation_target[np.abs(correlation_target) > 0.6].sort_values(ascending = False))
attributes = pd.DataFrame(correlation_target[np.abs(correlation_target) > 0.6].sort_values(ascending = False)).index.tolist()



correlation_attributes = train.corr()[attributes]



grid_kws = {"width_ratios": (.9, .05), "wspace": 0.2}

f, (ax1, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize = (9, 9))



cmap = sns.diverging_palette(220, 8, as_cmap=True)



ax1 = sns.heatmap(correlation_attributes, vmin = -1, vmax = 1, cmap = cmap, ax = ax1, square = False, linewidths = 0.5, yticklabels = True, \

    cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', \

                                 'ticks': [-1, -0.5, 0, 0.5, 1]})

ax1.set_xticklabels(ax1.get_xticklabels(), size = 10); 

ax1.set_title('Correlation Heatmap', size = 15);

cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 12);
#As we know, MSubClass is not a number, it is a class. Let's transform it

train['MSSubClass'] = train['MSSubClass'].astype(object)

test['MSSubClass'] = test['MSSubClass'].astype(object)



dtypes_ = pd.DataFrame(train.dtypes)



print('We have {:.0f} different categorical features'.format(dtypes_[dtypes_[0] == 'object'].count().iloc[0]))
f, axes = plt.subplots(44, 1, figsize=(15,250))



counter_for_axes = 0



for item in dtypes_[dtypes_[0] == 'object'][0].index.tolist():

    sns.boxplot(y='SalePrice', x=item, data=train,  orient='v' , ax=axes[counter_for_axes])

    counter_for_axes = counter_for_axes + 1
missing_ = pd.DataFrame(train.isna().sum()/len(train)).sort_values(0, ascending = False)

missing_values = missing_[missing_[0] > 0]



plt.figure(figsize=(15,5))

plt.bar(range(len(missing_values)),missing_values[0])

plt.xticks(range(len(missing_values)), missing_values.index, rotation = 'vertical')

plt.title('Missing Values %')



missing_values.index.tolist()
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer



#Here I will make a class to select some columns according to their types

class SelectType(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):

        self.dtype = dtype

    

    def fit(self, X, y = None):

        return self

    

    def transform(self, X):

        return X.select_dtypes(include = [self.dtype])
#Here I am selecting the right columns



class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        assert isinstance(X, pd.DataFrame)



        try:

            return X[self.columns]

        except KeyError:

            cols_error = list(set(self.columns) - set(X.columns))

            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
#Here I am applying the pipeline and testing the first model, the ridge one.



from sklearn.pipeline import make_pipeline



preprocessing_pipeline = make_pipeline(

    ColumnSelector(columns=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',

       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',

       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',

       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',

       'MoSold', 'YrSold', 'SaleType', 'SaleCondition']),

    FeatureUnion(transformer_list = [

    ('Numbers', make_pipeline(

        SelectType(np.number), SimpleImputer(strategy='constant', fill_value = 0), StandardScaler())),

    ('Object', make_pipeline(

        SelectType('object'), SimpleImputer(strategy='constant', fill_value = 'No'), OneHotEncoder(handle_unknown="ignore")))

])

)





from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



classifier_pipeline_ridge = make_pipeline(preprocessing_pipeline,

                                    Ridge()

)



param_grid = {"ridge__alpha": [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 100]}



X = train.iloc[:,:79]

y = train.iloc[:,79:]



classifier_model_ridge = GridSearchCV(classifier_pipeline_ridge, param_grid, cv=5, scoring = 'neg_mean_squared_error')

classifier_model_ridge.fit(X,y)
#Let's go for the Lasso



from sklearn.linear_model import Lasso



classifier_pipeline_lasso = make_pipeline(preprocessing_pipeline,

                                      Lasso())



param_grid = {"lasso__alpha": [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]}



classifier_model_lasso = GridSearchCV(classifier_pipeline_lasso, param_grid, cv=5, scoring = 'neg_mean_squared_error')

classifier_model_lasso.fit(X,y)

classifier_model_lasso.best_score_
#Now its RF time



from sklearn.ensemble import RandomForestRegressor



classifier_pipeline_rf = make_pipeline(preprocessing_pipeline,

                                      RandomForestRegressor())



param_grid = {"randomforestregressor__max_depth": [2,3,4]}



classifier_model_rf = GridSearchCV(classifier_pipeline_rf, param_grid, cv=5, scoring = 'neg_mean_squared_error')

classifier_model_rf.fit(X,y)

classifier_model_rf.best_score_
#And Kernel Ridge



from sklearn.kernel_ridge import KernelRidge



classifier_pipeline_krr = make_pipeline(preprocessing_pipeline,

                                      KernelRidge())



param_grid = {"kernelridge__alpha": [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 100]}



classifier_model_krr = GridSearchCV(classifier_pipeline_krr, param_grid, cv=5, scoring = 'neg_mean_squared_error')

classifier_model_krr.fit(X,y)

classifier_model_krr.best_score_

#Taking a look at the scores



krr = {'model':'krr','result':classifier_model_krr.best_score_}

ridge = {'model':'ridge','result':classifier_model_ridge.best_score_}

lasso = {'model':'lasso','result':classifier_model_lasso.best_score_}

random_forest = {'model':'random_forest','result':classifier_model_rf.best_score_}



result = pd.DataFrame([krr,ridge,lasso,random_forest])

result.sort_values('result', ascending = False)
#It looks we are going for the lasso.



pd.DataFrame(classifier_model_lasso.cv_results_)
#Getting ready

classifier_pipeline_lasso_2 = make_pipeline(preprocessing_pipeline,

                                      Lasso(alpha = 200))
#Fitting it

classifier_pipeline_lasso_2.fit(X,y)
#Going for the test

ans = pd.DataFrame(classifier_pipeline_lasso_2.predict(test))
#Just making it according to the sample

ans['Id'] = ids_test

ans2 = ans.set_index('Id')
#Final adjustments

ans2.rename(columns={0:'SalePrice'}, inplace = True)
#Done!

ans2

