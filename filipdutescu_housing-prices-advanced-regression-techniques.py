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
# Load data from the csv file, droping columns if provided

def load_data(filename:str, columns:'list of strings' = None):

    result = pd.read_csv(filename)

    if columns is not None and len(columns) > 1:

        return result.drop(columns=columns)

    return result
# Print a brief, quick analysis of a dataframe to gain insight

def quick_analysis(data_frame:pd.DataFrame):

    print('\nAnalysis of dataframe:')

    print(data_frame.head())

    print(data_frame.info())

    print(data_frame.describe())
from matplotlib import pyplot as plt



%matplotlib inline



raw_data = load_data('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

quick_analysis(raw_data)



plt.hist(raw_data['SalePrice'])

plt.show()
non_numeric_cols = raw_data.loc[:, raw_data.dtypes == object]



for col in non_numeric_cols.columns:

    print(non_numeric_cols[col].value_counts())
corr_matrix = raw_data.corr()

sale_correl = corr_matrix['SalePrice'].sort_values(ascending=False)

print(sale_correl)
raw_data['Grade'] = raw_data['OverallQual'] / raw_data['OverallCond']
raw_data['Age'] = raw_data['YrSold'] - raw_data['YearBuilt']

raw_data['RemodAge'] = raw_data['YrSold'] - raw_data['YearRemodAdd']
raw_data['TotalSF'] = raw_data['TotalBsmtSF'] + raw_data['1stFlrSF'] + raw_data['2ndFlrSF']
corr_matrix = raw_data.corr()

sale_correl = corr_matrix['SalePrice'].sort_values(ascending=False)

print(sale_correl)
age_correl = corr_matrix['Age'].sort_values(ascending=False)

print('Age correlations:', age_correl, '\n')



remod_age_correl = corr_matrix['RemodAge'].sort_values(ascending=False)

print('RemodAge correlations:', remod_age_correl, '\n')



grade_correl = corr_matrix['Grade'].sort_values(ascending=False)

print('Grade correlations:', grade_correl, '\n')



totalsf_correl = corr_matrix['TotalSF'].sort_values(ascending=False)

print('TotalSF correlations:', totalsf_correl, '\n')
from matplotlib import ticker as tick



# Plot correlations

def corr_plot(data :pd.DataFrame, feature :str, threshold=0.5, plot_type :str = 'scatter', y_lower_scale=True, same_fig=True, fig_size=(3, 4)):

    fig = plt.figure()

    corr_matrix = data.corr()

    alpha = 0.3

    i = 1

    for feat in corr_matrix.columns:

        if abs(corr_matrix[feat][feature]) > threshold and feat != feature:

            if same_fig == True:

                ax = fig.add_subplot(fig_size[0], fig_size[1], i)

                if plot_type == 'scatter':

                    ax.scatter(x=feat, y=feature, data=data, alpha=alpha)

                elif plot_type == 'hist':

                    ax.hist(x=feat, data=data)

                ax.set_xlabel(feat)

                if y_lower_scale == True:

                    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.e'))

                plt.yticks(rotation=45)

                i = i + 1

            else:

                if plot_type == 'scatter':

                    plt.scatter(x=feat, y=feature, data=data, alpha=alpha)

                elif plot_type == 'hist':

                    plt.hist(x=feat, data=data)

                plt.xlabel(feat)

                plt.show()



    if same_fig == True:

        fig.tight_layout()

        plt.show()
corr_plot(raw_data, 'SalePrice', y_lower_scale=False, same_fig=False)
corr_plot(raw_data, 'SalePrice', plot_type='hist', y_lower_scale=False, same_fig=False)
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np





class FeatureCreator(BaseEstimator, TransformerMixin):

    def __init__(self, features :'list of strings', operation, as_dataframe :bool = False, feat_name :str = 'NewFeature'):

        self.features = features

        self.operation = operation

        self.as_dataframe = as_dataframe

        self.feat_name = feat_name



    def fit(self, X, y = None):

        return self



    def transform(self, X, y = None):

        no_feat = len(self.features)

        prev_feat = self.features[0]

        for i in range(1, no_feat):

            new_feature = self.operation(X[prev_feat], X[self.features[i]])

            prev_feat = self.features[i]



        if self.as_dataframe:

            X[self.feat_name] = new_feature

            return X



        return np.c_[X, new_feature]





class FeatureDropper(BaseEstimator, TransformerMixin):

    def __init__(self, features, as_dataframe :bool = False):

        self.features = features

        self.as_dataframe = as_dataframe



    def fit(self, X, y = None):

        return self



    def transform(self, X, y = None):

        if self.as_dataframe == True:

            return X.drop(columns=self.features)

        return np.c_[X.drop(columns=self.features)]





class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, features, as_dataframe :bool = False):

        self.features = features

        self.as_dataframe = as_dataframe



    def fit(self, X, y = None):

        return self



    def transform(self, X, y = None):

        if self.as_dataframe == True:

            return X[self.features]

        return np.c_[X[self.features]]





class CategoricalImputer(BaseEstimator, TransformerMixin):



    def __init__(self, strategy :str = 'most_frequent', value :str = None):

        self.strategy = strategy

        self.value = value

        

    def fit(self, X, y=None):

        if self.strategy == 'most_frequent':

            self.fill = pd.Series([X[col].mode()[0] for col in X], index=X.columns)

        elif self.strategy == 'nan_to_none':

            self.fill = pd.Series(['None' for col in X], index=X.columns)

        elif self.strategy == 'custom_val' and self.value is not None:

            self.fill = pd.Series([self.value for col in X], index=X.columns)

            

        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)

   

raw_data['YrSold_C'] = raw_data['YrSold'].copy().astype(str)

raw_data['MoSold'] = raw_data['MoSold'].astype(str)

raw_data['MSZoning'] = raw_data['MSZoning'].astype(str)

raw_data['OverallCond_C'] = raw_data['OverallCond'].copy().astype(str)



num_cols = [

    'OverallQual', 

    'OverallCond', 

    'YearBuilt', 

    'YearRemodAdd', 

    'TotalBsmtSF', 

    '1stFlrSF', 

    '2ndFlrSF', 

    'GarageCars',

    'GarageArea',

    'FullBath',

    'YrSold', 

] 

cat_cols = [

    'MSZoning', 

    'Street', 

    'Utilities', 

    'Neighborhood', 

    'ExterQual', 

    'ExterCond', 

    'BsmtQual', 

    'BsmtCond', 

    'Heating', 

    'CentralAir', 

    'PavedDrive', 

    'SaleType', 

    'SaleCondition',

    'YrSold_C', 

    'MoSold',

    'OverallCond_C',

]
cat_cols_categs = [raw_data[col].unique() for col in cat_cols]

cat_cols_categs
from sklearn.pipeline import Pipeline, FeatureUnion 

from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

from sklearn.impute import SimpleImputer as Imputer



num_pipeline = Pipeline([

    ('feat_sel', FeatureSelector(num_cols, True)),

    ('grade', FeatureCreator(['OverallCond', 'OverallQual'], lambda x, y: x / y, as_dataframe=True, feat_name='Grade')),

    ('age', FeatureCreator(['YrSold', 'YearBuilt'], lambda x,y: x - y, as_dataframe=True, feat_name='Age')),

    ('remod_age', FeatureCreator(['YrSold', 'YearRemodAdd'], lambda x,y: x - y, as_dataframe=True, feat_name='RemodAge')),

    ('total_sf', FeatureCreator(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], lambda x,y: x + y, as_dataframe=True, feat_name='TotalSF')),

    ('drop_cat_feat', FeatureDropper(['YrSold', 'OverallCond'], as_dataframe=True)),

    ('imputer_mean', Imputer(strategy='mean')),

    ('robust_scalar', RobustScaler())

]) 



cat_pipeline = Pipeline([

    ('feat_sel', FeatureSelector(cat_cols, True)),

    ('imputer_most_frequent', CategoricalImputer()),

    ('encode', OneHotEncoder(categories=cat_cols_categs, sparse=False)),

])

feat_union = FeatureUnion(transformer_list=[

    ('num_features', num_pipeline),

    ('cat_features', cat_pipeline),

])
train_labels = raw_data['SalePrice'].copy()

train_feat = feat_union.fit_transform(raw_data)
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression





lin_reg = LinearRegression()

grid_search = GridSearchCV(lin_reg, [{}], cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_feat, train_labels)

print('Linear regression best hyperparameters:')

print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))



final_lr_model = grid_search.best_estimator_
from sklearn.tree import DecisionTreeRegressor





hyperparams_vals = [

    {'max_features': [6, 10, 12, 16, 18, 20, 24]},

]

    

dt_reg = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(dt_reg, hyperparams_vals, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_feat, train_labels)

print('Decision tree best hyperparameters:')

print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))



final_dt_model = grid_search.best_estimator_
from sklearn.ensemble import RandomForestRegressor





hyperparams_vals = [

    {'n_estimators': [200, 225, 250], 'max_features': [16, 24, 30]},

    {'bootstrap': [False], 'n_estimators': [220, 225], 'max_features': [24, 28]},

]

    

forest_reg = RandomForestRegressor(n_jobs=-1, random_state=42)

grid_search = GridSearchCV(forest_reg, hyperparams_vals, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_feat, train_labels)

print('Random forest best hyperparameters:')

print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))



final_rf_model = grid_search.best_estimator_
from xgboost import XGBRegressor





hyperparams_vals = [

    {'n_estimators': [450, 500, 750], 'max_features': [2, 4, 8], 'max_depth': [3, 4, None]},

]

    

xgbr_reg = XGBRegressor(learning_rate=0.05, n_threads=-1, random_state=42)

grid_search = GridSearchCV(xgbr_reg, hyperparams_vals, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_feat, train_labels)

print('XGBoost regressor best hyperparameters:')

print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))



final_xgb_model = grid_search.best_estimator_
from sklearn.svm import SVR





hyperparams_vals = [

    {'kernel': ['linear', 'sigmoid', 'rbf'], 'gamma': ['auto', 'scale']},

    {'kernel': ['poly'], 'gamma': ['auto', 'scale'], 'degree': [3, 4, 5]},

]

    

svm_reg = SVR()

grid_search = GridSearchCV(svm_reg, hyperparams_vals, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_feat, train_labels)

print('Support vector machine best hyperparameters:')

print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))



final_svm_model = grid_search.best_estimator_
from sklearn.linear_model import ElasticNet





hyperparams_vals = [

    {'alpha': [0.0005, 0.005, 0.05, 0.2], 'l1_ratio': [0.1, 0.25, 0.75, 0.9]},

]



enet_reg = ElasticNet(max_iter=100000000, tol=0.001)

grid_search = GridSearchCV(enet_reg, hyperparams_vals, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_feat, train_labels)

print('ElasticNet best hyperparameters:')

print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))



final_enet_model = grid_search.best_estimator_
rf_feat_imp = final_rf_model.feature_importances_

xgb_feat_imp = final_xgb_model.feature_importances_



other_feat = ['Grade', 'RemodAge', 'TotalSF']

all_features = num_cols.copy()

print(num_cols)

for cat_values in cat_cols_categs.copy():

    all_features.extend(cat_values)

all_features.extend(other_feat.copy())



print('Random forest feature importances:')

for feat in sorted(zip(rf_feat_imp, all_features), reverse=True):

    print(feat)

    

print('\nXGBoost feature importances:')

for feat in zip(xgb_feat_imp, all_features):

    print(feat)
test_data = load_data('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_data['YrSold_C'] = test_data['YrSold'].copy().astype(str).replace('nan', None)

test_data['MoSold'] = test_data['MoSold'].astype(str).replace('nan', None)

test_data['MSZoning'] = test_data['MSZoning'].astype(str).replace('nan', None)

test_data['OverallCond_C'] = test_data['OverallCond'].copy().astype(str).replace('nan', None)

test_feat = feat_union.transform(test_data)



rf_predictions = final_rf_model.predict(test_feat)

xgb_predictions = final_xgb_model.predict(test_feat)

predictions = rf_predictions * 0.35 + xgb_predictions * 0.65



pred_df = pd.DataFrame()

pred_df['Id'] = test_data['Id']

pred_df['SalePrice'] = predictions.flatten()



print(pred_df)