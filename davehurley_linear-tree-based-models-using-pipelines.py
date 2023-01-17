import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import (train_test_split, cross_val_score)

from sklearn.metrics import mean_squared_error

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (OneHotEncoder, FunctionTransformer, StandardScaler, OrdinalEncoder, LabelEncoder)

from sklearn.pipeline import (Pipeline, FeatureUnion)

from IPython.display import display

from sklearn.linear_model import (ElasticNetCV, LassoCV, RidgeCV, LinearRegression)

from sklearn.ensemble import (RandomForestRegressor, StackingRegressor)

from xgboost import XGBRegressor

from scipy.stats import skew

from lightgbm import LGBMRegressor

from sklearn.base import TransformerMixin



%matplotlib inline

plt.rcParams['figure.figsize'] = (12.0, 6.0) #  set defualt figure size
#  Load train and test data



df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#  Remove any duplicate target values in the training dataset



if len(set(df_train.Id)) == len(df_train):

    print('There are no duplicates of the target variable')

else:

    df_train.drop_duplicates(subset=['Id'], inplace=True)

    

#  Create new variable test_id and remove Id from train and test data as it says nothing about sale price



test_id = df_test.Id

df_train.drop(columns=['Id'], inplace=True)

df_test.drop(columns=['Id'], inplace=True)
#  Calculate percent missing data in train dataset

missing = df_train.isnull().sum()

missing = missing[missing > 0] / len(df_train) * 100



#  Sort data

missing.sort_values(inplace=True)



#  Create bar plot

missing.plot.bar(zorder=2)

plt.title('Training Dataset - Missing Values')

plt.ylabel('Percent Missing (%)')

plt.grid(zorder=0)



#  Print number of categorical and numeric features with missing data

num_cols = df_train.select_dtypes(exclude='object').columns

cat_cols = df_train.select_dtypes('object').columns



num_missing = len(df_train[num_cols].columns[df_train[num_cols].isnull().any()])

cat_missing = len(df_train[cat_cols].columns[df_train[cat_cols].isnull().any()])



print(f"Number of numerical features with missing data: {num_missing}")

print(f"Number of categorical features with missing data: {cat_missing}")
#  Plot distribution of sale price in training dataset

f, axs = plt.subplots(1,2)

axs[0].hist(df_train.SalePrice, bins=36, zorder=2)

axs[0].set_title('Positively Skewed')

axs[0].set_xlabel('Sale Price ($)')

axs[0].set_ylabel('Frequency')

axs[0].set_xticks(ticks=[1e5, 3e5, 5e5, 7e5])

axs[0].grid(zorder=0)



#  Plot distribution of log of sale price in training dataset

axs[1].hist(np.log(df_train.SalePrice), bins=36, zorder=2)

axs[1].set_title('Normally Distributed')

axs[1].set_xlabel('Log of Sale Price ($)')

axs[1].grid(zorder=0)
#  Only numeric columns and features (i.e. no sale price)

num_cols = list(df_train.select_dtypes(exclude='object').columns)

num_cols.remove('SalePrice')



#  Plot all feature data

df_train[num_cols].hist(bins=36, figsize=(12,12))

plt.tight_layout()
#  Find outliers



outliers = df_train.loc[(df_train.SalePrice < 200000) & (df_train.GrLivArea > 4000)]



#  Plot Sale Price against Above Ground Square Footage



plt.figure(figsize=(8,8))



plt.plot(df_train.GrLivArea, df_train.SalePrice,'b.')

plt.plot(outliers.GrLivArea, outliers.SalePrice, 'ro', markerfacecolor='none', markersize=10, label='outliers')

plt.xlabel('Above Ground Square Footage (GrLivArea)')

plt.ylabel('Sale Price ($)')

plt.legend()

plt.grid()



#  Drop outliers from training data

df_train.drop(outliers.index, inplace=True)
#  Plot cardinality of categorical columns



cat_cols = df_train.select_dtypes('object').columns



df_train[cat_cols].nunique().plot.bar(zorder=2)

plt.ylabel('Count')

plt.title('Cardinality of Categorical Data')

plt.grid(zorder=0)
#  Remove outliers from training dataset



outliers = df_train.loc[(df_train.SalePrice < 200000) & (df_train.GrLivArea > 4000)]

df_train.drop(outliers.index, inplace=True)



#  Use log of sale price in training dataset



df_train.SalePrice = np.log(df_train.SalePrice)
#  Custom transformer to extract specific features

    

class ColumnExtractor(TransformerMixin):

    

    def __init__(self, cols):

        self.cols = cols

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        Xcols = X[self.cols]

        return Xcols

    

#  Custom transformer that inherits from SimpleImputer class and returns a dataframe



class DFSimpleImputer(SimpleImputer):

    

    def transform(self, X):

        Xim = super(DFSimpleImputer, self).transform(X)

        Xim = pd.DataFrame(Xim, index=X.index, columns=X.columns)

        return Xim

    

#  Custom transformer that inherits from OneHotEncoder and return a dataframe

    

class DFOneHotEncoder(OneHotEncoder):

    

    def transform(self, X):

        Xoh = super(DFOneHotEncoder, self).transform(X)

        Xoh = pd.DataFrame(Xoh, X.index)

        return Xoh



#  Custom transformer that creates a new feature TotalSquareFootage



class TotalSF(TransformerMixin):



    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        Xadd = X.copy()

        Xadd['TotalSF'] = Xadd.GrLivArea + Xadd.TotalBsmtSF + Xadd.GarageArea

        return Xadd
#  Helper function that computes the average RMSE over X folds using Cross-Validation. Cross-Validation function will fit and score data.



def get_RMSE(pipeline, X, y, folds):



    MSE_scores = -1 * cross_val_score(pipeline, X, y, cv=folds, scoring='neg_mean_squared_error')

    RMSE_scores = np.sqrt(MSE_scores)

    

    return RMSE_scores
#  Training and testing datasets



X_train = df_train.drop(columns=['SalePrice'])

y_train = df_train.SalePrice

X_test = df_test
#  Numerical features



numeric_columns = X_train.select_dtypes(exclude='object').columns



#  Categorical features



categorical_columns = X_train.select_dtypes('object').columns
#  Define the preprocessing pipeline



pipeline = Pipeline([

    ('features', FeatureUnion([

        ('numeric', Pipeline([

            ('extract', ColumnExtractor(numeric_columns)),

            ('imputer', DFSimpleImputer()),

            ('totalSF', TotalSF()), #  create a new feature total square footage

            ('logger', FunctionTransformer(np.log1p)) #  take the log of all numeric features to create a normal distribution

        ])),

        ('categorical', Pipeline([

            ('extract', ColumnExtractor(categorical_columns)),

            ('imputer', DFSimpleImputer(strategy='constant', fill_value='None')), #  we determined earlier that 'NA' for categorical really means 'None'

            ('encode', DFOneHotEncoder(handle_unknown='ignore', sparse=False))

        ])),

    ])),

    ('scale', StandardScaler())  #  scale all features

])
#  Define the models to fit and evaluate.



models = [

    LassoCV(),

    RidgeCV(),

    ElasticNetCV(),

    RandomForestRegressor(),

    XGBRegressor(), 

    LGBMRegressor()

]



#  Preprocess the data for each model, fit the model, and evaluate



print('RMSE Cross-Validation Training Scores \n')



RMSE = []

model_names = []

for i, model in enumerate(models):

    

    full_pipeline = Pipeline(steps=[('pipeline', pipeline),

                                    ('model', model)])



    #  Fit training data and score

    

    RMSE.append(get_RMSE(full_pipeline, X_train, y_train, 5))

    

    #  Print the scores

    

    model_names.append(str(model).split('(')[0])

    print('{} Training Score: {}'.format(model_names[i], round(np.mean(RMSE[i]),4)))

    

#  Create a boxplot of the scores



plt.figure(figsize=(18,7))

plt.boxplot(RMSE, labels=model_names, showmeans=True)

plt.xlabel('Models', fontsize=16)

plt.ylabel('Root Mean Square Error (RMSE)', fontsize=16)

plt.title('Cross-Validation Scores', fontsize=18)

plt.tick_params(axis = 'both', which = 'major', labelsize = 14)

plt.tick_params(axis = 'both', which = 'minor', labelsize = 14)



x = '' #  Hack to stop figure vomit
#  Define models that we want to stack



models = [

    LassoCV(),

    RidgeCV(),

    ElasticNetCV(),

    RandomForestRegressor(),

    XGBRegressor(), 

    LGBMRegressor()

]



#  Define estimators to stacked regressor

estimator_names = ['lassoCV', 'ridgeCV', 'elasticnetCV', 'random_forest', 'xgbregressor', 'lgbmregressor']

estimators = [(estimator_names[i], model) for i, model in enumerate(models)]

    

#  Define stacked model with Linear Regression as the final estimator

Stacked = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=3)



#  Add stacked model to models list from previous step

models.append(Stacked)



#  Preprocess the data for each model, fit the model, and evaluate



print('RMSE Cross-Validation Training Scores With Stacked Model \n')



RMSE = []

model_names = []

for i, model in enumerate(models):

    

    full_pipeline = Pipeline(steps=[('pipeline', pipeline),

                                    ('model', model)])



    #  Fit training data and score

    RMSE.append(get_RMSE(full_pipeline, X_train, y_train, 5))

    

    model_names.append(str(model).split('(')[0])

    print('{} Training Score: {}'.format(model_names[i], round(np.mean(RMSE[i]),4)))

    

#  Create a boxplot of the scores

plt.figure(figsize=(18,7))

plt.boxplot(RMSE, labels=model_names, showmeans=True)

plt.xlabel('Models', fontsize=16)

plt.ylabel('Root Mean Square Error (RMSE)', fontsize=16)

plt.title('Cross-Validation Scores With Stacked Model', fontsize=18)

plt.tick_params(axis = 'both', which = 'major', labelsize = 14)

plt.tick_params(axis = 'both', which = 'minor', labelsize = 14)



x = '' #  Hack to stop figure vomit
#  Preprocess and fit entire training dataset



full_pipeline = Pipeline(steps=[('pipeline', pipeline),

                                ('model', Stacked)])



full_pipeline.fit(X_train, y_train) #  fit to entire training dataset and not just K folds of it



#  Predict the test dataset target values



y_predict = full_pipeline.predict(df_test)

y_predict = np.expm1(y_predict)  #  Kaggle will take the log of sale price to compare



my_submission = pd.DataFrame({'Id': test_id, 'SalePrice': y_predict})

my_submission.to_csv('submission.csv', index=False)