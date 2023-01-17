import math

import pandas as pd

import numpy as np

import seaborn as sns

import pandas_profiling as pp

import missingno as msno

from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', None) 



# Read the datahome-data-for-ml-course/

X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)
print('missing value:(missing count, total element,percentage)')

[{x:(X_full[x].isna().sum(),len(X_full.index),math.ceil(X_full[x].isna().sum()*100/len(X_full.index)*100)/100)} for x in X_full.columns[X_full.isna().any()]]
msno.heatmap(X_full)
# pp.ProfileReport(pd.concat([y,X_full], axis=1))
# This is skew cols we saw when profiling the data 

# skew_cols = ['1stFlrSF','BsmtUnfSF','GrLivArea','LotArea','OpenPorchSF','TotalBsmtSF']

# skew_cols = ['BsmtUnfSF','GrLivArea','LotArea','OpenPorchSF']

skew_cols = []



print(X_full['Exterior1st'].unique())

print(X_full['Exterior2nd'].unique())

print(X_full['Condition1'].unique())

print(X_full['Condition2'].unique())

print(X_full['FullBath'].unique())

print(X_full['HalfBath'].unique())

print(X_full['SaleType'].unique())

X_full[['Exterior1st','Exterior2nd','Condition1','Condition2','YearBuilt','YearRemodAdd','YrSold','GarageYrBlt']].head()
# Convert numeric cols to categories

X_full['MSSubClass'] = X_full['MSSubClass'].apply(str)

X_test_full['MSSubClass'] = X_test_full['MSSubClass'].apply(str)



X_full['MoSold'] = X_full['MoSold'].apply(str)

X_test_full['MoSold'] = X_test_full['MoSold'].apply(str)



# Changing OverallCond into a categorical variable

# X_full['OverallCond'] = X_full['OverallCond'].astype(str)

# X_test_full['OverallCond'] = X_test_full['OverallCond'].astype(str)
# Merge 'Exterior1st', 'Exterior2nd' to 'Exterior'

X_full['Exterior'] =  X_full.apply(lambda x: x['Exterior1st'] if (pd.isnull(x['Exterior2nd'])) else str(x['Exterior1st'])+'-'+str(x['Exterior2nd']), axis=1)

X_test_full['Exterior'] =  X_test_full.apply(lambda x: x['Exterior1st'] if (pd.isnull(x['Exterior2nd'])) else str(x['Exterior1st'])+'-'+str(x['Exterior2nd']), axis=1)

X_full.drop(['Exterior1st', 'Exterior2nd'],axis=1,inplace=True)

X_test_full.drop(['Exterior1st', 'Exterior2nd'],axis=1,inplace=True)



# Merge 'Condition1', 'Condition2' to 'Condition'

X_full['Condition'] =  X_full.apply(lambda x: x['Condition1'] if (pd.isnull(x['Condition2'])) else str(x['Condition1'])+'-'+str(x['Condition2']), axis=1)

X_test_full['Condition'] =  X_test_full.apply(lambda x: x['Condition1'] if (pd.isnull(x['Condition2'])) else str(x['Condition1'])+'-'+str(x['Condition2']), axis=1)

X_full.drop(['Condition1', 'Condition2'],axis=1,inplace=True)

X_test_full.drop(['Condition1', 'Condition2'],axis=1,inplace=True)



# Caculate YearRemodAdd and YrSold

# X_full['YearRemodAdd'] = X_full.apply(lambda x: x['YearRemodAdd'] - x['YearBuilt'], axis=1)

# X_full['GarageYrBlt'] = X_full.apply(lambda x: x['GarageYrBlt'] - x['YearBuilt'], axis=1)

# X_full['YrSold'] = X_full.apply(lambda x: x['YrSold'] - x['YearBuilt'], axis=1)



# X_test_full['YearRemodAdd'] = X_test_full.apply(lambda x: x['YearRemodAdd'] - x['YearBuilt'], axis=1)

# X_test_full['GarageYrBlt'] = X_test_full.apply(lambda x: x['GarageYrBlt'] - x['YearBuilt'], axis=1)

# X_test_full['YrSold'] = X_test_full.apply(lambda x: x['YrSold'] - x['YearBuilt'], axis=1)



# Generate total square

X_full['TotalSF'] = X_full['TotalBsmtSF'] + X_full['1stFlrSF'] + X_full['2ndFlrSF']

X_test_full['TotalSF'] = X_test_full['TotalBsmtSF'] + X_test_full['1stFlrSF'] + X_test_full['2ndFlrSF']

X_full.drop(columns=['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)

X_test_full.drop(columns=['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)



# # Generate total bathroom

X_full['TotalBathroom'] = 1*X_full['FullBath'] + X_full['HalfBath']

X_test_full['TotalBathroom'] = 1*X_test_full['FullBath'] + X_test_full['HalfBath']

X_full.drop(['FullBath', 'HalfBath'],axis=1,inplace=True)

X_test_full.drop(['FullBath', 'HalfBath'],axis=1,inplace=True)



# # Generate BsmtBath

# X_full['BsmtBath'] = X_full['BsmtHalfBath'] + 1.1*X_full['BsmtFullBath']

# X_test_full['BsmtBath'] = X_test_full['BsmtHalfBath'] + 1.1*X_test_full['BsmtFullBath']

# X_full.drop(['BsmtHalfBath', 'BsmtFullBath'],axis=1,inplace=True)

# X_test_full.drop(['BsmtHalfBath', 'BsmtFullBath'],axis=1,inplace=True)



# Generate TotalPorch

# X_full['TotalPorch'] = X_full['EnclosedPorch'] + X_full['SsnPorch'] + X_full['ScreenPorch']

# X_test_full['TotalPorch'] = X_test_full['EnclosedPorch'] + X_test_full['SsnPorch'] + X_test_full['ScreenPorch']

# X_full.drop(['EnclosedPorch','SsnPorch','ScreenPorch'],axis=1,inplace=True)

# X_test_full.drop(['EnclosedPorch','SsnPorch','ScreenPorch'],axis=1,inplace=True)



# Generate Bsmt

# X_full['Bsmt'] = X_full['BsmtCond'] + X_full['BsmtExposure'] + X_full['BsmtFinType1'] + X_full['BsmtFinType2']

# X_test_full['Bsmt'] = X_test_full['BsmtCond'] + X_test_full['BsmtExposure'] + X_test_full['BsmtFinType1'] + X_full['BsmtFinType2']

# X_full['BsmtFinType2'] =  X_full['BsmtFinType1'] + X_full['BsmtFinType2']

# X_test_full['BsmtFinType2'] =  X_test_full['BsmtFinType1'] + X_test_full['BsmtFinType2']

# X_full.drop(['BsmtFinType1','BsmtFinType2'],axis=1,inplace=True)

# X_test_full.drop(['BsmtFinType1','BsmtFinType2'],axis=1,inplace=True_HouseStyle





# Generagte BldgType_HouseStyle

# X_full['BldgType_HouseStyle'] = X_full['BldgType'] + '_' + X_full['HouseStyle']

# X_test_full['BldgType_HouseStyle'] = X_test_full['BldgType'] + '_' + X_test_full['HouseStyle']

# X_full.drop(['BldgType','HouseStyle'],axis=1,inplace=True)

# X_test_full.drop(['BldgType','HouseStyle'],axis=1,inplace=True)



# Merge PoolArea and PoolQC

# X_full['PoolQC'] = X_full['PoolQC'].map({'Ex':3,'Gd':2,'Fa':1})

# X_full['PoolQC'] = X_full['PoolQC'].fillna(0)

# X_test_full['PoolQC'] = X_test_full['PoolQC'].map({'Ex':3,'Gd':2,'Fa':1})

# X_test_full['PoolQC'] = X_test_full['PoolQC'].fillna(0)



# X_full.drop(columns=['PoolArea', 'PoolQC'],axis=1,inplace=True)

# X_test_full.drop(columns=['PoolArea', 'PoolQC'],axis=1,inplace=True)



# # Calculate log of SalePrice

# y = np.log(y)



# Drop columns that have too many missing value

X_full.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea'],axis=1,inplace=True)

X_test_full.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea'],axis=1,inplace=True)

# pp.ProfileReport(pd.concat([y,X_full], axis=1))
# select caterical columns

categorical_cols = [cname for cname in X_full.columns if X_full[cname].dtype == "object"]

# Select numerical columns

numerical_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]

X_full[numerical_cols].head(5)
print(sorted({x:X_full[x].nunique() for x in categorical_cols}.items(), key=lambda x: x[1],reverse=True))

X_full[categorical_cols].head(5)
print(sorted({x:X_full[x].nunique() for x in numerical_cols}.items(), key=lambda x: x[1],reverse=True))

print(X_full['MSSubClass'].unique())

X_full[numerical_cols].head(5)
# Break off validation set from training data

# X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

#                                                                 train_size=0.8, test_size=0.2,

#                                                                 random_state=0)



categorical_small_variety_cols = [cname for cname in X_full.columns if

                    X_full[cname].nunique() <= 15 and

                    X_full[cname].dtype == "object"]



categorical_large_variety_cols = [cname for cname in X_full.columns if

                    X_full[cname].nunique() > 15 and

                    X_full[cname].dtype == "object"]



# categorical_l_cols = [cname for cname in X_full.columns if

#                     X_full[cname].nunique() > 10 and 

#                     X_full[cname].nunique() <= 15 and 

#                     X_full[cname].dtype == "object"]

categorical_label_cols = []



print('numerical_cols: ',numerical_cols)

print('categorical_cols: ',categorical_cols)

print('categorical_label_cols: ',categorical_label_cols )

print('categorical_small_variety_cols: ', categorical_small_variety_cols)

print('categorical_large_variety_cols: ',categorical_large_variety_cols)
from sklearn.pipeline import Pipeline, TransformerMixin

from sklearn.neighbors import LocalOutlierFactor



class OutlierExtractor(TransformerMixin):

    def __init__(self, **kwargs):

        """

        Create a transformer to remove outliers. A threshold is set for selection

        criteria, and further arguments are passed to the LocalOutlierFactor class



        Keyword Args:

            neg_conf_val (float): The threshold for excluding samples with a lower

               negative outlier factor.



        Returns:

            object: to be used as a transformer method as part of Pipeline()

        """

        try:

            self.threshold = kwargs.pop('neg_conf_val')

        except KeyError:

            self.threshold = -10.0

        pass

        self.kwargs = kwargs



    def transform(self, X):

        """

        Uses LocalOutlierFactor class to subselect data based on some threshold



        Returns:

            ndarray: subsampled data



        Notes:

            X should be of shape (n_samples, n_features)

        """

        x = np.asarray(X)

        lcf = LocalOutlierFactor(**self.kwargs)

        lcf.fit(X)

        return x[lcf.negative_outlier_factor_ > self.threshold, :]



    def fit(self, *args, **kwargs):

        return self
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer

import category_encoders as ce

from xgboost import XGBRegressor



# Preprocessing for numerical data

numerical_transformer = Pipeline(verbose=False,steps=[

    ('imputer_num', SimpleImputer(strategy='median')),

#     ('remove_outlier', OutlierExtractor())

])



# Preprocessing for categorical data

categorical_onehot_transformer = Pipeline(verbose=False,steps=[

    ('imputer_onehot', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



categorical_label_transformer = Pipeline(verbose=False,steps=[

    ('imputer_label', SimpleImputer(strategy='most_frequent')),

    ('label', ce.OrdinalEncoder())

    

])



categorical_count_transformer = Pipeline(verbose=False,steps=[

    ('imputer_count', SimpleImputer(strategy='most_frequent')),

    ('count', ce.TargetEncoder(handle_missing='count'))

#     ('count', ce.CountEncoder(min_group_size = 1,handle_unknown=0,handle_missing='count'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(verbose=False,

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cox_box', PowerTransformer(method='yeo-johnson', standardize=False),skew_cols),

        ('cat_label', categorical_label_transformer, categorical_label_cols),

        ('cat_onehot', categorical_onehot_transformer, categorical_small_variety_cols),

        ('cat_count', categorical_count_transformer, categorical_large_variety_cols),

    ])



train_pipeline = Pipeline(verbose=False,steps=[

                    ('preprocessor', preprocessor),   

                    ('scale', StandardScaler(with_mean=True,with_std=True)),

                    ('model', XGBRegressor(random_state=0))

                    ])
import eli5

from eli5.sklearn import PermutationImportance

from sklearn.ensemble import RandomForestRegressor

from IPython.display import display



# Split dataset to train and test

X_train, X_valid, y_train, y_valid = train_test_split(X_full[numerical_cols], y,train_size=0.8, test_size=0.2,random_state=0)



# Define pipeline to do transformation

transform_pipeline = Pipeline(verbose=False,steps=[

                    ('imputer_num', SimpleImputer(strategy='median')),

                    ('scale', StandardScaler(with_mean=True,with_std=True)),

                    ])



# Transform data

transform_pipeline.fit(X_train,y_train)

pi_X_train = pd.DataFrame(transform_pipeline.transform(X_train))

pi_X_valid = pd.DataFrame(transform_pipeline.transform(X_valid))

pi_X_train.columns = X_train.columns

pi_X_valid.columns = X_valid.columns



# Define a model and calculate permutation importance of all numeric columns

pi_model = RandomForestRegressor(n_estimators=700,max_depth=4,random_state=0)

pi_model.fit(pi_X_train,y_train)

perm = PermutationImportance(pi_model, random_state=1).fit(pi_X_valid, y_valid)

eli5.show_weights(perm, feature_names = pi_X_valid.columns.tolist(),top=100)
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV



param_grid = {'model__nthread':[2], #when use hyperthread, xgboost may become slower

              'model__learning_rate': [0.04, 0.05], #so called `eta` value

              'model__max_depth': range(3,5,1),

#               'model__importance_type': ['weight', 'gain', 'cover'],

#               "model__min_child_weight" : [ 1 ],

#               "model__gamma"            : [ 0.0],

              "model__colsample_bytree" : [ 0.2 ],

              'model__silent': [1],

              'model__n_estimators': [700], #number of trees

#               'model__n_estimators': range(595,600,1), #number of trees

#               'model__n_estimators': range(550,1000,5), #number of trees

             }

searched_model = GridSearchCV(estimator=train_pipeline,param_grid = param_grid, scoring="neg_mean_absolute_error", cv=5, error_score='raise', verbose = 1)

searched_model.fit(X_full,y)



print(searched_model.best_estimator_)

print(searched_model.best_score_)

preds_test = searched_model.predict(X_test_full)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test_full.index,'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)

output