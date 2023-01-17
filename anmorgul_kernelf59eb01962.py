# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset_path = '/kaggle/input'

data_train_file_name = 'train.csv'

data_test_file_name = 'test.csv'

data_sample_submission_file_name = 'sample_submission.csv'

data_description_file_name = 'data_description.txt'



#import data

train = pd.read_csv(os.path.join(dataset_path,data_train_file_name),index_col=0)

test = pd.read_csv(os.path.join(dataset_path,data_test_file_name),index_col=0)

#Check size and look

print(train.shape)

train.head()
y_train = train.SalePrice.copy()

y_train.head()
#Check size and look

print(test.shape)

test.head()
# Combine train and test for pre-processing

datset_all = pd.concat([train[train.columns[:-1]],test])

datset_columns = datset_all.columns

print(datset_all.shape)

datset_all.head(5)
datset_all[['Condition1', 'Condition2']]
#Number and types of columns

datset_all.info()
#dd = map(lambda x: str(x), datset_all['Fireplaces']) + datset_all['FireplaceQu']

ff = lambda x: str(x)

datset_all['Fireplaces'].map(ff)

#dd['Fireplaces'].map(ff)

datset_all['Fireplaces'] = datset_all['Fireplaces'].map(ff).fillna('unknown') +'_'+ datset_all['FireplaceQu'].fillna('unknown')

datset_all['Fireplaces'].unique()

datset_all['Street'].unique()
datset_all['Alley'].unique()
roads_not_na = datset_all[['Street', 'Alley', 'Condition1', 'Condition2', 'PoolQC']].copy()

roads_not_na['road'] = '1' + roads_not_na['Street'].fillna('None') + '2' + roads_not_na['Alley'].fillna('None')

roads_not_na['Conditions'] = roads_not_na['Condition1'] + roads_not_na['Condition2']

roads_not_na['Street'] = roads_not_na['Street'].fillna('None')

roads_not_na['Alley'] = roads_not_na['Alley'].fillna('None')

roads_not_na['PoolQC'] = roads_not_na['PoolQC'].fillna('None')

#roads_not_na

roads_not_na['road'].unique()
roads_not_na['Conditions'].unique()
roads_not_na = roads_not_na[0:1460]

#roads_not_na['y'] = y_train

label_encoder = LabelEncoder()

label_roads_not_na = roads_not_na.copy()

roads_not_na.columns

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(roads_not_na[list(roads_not_na.columns)]))

OH_cols_train.index = roads_not_na.index

for col in list(roads_not_na.columns):

    print(col)

    

    label_roads_not_na[col] = label_encoder.fit_transform(roads_not_na[col])

label_roads_not_na['y'] = y_train  

label_roads_not_na['ylog'] = np.log(label_roads_not_na['y'])

print(label_roads_not_na['Street'].corr(label_roads_not_na['y']))

print(label_roads_not_na['Alley'].corr(label_roads_not_na['y']))

print(label_roads_not_na['road'].corr(label_roads_not_na['y']))

print(label_roads_not_na['Condition1'].corr(label_roads_not_na['y']))

print(label_roads_not_na['Condition2'].corr(label_roads_not_na['y']))

print(label_roads_not_na['Conditions'].corr(label_roads_not_na['y']))

print(label_roads_not_na['PoolQC'].corr(label_roads_not_na['y']))
#train['PoolQC'].fillna('none').corr(y_train)
f = plt.figure(figsize=(19, 15))

plt.matshow(label_roads_not_na.corr(), fignum=f.number)

plt.xticks(range(label_roads_not_na.shape[1]), label_roads_not_na.columns, fontsize=14, rotation=45)

plt.yticks(range(label_roads_not_na.shape[1]), label_roads_not_na.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);


corr = label_roads_not_na.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
sns.pairplot(label_roads_not_na)
pool = pd.DataFrame()

pool['PoolArea'] = train['PoolArea'].copy()

pool['PoolQC'] = train['PoolQC'].fillna('none').copy()



pool = pool.replace({"PoolQC" : {'none' : 0, 'Fa' : 1, 'TA' : 2, 'Gd' : 3, 'Ex' : 4}})

pool['y'] = y_train

pool['ylog'] = np.log(pool['y'])

sns.pairplot(pool)
OH_cols_train['y'] = y_train    

OH_cols_train['ylog'] = np.log(OH_cols_train['y'])
f = plt.figure(figsize=(19, 15))

plt.matshow(OH_cols_train.corr(), fignum=f.number)

plt.xticks(range(OH_cols_train.shape[1]), OH_cols_train.columns, fontsize=14, rotation=45)

plt.yticks(range(OH_cols_train.shape[1]), OH_cols_train.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
#OH_cols_train.corr()

corr = OH_cols_train.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# Look for missing data

plt.figure(figsize=[20,15])

sns.heatmap(datset_all.isnull(),yticklabels=False,cbar=False)

# Looking at distribution of house prices

plt.figure(figsize=[20,10])



# Histogram plot

plt.subplot(1,2,1)

sns.distplot(y)

plt.title('Standard')



# Skewness and kurtosis

print("Skewness: %f" % y.skew())

print("Kurtosis: %f" % y.kurt())



# Due to skew (>1), we'll log it and show it now better approximates a normal distribution

plt.subplot(1,2,2)

sns.distplot(np.log(y))

plt.title('Log transformation')
datset_all.dtypes.unique()
def get_predictors_not_objects(dataset):

    return dataset.select_dtypes(exclude = ['object'])



def drop_y(dataset, colname):

    try:

        return dataset.drop([colname], axis = 1)

    except:

        return dataset



def get_y(dataset, colname):

    return dataset[colname]



def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Get names of columns with missing values

def get_cols_with_missing(X):

    return [col for col in X.columns if X[col].isnull().any()]



def drop_na(X, cols_with_missing):

    X.drop(cols_with_missing, axis=1, inplace=True)

    #print(X_dropped)

    return X



def get_low_cardinality_cols(X):

    return [cname for cname in X.columns if X[cname].nunique() < 10 and 

                        X[cname].dtype == "object"]



# Select numerical columns

def get_numerical_cols(X):

    return [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]



def get_objects_cols(X):

    return [cname for cname in X.columns if X[cname].dtype in ['object']]



#danootobj = 

#sns.pairplot(get_predictors_not_objects(X_train))
cols = ['LotFrontage', 'LotArea', 'GrLivArea','OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']

df = train[cols]

print(df.head())

df['y'] = y_train

df['logy'] = np.log(df['y'])

df['MSZoning'] = train["MSZoning"]

df = df.drop(df[(df['GrLivArea']>4000) & (df['y']<300000)].index)

df = df.drop(df[(df['LotFrontage']>250)].index)

df = df.drop(df[(df['LotArea']>60000)].index)

df = df.drop(df[(df['BsmtFinSF2']>1250)].index)

print(df.head())

sns.pairplot(df, vars=df.columns[:-1], hue="MSZoning")
#********************************
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

import warnings

warnings.filterwarnings('ignore')

warnings.simplefilter('ignore')

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import StandardScaler



def drop_tails(df):

    df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)

    df = df.drop(df[(df['LotFrontage']>250)].index)

    df = df.drop(df[(df['LotArea']>60000)].index)

    df = df.drop(df[(df['BsmtFinSF2']>1250)].index)

    return df



X = pd.read_csv('../input/train.csv', index_col='Id')

X = drop_tails(X)

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

print(numerical_cols)

categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

print(categorical_cols)







from sklearn.base import BaseEstimator, TransformerMixin



class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, subset):

        #print(subset)

        self.subset = subset



    def transform(self, X, *_):

        #print(X)

        return X.loc[:, self.subset]



    def fit(self, *_):

        return self



class ColumnSelectorByType(BaseEstimator, TransformerMixin):

    def __init__(self, ColumnType):

        #print(subset)

        self.ColumnType = ColumnType



    def transform(self, X, *_):

        #print(X)

        return X.select_dtypes(include=[ColumnType])



    def fit(self, *_):

        return self    

    

class MyTransformer_imput_none(TransformerMixin, BaseEstimator):

    '''A template for a custom transformer.'''



    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        # transform X via code or additional methods

        for col in ("PoolQC", 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 

                    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 

                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):

            X[col] = X[col].fillna('None')

        #my_imputer = SimpleImputer()

        #imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

        # Imputation removed column names; put them back

        #imputed_X = X.columns

        #print(imputed_X)

        return X





    

imp_none2cat =     MyTransformer_imput_none()



# Preprocessing for numerical data

numerical_transformer =  Pipeline(steps=[

        ('selector1', ColumnSelector(numerical_cols)),

        ('imputer1', SimpleImputer(strategy='constant')),

        ('scaler', StandardScaler()),

])

    

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

        ('selector2', ColumnSelector(categorical_cols)),

        ('imputer_none1', imp_none2cat),

        ('imputer2', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore')),

])



# Bundle preprocessing for numerical and categorical data

preprocessor = FeatureUnion([

    ('numerical', numerical_transformer),            

    ('categorical', categorical_transformer), 

])



my_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', XGBRegressor())

])

    

param_grid = {

    'model__n_estimators': [x for x in range(285, 287, 1)], #285, 315, 1

    'model__learning_rate': [x for x in np.arange(0.123, .124, .001)], #(0.123, .128, .001)

    'model__max_depth': [3,4,5],

    'preprocessor__numerical__imputer1__strategy': ['constant', 'mean', 'median'],

}



print(param_grid)



grid = GridSearchCV(my_pipeline, cv=3, param_grid=param_grid, verbose=4)



#import pprint as pp



#pp.pprint(sorted(grid.get_params().keys()))



print(grid.estimator.get_params().keys())





#print(numerical_cols)

#print(categorical_cols)

#print(X_train.head())

#print(X_train[categorical_cols].head())



#my_pipeline.fit(X_train[categorical_cols],y_train)



grid.fit(X,y)



# summarize results

print("Best: %f using %s" % (grid.best_score_, 

    grid.best_params_))

means = grid.cv_results_['mean_test_score']

stds = grid.cv_results_['std_test_score']

params = grid.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    #print("%f (%f) with: %r" % (mean, stdev, param))

    pass
numerical_transformer =  Pipeline(steps=[

        ('selector1', ColumnSelector(numerical_cols)),

        ('imputer1', SimpleImputer(strategy=grid.best_params_['model__learning_rate'])),

        ('scaler', StandardScaler()),

])



my_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', XGBRegressor(

        learning_rate = grid.best_params_['model__learning_rate'],

        max_depth = grid.best_params_['model__max_depth'],

        n_estimators = grid.best_params_['model__n_estimators'],

        #learning_rate = grid.best_params_['model__learning_rate'],

    ))

])

my_pipeline.fit(X, y)

#'model__max_depth': 4, 'model__n_estimators': 286, 'preprocessor__numerical__imputer1__strategy': 'constant'}
preds_test = my_pipeline.predict(test)
#*****************************************************************************
# Save test predictions to file

output = pd.DataFrame({'Id': test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
#*************************
class LabelEncoderExt(object):

    def __init__(self):

        """

        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]

        Unknown will be added in fit and transform will take care of new item. It gives unknown class id

        """

        self.label_encoder = LabelEncoder()

        # self.classes_ = self.label_encoder.classes_



    def fit(self, data_list):

        """

        This will fit the encoder for all the unique values and introduce unknown value

        :param data_list: A list of string

        :return: self

        """

        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])

        self.classes_ = self.label_encoder.classes_



        return self



    def transform(self, data_list):

        """

        This will transform the data_list to id list where the new values get assigned to Unknown class

        :param data_list:

        :return:

        """

        new_data_list = list(data_list)

        for unique_item in np.unique(data_list):

            if unique_item not in self.label_encoder.classes_:

                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]



        return self.label_encoder.transform(new_data_list)
country_list = ['Argentina', 'Australia', 'Canada', 'France', 'Italy', 'Spain', 'US', 'Canada', 'Argentina, ''US']



label_encoder = LabelEncoderExt()



label_encoder.fit(country_list)

print(label_encoder.classes_) # you can see new class called Unknown

print(label_encoder.transform(country_list))





new_country_list = ['Canada', 'France', 'Italy', 'Spain', 'US', 'India', 'Pakistan', 'South Africa']

print(label_encoder.transform(new_country_list))