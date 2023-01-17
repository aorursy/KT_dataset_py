# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline # for making pipleine 

from sklearn.impute import SimpleImputer # for handling missing variables either categorical or numerical

from sklearn.preprocessing import OneHotEncoder # for one hot encoding categorical variables

from sklearn.metrics import mean_absolute_error # for Mean absolute error

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor as xgbr # for modelling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')

train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
train.head()
null = train.isnull().sum()

null
train.columns
train1 = train

test1 = test

train1
# Drop columns that have too many missing value

train1.drop(columns=['Alley','MiscFeature','PoolArea'],axis=1,inplace=True)

test1.drop(columns=['Alley','MiscFeature','PoolArea'],axis=1,inplace=True)

train1
train1.isnull().sum()


y = train1['SalePrice']

X = train1.drop('SalePrice',axis=1,inplace=True)

train1['Fence'].unique()

train1['GarageCond'].value_counts()

train1['GarageCond'] = train1['GarageCond'].fillna('TA')

test1['GarageCond'] = test1['GarageCond'].fillna('TA')

test1.head()


train1['MoSold'] = train1['MoSold'].apply(str)

test1['MoSold'] = test1['MoSold'].apply(str)



test['MSZoning'].value_counts()
test1.isnull().sum()

test1.describe(include='all')
test1.head()
train1['Functional'].value_counts()
train1['Fireplaces'].value_counts()
train1['Functional'] = train1['Functional'].fillna('Typ')

test1['Functional'] = test1['Functional'].fillna('Typ')
train2 = train1.filter(['LotArea','MSZoning','YearBuilt','PoolArea','TotRmsAbvGrd','SalePrice'

                      ,'OverallCond'], axis=1)

test2 = test1.filter(['LotArea','MSZoning','YearBuilt','PoolArea','TotRmsAbvGrd','SalePrice'

                      ,'OverallCond'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



test1.isnull().sum()
# Merge 'Exterior1st', 'Exterior2nd' to 'Exterior'

train1['Exterior'] =  train1.apply(lambda x: x['Exterior1st'] if (pd.isnull(x['Exterior2nd'])) else str(x['Exterior1st'])+'-'+str(x['Exterior2nd']), axis=1)

test1['Exterior'] =  test1.apply(lambda x: x['Exterior1st'] if (pd.isnull(x['Exterior2nd'])) else str(x['Exterior1st'])+'-'+str(x['Exterior2nd']), axis=1)

train1.drop(['Exterior1st', 'Exterior2nd'],axis=1,inplace=True)

test1.drop(['Exterior1st', 'Exterior2nd'],axis=1,inplace=True)



# Merge 'Condition1', 'Condition2' to 'Condition'

train1['Condition'] =  train1.apply(lambda x: x['Condition1'] if (pd.isnull(x['Condition2'])) else str(x['Condition1'])+'-'+str(x['Condition2']), axis=1)

test1['Condition'] =  test1.apply(lambda x: x['Condition1'] if (pd.isnull(x['Condition2'])) else str(x['Condition1'])+'-'+str(x['Condition2']), axis=1)

train1.drop(['Condition1', 'Condition2'],axis=1,inplace=True)

test1.drop(['Condition1', 'Condition2'],axis=1,inplace=True)



# Generate total square

train1['TotalSF'] = train1['TotalBsmtSF'] + train1['1stFlrSF'] + train1['2ndFlrSF']

test1['TotalSF'] = test1['TotalBsmtSF'] + test1['1stFlrSF'] + test1['2ndFlrSF']

train1.drop(columns=['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)

test1.drop(columns=['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)



# # Generate total bathroom

train1['TotalBathroom'] = 1*train1['FullBath'] + train1['HalfBath']

test1['TotalBathroom'] = 1*test1['FullBath'] + test1['HalfBath']

train1.drop(['FullBath', 'HalfBath'],axis=1,inplace=True)

test1.drop(['FullBath', 'HalfBath'],axis=1,inplace=True)









# Generate Bsmt

train1['Bsmt'] = train1['BsmtCond'] + train1['BsmtExposure'] + train1['BsmtFinType1'] + train1['BsmtFinType2']

test1['Bsmt'] = test1['BsmtCond'] + test1['BsmtExposure'] + test1['BsmtFinType1'] + train1['BsmtFinType2']

train1['BsmtFinType2'] =  train1['BsmtFinType1'] + train1['BsmtFinType2']

test1['BsmtFinType2'] =  test1['BsmtFinType1'] + test1['BsmtFinType2']

train1.drop(['BsmtFinType1','BsmtFinType2'],axis=1,inplace=True)

test1.drop(['BsmtFinType1','BsmtFinType2'],axis=1,inplace=True)

# Generate TotalPorch

train1['TotalPorch'] = train1['EnclosedPorch'] + train1['ScreenPorch']

test1['TotalPorch'] = test1['EnclosedPorch'] + test1['ScreenPorch']

train1.drop(['EnclosedPorch','ScreenPorch'],axis=1,inplace=True)

test1.drop(['EnclosedPorch','ScreenPorch'],axis=1,inplace=True)
#Generagte BldgType_HouseStyle

train1['BldgType_HouseStyle'] = train1['BldgType'] + '_' + train1['HouseStyle']

test1['BldgType_HouseStyle'] = test1['BldgType'] + '_' + test1['HouseStyle']

train1.drop(['BldgType','HouseStyle'],axis=1,inplace=True)

test1.drop(['BldgType','HouseStyle'],axis=1,inplace=True)



#Merge PoolArea and PoolQC

train1['PoolQC'] = train1['PoolQC'].map({'Ex':3,'Gd':2,'Fa':1})

train1['PoolQC'] = train1['PoolQC'].fillna(0)

test1['PoolQC'] = test1['PoolQC'].map({'Ex':3,'Gd':2,'Fa':1})

test1['PoolQC'] = test1['PoolQC'].fillna(0)



train1.drop(columns=['PoolQC'],axis=1,inplace=True)

test1.drop(columns=[ 'PoolQC'],axis=1,inplace=True)



# Calculate log of SalePrice



train1.isnull().sum()
train1['Bsmt'].value_counts()





skew_cols = []





# select caterical columns

categorical_cols = [cname for cname in train1.columns if train1[cname].dtype == "object"]

# Select numerical columns

numerical_cols = [cname for cname in train1.columns if train1[cname].dtype in ['int64', 'float64']]

train1[numerical_cols].head(5)



categorical_small_variety_cols = [cname for cname in train1.columns if

                    train1[cname].nunique() <= 15 and

                    train1[cname].dtype == "object"]



categorical_large_variety_cols = [cname for cname in train1.columns if

                    train1[cname].nunique() > 15 and

                    train1[cname].dtype == "object"]



categorical_l_cols = [cname for cname in train1.columns if

                     train1[cname].nunique() > 10 and 

                     train1[cname].nunique() <= 15 and 

                     train1[cname].dtype == "object"]

categorical_label_cols = []





from sklearn.pipeline import Pipeline, TransformerMixin

from sklearn.neighbors import LocalOutlierFactor



class OutlierExtractor(TransformerMixin):

    def __init__(self, **kwargs):

        

       

        try:

            self.threshold = kwargs.pop('neg_conf_val')



        except KeyError:

            self.threshold = -10.0

        pass

        self.kwargs = kwargs



    def transform(self, X):

        



        x = np.asarray(X)

        lcf = LocalOutlierFactor(**self.kwargs)

        lcf.fit(X)

        return x[lcf.negative_outlier_factor_ > self.threshold, :]



    def fit(self, *args, **kwargs):

        return self



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

X_train, X_valid, y_train, y_valid = train_test_split(train1[numerical_cols], y,train_size=0.8, test_size=0.2,random_state=0)



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



from sklearn.ensemble import GradientBoostingRegressor

# Define a model and calculate permutation importance of all numeric columns

pi_model = GradientBoostingRegressor(max_depth=4,random_state=0)

pi_model.fit(pi_X_train,y_train)

perm = PermutationImportance(pi_model, random_state=1).fit(pi_X_valid, y_valid)

eli5.show_weights(perm, feature_names = pi_X_valid.columns.tolist(),top=100)

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV



param_grid = {'model__nthread':[2], #when use hyperthread, xgboost may become slower

               "model__min_child_weight" : [ 1 ],

               "model__gamma"            : [ 0.0],

              "model__colsample_bytree" : [ 0.2 ],

              'model__silent': [1],

              'model__n_estimators': [500], #number of trees

             }

searched_model = GridSearchCV(estimator=train_pipeline,param_grid = param_grid, scoring="neg_mean_absolute_error", cv=10, error_score='raise', verbose = 1)

searched_model.fit(train1,y)



print(searched_model.best_estimator_)

print(searched_model.best_score_)
preds_test = searched_model.predict(test1)

# Save test predictions to file

output = pd.DataFrame({'Id': test1['Id'],'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)

output
