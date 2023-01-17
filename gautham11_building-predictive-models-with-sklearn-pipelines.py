%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



plt.rcParams["figure.figsize"] = [8,8]
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
plt.plot(train_data['GrLivArea'], train_data['SalePrice'], 'ro')

plt.show()
train_data = train_data[train_data['GrLivArea']<4000]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

                                        train_data.drop('SalePrice',axis=1),

                                        train_data['SalePrice'],

                                        test_size=0.3,

                                        random_state=0

                                    )
pd.DataFrame(X_train.dtypes.values, columns=['dtype']).reset_index().groupby('dtype').count().plot(kind='bar')
#I Can't pip install a library in kernel, hence embedding source of the package

#This can be assumed to be lib code

import numpy as np

from sklearn.preprocessing import FunctionTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_pandas import DataFrameMapper



class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy='mean', filler='NA'):

        self.strategy = strategy

        self.fill = filler



    def fit(self, X, y=None):

        if self.strategy in ['mean', 'median']:

            if not all([dtype in [np.number, np.int] for dtype in X.dtypes]):

                raise ValueError('dtypes mismatch np.number dtype is required for ' + self.strategy)

        if self.strategy == 'mean':

            self.fill = X.mean()

        elif self.strategy == 'median':

            self.fill = X.median()

        elif self.strategy == 'mode':

            self.fill = X.mode().iloc[0]

        elif self.strategy == 'fill':

            if type(self.fill) is list and type(X) is pd.DataFrame:

                self.fill = dict([(cname, v) for cname, v in zip(X.columns, self.fill)])

        return self



    def transform(self, X, y=None):

        if self.fill is None:

            self.fill = 'NA'

        return X.fillna(self.fill)

    

def CustomMapper(result_column='mapped_col', value_map={}, default=np.nan):

    def mapper(X, result_column, value_map, default):

        def colmapper(col):

            return col.apply(lambda x: value_map.get(x, default))

        mapped_col = X.apply(colmapper).values

        mapped_col_names = [result_column + '_' + str(i) for i in range(mapped_col.shape[1])]

        return pd.DataFrame(mapped_col, columns=[mapped_col_names])

    return FunctionTransformer(

        mapper,

        validate=False,

        kw_args={'result_column': result_column, 'value_map': value_map, 'default': default}

    )
#numerical features

X_train.select_dtypes([int, float]).columns
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.pipeline import FeatureUnion, make_union

from sklearn_pandas import DataFrameMapper, gen_features

#import sklearn_pipeline_utils as skutils



# Using dataFrameMapper to map the chosen imputer to the columns

# we are imputing the columns that doesn't have missing values also, this is generally 

# a good practice because we are not making any assumptions on the test data



numerical_data_pipeline = DataFrameMapper(

        [

            (['LotFrontage',

              'LotArea',

              'OverallQual',

              'OverallCond', 

              'YearBuilt',

              'YearRemodAdd'],CustomImputer(strategy='median'), {'alias': 'num_data1'}

            ),

            (['BsmtFinSF1',

              'BsmtFinSF2',

              'BsmtUnfSF',

              'GrLivArea',

              '1stFlrSF',

              '2ndFlrSF',

              'BedroomAbvGr',

              'TotRmsAbvGrd',

              'Fireplaces',

              'GarageCars',

              'GarageArea',

              'WoodDeckSF'], CustomImputer(strategy='fill', filler=0), {'alias': 'num_data2'}

            )

        ],input_df=True ,df_out=True)
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



scaled_numerical_pipeline = make_pipeline(

    numerical_data_pipeline,

    StandardScaler(),

    MinMaxScaler()

)
numerical_data_pipeline.fit_transform(X_train).head()
scaled_numerical_pipeline.fit_transform(X_train)[0:2]
train_data.select_dtypes('object').columns
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from sklearn_pandas import gen_features



impute_mode_cols = gen_features(

    columns=['MSSubClass', 'MSZoning', 'LotShape', 'LandContour',

             'LotConfig', 'LandSlope', 'Foundation', 'Condition1',

             'Condition2', 'BldgType', 'HouseStyle'],

    classes=[

        {'class':CustomImputer,'strategy':'mode'},

        {'class':LabelBinarizer}

    ]

)



impute_NA_cols = gen_features(

    columns=['Neighborhood', 'SaleType', 'SaleCondition', 'RoofStyle', 'GarageType'],

    classes=[

        {'class':CustomImputer, 'strategy':'fill', 'filler':'NA'},

        {'class':LabelBinarizer}

    ]

)



categorical_data_pipeline = make_union(

    DataFrameMapper(impute_mode_cols, input_df=True, df_out=True),

    DataFrameMapper(impute_NA_cols, input_df=True, df_out=True)

)
# we wrote this manually in case of numerical data pipeline, we have used gen_features to genrate this here

# printing the first two, similar transformers mapping is generated for all columns

impute_mode_cols[0:2]
categorical_data_pipeline.fit_transform(X_train).shape
score_map = {

    'Ex' : 5.0, 'Gd' : 4.0,

    'TA' : 3.0,'Av' : 3.0,

    'Fa' : 2.0, 'Po' : 1.0,

    'NA' : 0, 'No' : 1.0,

    'GLQ': 6,'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA':0,

    'Fin' : 3, 'RFn' : 2,

    'Typ' : 6 ,'Min2': 5,

    'Min1': 4, 'Mod' : 3,

    'Maj1': 2, 'Maj2': 1,

    'Sev' : 0, 'Mn' : 2.0,

}



score_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 

             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

             'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond',

             'FireplaceQu', 'GarageFinish', 'Functional'

            ]



score_data_pipeline = DataFrameMapper([

    (score_cols, [CustomImputer(strategy='fill', filler='NA'),

                  CustomMapper(value_map=score_map, default=0)], {'alias': 'score_col'})

], input_df=True, df_out=True)
score_data_pipeline.fit_transform(X_train).head()
tdf = train_data.copy()
tdf['remod'] = tdf['YearBuilt']!=tdf['YearRemodAdd']

tdf.boxplot(column=['SalePrice'], by=['remod'])
tdf['recent_remod'] = tdf['YrSold'] == tdf['YearRemodAdd']

tdf.boxplot(column=['SalePrice'], by=['recent_remod'])
tdf['garage_remod'] = tdf['YearBuilt'] != tdf['GarageYrBlt']

tdf.boxplot(column=['SalePrice'], by=['garage_remod'])
### How are recently built house treated in the market



tdf['recentbuilt'] = tdf['YrSold']==tdf['YearBuilt']

tdf.boxplot(column=['SalePrice'], by=['recentbuilt'])
from sklearn.preprocessing import FunctionTransformer



def ColumnsEqualityChecker(result_column='equality_col', inverse=False):

    def equalityChecker(X, result_column, inverse=False):

        def roweq(row):

            eq = all(row.values == row.values[0])

            return eq

        eq = X.apply(roweq, axis=1)

        if inverse:

            eq = eq.apply(np.invert)

        return pd.DataFrame(eq.values.astype(int), columns=[result_column])

    return FunctionTransformer(

        equalityChecker,

        validate=False,

        kw_args={'result_column': result_column, 'inverse': inverse}

    )
engineered_feature_pipeline = DataFrameMapper([

    (['YearBuilt','YearRemodAdd'], ColumnsEqualityChecker(inverse=True)),

    (['YearRemodAdd', 'YrSold'], ColumnsEqualityChecker()),

    (['YearBuilt', 'GarageYrBlt'], ColumnsEqualityChecker(inverse=True)),

    (['YearBuilt', 'YrSold'], ColumnsEqualityChecker(inverse=True)),

], input_df=True, df_out=True)
engineered_feature_pipeline.fit_transform(X_train).head()
features_pipeline = make_union(scaled_numerical_pipeline, 

                      categorical_data_pipeline, 

                      score_data_pipeline,

                      engineered_feature_pipeline)
features_pipeline.fit_transform(X_train).shape
import xgboost as xgb

from sklearn.svm import SVR

from sklearn.linear_model import SGDRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import ExtraTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor 



regressors = [

    SVR(),

    SGDRegressor(),

    KNeighborsRegressor(),

    DecisionTreeRegressor(),

    ExtraTreeRegressor(),

    GradientBoostingRegressor(),

    AdaBoostRegressor(),

    xgb.XGBRegressor()

]
Regression_pipeline = Pipeline([

    ('features', features_pipeline),

    ('regressor', regressors[0])

])
from sklearn.model_selection import cross_validate

from pprint import pprint



for reg in regressors:

    Regression_pipeline.set_params(regressor=reg)

    scores = cross_validate(Regression_pipeline, X_train, y_train, scoring='neg_mean_squared_log_error', cv=10)

    print('----------------------')

    print(str(reg))

    print('----------------------')

    pprint('Leaderboard score - mean log rmse train '+str((-scores['train_score'].mean())**0.5))

    pprint('Leaderboard score - mean log rmse test '+str((-scores['test_score'].mean())**0.5))
Regression_pipeline.fit(X_train, y_train)
y_validation_predict = Regression_pipeline.predict(X_test)
from sklearn.metrics import mean_squared_log_error



score = (mean_squared_log_error(y_validation_predict, y_test))**0.5
print('Validation Score '+str(score))
X = train_data.drop('SalePrice', axis=1)

y = train_data['SalePrice']
Regression_pipeline.fit(X,y)

result = Regression_pipeline.predict(test_data)
def generate_submission(filename, y_predict):

    df = pd.DataFrame({'Id': range(1461,2920), 'SalePrice': y_predict})

    df.to_csv(filename, index=False)
#generate_submission('improved_pipe2.csv', result)