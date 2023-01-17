import sys

import pandas as pd

import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (

    OneHotEncoder, LabelEncoder, FunctionTransformer,

    LabelBinarizer, StandardScaler, Imputer, MinMaxScaler

)

from sklearn.kernel_approximation import Nystroem, RBFSampler

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

%matplotlib inline



class TransformWrapper(TransformerMixin):

    def __init__(self, enc):

        self.enc = enc



    def fit(self, X, y=None):

        self.d = {}

        for col in X.columns:

            self.d[col] = self.enc()

            self.d[col].fit(X[col].values)

        return self

        

    def transform(self, X, y=None):



        l = list()

        for col in X.columns:

            l.append(self.d[col].transform(X[col].values).reshape(-1,1))

        result = np.hstack(l)

        return result





def write_answer(predict, output="answers.csv"):

    answer = pd.DataFrame({

        'Id': test.index.values,

        'SalePrice': predict.reshape(-1)

    })

    answer.to_csv(output, index=False)



class DataFrameImputer(TransformerMixin):

    """http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn"""



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value

        in column.



        Columns of other types are imputed with mean of column.



        """



    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

                              index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)

 

    

dtypes = {

    "BsmtFullBath": np.float64,

    "BsmtHalfBath": np.float64,

    "GarageCars": np.float64,

}

    

train = pd.read_csv('../input/train.csv', index_col="Id", dtype=dtypes)

test = pd.read_csv('../input/test.csv', index_col="Id", dtype=dtypes)



train_target = train['SalePrice']

train.drop(['SalePrice'], axis=1, inplace=True)

data = pd.concat([train, test])
categorical_cols = [

    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 

    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 

    'BldgType', 'HouseStyle',

    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 

    'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

    'Heating', 'HeatingQC', 'Electrical',  'KitchenQual', 'Functional', 

     'FireplaceQu', 

    'GarageType',  'GarageFinish',  'GarageQual', 'GarageCond', 

    'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',   'SaleType', 

    'SaleCondition',  'BsmtFinType2', 'CentralAir']



numerical_cols = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GrLivArea', 

       'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal',

       'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF',

       'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',

       'WoodDeckSF']

order_cols = [

    'OverallQual', 'OverallCond',  'MoSold',

    'YrSold', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'BedroomAbvGr',

    'HalfBath','TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'KitchenAbvGr',

]
pipe_cat = Pipeline([

    ("select", FunctionTransformer(

        lambda x: x[categorical_cols], validate=False)),    

    ("fillna", DataFrameImputer()),



    ("enc", TransformWrapper(LabelEncoder)),

    ("bin", OneHotEncoder(sparse=False)),

])



pipe_ordering = Pipeline([

    ("select", FunctionTransformer(

        lambda x: x[order_cols], validate=False)),

    ("fillna", DataFrameImputer()),

    ("enc", TransformWrapper(LabelEncoder)),

])



pipe_num = Pipeline([

    ("select", FunctionTransformer(

        lambda x: x[numerical_cols], validate=False)),

    ("fillna", Imputer(strategy="median")),

    ("scale",  StandardScaler()),

])





union = FeatureUnion([

    ('cat', pipe_cat),

    ('num', pipe_num),

    ('order', pipe_ordering)

])



pipe = Pipeline([

    ('union', union),

#    ('pca', PCA()),

])

pipe.fit(data)

X = pipe.transform(train)

y = np.log1p(train_target)

X_train, X_test, y_train, y_test = train_test_split(

    X, y,  test_size=0.33, random_state=42)

X.shape
from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor(

    learning_rate=0.1, n_estimators=100, 

    max_depth=2, min_samples_leaf=2, random_state=42

)

clf.fit(X_train, y_train)

predict = clf.predict(X_test)

mean_squared_error(y_test, predict)
clf_predict = clf.predict(pipe.transform(test))

write_answer(np.expm1(clf_predict), output="clf.csv")
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping





def make_model(input_dim=685):

    adam = Adam(lr=0.001)

    model = Sequential()

    model.add(Dense(120, input_dim=input_dim, init='normal', activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(60, init='normal', activation='relu'))

    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error', optimizer=adam)

    return model



early = EarlyStopping(patience=30)

model = make_model(X.shape[1])

h = model.fit(X_train, y_train, nb_epoch=1500, batch_size=50, verbose=0,

#             callbacks=[early,],

             validation_split=0.2)

plt.plot(h.history['val_loss'])

plt.ylim(ymax=20)

plt.plot(h.history['val_loss'])

plt.ylim(ymax=1)

plt.xlim(xmin=500)
predict = model.predict(X_test)

print(mean_squared_error(y_test, predict))
mlp_predict = model.predict(pipe.transform(test))

write_answer(np.expm1(mlp_predict), output="mlp.csv")
import xgboost

xgb = xgboost.XGBRegressor(

    learning_rate=0.1, 

    n_estimators=200, 

    reg_alpha=.15, 

    reg_lambda=1)

xgb.fit(X_train, y_train)

predict = xgb.predict(X_test)

print(mean_squared_error(y_test, predict))
xgb_predict = xgb.predict(pipe.transform(test))

write_answer(np.expm1(xgb_predict), output="xgb.csv")