import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings

from fancyimpute import SimpleFill

from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)
#product data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#sales, exchange rates, social network data
sales = pd.read_csv('../input/sales.csv')

#website navigation data
navigation = pd.read_csv('../input/navigation.csv')

#product images vectorized with ResNet50
vimages = pd.read_csv('../input/vimages.csv')
sales_float_columns = sales.dtypes[sales.dtypes == 'float64'].index.tolist()
sales.loc[:,sales_float_columns] = SimpleFill(fill_method='random').fit_transform(sales.loc[:,sales_float_columns])
navigation.loc[navigation.website_version_zone_number.isna(), 'website_version_zone_number'] = 'unknown'
navigation.loc[navigation.website_version_country_number.isna(), 'website_version_country_number'] = 'unknown'
train.loc[train.color.isna(), 'color'] = 'unknown'
test.loc[test.color.isna(), 'color'] = 'unknown'
currency_and_social_columns = sales.columns[9:].tolist()
first_day = sales.loc[sales.Date == 'Day_1',:]
all_currency_and_social = sales.groupby('sku_hash').mean()[currency_and_social_columns]
first_day_currency_and_social = first_day.groupby('sku_hash').mean()[currency_and_social_columns]
first_day_currency_and_social.columns = ['first_day_' + col for col in first_day_currency_and_social.columns]
all_sales = sales.groupby('sku_hash').sum()['sales_quantity']
all_sales = pd.DataFrame(all_sales)
first_day_sales = first_day.groupby(['sku_hash', 'day_transaction_date', 'Month_transaction']).sum()['sales_quantity']
first_day_sales = pd.DataFrame(first_day_sales)
first_day_sales.columns = ['first_day_sales']
first_day_sales.reset_index(inplace=True)
first_day_sales.set_index('sku_hash', inplace=True)
sales_data = pd.merge(all_sales, first_day_sales, left_index=True, right_index=True)
sales_data = pd.merge(sales_data, all_currency_and_social, left_index=True, right_index=True)
sales_data = pd.merge(sales_data, first_day_currency_and_social, left_index=True, right_index=True)
monthDict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
sales_data.Month_transaction = sales_data.Month_transaction.astype('object').map(monthDict)
sales_data.head()
first_day_navigation = navigation.loc[navigation.Date == 'Day 1',:]
first_day_views = first_day_navigation.groupby('sku_hash').sum()[['page_views', 'addtocart']]
first_day_views.columns = ['first_day_page_views', 'first_day_addtocart']
views = navigation.groupby('sku_hash').sum()[['page_views', 'addtocart']]
navigation_data = pd.merge(views, first_day_views, left_index=True, right_index=True)
sales_data.sales_quantity = sales_data.sales_quantity.astype('float64')
sales_data.first_day_sales = sales_data.first_day_sales.astype('float64')
sales_data['sales_quantity_log'] = (sales_data.sales_quantity + 1).apply(np.log)
sales_data['first_day_sales_log'] = (sales_data.first_day_sales + 1).apply(np.log)
train_data = pd.merge(train, sales_data, left_on='sku_hash', right_index=True)
train_data = pd.merge(train_data, navigation_data, how='left', left_on='sku_hash', right_index=True)
train_data = pd.merge(train_data, vimages, left_on='sku_hash', right_on='sku_hash')
test_data = pd.merge(test, sales_data, left_on='sku_hash', right_index=True)
test_data = pd.merge(test_data, navigation_data, how='left', left_on='sku_hash', right_index=True)
test_data = pd.merge(test_data, vimages, left_on='sku_hash', right_on='sku_hash')
train_data[navigation_data.columns] = train_data[navigation_data.columns].fillna(0)
test_data[navigation_data.columns] = test_data[navigation_data.columns].fillna(0)


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

        if isinstance(self.columns, str):
            self.columns = [self.columns]

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return list(x[selected_cols[0]])
        else:
            return x[selected_cols]
train_data1 = train_data.loc[train_data.month == 1, :].copy()
train_data1.drop(['month', 'sku_hash', 'ID'], axis=1, inplace=True)

X_test1 = test_data.loc[test_data.month == 1, :].copy()
X_test1.drop(['month', 'sku_hash'], axis=1, inplace=True)
X_test1.set_index('ID', inplace=True)

y_train1 = (train_data1.target + 1).apply(np.log)
X_train1 = train_data1.drop('target', axis=1)

train_data2 = train_data.loc[train_data.month == 2, :].copy()
train_data2.drop(['month', 'sku_hash', 'ID'], axis=1, inplace=True)

X_test2 = test_data.loc[test_data.month == 2, :].copy()
X_test2.drop(['month', 'sku_hash'], axis=1, inplace=True)
X_test2.set_index('ID', inplace=True)

y_train2 = (train_data2.target + 1).apply(np.log)
X_train2 = train_data2.drop('target', axis=1)


train_data3 = train_data.loc[train_data.month == 3, :].copy()
train_data3.drop(['month', 'sku_hash', 'ID'], axis=1, inplace=True)

X_test3 = test_data.loc[test_data.month == 3, :].copy()
X_test3.drop(['month', 'sku_hash'], axis=1, inplace=True)
X_test3.set_index('ID', inplace=True)

y_train3 = (train_data3.target + 1).apply(np.log)
X_train3 = train_data3.drop('target', axis=1)
images_cols = vimages.columns[1:].tolist()
float_cols = X_train1.dtypes[X_train1.dtypes == 'float64'].index.tolist()
float_cols = list(set(float_cols) - set(images_cols))
float_cols.remove('sales_quantity_log')
float_cols.remove('first_day_sales_log')
float_cols.remove('sales_quantity')
float_cols.remove('first_day_sales')
categorical_cols = X_train1.dtypes[X_train1.dtypes == 'object'].index.tolist()
categorical_cols.remove('en_US_description')
categorical_cols.remove('color')
model = make_pipeline(
    make_union(
        make_pipeline(PandasSelector(columns='en_US_description'), 
                      CountVectorizer(stop_words='english'),
                      LatentDirichletAllocation(n_components=10)),
        make_pipeline(PandasSelector(columns='color'), 
                      CountVectorizer()
                     ),
        make_pipeline(PandasSelector(columns=images_cols), 
                      PCA(10)),
        make_pipeline(PandasSelector(columns=float_cols),
                      PCA(10)),
        make_pipeline(PandasSelector(columns=['sales_quantity_log', 
                                              'first_day_sales_log', 
                                              'sales_quantity', 
                                              'first_day_sales'])),
        make_pipeline(PandasSelector(columns=categorical_cols), 
                      OneHotEncoder(handle_unknown='ignore'),
                      LatentDirichletAllocation(n_components=10))
    ),
    SelectFromModel(RandomForestRegressor(n_estimators=100)),
    DecisionTreeRegressor()
)
params = {'decisiontreeregressor__min_samples_split': [40, 60, 80],
          'decisiontreeregressor__max_depth': [4, 6, 8]}
gs = GridSearchCV(model, param_grid=params, cv=4, verbose=3, n_jobs=-1)

gs.fit(X_train1, y_train1)
y_test1 = gs.predict(X_test1)

print('metric cv: ', np.round(np.sqrt(gs.best_score_),4))
print('metric train: ', np.round(np.sqrt(mean_squared_error(y_train1, gs.predict(X_train1))),4))
print('params: ', gs.best_params_)
gs.fit(X_train2, y_train2)
y_test2 = gs.predict(X_test2)

print('metric cv: ', np.round(np.sqrt(gs.best_score_),4))
print('metric train: ', np.round(np.sqrt(mean_squared_error(y_train2, gs.predict(X_train2))),4))
print('params: ', gs.best_params_)
gs.fit(X_train3, y_train3)
y_test3 = gs.predict(X_test3)

print('metric cv: ', np.round(np.sqrt(gs.best_score_),4))
print('metric train: ', np.round(np.sqrt(mean_squared_error(y_train3, gs.predict(X_train3))),4))
print('params: ', gs.best_params_)
y_test1 = pd.Series(y_test1)
y_test1 = (y_test1).apply(np.exp)  - 1
y_test1.index = X_test1.index

y_test2 = pd.Series(y_test2)
y_test2 = (y_test2).apply(np.exp)  - 1
y_test2.index = X_test2.index

y_test3 = pd.Series(y_test3)
y_test3 = (y_test3).apply(np.exp) - 1
y_test3.index = X_test3.index

submission = pd.DataFrame(pd.concat([y_test1, y_test2, y_test3]))
submission.columns = ['target']
submission.to_csv('submission.csv')
