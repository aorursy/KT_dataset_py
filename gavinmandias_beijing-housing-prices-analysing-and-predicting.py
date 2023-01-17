# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import LinearSVR, SVR

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.externals import joblib

from random import randint

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def load_housing_data(file_path):

    return pd.read_csv(file_path, sep=',', encoding='iso-8859-1', low_memory=False)

housing = load_housing_data("../input/new.csv")

housing.info()

housing.head()
housing.describe()
housing.hist(bins=50, figsize=(20,15))

plt.show()
fig = plt.scatter(x=housing['Lat'], y=housing['Lng'], alpha=0.4, \

    s=housing['totalPrice']/100, label='Price', \

    c=housing['district'], cmap=plt.get_cmap('jet'))

plt.colorbar(fig)

plt.legend()

plt.show()
fig = plt.scatter(x=housing['Lat'], y=housing['Lng'], alpha=0.4, \

    c=housing['totalPrice'], cmap=plt.get_cmap('jet'))

plt.colorbar(fig)

plt.show()
f,ax = plt.subplots(figsize=(25, 25))

sns.heatmap(housing.corr(), annot = True, linewidth = .5, fmt = ".3f",ax = ax)

plt.show()
price_by_trade_time = pd.DataFrame()

price_by_trade_time['totalPrice'] = housing['totalPrice']

price_by_trade_time.index = housing['tradeTime'].astype('datetime64[ns]')

price_by_trade_month = price_by_trade_time.resample('M').mean().to_period('M').fillna(0)

price_by_trade_month.plot(kind='line')

plt.show()
price_stat_trade_month_index = [x.strftime('%Y-%m') for x in set(price_by_trade_time.to_period('M').index)]

price_stat_trade_month_index.sort()

price_stat_trade_month = []

for month in price_stat_trade_month_index:

    price_stat_trade_month.append(price_by_trade_time[month]['totalPrice'].values)

price_stat_trade_month = pd.DataFrame(price_stat_trade_month)

price_stat_trade_month.index = price_stat_trade_month_index

price_stat_trade_month = price_stat_trade_month.T

price_stat_trade_month.boxplot(figsize=(15,10))

plt.xticks(rotation=90,fontsize=7)

plt.show()
price_by_cons_time = pd.DataFrame()

price_by_cons_time['totalPrice'] = housing['totalPrice']

price_by_cons_time['constructionTime'] = housing['constructionTime']

price_by_cons_time = price_by_cons_time[

	(price_by_cons_time.constructionTime != '0')

	& (price_by_cons_time.constructionTime != '1')

	& (price_by_cons_time.constructionTime != 'Î´Öª')

]

price_by_cons_time['constructionTime'] = price_by_cons_time['constructionTime'].astype('int64')

price_by_cons_time['constructionTime'] = 2018 - price_by_cons_time['constructionTime']

price_by_cons_time_index = list(set(price_by_cons_time['constructionTime']))

price_by_cons_time_index.sort()

price_by_cons_time.index = price_by_cons_time['constructionTime']

price_by_cons_time = price_by_cons_time.drop('constructionTime', axis=1)

price_by_cons_time_line = []

price_by_cons_time_stat = []

for years in price_by_cons_time_index:

	price_by_cons_time_line.append(price_by_cons_time.loc[years]['totalPrice'].mean())

	try:

		price_by_cons_time_stat.append(price_by_cons_time.loc[years]['totalPrice'].values)

	except Exception:

		price_by_cons_time_stat.append(np.array([price_by_cons_time.loc[years]['totalPrice']]))

plt.plot(list(price_by_cons_time_index), price_by_cons_time_line)

plt.show()
price_by_cons_time_stat = pd.DataFrame(price_by_cons_time_stat)

price_by_cons_time_stat.index = price_by_cons_time_index

price_by_cons_time_stat = price_by_cons_time_stat.T

price_by_cons_time_stat.boxplot(figsize=(20,15))

plt.show()
price_by_cons_time_stat.boxplot(figsize=(20,15))

plt.ylim(0,2500)

plt.show()
#square and price

price_by_square = pd.DataFrame()

price_by_square['totalPrice'] = housing['totalPrice']

price_by_square['square'] = housing['square']

price_by_square['square'] = np.ceil(price_by_square['square'])

price_by_square['square'] = price_by_square['square'] - (price_by_square['square'] % 10)

price_by_square_index = list(set(price_by_square['square']))

price_by_square_index.sort()

price_by_square.index = price_by_square['square']

price_by_square_line = []

price_by_square_stat = []

for squares in price_by_square_index:

	price_by_square_line.append(price_by_square.loc[squares]['totalPrice'].mean())

	try:

		price_by_square_stat.append(price_by_square.loc[squares]['totalPrice'].values)

	except Exception:

		price_by_square_stat.append(np.array([price_by_square.loc[squares]['totalPrice']]))

plt.plot(price_by_square_index, price_by_square_line)

plt.show()
price_by_square['square'].hist(bins=50, figsize=(20,15))

plt.show()
price_by_square['square'].hist(bins=100, figsize=(20,15))

plt.xlim(0,500)

plt.show()
price_by_square_stat = pd.DataFrame(price_by_square_stat).T

price_by_square_index = [int(x) for x in price_by_square_index]

price_by_square_stat.columns = price_by_square_index

price_by_square_stat.boxplot(figsize=(20,15))

plt.xticks(rotation=90)

plt.ylim(0,5000)

plt.show()
def get_mean(price_by_square):

	try:

		price_by_square_index = list(set(price_by_square['square']))

		price_by_square_index.sort()

		price_by_square_line = []

		price_by_square.index = price_by_square['square']

		for squares in price_by_square_index:

			price_by_square_line.append(price_by_square.loc[squares]['totalPrice'].mean())

		price_by_square_index = [int(x) for x in price_by_square_index]

	except Exception:

		price_by_square_line = [price_by_square.loc['totalPrice']]

		price_by_square_index = [int(price_by_square['square'])]

	return price_by_square_line, price_by_square_index
price = pd.DataFrame()

price['totalPrice'] = housing['totalPrice']

price['square'] = housing['square']

price.index = housing['tradeTime'].astype('datetime64[ns]')

price['square'] = np.ceil(price['square'])

price['square'] = price['square'] - (price['square'] % 10)

price = price.to_period('Y')

price_time_index = [x.strftime('%Y') for x in set(price.index)]

price_time_index.sort()

colormap = mpl.cm.Dark2.colors

m_styles = ['','.','o','^','*']

for year, (maker, color) in zip(price_time_index, itertools.product(m_styles, colormap)):

    y, x = get_mean(price.loc[year])

    plt.plot(x, y, color=color, marker=maker, label=year)

plt.xticks(rotation=90)

plt.legend(price_time_index)

plt.show()
for year, (maker, color) in zip(price_time_index, itertools.product(m_styles, colormap)):

    y, x = get_mean(price.loc[year])

    plt.plot(x, y, color=color, marker=maker, label=year)

plt.xticks(rotation=90)

plt.legend(price_time_index)

plt.xlim(0,750)

plt.show()
for year, (maker, color) in zip(price_time_index, itertools.product(m_styles, colormap)):

    y, x = get_mean(price.loc[year])

    plt.plot(x, y, color=color, marker=maker, label=year)

plt.xticks(rotation=90)

plt.legend(price_time_index)

plt.xlim(0,750)

plt.ylim(0,5000)

plt.show()
class DataNumCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, clean=True):

        self.clean = clean

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        if self.clean:

            X = X[(X.constructionTime != '0') & (X.constructionTime != '1') & (X.constructionTime != 'Î´Öª')]

            X['constructionTime'] = 2018 - X['constructionTime'].astype('int64')

            X = X[(X.buildingType == 1) | (X.buildingType == 2) | (X.buildingType == 3) | (X.buildingType == 4)]

            X = X[X.livingRoom != '#NAME?']

            X = X[(X.drawingRoom == '0') | (X.drawingRoom == '1') | (X.drawingRoom == '2') | (X.drawingRoom == '3') | (X.drawingRoom == '4') | (X.drawingRoom == '5')]

            X = X[(X.bathRoom == '0') | (X.bathRoom == '1') | (X.bathRoom == '2') | (X.bathRoom == '3') | (X.bathRoom == '4') | (X.bathRoom == '5') | (X.bathRoom == '6') | (X.bathRoom == '7')]

            X.bathRoom = X.bathRoom.astype('float64')

            X.drawingRoom = X.drawingRoom.astype('float64')

            X.livingRoom = X.livingRoom.astype('float64')

            return X

        else:

            return X
class DataFrameSelector(BaseEstimator, TransformerMixin):

    """docstring for DataFrameSelector"""

    def __init__(self, attribute_name):

        self.attribute_name = attribute_name

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_name].values
def display_scores(scores):

    print('Scores: ', scores)

    print('Mean: ', scores.mean())

    print('Standard deviation: ', scores.std())
housing = load_housing_data('../input/new.csv')

housing = housing.drop(['url','id','price','Cid','DOM','tradeTime', 'floor', 'followers', 'communityAverage'], axis=1)

housing.head()
spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in spliter.split(housing, housing['district']):

    train_set = housing.loc[train_index]

    test_set = housing.loc[test_index]



housing = train_set.copy()
num_attributes = ['Lng', 'Lat', 'square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'constructionTime',  'ladderRatio', 'elevator', 'fiveYearsProperty', 'subway']

cat_attributes = ['buildingType', 'renovationCondition', 'buildingStructure', 'district']



num_pipeline = Pipeline([

    ('cleaner', DataNumCleaner()),

    ('selector', DataFrameSelector(num_attributes)),

    ('imputer', Imputer(strategy='most_frequent')),

    ('std_scaler', StandardScaler())

])



cat_pipeline = Pipeline([

    ('cleaner', DataNumCleaner()),

    ('selector', DataFrameSelector(cat_attributes)),

    ('encoder', OneHotEncoder())

])



label_pipeline = Pipeline([

    ('cleaner', DataNumCleaner()),

    ('selector', DataFrameSelector(['totalPrice']))

])



full_pipeline = FeatureUnion([

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline)

])
housing_prepared = full_pipeline.fit_transform(housing)

housing_label = label_pipeline.fit_transform(housing)
lin_svm_reg = LinearSVR()

lin_svm_reg_scores = cross_val_score(lin_svm_reg, housing_prepared, housing_label, scoring='neg_mean_squared_error', cv=10)

lin_svm_reg_rmse_scores = np.sqrt(-lin_svm_reg_scores)

display_scores(lin_svm_reg_rmse_scores)
lin_svm_reg = LinearSVR(C=0.5,loss='squared_epsilon_insensitive')

lin_svm_reg.fit(housing_prepared,housing_label)
X_test_prepared = full_pipeline.fit_transform(test_set)

y_test = label_pipeline.fit_transform(test_set)

final_predictons = lin_svm_reg.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictons)
final_rmse = np.sqrt(final_mse)

print(final_rmse)
test_index = [randint(0,len(y_test)) for i in range(100)]

y_label = [y_test[index] for index in test_index]

y_predict = [lin_svm_reg.predict(X_test_prepared[index]) for index in test_index]

x = [i+1 for i in range(100)]

plt.plot(x, y_label, c='red', label='label')

plt.plot(x, y_predict, c='blue', label='predict')

plt.legend()

plt.show()
joblib.dump(lin_svm_reg,'BeijingHousingPricePredicter.pkl')