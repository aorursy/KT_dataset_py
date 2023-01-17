import warnings

warnings.filterwarnings("ignore")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



filepath = '../input/diamonds/diamonds.csv'

diamond_data = pd.read_csv(filepath)

# Drop the first column since it has indexes we don't need

diamond_data.drop(diamond_data.columns[0], axis=1, inplace=True)



num_cols = diamond_data.select_dtypes(exclude=['object']).columns

cat_cols = np.setdiff1d(diamond_data.columns, num_cols)
for feature in diamond_data.columns:

    isnull = diamond_data.loc[diamond_data[feature].isnull()].shape[0]

    print(feature + " - " + str(isnull))
diamond_data[num_cols].describe()
diamond_data.drop(diamond_data[diamond_data.x == 0].index, inplace=True)

diamond_data.drop(diamond_data[diamond_data.y == 0].index, inplace=True)

diamond_data.drop(diamond_data[diamond_data.z == 0].index, inplace=True)

diamond_data.reindex()

diamond_data.describe()
diamond_data[cat_cols].describe()
import seaborn as sns

sns.distplot(diamond_data.price, kde=False)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

sns.scatterplot(x=diamond_data['carat'], y=diamond_data['price'], ax=ax1)

sns.scatterplot(x=diamond_data['depth'], y=diamond_data['price'], ax=ax2)

sns.scatterplot(x=diamond_data['table'], y=diamond_data['price'], ax=ax3)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

sns.stripplot(x=diamond_data['clarity'], y=diamond_data['price'], ax=ax1)

sns.stripplot(x=diamond_data['color'], y=diamond_data['price'], ax=ax2)

sns.stripplot(x=diamond_data['cut'], y=diamond_data['price'], ax=ax3)
clarity_cut = diamond_data[['clarity', 'cut']]

clarity_cut_concat = clarity_cut['clarity'].map(str) + '_' + clarity_cut['cut'].map(str)

cl_cut_counts = clarity_cut.assign(concat=clarity_cut_concat).groupby(['clarity', 'cut']).concat.count()



clarity_color = diamond_data[['clarity', 'color']]

clarity_color_concat = clarity_color['clarity'].map(str) + '_' + clarity_color['color'].map(str)

cl_col_counts = clarity_color.assign(concat=clarity_color_concat).groupby(['clarity', 'color']).concat.count()



color_cut = diamond_data[['color', 'cut']]

color_cut_concat = color_cut['color'].map(str) + '_' + color_cut['cut'].map(str)

col_cut_counts = color_cut.assign(concat=color_cut_concat).groupby(['color', 'cut']).concat.count()



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(data=cl_cut_counts.unstack(), ax=ax1)

sns.heatmap(data=cl_col_counts.unstack(), ax=ax2)

sns.heatmap(data=col_cut_counts.unstack(), ax=ax3)
clarity_dict = {'FL': 11,'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7,

                'VS2': 6, 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1}

color_dict = {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1}

cut_dict = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}



diamond_data['clarity'] = diamond_data['clarity'].map(clarity_dict)

diamond_data['cut'] = diamond_data['cut'].map(cut_dict)

diamond_data['color'] = diamond_data['color'].map(color_dict)

diamond_data[['clarity', 'cut', 'color']].head()
diamond_data['xyz'] = diamond_data['x'] * diamond_data['y'] * diamond_data['z']

xyz_carat = diamond_data['xyz'] * diamond_data['carat']

diamond_data['xyz_carat'] = xyz_carat.map(np.log)



depth_table = diamond_data['depth'] * diamond_data['table']

diamond_data['depth_table'] = depth_table.map(np.log)



carat_depth = diamond_data['depth'] * diamond_data['carat']

diamond_data['carat_depth'] = carat_depth.map(np.log)



carat_table = diamond_data['carat'] * diamond_data['table']

diamond_data['carat_table'] = carat_table.map(np.log)



diamond_data.describe()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

sns.scatterplot(x=diamond_data['xyz_carat'], y=diamond_data['price'], ax=ax1)

sns.scatterplot(x=diamond_data['carat_table'], y=diamond_data['price'], ax=ax2)

sns.scatterplot(x=diamond_data['carat_depth'], y=diamond_data['price'], ax=ax3)
import itertools

from sklearn.preprocessing import LabelEncoder



interactions = pd.DataFrame(index=diamond_data.index)

for col1, col2 in itertools.combinations(cat_cols, 2):

    col_name = col1 + '_' + col2

    interaction = diamond_data[col1].map(str) + '_' + diamond_data[col2].map(str)

    encoder = LabelEncoder()

    interactions[col_name] = encoder.fit_transform(interaction)

    

diamond_data = diamond_data.join(interactions)

diamond_data.head()
def train_valid_test_split(data, train_percent, valid_percent):

    np.random.seed(6)

    perm = np.random.permutation(diamond_data.index)

    n = len(data.index)

    train_end = int(train_percent * n)

    valid_end = int(valid_percent * n) + train_end

    train = data.loc[perm[:train_end]]

    valid = data.loc[perm[train_end:valid_end]]

    test = data.loc[perm[valid_end:]]

    return train, valid, test



train, valid, test = train_valid_test_split(diamond_data, 0.7, 0.2)

train_X = train.drop('price', axis=1)

train_y = train.price

valid_X = valid.drop('price', axis=1)

valid_y = valid.price

test_X = test.drop('price', axis=1)

test_y = test.price
from sklearn.metrics import mean_absolute_error



def evaluate_model(model, train_X, train_y, valid_X, valid_y):

    model.fit(train_X, train_y)

    predictions = model.predict(valid_X)

    return mean_absolute_error(valid_y, predictions)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

import lightgbm as lgb



models = [('DecisionTreeRegressor', DecisionTreeRegressor()),

          ('RandomForestRegressor', RandomForestRegressor()),

          ('LinearRegression', LinearRegression()),

          ('XGBRegressor', XGBRegressor())]



for model_name, model in models:

    print('mae for ' + model_name + ": ", end='')

    print(evaluate_model(model, train_X, train_y, valid_X, valid_y))
model = RandomForestRegressor()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression



params = [6, 8, 10, 12, 14, 15, 16, 17]

for k in params:

    selector = SelectKBest(f_regression, k=k)

    X_new = selector.fit_transform(train_X, train_y)

    selected_features = pd.DataFrame(selector.inverse_transform(X_new),

                                     index=train_X.index, columns=train_X.columns)

    selected_columns = selected_features.columns[selected_features.var() != 0]

    print('mae for k = {}: '.format(k), end='')

    print(evaluate_model(model, train_X[selected_columns], train_y, valid_X[selected_columns], valid_y))
selector = SelectKBest(f_regression, k=15)

X_new = selector.fit_transform(train_X, train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new),

                                     index=train_X.index, columns=train_X.columns)

selected_columns = selected_features.columns[selected_features.var() != 0]

    

train_X = train_X[selected_columns]

valid_X = valid_X[selected_columns]

test_X = test_X[selected_columns]

selected_columns
n_estimators = range(5, 96, 15)



for num in n_estimators:

    print('mae for n_estimators = {}: '.format(num), end='')

    print(evaluate_model(RandomForestRegressor(n_estimators=num),train_X, train_y, valid_X, valid_y))
print('mae on test data: ')

print(evaluate_model(RandomForestRegressor(n_estimators=50),train_X, train_y, test_X, test_y))