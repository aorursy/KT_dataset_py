import numpy as np 

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import scale

import seaborn as sns

from matplotlib import pyplot as plt

import math

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head()
data.dtypes
data.isnull().sum()
#Split the data into real and categorical columns

categorical_columns = ['neighbourhood_group', 'neighbourhood']

real_columns = ['latitude', 'longitude', 'price', 'reviews_per_month', 'minimum_nights', 

                'number_of_reviews', 'calculated_host_listings_count', 'availability_365',

                'calculated_host_listings_count', 'number_of_reviews']
for cat_col in categorical_columns:

    print(cat_col)

    print(data[cat_col].unique())
for real_col in real_columns:

    print('{}, min = {}, max = {}, mean = {}, std = {}'.format(real_col, data[real_col].min(),

                                                             data[real_col].max(), data[real_col].mean(),

                                                             data[real_col].std()))
for real_col in real_columns:

    sns.pairplot(data,  y_vars=['price'], x_vars = [real_col], height = 4)
plt.imshow(plt.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png', 0),

          zorder = 0,  extent=[-74.24441999999999, -73.71299, 40.499790000000004,40.913059999999994])

ax = plt.gca()



plot_data = data[data.price < 1000]

plot_data.plot(kind='scatter', x='longitude', y='latitude', c='price', ax=ax,

               cmap=plt.get_cmap('plasma'), colorbar=True, alpha = 1, figsize=(10,8),

               zorder = 1)
sns.distplot(data['price'])
sns.distplot(np.log1p(data['price']))
sns.distplot(np.log1p(data['availability_365']))
sns.distplot(data['number_of_reviews'])
sns.distplot(np.log1p(data['number_of_reviews']))
sns.pairplot(data, x_vars = ['neighbourhood_group'], y_vars = ['price'], height = 10)
sns.violinplot(data = data, x = 'neighbourhood_group', y = 'price', height = 10)
sns.violinplot(data = data[data.price < 1000], x = 'neighbourhood_group', y = 'price', height = 10)
sns.violinplot(data = data[data.price < 500], x = 'neighbourhood_group', y = 'price', height = 10)
data[['neighbourhood', 'price']].groupby('neighbourhood').mean().plot(kind = 'box')
data[['neighbourhood_group', 'price']].groupby('neighbourhood_group').mean().plot(kind = 'box')
a4_dims = (12, 12)

fig, ax = plt.subplots(figsize=a4_dims)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.heatmap(data.corr(), cmap = cmap, annot=True, ax = ax)
#I omit neighbourhood and neighbourhood_group because lat and lon and these features are the same.  

#I omit last_review reviews_per_month, number_of_reviews because when a new listing is published, we do not know the number of reviews. 

cleared_data = data.drop(['neighbourhood', 'neighbourhood_group', 'id',	'name',	'host_id',	'host_name', 'last_review', 'reviews_per_month'], axis = 'columns')
cleared_data.head()
cleared_data = pd.get_dummies(cleared_data)
print(cleared_data.shape)
cleared_data.head()
X = cleared_data.drop('price', axis = 'columns')

y = cleared_data['price'].replace(np.nan, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 231)
def print_regressor_error(model):

    print('Square train error: ' + str(((model.predict(X_train) - y_train)**2).mean()))

    print('Abs train error: ' + str((abs(model.predict(X_train) - y_train).mean())))



    print('Square test error: ' + str(((model.predict(X_test) - y_test)**2).mean()))

    print('Abs test error: ' + str((abs(model.predict(X_test) - y_test).mean())))
lrgr = LinearRegression().fit(X_train, y_train)
lrgr.coef_
#We can see that we are mistaken by 70$, that is quite big mistake

#Lets try ridge and lasso

print_regressor_error(lrgr)
#same error...

lasso = Lasso(alpha=1).fit(X_train, y_train)

lasso.coef_
for column, weight in zip(X.columns, lasso.coef_):

    if (weight != 0):

        print('{} = {}'.format(column, weight))
lasso.predict(X_test)
print_regressor_error(lasso)
#also the same

ridge = Ridge(max_iter  = 10000).fit(X_train, y_train)
print_regressor_error(ridge)
ridge.coef_
random_forest = RandomForestRegressor(n_estimators = 300, max_depth = 20, criterion = 'mse', bootstrap = True).fit(X_train, y_train)
print_regressor_error(random_forest)
sns.distplot(np.log1p(y_train))

sns.distplot(np.log1p(random_forest.predict(X_train)))
sns.distplot(np.log1p(y_test))

sns.distplot(np.log1p(random_forest.predict(X_test)))
def get_features_importances(classifier, curr_data):

    importance = []

    for feature, weight in zip(curr_data.columns, classifier.feature_importances_):

        importance.append((feature, weight))



    importance = sorted(importance, key = lambda x : -x[1])

    

    return importance
get_features_importances(random_forest, X)
bagging_regressor = BaggingRegressor(n_estimators = 40).fit(X_train, y_train)
print_regressor_error(bagging_regressor)
#now let us try mlp

mlp = MLPRegressor(hidden_layer_sizes = (230), solver = 'lbfgs', alpha = 0.01).fit(X_train, y_train)
print_regressor_error(mlp)
mlp.predict(X_test)
grad_boost_rgr = GradientBoostingRegressor(n_estimators = 150).fit(X_train, y_train)
print_regressor_error(grad_boost_rgr)
first_bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 3000,

       4000]

second_bins = np.linspace(0, 10000, 65)
sns.distplot(first_bins)
sns.distplot(second_bins)
def get_classes(bins, prices):

    prices_classes = []

    for i in range(len(prices)):

        curr_price = prices[i]



        index = 0

        while (index + 1 < len(bins) and curr_price > bins[index + 1]):

            index += 1

        

        prices_classes.append(index)



    return prices_classes



first_prices_classes = get_classes(first_bins, np.array(cleared_data['price']))

second_prices_classes = get_classes(second_bins, np.array(cleared_data['price']))
X_first, y_first = X, np.array(first_prices_classes).reshape(len(y))
X_train_first, X_test_first, y_train_first, y_test_first = train_test_split(X_first, y_first, test_size = 0.2)
from sklearn.ensemble import RandomForestClassifier



rnd_first_forest_classifier = RandomForestClassifier(n_estimators = 500, max_depth = 20).fit(X_train_first, y_train_first)
accuracy_score(rnd_first_forest_classifier.predict(X_test_first), y_test_first)
get_features_importances(rnd_first_forest_classifier, X_first)
X_second, y_second = X, np.array(second_prices_classes).reshape(len(y))
X_train_second, X_test_second, y_train_second, y_test_second = train_test_split(X_second, y_second, test_size = 0.2)
from sklearn.ensemble import RandomForestClassifier



rnd_second_forest_classifier = RandomForestClassifier(n_estimators = 500, max_depth = 20).fit(X_train_second, y_train_second)
accuracy_score(rnd_second_forest_classifier.predict(X_test_second), y_test_second)
get_features_importances(rnd_second_forest_classifier, X_second)
def normalize(curr_data):

    for column in curr_data.columns:

        mean = curr_data[column].mean()

        std = curr_data[column].std()



        curr_data[column] = (curr_data[column] - mean) / std
new_columns = ['longitude', 'room_type', 'latitude', 'availability_365', 'number_of_reviews', 'calculated_host_listings_count', 'minimum_nights', 'price']
new_reg_data = data[new_columns]
new_reg_data = pd.get_dummies(new_reg_data)
new_reg_data.head()
new_reg_data = new_reg_data.drop(['room_type_Shared room'], axis = 'columns')
new_reg_data.head()
new_price_std = new_reg_data['price'].std()

new_price_mean = new_reg_data['price'].mean()
normalize(new_reg_data)
new_reg_data.tail()
X_new, y_new = new_reg_data.drop('price', axis = 'columns'), new_reg_data['price']
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, train_size = 0.2)
new_reg = LinearRegression().fit(X_new_train, y_new_train)
new_reg.predict(X_new_train)
new_reg.coef_
mse_err = ((new_reg.predict(X_new_train) - y_new_train)**2).mean()

abs_err = abs(new_reg.predict(X_new_test) - y_new_test).mean()



real_abs_err = abs_err * new_price_std + new_price_mean



print('Square train error: ' + str((mse_err)))

print('Abs train error: ' + str((abs_err)))

      

print('Real Abs train error: ' + str((real_abs_err)))
new_lasso = Lasso().fit(X_new_train, y_new_train)
new_lasso.coef_
new_lasso.predict(X_new_train)
ridge = Ridge().fit(X_new_train, y_new_train)
ridge.coef_
ridge.predict(X_new_train)