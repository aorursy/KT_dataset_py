import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn as sns

import warnings



# Do not limit number of columns in output.

pd.set_option('display.max_columns', None)



warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/train.csv')
train_data.head()
import ast



def flatten_field(field, attribute):

    """

    Convert a field from Python AST representation to a plain CSV string. To do that, project

    only `attribute` and ignore remaining field of the object.

    

    For example, the original field

        {'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Action'}

    is converted to

        "comedy,action"

        

    Note that this function converts all values to lower case.

    """

    if pd.isna(field):

        return ''

    else:

        obj_list = ast.literal_eval(field)

        result = []

        for obj in obj_list:

            result.append(obj[attribute].lower())

            

    return ','.join(result)





def flatten_data(data):

    col_attribute_mapping = {'belongs_to_collection': 'name', 

                             'genres': 'name', 

                             'production_countries': 'iso_3166_1',

                             'production_companies': 'name',

                             # For spoken language we actually want the ISO code instead of name to 

                             # avoid non-ASCII characters.

                             'spoken_languages': 'iso_639_1',

                             'Keywords': 'name', 

                             'cast': 'name', 

                             'crew': 'name'}

    

    for col in col_attribute_mapping.keys():

        data[col] = data.apply(lambda row: flatten_field(row[col], col_attribute_mapping[col]), 

                                           axis=1)
flatten_data(train_data)

train_data.head()
train_data.to_csv('train_flatten.csv', index=False)
fig, ax = plt.subplots(figsize=(10,5))

fig.suptitle('Revenue Distribution', fontsize=15)

sns.distplot(train_data['revenue'], bins=50, kde=False)

ax.grid()
sns.jointplot(x="budget", y="revenue", data=train_data, height=11, ratio=4, color="g")

plt.show()
sns.jointplot(x="popularity", y="revenue", data=train_data, height=11, ratio=4, color="g")

plt.show()
# Copied from https://www.kaggle.com/kamalchhirang/eda-feature-engineering-lgb-xgb-cat.

#Since only last two digits of year are provided, this is the correct way of getting the year.

train_data[['release_month','release_day','release_year']] = train_data['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

# Some rows have 4 digits of year instead of 2, that's why I am applying (train['release_year'] < 100) this condition

train_data.loc[ (train_data['release_year'] <= 19) & (train_data['release_year'] < 100), "release_year"] += 2000

train_data.loc[ (train_data['release_year'] > 19)  & (train_data['release_year'] < 100), "release_year"] += 1900
train_data.head()
avg_revenue_by_year = train_data.groupby(['release_year'], as_index=False)['revenue'].mean()



sns.jointplot(x='release_year', y='revenue', data=avg_revenue_by_year, height=11, ratio=4, color="g")

plt.show()
avg_revenue_by_language = train_data.groupby(['original_language'], as_index=False)['revenue'].mean()

plt.figure(figsize=(16, 6))

sns.barplot(x='original_language', y='revenue', data=avg_revenue_by_language)

plt.show()
train_data.describe()
# Note we are not including 'id' column.

train_data_subset = train_data[['budget', 'popularity', 'revenue']]
np.sum(pd.isna(train_data_subset))
train_data_subset = train_data_subset.fillna(0)



np.sum(pd.isna(train_data_subset))
from scipy import stats



revenue = stats.boxcox(train_data_subset['revenue'])[0]



revenue.head()

import statsmodels

import statsmodels.formula.api as smf





lr_model = smf.ols('revenue ~ budget + popularity', data=train_data_subset).fit()
lr_model.summary()
# Number of observations, number of columns

n, n_cols = train_data_subset.shape

# Number of predictors

p = n_cols - 1

rss = np.sum(np.power(lr_model.resid, 2))

rse = np.sqrt((1/(n-p-1)) * rss)

rse
rse / np.mean(train_data_subset['revenue'])
#fig, ax = plt.subplots(figsize=(6,2.5))

#ax.scatter(lr_model.fittedvalues, lr_model.resid)

import seaborn as sns



fit_data = pd.DataFrame({'Fitted': lr_model.fittedvalues, 'Residuals': lr_model.resid})



sns.regplot(x='Fitted', y='Residuals', data=fit_data)
import scipy as sp

fig, ax = plt.subplots(figsize=(6,2.5))

_, (__, ___, r) = sp.stats.probplot(lr_model.resid, plot=ax, fit=True)

r**2
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

from sklearn.model_selection import train_test_split



y = train_data['revenue']

X = train_data[['budget', 'popularity']]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lm = LinearRegression().fit(X_train, y_train)

y_predict = lm.predict(X_test)

y_predict[y_predict < 0] = 0

print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))

print("Mean squared log error (linear model): {:.2f}".format(mean_squared_log_error(y_test, y_predict)))

print("r2_score (linear model): {:.2f}".format(r2_score(y_test, y_predict)))
from sklearn.preprocessing import StandardScaler



y = train_data['revenue']

X = train_data[['budget', 'popularity']]



X.head()



X.loc[:,['budget', 'popularity']] = StandardScaler().fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lm = LinearRegression().fit(X_train, y_train)

y_predict = lm.predict(X_test)

y_predict[y_predict < 0] = 0

print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))

print("Mean squared log error (linear model): {:.2f}".format(mean_squared_log_error(y_test, y_predict)))

print("r2_score (linear model): {:.2f}".format(r2_score(y_test, y_predict)))
y = np.log(train_data['revenue'] + 1)

X = train_data[['budget', 'popularity']]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lm_log = LinearRegression().fit(X_train, y_train)

y_predict = lm_log.predict(X_test)

y_predict[y_predict < 0] = 0

print("Mean squared error (linear model, log(revenue)): {:.2f}".format(mean_squared_error(y_test, y_predict)))

print("Mean squared log error (linear model, log(revenue)): {:.2f}".format(mean_squared_log_error(y_test, y_predict)))

print("r2_score (linear model(log(revenue))): {:.2f}".format(r2_score(y_test, y_predict)))
# Preparing submission from linear model and linear model on log-transformed data.



test_data = pd.read_csv('../input/test.csv')

ids_test = test_data['id']

test_data_subset = test_data[['budget', 'popularity']]



y = train_data['revenue']

X = train_data[['budget', 'popularity']]



mean_y = np.mean(y)



lm = LinearRegression().fit(X, y)

y_predict = lm.predict(test_data_subset)



pred_df = pd.DataFrame({'id': ids_test, 'revenue': y_predict})

pred_df.loc[pred_df.revenue < 0, ['revenue']] = mean_y

pred_df.to_csv('lm_predictions.csv', index=False)



y = np.log(train_data['revenue'] + 1)

X = train_data[['budget', 'popularity']]



lm_log = LinearRegression().fit(X, y)

y_predict = lm_log.predict(test_data_subset)



pred_df = pd.DataFrame({'id': ids_test, 'revenue': np.exp(y_predict) - 1})

pred_df.loc[pred_df.revenue < 0, ['revenue']] = mean_y

pred_df.to_csv('lm_predictions_log.csv', index=False)

pred_df.head()
mean_y
train_data = pd.read_csv('../input/train.csv')

flatten_data(train_data)
train_data.head()
def count_unique(data, field_name):

    s = set()

    

    def count(field):

        for genre in field.split(','):

            s.add(genre)

        

    data.apply(lambda x: count(x[field_name]), axis=1)

    return s



print("Distinct values for 'genres':", len(count_unique(train_data, 'genres')))

print("Distinct values for 'original_language':", len(count_unique(train_data, 'original_language')))

print("Distinct values for 'production_companies':", len(count_unique(train_data, 'production_companies')))

print("Distinct values for 'production_countries':", len(count_unique(train_data, 'production_countries')))

print("Distinct values for 'spoken_languages':", len(count_unique(train_data, 'spoken_languages')))

print("Distinct values for 'Keywords':", len(count_unique(train_data, 'Keywords')))

print("Distinct values for 'cast':", len(count_unique(train_data, 'cast')))
def is_high_variance(data):

    return (np.sum(data) / data.shape[0]) > 0.15





def dummify_columns(data, add_label=True):

    # production_companies, Keywords and cast columns will be dealt with later.

    columns_to_dummify = ['genres', 'original_language', 'production_countries', 'spoken_languages']

    processed_data = data



    for column in columns_to_dummify:

        # Split column in values separated by comma and create dummies based on that instead

        # of considering the column value as a whole.

        dummies = processed_data[column].str.get_dummies(sep=",")

        processed_data.drop([column], axis=1, inplace=True)

        # Prefix dummy columns with the name of the original column so we don't end up with duplicated

        # column names when columns share the same level (what breaks XGBoost).

        dummies.columns = [column + "_" + dummy_column for dummy_column in dummies.columns ]

        before_len = dummies.shape[1]

        # Get rid of columns that do not provide much information.

        dummies = dummies.loc[:, is_high_variance(dummies)]

        after_len = dummies.shape[1]

        print(column, before_len, ' -> ', after_len)

        processed_data = pd.concat([processed_data, dummies], axis=1)

        

    return processed_data
processed_train_data = dummify_columns(train_data)

processed_train_data.head()
from datetime import datetime



def add_delta_col(row):

    format = "%m/%d/%Y"

    first_day_str = '1/1/' + str(row['release_year'])

    first_date = datetime.strptime(first_day_str, format)

    release_date_str = str(row['release_month']) + '/' + str(row['release_day']) + '/' + str(row['release_year'])

    release_date = datetime.strptime(release_date_str, format)

    

    return (release_date - first_date).days

    



def add_date_features(data):

    date_format = "%m/%d/%Y"

    

    # Split 'release_date' into separate fields for month and year, ignoring day.

    date_features = data['release_date'].str.split('/', expand=True).replace(np.nan, 1).astype(int)

    data['release_month'] = date_features[0]

    data['release_day'] = date_features[1]

    data['release_year'] = date_features[2]

    data.loc[(data['release_year'] <= 19) & (data['release_year'] < 100), "release_year"] += 2000

    data.loc[(data['release_year'] > 19)  & (data['release_year'] < 100), "release_year"] += 1900

    data['release_date_delta'] = data.apply(add_delta_col, axis=1)
add_date_features(processed_train_data)

processed_train_data.head()
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

from sklearn.model_selection import train_test_split





def report_scores(y, y_hat):

    print("Mean squared error (XGBoost): {:.2f}".format(mean_squared_error(y_test, y_predict)))

    print("Mean squared log error (XGBoost): {:.2f}".format(mean_squared_log_error(y_test, y_predict)))

    print("r2_score (XGBoost): {:.2f}".format(r2_score(y_test, y_predict)))    



    

y = processed_train_data['revenue']

X = processed_train_data.drop(['revenue'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

xgb_model = XGBRegressor(objective="reg:linear", random_state=0).fit(X_train, y_train)

y_predict = xgb_model.predict(X_test)

y_predict[y_predict < 0] = 0

report_scores(y_test, y_predict)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, make_scorer



train_data = pd.read_csv('../input/train.csv')

flatten_data(train_data)

add_date_features(train_data)

columns_to_drop = ['id', 'belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'overview',

                   'poster_path', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew',

                  'production_companies', 'release_date', 'release_day']



train_data = train_data.drop(columns_to_drop, axis=1)

train_data = dummify_columns(train_data)



train_data.head()
#scorer = make_scorer(mean_squared_error, greater_is_better=False)

from scipy import stats, special



grid_values = {

    'colsample_bytree': [0.5],

    'subsample': [0.9],

    'max_depth': [3, 5, 7],

    'n_estimators': [300, 500, 600, 700, 800],

    'gamma': [0],

    'min_child_weight': [2, 3, 4, 5],

    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05]

}



y = np.log(train_data['revenue'] + 1)

X = train_data.drop(['revenue'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



# UNCOMMENT NEXT LINE TO PERFORM GRID SEARCH.

#xgb_model = XGBRegressor(objective="reg:linear", n_jobs=4, random_state=0)

# Best so far.

# xgb_model = XGBRegressor(objective="reg:linear", n_jobs=4, learning_rate=0.05, n_estimators=300, max_depth=3, min_child_weight=5, random_state=0)

# Experimenting with tuning colsample_bytree and subsample first.

#grid_clf_acc = GridSearchCV(xgb_model, grid_values, fit_params={'eval_metric': 'rmse'}, scoring=scorer)

grid_clf_acc = GridSearchCV(xgb_model, grid_values, n_jobs=4, cv=5, fit_params={'eval_metric': 'rmse'}, scoring='neg_mean_squared_error')

grid_clf_acc.fit(X_train, y_train)

#y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 



print('Grid best parameter (max. MSE): ', grid_clf_acc.best_params_)

print('Grid best score (MSE): ', grid_clf_acc.best_score_)
xgb.plot_importance(grid_clf_acc.best_estimator_)
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

len_train_data = train_data.shape[0]

y = np.log(train_data['revenue'] + 1)

train_data = train_data.drop(['revenue'], axis=1)

ids_test = test_data['id']



# Create a single dataset to do preprocessing and split for training and prediction.

data = pd.concat([train_data, test_data], axis=0)

flatten_data(data)

add_date_features(data)

columns_to_drop = ['id', 'belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'overview',

                   'poster_path', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew',

                  'production_companies', 'release_date', 'release_day']

data = data.drop(columns_to_drop, axis=1)

data = dummify_columns(data)
data.head()
import xgboost as xgb



X_train = data[0:len_train_data]

X_test = data[len_train_data:data.shape[0]]

xgb_model = XGBRegressor(objective="reg:linear", n_jobs=4, learning_rate=0.03, n_estimators=500, max_depth=3, min_child_weight=5, colsample_bytree=0.5, subsample=0.9, random_state=0)

xgb_model.fit(X_train, y, eval_metric='rmse')



y_predict = np.exp(xgb_model.predict(X_test)) - 1



pred_df = pd.DataFrame({'id': ids_test, 'revenue': y_predict})

pred_df.loc[pred_df.revenue < 0, ['revenue']] = 0

pred_df.to_csv('xgb_predictions.csv', index=False)

pred_df.head()
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



X_normalized = StandardScaler().fit(train_data).transform(train_data)



pca = PCA(n_components=2).fit(X_normalized)



X_pca = pca.transform(X_normalized)



col = X_train['release_month'].astype(str)