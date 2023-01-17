import numpy as np

import pandas as pd



# read in the csv into a DataFrame

df = pd.read_csv('../input/train.csv')
# view the first 5 rows

df.head()
# view a summary of column names, counts, and data types

df.info()
# view statistical summary of numerical features

df.describe()
import matplotlib.pyplot as plt

df.hist(bins=50, figsize = (30,20))

plt.show()
housing = df.copy()
corr_matrix = housing.corr()
# view list of attributes correlating with the sales price

attributes = corr_matrix['SalePrice'].sort_values(ascending = False)

attributes
# top 10 attributes

top10 = attributes[1:11]

attributes[:11]
from pandas.plotting import scatter_matrix



# scatter plot of the top 5 attributes

scatter_matrix(housing[attributes[:6].index], figsize = (20,20));
housing.plot(kind = 'scatter', x = 'GrLivArea', y = 'SalePrice', alpha = 0.1);
# copy the training data without the final sales price

housing = df.drop('SalePrice', axis = 1)



# copy only the final sales price

housing_labels = df['SalePrice'].copy()
housing.head()
housing_labels.head()
# subset of housing data with only the top 10 features

top10 = list(top10.index)

housing = housing[top10]
housing.describe()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



# Pipeline constructor

full_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy = 'median')),

        ('std_scaler', StandardScaler()),

    ])



# call the pipeline, fit and transform our housing data

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared
# place the standardized housing data into a DataFrame with the same columns and indices as our unprepared data

pd.DataFrame(housing_prepared, columns = housing.columns, index = housing.index).head()
# compare to non-standardized

housing.head()
from sklearn.linear_model import LinearRegression



# Linear Regression constructor

lin_reg = LinearRegression()



# fit our training data

lin_reg.fit(housing_prepared, housing_labels)
# retrieve the first 5 rows and first 5 labels

some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]



# predict the first 5 labels

some_data_prepared = full_pipeline.transform(some_data)



print('Predictions: ', lin_reg.predict(some_data_prepared))

print('Labels: ', list(some_labels))
from sklearn.metrics import mean_squared_error



# use our model to predict the final sales price

housing_predictions = lin_reg.predict(housing_prepared)



# calculate the mean squared error between our prediction and the label

lin_mse = mean_squared_error(housing_labels, housing_predictions)



# take the square root of our error

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score



# use our decision tree model with the prepared data, the labels, and give us 10 evaluation scores

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)



# calculate our error

tree_rmse_scores = np.sqrt(-scores)
# function to display scores

def display_scores(scores):

    print('Scores: ', scores)

    print('Mean: ', scores.mean())

    print('Standard Deviation: ', scores.std())

    

display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)



lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)



forest_mse = mean_squared_error(housing_labels, housing_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
from sklearn.model_selection import RandomizedSearchCV



# number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# number of features to consider at every split

max_features = ['auto', 'sqrt']

# maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# method of selecting samples for training each tree

bootstrap = [True, False]



# create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



# iterate to find the best hyperparameters

forest_reg = RandomForestRegressor()

rnd_search = RandomizedSearchCV(forest_reg, param_distributions = random_grid, n_iter = 5, cv = 5, scoring = 'neg_mean_squared_error')



# fit the data

rnd_search.fit(housing_prepared, housing_labels)
negative_mse = rnd_search.best_score_

rmse = np.sqrt(-negative_mse)

rmse
rnd_search.best_estimator_
# print every score and the corresponding parameters

cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):

    print(np.sqrt(-mean_score), params)
feature_importances = rnd_search.best_estimator_.feature_importances_

feature_importances
sorted(zip(feature_importances, housing.columns), reverse = True)
# get our best model

final_model = rnd_search.best_estimator_



# load in the test set

X_test = pd.read_csv('../input/test.csv')



# extract the top 10 features

X_test_top10 = X_test[top10]



# call our transformation pipeline

X_test_prepared = full_pipeline.transform(X_test_top10)



# make our final predictions

final_predictions = final_model.predict(X_test_prepared)
# create a download link for our predictions



from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create submission DataFrame with the corresponding Id

submission = pd.DataFrame()

submission['Id'] = X_test['Id']

submission['SalePrice'] = final_predictions



create_download_link(submission)
from tpot import TPOTRegressor



# pipeline caching

from tempfile import mkdtemp

from joblib import Memory

from shutil import rmtree



tpot = TPOTRegressor(generations = 100, population_size = 100, memory = 'auto', 

                     warm_start = True, verbosity = 2, periodic_checkpoint_folder = 'checkpoint',

                     scoring = 'neg_mean_squared_error', max_time_mins = 240)



tpot.fit(housing_prepared, housing_labels)

tpot.export('tpot_housing_data.py')
# import best TPOT model

from sklearn.linear_model import ElasticNetCV, LassoLarsCV

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline, make_union

from tpot.builtins import StackingEstimator

from xgboost import XGBRegressor



# Average CV score on the training set was:-833332153.5462229

exported_pipeline = make_pipeline(

    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.65, tol=0.001)),

    StackingEstimator(estimator=LassoLarsCV(normalize=False)),

    XGBRegressor(learning_rate=0.1, max_depth=8, min_child_weight=3, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.9000000000000001)

)



exported_pipeline.fit(housing_prepared, housing_labels)

results = exported_pipeline.predict(X_test_prepared)
# download best TPOT model predictions

tpot_submission = pd.DataFrame()

tpot_submission['Id'] = X_test['Id']

tpot_submission['SalePrice'] = results



create_download_link(tpot_submission)