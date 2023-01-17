import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import json

import math

import xgboost as xgb

import time



from xgboost import XGBRegressor

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from statsmodels.graphics.gofplots import qqplot
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.info()
data.describe()
data[:3]
class DataHelper:

    '''

    Helper class for plots and data handling functions.

    '''

    

    def __init__(self, data):

        self.data = data

        

    def update_data(self, data):

        self.data = data

        

    def drop_columns(self, column_names):

        '''

        Drop some columns. Must update data with the returned new data set

        '''

        self.data = self.data.drop(column_names, axis = 1)

        return self.data

        

    def get_count_unique_vals(self, column_names):

        '''

        For each of the column names as parameter, count the unique different values

        '''

        counts = {}

        for column_name in column_names:

            counts[column_name] = self.data.groupby(column_name)[column_name].nunique().count()

        return counts

    

    def show_price_distribution(self):

        '''

        Show the price distribution in 3 graphs:

        - price histogram distribution

        - log of price histogram distribution

        - how close the distribution matches a normal distribution (the red line)

        '''

        fig, ax = plt.subplots(1, 3, figsize=(23, 5))

        sns.distplot(self.data['price'], ax = ax[0])

        sns.distplot(self.data['log_price'], ax = ax[1])

        qqplot(self.data['log_price'], line ='s', ax = ax[2])

        ax[2].set_title('Comparison with theoritical quantils of normal distribution')

        

    def show_count_and_distrib_categorical_feature(self, column_name):

        '''

        For a categorical feature (with not too many categories to be rendered), display the count

        and the price distribution for each category.

        '''

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        sns.countplot(self.data[column_name], ax=ax[0])

        ax[0].set_title("Offers count per " + column_name)

        ax[0].set_xlabel(column_name)

        sns.boxplot(x=column_name, y='log_price', data = self.data, ax=ax[1])

        ax[1].set_title("Price per " + column_name)

        ax[1].set_xlabel(column_name)

        ax[1].set_ylabel("Log of price")

        fig.show()

        

    def show_numerical_feature_distribution(self, column_name, bins=30):

        '''

        Show the count distribution for a numerical feature in two plots, one with a log xscale and another

        with normal xscale.

        '''

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].hist(self.data[column_name], bins)

        ax[0].set_title('count of ' + column_name)

        ax[0].set_xlabel(column_name)

        ax[0].set_ylabel("count")

        ax[0].set_yscale('log')

        ax[1].hist(np.log(1 + self.data[column_name]), bins)

        ax[1].set_title('count of ' + column_name)

        ax[1].set_yscale('log')

        ax[1].set_xlabel('Log of ' + column_name)

        ax[1].set_ylabel("count")

        ax[1].set_yscale('log')

        

    def get_pearson_features_correlation(self):

        '''

        Show matrix of pearson features correlation

        '''

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        ax.grid(True)

        ax.set_title("Pearson's correlation")

        corr = self.data.corr()

        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax)

        

    def get_dummies(self):

        '''

        Get dummies for every categorical feature in the data set. Remove the categorical columns.

        Must update data with the returned new data set

        '''

        categorical_columns = list(self.data.select_dtypes(include=['object']).columns)

        for column in categorical_columns:

            self.data = self.data.drop(column, axis = 1).join(pd.get_dummies(self.data[column]))

        return self.data

            

    def standardize_columns(self, column_names):

        '''

        Standardize every columns passed in arg.

        '''

        self.data[column_names] = StandardScaler().fit_transform(self.data[column_names])

        

# Log price will be need in many places later

data['log_price'] = np.log(1 + data['price'])

data_helper = DataHelper(data)
data_helper.get_count_unique_vals(['name', 'host_name', 'host_id', 'neighbourhood'])
data = data_helper.drop_columns(['host_id', 'id', 'name', 'host_name'])
# Count missing values

data[pd.isnull(data['last_review'])]['price'].count()
data[pd.isnull(data['last_review'])]['number_of_reviews'].mean()
data['never_reviewed'] = pd.isnull(data['last_review'])

data['reviews_per_month'] = data['reviews_per_month'].fillna(0)



# Remove na and convert to int for usage

data['last_review'] = pd.to_datetime(data['last_review'])

min_date_review = data['last_review'].min()

data['last_review'] = data['last_review'].fillna(min_date_review)

data['last_review'] = data[['last_review']].apply(lambda x: x[0].timestamp(), axis=1)
data.info()
data_helper.show_price_distribution()
oldCount = data['price'].count()

data = data[data['log_price'] > 2.6][data['log_price'] < 8.5]

data_helper.update_data(data)

print("Data points lost: ", oldCount - data['price'].count())



#qqplot(np.log(1 + data['price']), line ='s').show()

data_helper.show_price_distribution()
data_helper.show_count_and_distrib_categorical_feature('neighbourhood_group')
# Density distribution of the offers geographical position

sns.jointplot(x="longitude", y="latitude", data=data, kind="kde")
data_helper.show_count_and_distrib_categorical_feature('room_type')
data_helper.show_numerical_feature_distribution('minimum_nights')
data_helper.show_numerical_feature_distribution('number_of_reviews')
fig, ax = plt.subplots(1, 1, figsize=(15, 5))

ax.hist(data['last_review'], 100)

ax.set_title('count of last_review')

ax.set_xlabel('last_review (timestamps)')

ax.set_ylabel("count")

ax.set_yscale('log')
data_helper.show_numerical_feature_distribution('reviews_per_month')
data_helper.show_numerical_feature_distribution('calculated_host_listings_count')
data_helper.show_numerical_feature_distribution('availability_365')
data[data['availability_365'] == 0]['availability_365'].count()
data['never_available'] = data['availability_365'] == 0
data_helper.get_pearson_features_correlation()
# Standardize the data (necessary for Ridge and does not afect xgBoost)

data_helper.standardize_columns(['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365'])



# Get dummies

data = data_helper.get_dummies()
# Important to use a seed to have the same base of comparison and data reproducability

seed = 7805



# Train set and test set that will only be used at the end to compare the results at the very end.

X_train, X_test, y_train, y_test = train_test_split(data.drop(['price', 'log_price'], axis=1), data['log_price'], test_size=0.20, random_state=seed)
class RidgeRegressionHelper:

    def __init__(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train

        self.y_train = y_train

        self.X_test = X_test

        self.y_test = y_test

        

    def find_best_params(self, seed=None, kfolds=5, alphas=[1]):

        '''

        Perform grid search over the list of alphas

        '''

        kfolds = KFold(n_splits=kfolds, random_state=seed)

        results = []

        for alpha in alphas:

            model = Ridge(alpha = alpha)

            scores = cross_val_score(model, self.X_train, self.y_train, cv=kfolds, scoring='neg_mean_absolute_error')

            results.append([alpha, -scores.mean(), scores.std()])

            

        return pd.DataFrame(results, columns=['alpha', 'mae-mean', 'mae-std'])

    

    def get_performance_score(self, seed=None, kfolds=5, alpha=0.3):

        '''

        Display performance metrics on the training and testing data sets after training the model using the training data and the parameters passed as arguments.

        '''

        # Cross validation

        kfolds = KFold(n_splits=kfolds, random_state=seed)

        model = Ridge(alpha = alpha)

        scores = cross_val_score(model, self.X_train, self.y_train, cv=kfolds, scoring='neg_mean_absolute_error')

        

        # Train and fit to testing set

        model.fit(self.X_train, self.y_train)

        

        y_pred = model.predict(self.X_test)

        

        results = []

        results.append(['Ridge', -scores.mean(), scores.std(), mean_absolute_error(self.y_test, y_pred), r2_score(self.y_test, y_pred)])

        return pd.DataFrame(results, columns=['Model', 'CV MAE', 'CV MAE std', 'test MAE', 'r2 test score'])

    

    def show_results(self, results):

        '''

        Plot the results MAE relative to the alpha values (use log x scale).

        '''

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.plot(results['alpha'], results['mae-mean'])

        ax.set_xlabel('alpha')

        ax.set_ylabel('MAE')

        ax.set_xscale('log')
class XGBoostRegressionHelper:

    def __init__(self, X_train, y_train, X_test, y_test):

        self.dtrain = xgb.DMatrix(X_train, label=y_train)

        self.dX_test = xgb.DMatrix(X_test)

        self.y_test = y_test

        

    def find_best_params(self, seed=None, kfolds=5, learning_rates=[0.3], max_depths=[6], min_child_weights=[1], num_boost_round=1000):

        '''

        Performs a grid search given the list of parameters.

        

        :param int seed: number used for replicable results

        :param int kfolds: number of folds to use during the cross validation

        :param array of floats learning_rates: different learning rates values to test

        :param array of int max_depths: different max depth values to test

        :param array of int min_child_weights: different mi child weight values to test

        :param int num_boost_round: (same a n_estimators) max number of boosting rounds to do, can stop before

        :returns Dataframe: result set containing mae results and time taken for each combinaton of parameters

        '''

        results = []

        for learning_rate in learning_rates:

            for max_depth in max_depths:

                for min_child_weight in min_child_weights:

                    params = {

                        'max_depth':max_depth,

                        'min_child_weight': min_child_weight,

                        'eta':learning_rate,

                        'objective':'reg:squarederror',

                    }

                    timestamp = time.time()

                    cv_results = xgb.cv(

                        params,

                        self.dtrain,

                        num_boost_round=num_boost_round,

                        seed=seed,

                        nfold=kfolds,

                        metrics={'mae'},

                        early_stopping_rounds=10

                    )

                    totalTime = time.time() - timestamp

                    boostRounds = cv_results['test-mae-mean'].idxmin()

                    results.append([learning_rate, max_depth, min_child_weight, cv_results['test-mae-mean'][boostRounds], cv_results['test-mae-std'][boostRounds], boostRounds, totalTime])

        

        return pd.DataFrame(results, columns=['learning_rate', 'max_depth', 'min_child_weight', 'mae-mean', 'mae-std', 'boostRounds', 'totalTime_seconds'])

    

    def show_results_and_times_relative_to_parameter(self, results, parameter_name):

        '''

        Given a result set from find_best_params, show the mae-mean score and time taken given the best score of each value of one of the parameters.

        

        :param DataFrame results: result set obtained from find_best_params

        :param str parameter_name: name of one of the parameter to optimize in find_best_params

        '''

        best_scores = results.loc[results.groupby(parameter_name)['mae-mean'].idxmin()]

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].plot(best_scores[parameter_name], best_scores['mae-mean'])

        ax[0].set_xlabel(parameter_name)

        ax[0].set_ylabel('MAE')

        ax[1].plot(best_scores[parameter_name], best_scores['totalTime_seconds'])

        ax[1].set_xlabel(parameter_name)

        ax[1].set_ylabel('Total time for CV in seconds')

        

    def get_performance_score(self, seed=None, kfolds=5, learning_rate=0.3, max_depth=6, min_child_weight=1, num_boost_round=1000):

        '''

        Display performance metrics on the training and testing data sets after training the model using the training data and the parameters passed as arguments.

        '''

        params = {

            'max_depth':max_depth,

            'min_child_weight': min_child_weight,

            'eta':learning_rate,

            'objective':'reg:squarederror',

        }

        

        # Cross validation part

        cv_results = xgb.cv(

            params,

            self.dtrain,

            num_boost_round=num_boost_round,

            seed=seed,

            nfold=kfolds,

            metrics={'mae'},

            early_stopping_rounds=10

        )

        boostRounds = cv_results['test-mae-mean'].idxmin()

        

        # Train and fit to testing set

        xgb_reg = xgb.train(

            params,

            self.dtrain,

            num_boost_round=num_boost_round,

        )

        

        y_pred = xgb_reg.predict(self.dX_test)

        

        results = []

        results.append(['XGBoost', cv_results['test-mae-mean'][boostRounds], cv_results['test-mae-std'][boostRounds], mean_absolute_error(self.y_test, y_pred), r2_score(self.y_test, y_pred)])

        return pd.DataFrame(results, columns=['Model', 'CV MAE', 'CV MAE std', 'test MAE', 'r2 test score'])

    

    def get_features_importance(self, learning_rate=0.3, max_depth=6, min_child_weight=1, num_boost_round=50, max_num_features=10):

        '''

        Get the ranking of feature importance as a barh graph.

        '''

        params = {

            'max_depth':max_depth,

            'min_child_weight': min_child_weight,

            'eta':learning_rate,

            'objective':'reg:squarederror',

        }

        

        xgb_reg = xgb.train(

            params,

            self.dtrain,

            num_boost_round=num_boost_round,

        )

        

        xgb.plot_importance(xgb_reg, max_num_features=max_num_features)
# Ridge baseline

ridgeHelper = RidgeRegressionHelper(X_train, y_train, X_test, y_test)

ridgeHelper.get_performance_score(seed=seed)
# XGBoost baseline

xgbHelper = XGBoostRegressionHelper(X_train, y_train, X_test, y_test)

xgbHelper.get_performance_score(seed=seed)
# Finding best alpha

results = ridgeHelper.find_best_params(seed = seed, alphas=[0.5, 1, 2, 5, 10, 20])

ridgeHelper.show_results(results)
# Finer tuning around 5

finer_results = ridgeHelper.find_best_params(seed = seed, alphas=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])

ridgeHelper.show_results(finer_results)
best_result = finer_results.loc[finer_results['mae-mean'].idxmin()]

best_alpha = best_result['alpha']

best_result
ridgePerformance = ridgeHelper.get_performance_score(seed=seed, alpha=best_alpha)

ridgePerformance
# Try to find the best parameters for XGBoost

results = xgbHelper.find_best_params(seed = seed, learning_rates=[0.05, 0.1, 0.3], max_depths=[5, 7, 9], min_child_weights=[1, 3, 5])
# Let's have a look at how the learning rate influence the results and the time it takes

xgbHelper.show_results_and_times_relative_to_parameter(results, 'learning_rate')
# For a better understanding, we can directly look at the results

results
finer_results = xgbHelper.find_best_params(seed = seed, learning_rates=[0.01, 0.025, 0.05], max_depths=[9, 11, 13], min_child_weights=[1])
xgbHelper.show_results_and_times_relative_to_parameter(finer_results, 'learning_rate')

finer_results
best_result = finer_results.loc[finer_results['mae-mean'].idxmin()]

best_result
xgbPerformance = xgbHelper.get_performance_score(seed=seed, learning_rate=0.01, max_depth=11, min_child_weight=1, num_boost_round=500)

xgbPerformance
xgbHelper.get_features_importance(learning_rate=0.01, max_depth=11, min_child_weight=1, num_boost_round=500)