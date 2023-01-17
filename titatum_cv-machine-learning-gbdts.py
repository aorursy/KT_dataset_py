import os

from time import time

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn import ensemble

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.ensemble.partial_dependence import partial_dependence
input_dir = "../input"

examples_dir = os.path.join(input_dir,'cv-data-augmentation-text-scores')

examples = pd.read_parquet(os.path.join(examples_dir,'positive_negative_examples.parquet.gzip'))
examples.sample(10)
examples.shape
examples.dropna().shape
test_period_start = dt.datetime(2018, 7, 1)

test_period_end = dt.datetime(2019, 1, 31)



# the response or target column

y_col = 'matched' 



# 'questions_id' is only used in the Conditional Logistic Regression model

# 'answer_user_id' and 'emails_date_sent' are only used for comparison statistics

feature_cols = ['questions_id', 'answer_user_id', 'emails_date_sent',

                'days_from_joined_dates', 'days_from_last_activities',

                'professional_activities_sum_100000', 'professional_activities_sum_365', 'professional_activities_sum_30', 

                'questioner_answerer_shared_schools', 'questioner_answerer_shared_tags', 'questioner_answerer_shared_groups', 

                'questioners_answerers_paths', 'commenters_questioners_paths', 'commenters_answerers_paths',

                'LSI_Score']
def split_train_test(examples, feature_cols, y_col,

                     test_period_start, test_period_end):

    

    train_data = examples[examples['questions_date_added'] < test_period_start]

    train_x = train_data[feature_cols]

    train_y = train_data[y_col]

    print('Train Data Set Size: {}'.format(train_x.shape))

    

    test_data = examples[(examples['questions_date_added'] >= test_period_start) & (examples['questions_date_added'] <= test_period_end)]

    test_x = test_data[feature_cols]

    test_y = test_data[y_col]

    print('Test Data Set Size: {}'.format(test_x.shape))

    

    return train_x, train_y, test_x, test_y
# Slipt the original data set 

train_x, train_y, test_x, test_y = split_train_test(examples, feature_cols, y_col, test_period_start, test_period_end)
# Save train and test data sets in CSV format

train_x.to_csv('train_x.gz', compression='gzip')

train_y.to_csv('train_y.gz', compression='gzip')

test_x.to_csv('test_x.gz', compression='gzip')

test_y.to_csv('test_y.gz', compression='gzip')
# Drop 'questions_id' column which is only used in the Conditional Logistic Regression model

# Drop 'answer_user_id' and 'emails_date_sent' columns which are only used for comparison statistics

for dropped_col in ['questions_id', 'answer_user_id', 'emails_date_sent']:    

    train_x = train_x.drop(dropped_col, axis=1)

    test_x = test_x.drop(dropped_col, axis=1)

    feature_cols.remove(dropped_col)

print(feature_cols)
train_sample_size = None

if train_sample_size is not None:    

    train_sample_index = np.random.choice(train_x.index, size=train_sample_size, replace=False)

    train_x = train_x.iloc[train_sample_index]

    train_y = train_y.iloc[train_sample_index]

print('Label distribution: ')

print(train_y.value_counts())
def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
# This hyper parameter set is selected by the below randomized grid search in the previous runs of this notebook.

# Therefore, we use this set of hyper parameters from now on to reduce the running time

# In production, this step needs to be fine tuned further

best_params = {"learning_rate": 0.1,

               "n_estimators": 100,

               "max_depth": 3}



# if we would like to search for best hyper parameters, uncomment the below line to set best_params to None #

# best_params = None         

if best_params is None:    

    # a sample grid of hyper parameters

    param_dist = {"learning_rate": [0.01, 0.05, 0.1, 0.2],

                  "n_estimators": [50, 100, 200],

                  "max_depth": [3]}

    # run randomized search

    n_iter_search = 5

    random_search = RandomizedSearchCV(ensemble.GradientBoostingClassifier(), param_distributions=param_dist,

                                       n_iter=n_iter_search, cv=3)

    start = time()

    random_search.fit(train_x, train_y)

    print("RandomizedSearchCV took %.2f seconds for %d candidates"

          " parameter settings." % ((time() - start), n_iter_search))

    report(random_search.cv_results_)

    best_params = random_search.best_params_
%%time

gbc_model = ensemble.GradientBoostingClassifier(**best_params)

print(gbc_model)

gbc_model.fit(train_x, train_y)
feature_importance = gbc_model.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, np.asarray(feature_cols)[np.asarray(sorted_idx)])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.savefig('feature_importance.jpg')

plt.show()
feature_names = pd.DataFrame({'Name':feature_cols, 

                              'ID': range(0, len(feature_cols))}).set_index('Name')
plot_partial_dependence(gbc_model, train_x, [feature_names.loc['days_from_joined_dates']],

                        feature_names=feature_cols,

                        n_jobs=3, grid_resolution=50)
plot_partial_dependence(gbc_model, train_x, [feature_names.loc['professional_activities_sum_30']],

                        feature_names=feature_cols,

                        n_jobs=3, grid_resolution=50)
plot_partial_dependence(gbc_model, train_x, [feature_names.loc['questioners_answerers_paths']],

                        feature_names=feature_cols,

                        n_jobs=3, grid_resolution=50)
plot_partial_dependence(gbc_model, train_x, [feature_names.loc['LSI_Score']],

                        feature_names=feature_cols,

                        n_jobs=3, grid_resolution=50)
for feature_name in feature_names.index:

    plot_partial_dependence(gbc_model, train_x, [feature_names.loc[feature_name]],

                            feature_names=feature_cols,

                            n_jobs=3, grid_resolution=50)

    plt.savefig('{}.jpg'.format(feature_name))

os.listdir()
predicted_test_y = pd.DataFrame(gbc_model.predict_proba(test_x)[:,1], index=test_y.index, columns=['Predicted_Match_Prob'])
predicted_test_y.sample(10)
predicted_test_y.to_csv('predicted_test_y.gz', compression='gzip')
os.listdir()