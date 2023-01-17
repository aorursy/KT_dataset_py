!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887

%load_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np 

import pandas as pd 

from fastai.imports import*

from fastai.structured import *

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor

from IPython.display import display

from sklearn import metrics

from pprint import pprint

import os

import shap

import eli5

from eli5.sklearn import PermutationImportance

from pdpbox import pdp, get_dataset, info_plots



print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train['SalePrice'] = np.log(df_train['SalePrice'])
train_cats(df_train)#Change any columns of strings in a panda's dataframe to a column of categorical values
apply_cats(df_test, df_train)
train_df, y_trn, nas = proc_df(df_train, 'SalePrice')

test_df, _, _ = proc_df(df_test, na_dict=nas)

train_df.head()
df_test.columns[df_test.isnull().any()]
df_train.columns[df_train.isnull().any()]
test_df.columns
train_df.columns
test_df.drop(['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na'], axis =1, inplace = True)

train_df.drop(['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na'], axis = 1, inplace = True)
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(train_X), train_y), rmse(m.predict(val_X), val_y),     ## RMSE of log of prices

                m.score(train_X, train_y), m.score(val_X, val_y)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
train_X, val_X, train_y, val_y = train_test_split(train_df, y_trn, test_size=0.33, random_state=42)
%time

m = RandomForestRegressor(n_estimators=1, min_samples_leaf=3, n_jobs=-1, max_depth = 3, oob_score=True) ## Use all CPUs available

m.fit(train_X, train_y)



print_score(m)
draw_tree(m.estimators_[0], train_X, precision=3)
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model

rf_random.fit(train_X, train_y)
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy



best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, val_X, val_y)
perm = PermutationImportance(rf_random, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
preds = np.stack([t.predict(val_X) for t in rf_random.best_estimator_])
preds.shape
preds[:,0], np.mean(preds[:,0]), val_y[0]
plt.plot([metrics.r2_score(val_y, np.mean(preds[:i+1], axis=0)) for i in range(20)]);
for feat_name in val_X.columns:

#for feat_name in base_features:

    #pdp_dist = pdp.pdp_isolate(model=rf_random.best_estimator_, dataset=val_X, model_features=base_features, feature=feat_name)

    pdp_dist = pdp.pdp_isolate(model = rf_random.best_estimator_, dataset=val_X, model_features=val_X.columns, feature=feat_name)



    pdp.pdp_plot(pdp_dist, feat_name)



    plt.show()
explainer = shap.TreeExplainer(rf_random.best_estimator_)

shap_values = explainer.shap_values(val_X)



# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

shap.force_plot(explainer.expected_value, shap_values[1,:], val_X.iloc[1,:], matplotlib=True) ## change shap and val_X
shap.summary_plot(shap_values, val_X)
pred = rf_random.best_estimator_.predict(test_df)

submission = pd.read_csv('../input/sample_submission.csv')

submission.head()
#submission['SalePrice'] = np.exp(pred)   ## Convert log back 

submission.to_csv('submission_v2.csv', index=False)