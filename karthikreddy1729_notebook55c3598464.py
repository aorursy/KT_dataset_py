# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import os
print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_train.describe()
df_train.head()
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
train_X = df_train.loc[:, 'F3':'F17']
train_y = df_train.loc[:, 'O/P']
from xgboost import XGBRegressor
XGBModel = XGBRegressor()
XGBModel.fit(train_X, train_y , verbose=False)

df_test = df_test.loc[:, 'F3':'F17']
XGBpredictions = XGBModel.predict(df_test)
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(XGBpredictions)
result.head()
result.to_csv('output.csv', index=False)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import os
print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
test_index=df_test['Unnamed: 0']
print("Dataset has {} entries and {} features".format(*df_train.shape))
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
train_X = df_train.loc[:, 'F3':'F17']
train_y = df_train.loc[:, 'O/P']

# Splitting the data into train and test datasets
# test:train = 3:7
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=0)
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
from sklearn.metrics import mean_absolute_error
import numpy as np
# "Learn" the mean from the training data
mean_train = np.mean(y_train)
# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train
# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear',
}
params['eval_metric'] = "mae"
num_boost_round = 999
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)
cv_results
cv_results['test-mae-mean'].min()
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]
# Define initial best params and MAE
min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth,min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
params['max_depth'] = 11
params['min_child_weight'] = 7
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]
min_mae = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(subsample,colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
params['subsample'] = 1.
params['colsample_bytree'] = 1.
%time
# This can take some time…
min_mae = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    %time cv_results = xgb.cv(params,dtrain,num_boost_round=num_boost_round,seed=42,nfold=5,metrics=['mae'],early_stopping_rounds=10)
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))
params['eta'] = .01
params
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
num_boost_round = model.best_iteration + 1
best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)
mean_absolute_error(best_model.predict(dtest), y_test)
best_model.save_model("my_model.model")
test = pd.read_csv('../input/wecrec2020/Test_data.csv')
test = test.loc[:, 'F3':'F17']
d_pred = xgb.DMatrix(test,label=y_train[:3379])
pred = best_model.predict(d_pred)
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output.csv', index=False)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import os
print((os.listdir('../input/')))
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
test_df  = pd.read_csv('../input/wecrec2020/Test_data.csv')
train_df = pd.read_csv('../input/wecrec2020/Train_data.csv')
train_df = train_df.loc[:,'F1':]
test_df = test_df.loc[:,'F1':]
train_df.head(5)
test_df.head(5)
test_df.isnull().values.any()
train_df.isnull().values.any()
train_columns = list(train_df.columns.values)
test_columns  = list(test_df.columns.values)
set(train_columns) - set(test_columns)
train_df.shape
test_df.shape
# A couple style settings
sns.set_style("whitegrid")
sns.set_context("poster")
plt.figure(figsize = (12, 6))
plt.hist(train_df['O/P'])
plt.title('Histogram of O/P in the training set')
plt.xlabel('Count')
plt.ylabel('O/P')
plt.show()
plt.clf()
x = train_df['O/P']

fig, ax = plt.subplots(figsize=(12, 6))
n_bins = 50

# plot the cumulative histogram
n, bins, patches = ax.hist(x, n_bins, normed=1, histtype='step',
                           cumulative=True, label='Empirical')
train_df['O/P'].describe()
train_df.columns.values
df = train_df.loc[:, (train_df != 0).any(axis=0)]
df.shape
df.describe()
nz = list(df.columns.values) 
nz.remove('F1')
nz.remove('F2')
nz.remove('O/P')
type(nz)
train_nz = pd.DataFrame({'Percentile':((df[nz].values)==0).mean(axis=0),
                           'Column' : nz})
train_nz.head(5)
plt.figure(figsize = (12,5))
plt.hist(train_nz['Percentile'], bins = 100)
plt.title('Percentge of column that has value 0')
plt.xlabel('Percentage zero')
plt.ylabel('Number of columns')
plt.show()
plt.clf()
train_nz['Percentile'].describe()
sns.set_style('ticks')
fig, ax = plt.subplots()

fig.set_size_inches(11, 1)
sns.boxplot(x="Percentile",
            data=train_nz, palette="Set3", ax = ax)
plt.show()
plt.clf()
X_train = train_df.loc[:, 'F3':'F17']
y_train = train_df.loc[:, 'O/P']
clf = RandomForestRegressor()
clf.fit(X_train, y_train)

test_df = test_df.loc[:, 'F3':'F17']
preds = clf.predict(test_df)
rf = RandomForestRegressor()

from pprint import pprint

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
# bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

pprint(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 1)

# Fit the random search model
# In order to test these models I will need to do a train test split with the training data-set. 
X_train, X_test, y_train, y_test = train_test_split(train_df[nz], train_df['O/P'], test_size=0.2)
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = np.sqrt(mean_squared_error(y_test, predictions))
    print('Model Performance')
    print('MSE of: ', errors)
    
    return errors
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)


best_random = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
best_random.fit(X_train , y_train)

random_accuracy = evaluate(best_random, X_test, y_test)

print('\n')
print('Base Accuracy: ', base_accuracy)
print('\n')
print('Random Accuracy: ', random_accuracy)
print('Improvement of {:0.2f}%.'.format((random_accuracy - base_accuracy) / base_accuracy))

print('\n')
print('RF_Randomized_Search_CV')
print('\n')
y_train = train_df['O/P']
X_train = train_df[nz]


clf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

clf.fit(X_train, y_train)

preds = clf.predict(test_df[nz])
train_nz['Percentile'].describe()
sub_seventy = pd.DataFrame(train_nz.loc[train_nz['Percentile'] < 0.7])
sub_seventy_col_series = sub_seventy['Column']
sub_seventy_col = list(sub_seventy_col_series)
plt.figure(figsize = (15,5))
plt.boxplot(sub_seventy['Percentile'], patch_artist = True, vert = False)
plt.title('Boxplot for percentage zero of columns sub seventy')
plt.xlabel('Percentage zero')
plt.show()
plt.clf()
len(sub_seventy_col)
sub_seventy_df = train_df[sub_seventy_col]
sub_seventy_df['O/P'] = train_df['O/P']
sub_seventy_df.head(3)
sub_seventy_y = sub_seventy_df['O/P']
sub_seventy_X = sub_seventy_df.loc[: , sub_seventy_df.columns != 'O/P']

train = train_df[sub_seventy_col]
# train['target'] = train_df['target']

test = test_df[sub_seventy_col]
from sklearn import model_selection

Y = train_df['O/P']
Y = np.log(Y+1)


def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


kf = model_selection.KFold(n_splits=10, shuffle=True)
def runRF(x_train, y_train,x_test, y_test,test):
    #model=RandomForestRegressor(bootstrap=True, max_features=0.75, min_samples_leaf=11, min_samples_split=13, n_estimators=100)
    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
    model.fit(x_train, y_train)
    y_pred_train=model.predict(x_test)
    mse=rmsle(np.exp(y_pred_train)-1,np.exp(y_test)-1)
    y_pred_test=model.predict(test)
    return y_pred_train,mse,y_pred_test

pred_full_test_RF = 0    
rmsle_RF_list=[]

for dev_index, val_index in kf.split(train):
    dev_X, val_X = train.loc[dev_index], train.loc[val_index]
    dev_y, val_y = Y.loc[dev_index], Y.loc[val_index]
    ypred_valid_RF,rmsle_RF,ytest_RF=runRF(dev_X, dev_y, val_X, val_y,test)
    print("fold_ RF _ok "+str(rmsle_RF))
    rmsle_RF_list.append(rmsle_RF)
    pred_full_test_RF = pred_full_test_RF + ytest_RF

rmsle_RF_mean=np.mean(rmsle_RF_list)
print("Mean cv score : ", np.mean(rmsle_RF_mean))
ytest_RF=pred_full_test_RF/10
ytest_RF = np.exp(ytest_RF)-1
ytest_RF
test = pd.read_csv('../input/wecrec2020/Test_data.csv')
test_index=test['Unnamed: 0']
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(ytest_RF)
result.head()
result.to_csv('result.csv', index=False)



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.preprocessing import LabelEncoder
from math import sqrt
%matplotlib inline
import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_train.head()
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
train_X = df_train.loc[:, 'F3':'F17']
train_y = df_train.loc[:, 'O/P']
test_X = df_test.loc[:, 'F3':'F17']
labels = train_y
features = train_X
# List of features for later use
feature_list = list(features.columns)
# Convert to numpy arrays
import numpy as np
features = np.array(features)
labels = np.array(labels)
# Training and Testing Sets
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# Find the original feature indices 
original_feature_indices = [feature_list.index(feature) for feature in feature_list]
# Create a test set of the original features
original_test_features = test_features[:, original_feature_indices]
from sklearn.ensemble import RandomForestRegressor
rf_exp = RandomForestRegressor(n_estimators= 100, random_state=100)
rf_exp.fit(train_features, train_labels)
predictions = rf_exp.predict(test_features)
# Performance metrics
errors = abs(predictions - test_labels)
print('Metrics for Random Forest Trained on Expanded Data')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
importances = list(rf_exp.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')
# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(x_values, sorted_features, rotation = 'vertical')
# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');
print('Number of features for 95% importance:', np.where(cumulative_importances > 0.95)[0][0] + 1)
# Extract the names of the most important features
important_feature_names = [feature[0] for feature in feature_importances[0:5]]
# Find the columns of the most important features
important_indices = [feature_list.index(feature) for feature in important_feature_names]
# Create training and testing sets with only the important features
important_train_features = train_features[:, important_indices]
important_test_features = test_features[:, important_indices]
# Sanity check on operations
print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)
# Train the expanded model on only the important features
rf_exp.fit(important_train_features, train_labels);
# Make predictions on test data
predictions = rf_exp.predict(important_test_features)
# Performance metrics
errors = abs(predictions - test_labels)
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

import numpy as np
test = np.array(test_X)
test = test[:, important_indices]

pred = rf_exp.predict(test)
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output.csv', index=False)
