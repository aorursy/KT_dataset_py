import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
# Change the working directory
# ATTN: You will need to change this locally.
# os.chdir('C:/Users/jonda/OneDrive/Documents/Springboard/Santander_Capstone')

# Read in the csv's for test, and train data sets.
test_df  = pd.read_csv('../input/test.csv')
train_df = pd.read_csv('../input/train.csv')
# Take a first look at the training set
train_df.head(5)
# Take a first look at the testing set
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
plt.hist(train_df['target'])
plt.title('Histogram of target values in the training set')
plt.xlabel('Count')
plt.ylabel('Target value')
plt.show()
plt.clf()
x = train_df['target']

fig, ax = plt.subplots(figsize=(12, 6))
n_bins = 50

# plot the cumulative histogram
n, bins, patches = ax.hist(x, n_bins, normed=1, histtype='step',
                           cumulative=True, label='Empirical')
train_df['target'].describe()
train_df.columns.values
# Drop all columns that consist of only zeros.
df = train_df.loc[:, (train_df != 0).any(axis=0)]
df.shape
df.describe()
nz = list(df.columns.values) 
nz.remove('ID')
nz.remove('target')
type(nz)
# This next bit was inspired by Bojan Tunguz's idea to determine just how sparse each column is
# https://www.kaggle.com/tunguz/yaeda-yet-another-eda
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
# The approach for machine learning was also inspired by Bojan Tunguz's work

y_train = train_df.target.values
print('y shape: ', y_train.shape)
X_train = train_df[nz]
print('\n')
print('X_train 5 line head below: ')
X_train.head(5)
clf = RandomForestRegressor()
clf.fit(X_train, y_train)

preds = clf.predict(test_df[nz])
sample_submission = pd.read_csv("C:/Users/jonda/OneDrive/Documents/Springboard/Santander_Capstone/sample_submission.csv")
sample_submission.target = preds
sample_submission.to_csv('simple_rfr_all_default.csv', index=False)
sample_submission.head()
train_df_no_target = train_df.loc[:, train_df.columns != 'target']
type(train_df_no_target)
train_df_target = train_df.loc[:, train_df.columns == 'target']
type(train_df_target)
############### Start: Randomized Search CV ##################################

# Look at parameters used by our current forest
# from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

from pprint import pprint

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_validation import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(train_df[nz], train_df.target.values, test_size=0.2)


# rf_random.fit(X_train, y_train)
# Evaluation of Random Search
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


# =============================================================================
# Best param set for random forest regression on Registered Users
# =============================================================================
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
#           max_features='sqrt', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=2, min_samples_split=5,
#           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
#           oob_score=False, random_state=None, verbose=0, warm_start=False)
# =============================================================================

################# End: Randomized Search CV ##################################
y_train = train_df.target.values
X_train = train_df[nz]


clf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=30,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

clf.fit(X_train, y_train)

preds = clf.predict(test_df[nz])

sample_submission = pd.read_csv("C:/Users/jonda/OneDrive/Documents/Springboard/Santander_Capstone/sample_submission.csv")
sample_submission.target = preds
sample_submission.to_csv('simple_rfr_searchCV_all_features.csv', index=False)
sample_submission.head()
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
sub_seventy_df['target'] = train_df['target']
sub_seventy_df.head(3)
sub_seventy_y = sub_seventy_df['target']
sub_seventy_X = sub_seventy_df.loc[: , sub_seventy_df.columns != 'target']

train = train_df[sub_seventy_col]
# train['target'] = train_df['target']

test = test_df[sub_seventy_col]
# Functions from Haim Feldman's kernal: https://www.kaggle.com/haimfeld87/randomforest-with-50-features
from sklearn import model_selection

Y = train_df['target']
Y = np.log(Y+1)

test_id = test_df.ID

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
out_df = pd.DataFrame(ytest_RF)
out_df.columns = ['target']
out_df.insert(0, 'ID', test_id)
out_df.to_csv("RF_" + str(rmsle_RF_mean) + "_.csv", index=False)
