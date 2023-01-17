import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
# Specify the input data here

# After download the full train.csv, you can use command

# head -n 50000 train.csv train-50k.csv to get a sample

# For running with full data, you should upload to kaggle/or server. It needs much more memory

# input_data = "../data/raw/force-well-logs/train-50k.csv"

input_data = "../input/forcedataset/force-ai-well-logs/train.csv"

# test_data = "../input/forcedataset/force-ai-well-logs/test.csv"
# There are two target columns, we should remove from learning

TARGET_1 = "FORCE_2020_LITHOFACIES_LITHOLOGY"

TARGET_2 = "FORCE_2020_LITHOFACIES_CONFIDENCE"

WELL_NAME = 'WELL'
def draw_confusion_matrix(model, X_valid, y_valid):

    fig, ax = plt.subplots(figsize=(6,6))

    disp = plot_confusion_matrix(model, X_valid, y_valid, normalize = None, xticks_rotation = 'vertical', ax = ax)

    disp.ax_.set_title("Plot Confusion Matrix Not Normalized")

    fig1, ax1 = plt.subplots(figsize=(6,6))

    disp1 = plot_confusion_matrix(model, X_valid, y_valid, normalize = 'true', values_format = ".2f", xticks_rotation = 'vertical', ax = ax1)

    disp1.ax_.set_title("Plot Confusion Matrix Normalized")

    plt.show()
def calculate_accuracy(y_true, y_pred):

    cm = confusion_matrix(y_true = y_true, y_pred = y_pred)

    tp = 0

    for i in range(len(cm)):

        tp += cm[i][i]

    accuracy = 1.0 * tp / np.sum(cm)

    return accuracy
df = pd.read_csv(input_data, sep=';', nrows = 50000)
df.columns
# See how many wells in the first 50k row

print('N wells = ', np.unique(df['WELL'].values))
# See how many rock types

print('N target classes = ', np.unique(df[TARGET_1].values))
#dictionary to translate number into rock types

lithology_keys = {30000: 'Sandstone',

                 65030: 'Sandstone/Shale',

                 65000: 'Shale',

                 80000: 'Marl',

                 74000: 'Dolomite',

                 70000: 'Limestone',

                 70032: 'Chalk',

                 88000: 'Halite',

                 86000: 'Anhydrite',

                 99000: 'Tuff',

                 90000: 'Coal',

                 93000: 'Basement'}
df[TARGET_1].value_counts()
# We view the NA columns, and drop some columns with many NA

df.isna().sum()
df.isna().sum()
# Describe serveral stats of df

df.describe()
# See the dtype, because we have to remove string values columns

df.dtypes
# We drop columns which have larger than 10k NA values (20%)

# and object values

unused_columns = ['RSHA', 'SGR', 'NPHI', 'BS', 'DTS', 'DCAL', 'RMIC', 'ROPA', 'RXO']

unused_columns += [WELL_NAME, 'GROUP', 'FORMATION']

# ADD two target columns into unused columns

unused_columns += [TARGET_1, TARGET_2]
# Get the columns features

all_columns = list(df.columns)



use_columns = [c for c in all_columns if c not in unused_columns]

print(use_columns)
# We provide a simple replace nan value with mean value.

# This is the simplest/standard preprocess methods



# with each column in use_columns

# df[c].mean() computes the mean values without na value

# df[c].fillna() fill with the above value, use option inplace to provide inplace replacing

for c in use_columns:

    df[c].fillna(df[c].mean(), inplace=True)
# train_wells = list(np.unique(df['WELL'].values))

train_wells = ['15/9-13', '15/9-15']

# Use this condition to find out which rows in the data is select for training

train_mask = df[WELL_NAME].isin(train_wells)
X_train = df[train_mask][use_columns].values

y_train = df[train_mask][TARGET_1].values

print(X_train.shape, y_train.shape)
X_valid = df[~train_mask][use_columns].values

y_valid = df[~train_mask][TARGET_1].values

print(X_valid.shape, y_valid.shape)
penalty_matrix = np.load("../input/penalty-matrix/penalty_matrix.npy")
# Position of each type of rock in the penalty_matrix

penalty_dict = {"Sandstone": 0,

                "Sandstone/Shale": 1,

                "Shale": 2, 

                "Marl": 3,

                "Dolomite": 4,

                "Limestone": 5,

                "Chalk": 6,

                "Halite": 7,

                "Anhydrite": 8,

                "Tuff": 9,

                "Coal": 10,

                "Basement": 11}
# Used for getting the right "rock number" from confusion matrix index

cm_rock_idx = np.unique(df[TARGET_1].values)
def calculate_penalty(cm = None, penalty_matrix = None, lithology_dict = None, penalty_dict = None, cm_rock_idx = None):

    sum_penalty = 0

    for i in range(len(cm)):

        for j in range(len(cm)):

            rock_i = lithology_dict[cm_rock_idx[i]]

            rock_j = lithology_dict[cm_rock_idx[j]]

            penalty_i = penalty_dict[rock_i]

            penalty_j = penalty_dict[rock_j]

            sum_penalty += cm[i][j] * penalty_matrix[penalty_i][penalty_j]

    return -1.0 * sum_penalty / np.sum(cm)
from xgboost import XGBClassifier
import time
tree_depth_report = []


for max_depth in [3,5,7,9]:

    

    xgb_model_maxdepth = XGBClassifier(max_depth = max_depth)

    start_time = time.time()

    xgb_model_maxdepth.fit(X_train, y_train)

    runtime = time.time() - start_time

    predict_y_train = xgb_model_maxdepth.predict(X_train)

    predict_y_valid = xgb_model_maxdepth.predict(X_valid)

    cm_train = confusion_matrix(y_true = y_train, y_pred = predict_y_train)

    cm_valid = confusion_matrix(y_true = y_valid, y_pred = predict_y_valid)

    accuracy_on_train = calculate_accuracy(y_train, predict_y_train)

    accuracy_on_valid = calculate_accuracy(y_valid, predict_y_valid)

    penalty_train = calculate_penalty(cm_train, penalty_matrix, lithology_keys, penalty_dict, cm_rock_idx)

    penalty_valid = calculate_penalty(cm_valid, penalty_matrix, lithology_keys, penalty_dict, cm_rock_idx)

    tree_depth_report.append([max_depth, accuracy_on_train, accuracy_on_valid, penalty_train, penalty_valid, runtime])

    

    

    

    
tree_depth_report = pd.DataFrame(tree_depth_report).rename(columns = {0: 'depth', 1: 'train_accuracy', 2: 'valid_accuracy', 3: 'penalty_train', 4: 'penalty_valid', 5: 'runtime'})
tree_depth_report
learning_rates_report = []
for learning_rate in [0.01, 0.03, 0.1, 0.3, 1, 3]:

    

    xgb_model_learning_rate = XGBClassifier(max_depth = 9, learning_rate = learning_rate)

    start_time = time.time()

    xgb_model_learning_rate.fit(X_train, y_train)

    runtime = time.time() - start_time

    predict_y_train = xgb_model_learning_rate.predict(X_train)

    predict_y_valid = xgb_model_learning_rate.predict(X_valid)

    cm_train = confusion_matrix(y_true = y_train, y_pred = predict_y_train)

    cm_valid = confusion_matrix(y_true = y_valid, y_pred = predict_y_valid)

    accuracy_on_train = calculate_accuracy(y_train, predict_y_train)

    accuracy_on_valid = calculate_accuracy(y_valid, predict_y_valid)

    penalty_train = calculate_penalty(cm_train, penalty_matrix, lithology_keys, penalty_dict, cm_rock_idx)

    penalty_valid = calculate_penalty(cm_valid, penalty_matrix, lithology_keys, penalty_dict, cm_rock_idx)

    learning_rates_report.append([learning_rate, accuracy_on_train, accuracy_on_valid, penalty_train, penalty_valid, runtime])
learning_rates_report = pd.DataFrame(learning_rates_report).rename(columns = {0: 'learning_rate', 1: 'train_accuracy', 2: 'valid_accuracy', 3: 'penalty_train', 4: 'penalty_valid', 5: 'runtime'})
learning_rates_report
n_estimators_report = []

max_depth = 9

learning_rate = 0.1
for n_estimators in [50, 100, 500]:

    

    xgb_model_nEstimators = XGBClassifier(max_depth = max_depth, learning_rate = learning_rate, n_estimators = n_estimators)

    start_time = time.time()

    xgb_model_nEstimators.fit(X_train, y_train)

    runtime = time.time() - start_time

    predict_y_train = xgb_model_nEstimators.predict(X_train)

    predict_y_valid = xgb_model_nEstimators.predict(X_valid)

    cm_train = confusion_matrix(y_true = y_train, y_pred = predict_y_train)

    cm_valid = confusion_matrix(y_true = y_valid, y_pred = predict_y_valid)

    accuracy_on_train = calculate_accuracy(y_train, predict_y_train)

    accuracy_on_valid = calculate_accuracy(y_valid, predict_y_valid)

    penalty_train = calculate_penalty(cm_train, penalty_matrix, lithology_keys, penalty_dict, cm_rock_idx)

    penalty_valid = calculate_penalty(cm_valid, penalty_matrix, lithology_keys, penalty_dict, cm_rock_idx)

    n_estimators_report.append([n_estimators, accuracy_on_train, accuracy_on_valid, penalty_train, penalty_valid, runtime])
n_estimators_report = pd.DataFrame(n_estimators_report).rename(columns = {0: 'n_estimators', 1: 'train_accuracy', 2: 'valid_accuracy', 3: 'penalty_train', 4: 'penalty_valid', 5: 'runtime'})
n_estimators_report
# from xgboost import XGBClassifier
# xgb_model = XGBClassifier()
# xgb_model.fit(X_train, y_train)
# param_grid = [{'max_depth':[6,7,8], 'min_child_weight':[2,3,4]

#                ,'objective' : ['reg:linear'],'colsample_bytree' : [0.6,0.8,1], 'learning_rate' : [0.001,0.01],

#               'reg_lambda' : [0.1,0.5], 'n_estimators' : [100, 500]}]

# model = xgboost.XGBClassifier()

# grid_search = GridSearchCV(model, param_grid, cv = 7, 

#                          scoring = 'neg_mean_squared_error', 

#                          return_train_score = True, refit = True)

# grid_search.fit(X_train, y_train)

# xgb_reg = grid_search.best_estimator_

# xgb_reg.fit(X_train, y_train, verbose=1)
# predict_y_XGB = xgb_model.predict(X_valid)
# cm_XGB = confusion_matrix(y_true = y_valid, y_pred = predict_y_XGB)
# cm_XGB
# draw_confusion_matrix(xgb_model, X_valid, y_valid)
# accuracy_XGB = calculate_accuracy(y_valid, predict_y_XGB)
# accuracy_XGB
# penalty_XGB = calculate_penalty(cm_XGB, penalty_matrix, lithology_keys, penalty_dict, cm_rock_idx)
# print("XGBoost Accuracy: {},".format(accuracy_XGB), "Penalty Score: {}".format(penalty_XGB))