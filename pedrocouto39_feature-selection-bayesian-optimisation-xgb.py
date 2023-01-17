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
# Loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

%matplotlib inline
# Loading the Train and Test datasets
df_train = pd.read_csv("../input/santander-customer-satisfaction/train.csv")
df_test = pd.read_csv("../input/santander-customer-satisfaction/test.csv")
# Checking the genearl infos of df_train
df_train.info()
# Checking the genearl infos of df_test
df_test.info()
# Checking the first 5 rows of df_train
df_train.head()
# Checking the first 5 rows of df_test
df_test.head()
# Checking if is there any missing value in both train and test datasets
df_train.isnull().sum().sum(), df_test.isnull().sum().sum()
# Investigating the proportion of unsatisfied customers on df_train
rate_insatisfied = df_train.TARGET.value_counts()[1] / df_train.TARGET.value_counts()[0]
rate_insatisfied * 100
from sklearn.model_selection import train_test_split

# Spliting the dataset on a proportion of 80% for train and 20% for test.
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('TARGET', axis = 1), df_train.TARGET, 
                                                    train_size = 0.8, stratify = df_train.TARGET,
                                                    random_state = 42)

#Checando o resultado do splot
X_train.shape, y_train.shape[0], X_test.shape, y_test.shape[0]
# Making copys of X_train and X_test to work with in this section
X_train_clean = X_train.copy()
X_test_clean = X_test.copy()
# Investigating if there are constant or semi-constat feature in X_train
from sklearn.feature_selection import VarianceThreshold

# Removing all features that have variance under 0.01
selector = VarianceThreshold(threshold = 0.01)
selector.fit(X_train_clean)
mask_clean = selector.get_support()
X_train_clean = X_train_clean[X_train_clean.columns[mask_clean]]
# Cheking if we realy removed something
(len(df_train.columns) - 1) - X_train_clean.shape[1]
# Total of remaning features
X_train_clean.shape[1]
# Checking if there is any duplicated column
remove = []
cols = X_train_clean.columns
for i in range(len(cols)-1):
    column = X_train_clean[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(column, X_train_clean[cols[j]].values):
            remove.append(cols[j])


# If yes, than they will be dropped here
X_train_clean.drop(remove, axis = 1, inplace=True)
# Checking if any column was dropped
X_train_clean.shape
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import cross_val_score
import xgboost as xgb
#Create an automated routine to test different K values for f_classif method

K_vs_score_fc = [] #List to store AUC of each K with f_classif

start = time.time()

for k in range(2, 247, 2):
    start = time.time()
    
    # Instantiating a KBest object for each of the metrics in order to obtain the K features with the highest value
    selector_fc = SelectKBest(score_func = f_classif, k = k)

    
    # Selecting K-features and modifying the dataset
    X_train_selected_fc = selector_fc.fit_transform(X_train_clean, y_train)

    
    # Instantiating an XGBClassifier object
    clf = xgb.XGBClassifier(seed=42)
    
    # Using 10-CV to calculate AUC for each K value avoinding overfitting
    auc_fc = cross_val_score(clf, X_train_selected_fc, y_train, cv = 10, scoring = 'roc_auc')

    
    # Adding the average values obtained in the CV for further analysis.
    K_vs_score_fc.append(auc_fc.mean())

    
    end = time.time()
    # Returning the metrics related to the tested K and the time spent on this iteration of the loop
    print("k = {} - auc_fc = {} - Time = {}s".format(k, auc_fc.mean(), end-start))
    
end = time.time()
print(end - start)

# Just for purpose of sharing this piece of code
# Create an automated routine to test different K values for mutual_info_classif

K_vs_score_mic = [] #List to store AUC of each K with mutual_info_classif


for k in range(2, 247, 2):
    start = time.time()
    
    # Instantiating a KBest object for each of the metrics in order to obtain the K features with the highest value
    selector_mic = SelectKBest(score_func = mutual_info_classif, k = k)
    
    # Selecting K-features and modifying the dataset
    X_train_selected_mic = selector_mic.fit_transform(X_train_clean, y_train) 
    
    # Instantiating an XGBClassifier object
    clf = xgb.XGBClassifier(seed=42)
    
    # Using 10-CV to calculate AUC for each K value avoinding overfitting
    auc_mic = cross_val_score(clf, X_train_selected_mic, y_train, cv = 10, scoring = 'roc_auc')
    
    # Adding the average values obtained in the CV for further analysis.
    K_vs_score_mic.append(auc_mic.mean())
    
    end = time.time()
    # Returning the metrics related to the tested K and the time spent on this iteration of the loop
    print("k = {} - auc_mic = {} - Time = {}s".format(k, auc_mic.mean(), end-start))
    

# Checking if both list have 123 elements each
len(K_vs_score_fc)
# Ploting K_vs_score_fc (# of K-Best features vs AUC)

# Figure setup
fig, ax = plt.subplots(figsize = (20, 8))
plt.title('Score valeus for each K with f_classif method', fontsize=18)
plt.ylabel('Score', fontsize = 16)
plt.xlabel('Value of K', fontsize = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

# Create the lines
plt.plot(np.arange(2, 247, 2), K_vs_score_fc, color='blue', linewidth=2)

plt.show()
# Ploting K_vs_score_fc (# of K-Best features vs AUC) 
import matplotlib.patches as patches

# Figure setup
fig, ax = plt.subplots(1, figsize = (20, 8))
plt.title('Score valeus for each K with f_classif method', fontsize=18)
plt.ylabel('Score', fontsize = 16)
plt.xlabel('Value of K', fontsize = 16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

# Create the lines
plt.plot(np.arange(2, 247, 2), K_vs_score_fc, color='blue', linewidth=2)
ax.set_ylim(0.80, 0.825);

# Create a Rectangle patch
rect = patches.Rectangle((82, 0.817), 20, (0.823 - 0.817), linewidth=2, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
# Selection the 96 best features aconrdingly to f_classif
selector_fc = SelectKBest(score_func = f_classif, k = 96)
selector_fc.fit(X_train_clean, y_train)
mask_selected = selector_fc.get_support()

# Saving the selected columns in a list
selected_col = X_train_clean.columns[mask_selected]
selected_col
#plotando o feature score das 96 melhores features
feature_score = pd.Series(selector_fc.scores_, index=X_train_clean.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(20, 12))
ax.barh(feature_score.index[0:30], feature_score[0:30])
plt.gca().invert_yaxis()


ax.set_xlabel('K-Score', fontsize=18);
ax.set_ylabel('Features', fontsize=18);
ax.set_title('30 best features by its K-Score', fontsize = 20)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False);
[print(i) for i in feature_score.index[0:10]];
# Creating datasets where only with the selected 96 features are included
X_train_selected = X_train[selected_col]
X_test_selected = X_test[selected_col]
# Checking the first 5 rows of X_train_selected and its shape
X_train_selected.head()
# Checking the first 5 rows of X_train_selected and its shape
X_test_selected.head()
# Using a random forest to optimize
from skopt import forest_minimize
# Function for hyperparamters tunning
# Implementation learned on a lesson of Mario Filho (Kagle Grandmaster) for parametes optmization.
# Link to the video: https://www.youtube.com/watch?v=WhnkeasZNHI
def tune_xgbc(params):
    """Function to be passed as scikit-optimize minimizer/maximizer input
    
    Parameters:
    Tuples with information about the range that the optimizer should use for that parameter, 
    as well as the behaviour that it should follow in that range.
    
    Returns:
    float: the metric that should be minimized. If the objective is maximization, then the negative 
    of the desired metric must be returned. In this case, the negative AUC average generated by CV is returned.
    """
    
    
    #Hyperparameters to be optimized
    print(params)
    learning_rate = params[0] 
    n_estimators = params[1] 
    max_depth = params[2]
    min_child_weight = params[3]
    gamma = params[4]
    subsample = params[5]
    colsample_bytree = params[6]
        
    
    #Model to be optimized
    mdl = xgb.XGBClassifier(learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth, 
                            min_child_weight = min_child_weight, gamma = gamma, subsample = subsample, 
                            colsample_bytree = colsample_bytree, seed = 42)
    

    #Cross-Validation in order to avoid overfitting
    auc = cross_val_score(mdl, X_train_selected, y_train, cv = 10, scoring = 'roc_auc')
    
    print(auc.mean())
    # as the function is minimization (forest_minimize), we need to use the negative of the desired metric (AUC)
    return -auc.mean()
# Creating a sample space in which the initial randomic search should be performed
space = [(1e-3, 1e-1, 'log-uniform'), # learning rate
          (100, 2000), # n_estimators
          (1, 10), # max_depth 
          (1, 6.), # min_child_weight 
          (0, 0.5), # gamma 
          (0.5, 1.), # subsample 
          (0.5, 1.)] # colsample_bytree 

# Minimization using a random forest with 20 random samples and 50 iterations for Bayesian optimization.
result = forest_minimize(tune_xgbc, space, random_state = 42, n_random_starts = 20, n_calls  = 25, verbose = 1)
# Hyperparameters optimized values
hyperparameters = ['learning rate', 'n_estimators', 'max_depth', 'min_child_weight', 'gamma', 'subsample',
                   'colsample_bytree']

for i in range(0, len(result.x)): 
    print('{}: {}'.format(hyperparameters[i], result.x[i]))
from skopt.plots import plot_convergence
# Setting up the figure
fig, ax = plt.subplots(figsize = (20,8))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(b = None)

# Ploting
plot_convergence(result)

# Setting up axes and title
ax.set_title('Convergence Plot', fontsize = 18)
ax.set_xlabel('Number of calls (n)', fontsize = 16)
ax.set_ylabel('min(x) after n calls', fontsize = 16);
# Generating the model with the optimized hyperparameters
clf_optimized = xgb.XGBClassifier(learning_rate = result.x[0], n_estimators = result.x[1], max_depth = result.x[2], 
                            min_child_weight = result.x[3], gamma = result.x[4], subsample = result.x[5], 
                            colsample_bytree = result.x[6], seed = 42)
# Fitting the model to the X_train_selected dataset
clf_optimized.fit(X_train_selected, y_train)
# Evaluating the performance of the model in the test data (which have not been used so far).
y_predicted = clf_optimized.predict_proba(X_test_selected)[:,1]
auc(y_test, y_predicted)
# making predctions on the test dataset (df_test), from Kaggle, with the selected features and optimized parameters
y_predicted_df_test = clf_optimized.predict_proba(df_test[selected_col])[:, 1]
# saving the result into a csv file to be uploaded into Kaggle late subimission 
# https://www.kaggle.com/c/santander-customer-satisfaction/submit
sub = pd.Series(y_predicted_df_test, index = df_test['ID'], name = 'TARGET')
sub.to_csv('submission.csv')
# Code base on this post: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
import sklearn.metrics as metrics

# Calculate FPR and TPR for all thresholds
fpr, tpr, threshold = metrics.roc_curve(y_test, y_predicted)
roc_auc = metrics.auc(fpr, tpr)

# Plotting the ROC curve
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (20, 8))
plt.title('Receiver Operating Characteristic', fontsize=18)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc = 'upper left', fontsize = 16)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize = 16)
plt.xlabel('False Positive Rate', fontsize = 16)
plt.show()