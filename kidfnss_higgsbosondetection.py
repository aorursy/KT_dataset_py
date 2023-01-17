import numpy as np # linear algebra
import matplotlib.pylab as plt # Plotting
import sklearn # Machine learning models.
from sklearn.neighbors import KNeighborsClassifier as KNN
import sklearn.metrics # Area under ROC curve calculations. 
from sklearn.model_selection import train_test_split as tts # Train test split 
from sklearn.model_selection import KFold # KFold cross validation 
import pandas as pd # Quick dataframe previewing 



filename = '/kaggle/input/higgs-boson-detection/train.csv'


data = np.loadtxt(filename, skiprows=1, delimiter=',')

X = data[:,1:]
Y = data[:,0:1]

# Split off validation set, size = 30% validation 70% training
Xtrain, Xvalid, Ytrain, Yvalid = tts(X, Y, test_size=.3)


# import numpy as np # linear algebra
# import matplotlib.pylab as plt # Plotting
# import sklearn # Machine learning models.
# from sklearn.neighbors import KNeighborsClassifier as KNN
# import sklearn.metrics # Area under ROC curve calculations. 
# from sklearn.model_selection import train_test_split as tts # Train test split 
# from sklearn.model_selection import KFold # KFold cross validation 
# import pandas as pd # Quick dataframe previewing 



# filename = '/kaggle/input/higgs-boson-detection/train.csv'


# data = np.loadtxt(filename, skiprows=1, delimiter=',')

# X = data[:,1:]
# Y = data[:,0:1]

# # Split off validation set, size = 30% validation 70% training 
# Xtrain, Xvalid, Ytrain, Yvalid = tts(X, Y, test_size=.3)


# Split training set with KFoldValidation in order to perform hyperparameter optimization
kf = KFold(n_splits = 15)
kFolds = kf.split(Xtrain)

# Set 'k' in kNN algo, declare dictionary to hold AUROC values {kval : auroc}
k = 1
aurocs = {}


# Use kFold technique on training set to test models with different values of k starting at 3
for train_index, test_index in kFolds:
    k += 2
    model = KNN(n_neighbors=k)
    model.fit(Xtrain[train_index], Ytrain[train_index][:,0])
    predictions = model.predict_proba(Xtrain[test_index])
    val = sklearn.metrics.roc_auc_score(Ytrain[test_index], predictions[:,1])
    print(f'Validation AUROC for KNN with k = {k}: {val}')
    aurocs[k] = val

# Choose optimized k-value and evaulate AUROC value on the untouched validation set (note: not the same validation set used in kFolds technique)
model = KNN(n_neighbors=max(aurocs, key=aurocs.get)) # Set model with optimized k-value
model.fit(Xtrain[train_index], Ytrain[train_index][:,0]) # Fit model on the training set
predictions = model.predict_proba(Xvalid) # Predict on validation set
val = sklearn.metrics.roc_auc_score(Yvalid, predictions[:,1]) # Calculate AUROC value
print(f'Max AUROC value at k = {max(aurocs, key=aurocs.get)}. AUROC value with optimized kValue on validation set = {val}') # Print AUROC value on untouched validation set

# Plot ROC curve.
fpr, tpr, thresholds = sklearn.metrics.roc_curve(Yvalid, predictions[:,1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'kNN with k = {max(aurocs, key=aurocs.get)}')
# Using decision tree classifier
from sklearn import tree # Decision tree algo

# Split training set with KFoldValidation
kf_two = KFold(n_splits = 10)
kFolds_two = kf_two.split(Xtrain)

# Set 'max_depth' in decision tree algo, declare dictionary to hold AUROC values {max_depth : auroc}
mdepth = 1
aurocs_two = {}


# Use kFold technique on training set to test models with different values of max_depth starting at 3
for train_index, test_index in kFolds_two:
    mdepth += 2
    model_two = tree.DecisionTreeClassifier(max_depth=mdepth)
    model_two.fit(Xtrain[train_index], Ytrain[train_index][:,0])
    predictions_two = model_two.predict_proba(Xtrain[test_index])
    val_two = sklearn.metrics.roc_auc_score(Ytrain[test_index], predictions_two[:,1])
    print(f'Validation AUROC for Decision Tree with max_depth = {mdepth}: {val_two}')
    aurocs_two[mdepth] = val_two
    
# Choose optimized max_depth and evaulate AUROC value on the untouched validation set (note: not the same validation set used in kFolds technique)
model_two = tree.DecisionTreeClassifier(max_depth=max(aurocs_two, key=aurocs_two.get)) # Set model with optimized max_depth value
model_two.fit(Xtrain[train_index], Ytrain[train_index][:,0]) # Fit model on the training set
predictions_two = model_two.predict_proba(Xvalid) # Predict on validation set
val_two = sklearn.metrics.roc_auc_score(Yvalid, predictions_two[:,1]) # Calculate AUROC value
print(f'Max AUROC value at max_depth = {max(aurocs_two, key=aurocs_two.get)}. AUROC value with optimized kValue on validation set = {val_two}') # Print AUROC value on untouched validation set

# Plot ROC curve.
fpr_two, tpr_two, thresholds_two = sklearn.metrics.roc_curve(Yvalid, predictions_two[:,1])
plt.plot(fpr_two, tpr_two)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Decision tree with max_depth = {max(aurocs_two, key=aurocs_two.get)}')
# Using decision tree classifier
from sklearn.ensemble import RandomForestClassifier

# Split training set with KFoldValidation
kf_two = KFold(n_splits = 10)
kFolds_two = kf_two.split(Xtrain)

# Set 'max_depth' in random forest algo, declare dictionary to hold AUROC values {max_depth : auroc}
mdepth = 1
aurocs_two = {}


# Use kFold technique on training set to test models with different values of max_depth starting at 3
for train_index, test_index in kFolds_two:
    mdepth += 2
    model_two = RandomForestClassifier(max_depth=mdepth)
    model_two.fit(Xtrain[train_index], Ytrain[train_index][:,0])
    predictions_two = model_two.predict_proba(Xtrain[test_index])
    val_two = sklearn.metrics.roc_auc_score(Ytrain[test_index], predictions_two[:,1])
    print(f'Validation AUROC for Random Forest with max_depth = {mdepth}: {val_two}')
    aurocs_two[mdepth] = val_two

# Assign a variable to the max_depth value that gave the best AUROC value
bestDepth = max(aurocs_two, key=aurocs_two.get)

n_est = 80 # n_estimators, the next hyperparameter that will be tested
aurocs_est = {} # Declare a dictionary to hold {n_estimators : auroc}

# Redo training set split with KFoldValidation to test the hyperparameter 'n_estimators'
kf_two = KFold(n_splits = 10)
kFolds_two = kf_two.split(Xtrain)

# Use kFold technique on training set to test models with different values of n_estimators starting at 100
# In this iteration we will be using the optimized value of max_depth found in the previous iteration
for train_index, test_index in kFolds_two:
    n_est += 20
    model_two = RandomForestClassifier(max_depth=bestDepth, n_estimators=n_est)
    model_two.fit(Xtrain[train_index], Ytrain[train_index][:,0])
    predictions_two = model_two.predict_proba(Xtrain[test_index])
    val_two = sklearn.metrics.roc_auc_score(Ytrain[test_index], predictions_two[:,1])
    print(f'Validation AUROC for Random Forest with max_depth = {bestDepth} and n_estimators = {n_est}: {val_two}')
    aurocs_est[n_est] = val_two # {n_est : auroc}

# Assign a variable to the n_estimators value that gave the best AUROC value
bestEst = max(aurocs_est, key=aurocs_est.get)
    
# Use optimized hyperparameters and evaulate AUROC value on the untouched validation set (note: not the same validation set used in kFolds technique)
model_two = RandomForestClassifier(max_depth=bestDepth, n_estimators=bestEst) # Set model with optimized hyperparameter values
model_two.fit(Xtrain[train_index], Ytrain[train_index][:,0]) # Fit model on the training set
predictions_two = model_two.predict_proba(Xvalid) # Predict on validation set
val_two = sklearn.metrics.roc_auc_score(Yvalid, predictions_two[:,1]) # Calculate AUROC value
print(f'Max AUROC value at max_depth = {bestDepth} and n_estimators = {bestEst}. AUROC value with optimized kValue on validation set = {val_two}') # Print AUROC value on untouched validation set

# Plot ROC curve.
fpr_two, tpr_two, thresholds_two = sklearn.metrics.roc_curve(Yvalid, predictions_two[:,1])
plt.plot(fpr_two, tpr_two)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Random Forest with max_depth = {bestDepth} and n_estimators = {bestEst}')
# Using gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier

# Split training set with KFoldValidation
kf_two = KFold(n_splits = 10)
kFolds_two = kf_two.split(Xtrain)

# Set 'max_depth' in random gradient boosting algo, declare dictionary to hold AUROC values {max_depth : auroc}
mdepth = 1
aurocs_two = {}


# Use kFold technique on training set to test models with different values of max_depth starting at 3
for train_index, test_index in kFolds_two:
    mdepth += 2
    model_two = GradientBoostingClassifier(max_depth=mdepth)
    model_two.fit(Xtrain[train_index], Ytrain[train_index][:,0])
    predictions_two = model_two.predict_proba(Xtrain[test_index])
    val_two = sklearn.metrics.roc_auc_score(Ytrain[test_index], predictions_two[:,1])
    print(f'Validation AUROC for Gradient Boosting with max_depth = {mdepth}: {val_two}')
    aurocs_two[mdepth] = val_two

# Assign a variable to the max_depth value that gave the best AUROC value
bestDepth = max(aurocs_two, key=aurocs_two.get)

n_est = 80 # n_estimators, the next hyperparameter which will be tested
aurocs_est = {} # Declare a dictionary to hold {n_estimators : auroc}

# Redo training set split with KFoldValidation to test the hyperparameter 'n_estimators'
kf_two = KFold(n_splits = 10)
kFolds_two = kf_two.split(Xtrain)

# Use kFold technique on training set to test models with different values of n_estimators starting at 100
# In this iteration we will be using the optimized value of max_depth found in the previous iteration
for train_index, test_index in kFolds_two:
    n_est += 20
    model_two = GradientBoostingClassifier(max_depth=bestDepth, n_estimators=n_est)
    model_two.fit(Xtrain[train_index], Ytrain[train_index][:,0])
    predictions_two = model_two.predict_proba(Xtrain[test_index])
    val_two = sklearn.metrics.roc_auc_score(Ytrain[test_index], predictions_two[:,1])
    print(f'Validation AUROC for Gradient Boosting with max_depth = {bestDepth} and n_estimators = {n_est}: {val_two}')
    aurocs_est[n_est] = val_two # {n_est : auroc}

# Assign a variable to the n_estimators value that gave the best AUROC value
bestEst = max(aurocs_est, key=aurocs_est.get)
    
# Use optimized hyperparameters and evaulate AUROC value on the untouched validation set (note: not the same validation set used in kFolds technique)
model_two = GradientBoostingClassifier(max_depth=bestDepth, n_estimators=bestEst) # Set model with optimized hyperparameter values
model_two.fit(Xtrain[train_index], Ytrain[train_index][:,0]) # Fit model on the training set
predictions_two = model_two.predict_proba(Xvalid) # Predict on validation set
val_two = sklearn.metrics.roc_auc_score(Yvalid, predictions_two[:,1]) # Calculate AUROC value
print(f'Max AUROC value at max_depth = {bestDepth} and n_estimators = {bestEst}. AUROC value with optimized kValue on validation set = {val_two}') # Print AUROC value on untouched validation set

# Plot ROC curve.
fpr_two, tpr_two, thresholds_two = sklearn.metrics.roc_curve(Yvalid, predictions_two[:,1])
plt.plot(fpr_two, tpr_two)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Gradient Boosting with max_depth = {bestDepth} and n_estimators = {bestEst}')
# Make probabilistic predictions.
filename = '/kaggle/input/higgs-boson-detection/test.csv' # This is the path if running on kaggle.com. Otherwise change this.
Xtest1 = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=range(1,29))
predictions = model_two.predict_proba(Xtest1) # Choose the best model
predictions = predictions[:,1:2] # Predictions has two columns. Get probability that label=1.
N = predictions.shape[0]
assert N == 50000, "Predictions should have length 50000."
submission = np.hstack((np.arange(N).reshape(-1,1), predictions)) # Add Id column.
np.savetxt(fname='submission.csv', X=submission, header='Id,Predicted', delimiter=',', comments='')

# If running on Kaggle.com, submission.csv can be downloaded from this Kaggle Notebook under Sessions->Data->output->/kaggle/working.