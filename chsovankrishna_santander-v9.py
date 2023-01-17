import pandas as pd

from pandas import Series, DataFrame
import numpy as np
import matplotlib as mpl

import seaborn as sn
# Form machine learning
import sklearn


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", index_col=None)
test = pd.read_csv("../input/test.csv", index_col=None)
train_ID = train.ID
test_ID = test.ID
train_TARGET = train.TARGET
# Copying the contents of the columns to separate DFs, later will combine after the pre processing
train.drop('TARGET', axis = 1, inplace = True)
train.drop('ID', axis = 1, inplace = True)
test.drop('ID', axis = 1, inplace = True)

total = pd.concat([train, test])
total = total.replace(-999999,2)
######################################### Checking the no of different datatype variables present in the dataset

floatlist = []
integerlist = []
objectlist = []

for i in total.columns:
    if total[i].dtypes==np.float64 or total[i].dtypes==np.float32:
        floatlist.append(i)
    elif total[i].dtypes==np.int64 or total[i].dtypes==np.int32:
        integerlist.append(i)
    else:
        objectlist.append(i)

print ("The number of float variables:", len(floatlist))
print ("The number of integer variables:", len(integerlist))
print ("The number of non-numeric/class variables:", len(objectlist))

########################################### Categorizing each variables according to their unique values
var_0 = []
var_1 = []
var_2 = []

for i in total.columns:
    if total[i].nunique() <= 10:
        var_0.append(i)
    elif total[i].nunique() > 10 & total[i].nunique() <= 100:
        var_1.append(i)
    else:
        var_2.append(i)
        
print ("The number of columns with <= 10 unique values:", len(var_0))
print ("The number of columns with 10<x<=100 unique values", len(var_1))
print ("The number of columns with >100 unique values:", len(var_2))

########################################## Checking each variable for presence of missing values

total_missing = total.isnull().sum()

total_missing_counter = 0
total_missing_varlist = []

for i in range(len(total_missing)):
    if total_missing[i]>0:
        total_missing_varlist.append(i)
    total_missing_counter += 1
    
print('No of variables checked for missing values:', total_missing_counter)
print('Variables having missing values:', total_missing_varlist)

########################################### Removing constant columns (std == 0 )

colsToRemove = []
for col in total.columns:
    if total[col].std() == 0:
        colsToRemove.append(col)

total.drop(colsToRemove, axis=1, inplace=True)

########################################### Drop dulicate columns
colsToRemove = []
columns = total.columns
for i in range(len(columns)-1):
    v = total[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v,total[columns[j]].values):
            colsToRemove.append(columns[j])

total.drop(colsToRemove, axis=1, inplace=True)

print('Duplicate variables:', colsToRemove)


total.shape
# the column size just decreased from 371 to 309 (27 variables)
Train = total[:train.shape[0]]
Train["TARGET"] = train_TARGET
test = total[train.shape[0]:]

print ("new train shape:", Train.shape)
print ("new test shape:", test.shape)
Train.shape

# Master split for training & test dataset

Train['is_Train'] = np.random.uniform(0, 1, len(Train)) <= .75
training, validation = Train[Train['is_Train']==True], Train[Train['is_Train']==False]
validation.shape
# Dependant variable
y_train = training['TARGET']
# Independant variable list
features = list(training.columns[:308])
x_train = training[features]
# Dependant variable
y_validation = validation['TARGET']
# Independant variable list
features = list(validation.columns[:308])
x_validation = validation[features]
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier #For Classification
# from sklearn.ensemble import AdaBoostRegressor #For Regression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier #For Classification
from sklearn.ensemble import GradientBoostingRegressor #For Regression
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
clf.fit(x_train, y_train)
y_pred_class = clf.predict(x_validation)
# Confusion Matrix
from sklearn import metrics

metrics.confusion_matrix(y_validation, y_pred_class)
# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_validation, y_pred_class)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
metrics.accuracy_score(y_validation, y_pred_class)
# store the predicted probabilities for class 1
y_pred_prob = clf.predict_proba(x_validation)[:, 1]
# allow plots to appear in the notebook
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# histogram of predicted probabilities
plt.hist(y_pred_prob, bins=8)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of ----')
plt.ylabel('Frequency')
# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_validation, y_pred_prob)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('------')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
# AUC is the percentage of the ROC plot that is underneath the curve:
# IMPORTANT: first argument is true values, second argument is predicted probabilities
metrics.roc_auc_score(y_validation, y_pred_prob)
################################
test.shape
y_pred_class_test = clf.predict(test)
y_pred_class_test.shape
y_predproba_test = clf.predict_proba(test)

submission = pd.DataFrame({"ID":test_ID, "TARGET": y_predproba_test[:,1]})
submission.to_csv("submission.csv", index=False)



