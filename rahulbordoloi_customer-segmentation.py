# Importing Libraries

# filtering out the warnings after cell execution
import warnings
warnings.filterwarnings('ignore')

# General Commonly Used Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing Libraries
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# EDA
from sklearn import base

# Feature Engineering and Selection
from sklearn.utils import class_weight

# Modeling & Accuracy Metrics
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

# Validation and Hyperparameter Tuning
from sklearn.model_selection import KFold, cross_val_score as cvs, GridSearchCV

# Utility Library
from collections import Counter
import os
# Kaggle cwd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importing Datasets

train_set = pd.read_csv("/kaggle/input/analytics-vidhya-janatahack-customer-segmentation/Train_aBjfeNk.csv", verbose = -1)
test_set = pd.read_csv("/kaggle/input/analytics-vidhya-janatahack-customer-segmentation/Test_LqhgPWU.csv", verbose = -1)
id_cols = test_set['ID']
train_set.head(5)
# Spliting Train Dataset into Train and Train-Remain (For further Validation and Test Spliting)
train, test = train_test_split(train_set, test_size = 0.3, random_state = 0, shuffle = True)
# Education

train["Profession"].fillna("Others", inplace = True)
test["Profession"].fillna("Others", inplace = True)
test_set["Profession"].fillna("Others", inplace = True)
# Ever Married

## Train
gg = train.index[((train.Ever_Married.isnull()) & (train.Family_Size == 1.0))].tolist()
train.at[gg, 'Ever_Married'] = 'No'
train.Ever_Married.fillna('Yes', inplace = True)

## Validation
gg = test.index[((test.Ever_Married.isnull()) & (test.Family_Size == 1.0))].tolist()
test.at[gg, 'Ever_Married'] = 'No'
test.Ever_Married.fillna('Yes', inplace = True)

## Test
gg = test_set.index[((test_set.Ever_Married.isnull()) & (test_set.Family_Size == 1.0))].tolist()
test_set.at[gg, 'Ever_Married'] = 'No'
test_set.Ever_Married.fillna('Yes', inplace = True)
# Family Size

train['Family_Size'].fillna(round(train.Family_Size.mean()), inplace = True)
test['Family_Size'].fillna(round(train.Family_Size.mean()), inplace = True)
test_set['Family_Size'].fillna(round(train.Family_Size.mean()), inplace = True)
# Profession

train['Profession'].fillna('Others', inplace = True)
test['Profession'].fillna('Others', inplace = True)
test_set['Profession'].fillna('Others', inplace = True)
# Graduation

## Train
gg = train.index[((train.Graduated.isnull()) & (train.Age <= 24) & (train.Family_Size == 1.0))].tolist()
train.at[gg, 'Graduated'] = 'No'
train.Graduated.fillna('Yes', inplace = True)

## Validation
gg = test.index[((test.Graduated.isnull()) & (test.Age <= 24) & (test.Family_Size == 1.0))].tolist()
test.at[gg, 'Graduated'] = 'No'
test.Graduated.fillna('Yes', inplace = True)

## Test
gg = test_set.index[((test_set.Graduated.isnull()) & (test_set.Age <= 24) & (test_set.Family_Size == 1.0))].tolist()
test_set.at[gg, 'Graduated'] = 'No'
test_set.Graduated.fillna('Yes', inplace = True)
# Work_Experience

train.Work_Experience.fillna(round(train.Work_Experience.mean()), inplace = True)
test.Work_Experience.fillna(round(train.Work_Experience.mean()), inplace = True)
test_set.Work_Experience.fillna(round(train.Work_Experience.mean()), inplace = True)
# Var_1

train.Var_1.fillna(train.Var_1.mode()[0], inplace = True)
test.Var_1.fillna(train.Var_1.mode()[0], inplace = True)
test_set.Var_1.fillna(train.Var_1.mode()[0], inplace = True)
train.isnull().mean(), test.isnull().mean(), test_set.isnull().mean()
# Determining Output Leakage -> Keeping the IDs
leak = list(set(train_set.ID) & set(test_set.ID))
len(leak)
# Storing the index in Train Set
ss = list()
for i in leak:
    ss.append(train_set.index[train_set.ID == i][0])
print(len(ss))

# Storing the Values for the Respective Indexes
op_values = list()
for i in ss:
    op_values.append(train_set.iloc[i, -1])
print(len(op_values))
# Label Encoding

l = LabelEncoder()

## Gender
train.loc[:, 'Gender'] = l.fit_transform(train.loc[:, 'Gender'])
test.loc[:, 'Gender'] = l.fit_transform(test.loc[:, 'Gender'])
test_set.loc[:, 'Gender'] = l.fit_transform(test_set.loc[:, 'Gender'])

## Ever Married
train.loc[:, 'Ever_Married'] = l.fit_transform(train.loc[:, 'Ever_Married'])
test.loc[:, 'Ever_Married'] = l.fit_transform(test.loc[:, 'Ever_Married'])
test_set.loc[:, 'Ever_Married'] = l.fit_transform(test_set.loc[:, 'Ever_Married'])

## Graduated
train.loc[:, 'Graduated'] = l.fit_transform(train.loc[:, 'Graduated'])
test.loc[:, 'Graduated'] = l.fit_transform(test.loc[:, 'Graduated'])
test_set.loc[:, 'Graduated'] = l.fit_transform(test_set.loc[:, 'Graduated'])
## Segmentation - Target Variable

train.loc[:, 'Segmentation'] = l.fit_transform(train.loc[:, 'Segmentation'])
test.loc[:, 'Segmentation'] = l.fit_transform(test.loc[:, 'Segmentation'])
# Defining K-Fold Target Encoding Class for Train (K-Fold as for Regularization)

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colname, targetName, n_fold = 5):

        self.colnames = colname
        self.targetName = targetName
        self.n_fold = n_fold

    def fit(self, x, y = None):
        return self

    def transform(self, x):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in x.columns)
        assert(self.targetName in x.columns)

        mean_of_target = x[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold, shuffle = False, random_state=0)

        col_mean_name = 'tgt_' + self.colnames
        x[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(x):
            x_tr, x_val = x.iloc[tr_ind], x.iloc[val_ind]
            x.loc[x.index[val_ind], col_mean_name] = x_val[self.colnames].map(x_tr.groupby(self.colnames)[self.targetName].mean())

        x[col_mean_name].fillna(mean_of_target, inplace = True)

        return x
# Defining K-Fold Target Encoding Class for Validation (K-Fold as for Regularization) [Mapping from Train]

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, train, colNames, encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
         
    def fit(self, X, y = None):
        return self

    def transform(self, X):

        mean = self.train[[self.colNames, self.encodedName]].groupby(self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]

        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})

        return X
# K-Fold Target Encoding

## Profession

### Train
targetc = KFoldTargetEncoderTrain('Profession', 'Segmentation', n_fold = 5)
train = targetc.fit_transform(train)

### Validation
targetc = KFoldTargetEncoderTest(train, 'Profession', 'tgt_Profession')
test = targetc.fit_transform(test)

### Validation
targetc = KFoldTargetEncoderTest(train, 'Profession', 'tgt_Profession')
test_set = targetc.fit_transform(test_set)

## Var_1

### Train
targetc = KFoldTargetEncoderTrain('Var_1', 'Segmentation', n_fold = 5)
train = targetc.fit_transform(train)

### Validation
targetc = KFoldTargetEncoderTest(train, 'Var_1', 'tgt_Var_1')
test = targetc.fit_transform(test)

### Test
targetc = KFoldTargetEncoderTest(train, 'Var_1', 'tgt_Var_1')
test_set = targetc.fit_transform(test_set)
# Weighted Encoding

## Spending Score
spend_enc = {"Low" : 0, "Average" : 1, "High": 2}

train['Spending_Score'] = train['Spending_Score'].map(spend_enc)
test['Spending_Score'] = test['Spending_Score'].map(spend_enc)
test_set['Spending_Score'] = test_set['Spending_Score'].map(spend_enc)
# Dropping off Redundant Features

train.drop(['Profession', 'Var_1', 'ID'], axis = 1, inplace = True)
test.drop(['Profession', 'Var_1', 'ID'], axis = 1, inplace = True)
test_set.drop(['Profession', 'Var_1', 'ID'], axis = 1, inplace = True)
# Reducing Skewness

## Train   
train.Age = stats.boxcox(train.Age)[0]
train.Family_Size += 1   
train.Family_Size = stats.boxcox(train.Family_Size)[0] 
train.tgt_Var_1 = stats.boxcox(train.tgt_Var_1)[0]    
train.tgt_Profession = stats.boxcox(train.tgt_Profession)[0]
train.Work_Experience = np.cbrt(train.Work_Experience)

## Validation   
test.Age = stats.boxcox(test.Age)[0]
test.Family_Size += 1   
test.Family_Size = stats.boxcox(test.Family_Size)[0]  
test.tgt_Var_1 = stats.boxcox(test.tgt_Var_1)[0]   
test.tgt_Profession = stats.boxcox(test.tgt_Profession)[0] 
test.Work_Experience = np.cbrt(test.Work_Experience)

## Test  
test_set.Age = stats.boxcox(test_set.Age)[0] 
test_set.Family_Size += 1
test_set.Family_Size = stats.boxcox(test_set.Family_Size)[0]      
test_set.tgt_Var_1 = stats.boxcox(test_set.tgt_Var_1)[0]  
test_set.tgt_Profession = stats.boxcox(test_set.tgt_Profession)[0] 
test_set.Work_Experience = np.cbrt(test_set.Work_Experience)
# Dividing into independent and dependent features

## Train
x = train.drop(['Segmentation'], axis = 1)
y = train.Segmentation.values.reshape(-1, 1)

## Validation
xx = train.drop(['Segmentation'], axis = 1)
yy = train.Segmentation.values.reshape(-1, 1)
# Standard Scaling

sc_x = StandardScaler()
x_scale = sc_x.fit_transform(x)
xx_scale = sc_x.fit_transform(xx)
t_scale = sc_x.fit_transform(test_set)
# Determining Class Weights

class_weights = class_weight.compute_class_weight('balanced', np.unique(y.reshape(-1, )), y.reshape(-1, ))
im_weight = dict(enumerate(class_weights))
im_weight
# CatBoost Classifier

# Fitting CatBoost Classifier to the Training Set
classifier = CatBoostClassifier(random_state = 0, eval_metric = 'Accuracy', class_weights = im_weight, od_type = "Iter", thread_count = -1)
classifier.fit(x_scale, y, eval_set = (xx_scale, yy))
y_pred = classifier.predict(x_scale)

## Classification Report - Train
print(classification_report(y, y_pred))

# Applying k-fold Cross Validation Score
from sklearn.model_selection import cross_val_score as cvs
accuracies = cvs(estimator = classifier, X = xx_scale, y = yy, cv = 10, scoring = 'accuracy', n_jobs = -1)
print(accuracies.mean())
print(accuracies.std())

## Classification Report - Validation
print(classification_report(yy, classifier.predict(xx_scale)))
# Setting up the Dictionary of Hyper-Paramters

hyperparams = {
    "eval_metric" : ["Accuracy"],              # Evaluation Metric
    "random_state" : [0],                      # Random State to retain the same configuration
    "iterations" : [100, 200, 500, 1000],      # Iterations is an alis for 'n_estimators' -> Maximum no of Trees
    "learning_rate" : [0.03, 0.1, 0.001],      # Learning Rate of our Model
    "l2_leaf_reg" : [3.0, 1.0, 5.0],           # L2 Regularization Parameter of our Cost Function to reduce overfitting
    "depth" : [6, 7, 8, 9, 10],                # Depth of our Trees
    "class_weights" : [im_weight],             # Class Weights
    "od_type" : ["Iter"],                      # Type of overfitting detector
    "od_wait" : [50, 100],                     # The No. of Iterations to continue the training after the iteration with the optimal metric value
    "task_type": ["CPU"],                      # Processing Unit
}
# Using Grid Search CV Method to find out the Best Set of Hyper-Parameters

classifier = CatBoostClassifier()
class_cv = GridSearchCV(classifier, hyperparams, verbose = 1, scoring = ['accuracy'], n_jobs = -1, cv = 5, refit = 'accuracy')
class_cv.fit(x_scale, y, eval_set = (xx_scale, yy))
# Dictionary of the Best Parameters
class_cv.best_params_
# Fitting CatBoost Classifier to the Training Set

classifier = CatBoostClassifier(**class_cv.best_params_)
classifier.fit(x_scale, y)

y_pred = classifier.predict(xx_scale)

print(classification_report(y, y_pred))
# Fitting CatBoost Classifier to the Test Set
classifier = CatBoostClassifier(**class_cv.best_params_)
classifier.fit(x_scale, y, eval_set = (xx_scale, yy))
y_pred = classifier.predict(t_scale)
# Creating Submission Dataframe
submission = pd.DataFrame()
submission['ID'] = id_cols
submission['Segmentation'] = y_pred
submission
# Maping the integer values with the objects
spend_enc = {0 : 'A', 1 : 'B', 2: 'C', 3 : 'D'}
submission['Segmentation'] = submission['Segmentation'].map(spend_enc)
# Checking out number of Each Prediction 
Counter(submission.Segmentation)
# Taking out the index in test Set
ss = list()
for i in leak:
    ss.append(submission.index[submission.ID == i][0])
len(ss)

# Imputing the Values
for (index, replacement) in zip(ss, op_values):
    submission.Segmentation[index] = replacement
# Checking out number of Each Prediction post mutation
Counter(submission.Segmentation)
# Checking out Submission Dataframe
submission.head(5)
# Generating Submission File
submission.to_csv("Final_Submission.csv", index = False)