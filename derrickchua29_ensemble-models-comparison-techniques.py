# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import Modules

# Foundational Packages
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.options.display.max_columns = 100
ZZ = 15
# Open Train & Test files
train_raw = pd.read_csv('../input/train.csv', na_values=-1) #FYI na_values are defined in the original data page
test_raw = pd.read_csv('../input/test.csv', na_values=-1)
# Copy Train file for workings
train_raw_copy = train_raw.copy(deep=True)
# Shape
print('Train Shape: ', train_raw_copy.shape)
print('Test Shape: ', test_raw.shape)
# Brief Head Output
display(train_raw_copy.head())
display(test_raw.head())
# Brief Sample Output
samples_show = 10
display(train_raw_copy.sample(samples_show))
display(test_raw.sample(samples_show))
# Heatmap of correlations
cor = train_raw_copy.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(cor)
plt.show()
# Function to output missing values & UniqueCounts & DataTypes
def basic_details(df):
    details = pd.DataFrame()
    details['Missing value'] = df.isnull().sum()
    details['N unique value'] = df.nunique()
    details['dtype'] = df.dtypes
    print('\n', details, '\n')
basic_details(train_raw_copy)
basic_details(test_raw)
##### C1 - Correction

# Combine both df for easy referencing
data_cleaner = [train_raw_copy, test_raw]

# Get List of Column names to drop
# 1.Drop those that missing values exceeds threshold
limit = 569  # ps_car_09_cat from "train_raw_copy" used as threshold for Missing Values
remove_cols_1 = [c for c in train_raw_copy.columns if train_raw_copy[c].isnull().sum() > limit]

# 2.Drop those that are uncorrelated from Heatmap
# **NOTE we will rectify this later during Feature Selection**
remove_cols_2 = train_raw_copy.columns[train_raw_copy.columns.str.startswith('ps_calc')]

# Dropping
for DataSet in data_cleaner:
    DataSet.drop(columns=remove_cols_1, axis=1, inplace=True)
    DataSet.drop(columns=remove_cols_2, axis=1, inplace=True)
# Check New Shape
print('Train New Shape: ',train_raw_copy.shape)
print('Test New Shape: ', test_raw.shape)
##### C2 - Completing (Missing)
# Choices : Median / Mean / Mode

# Easy referencing
for df in data_cleaner:
    # List Comprehension
    Residual_Missing = [c for c in df.columns if df[c].isnull().sum() > 0]
    for col in Residual_Missing:
        df[col].fillna(df[col].mode()[0], inplace=True)
# Check Missing
print('Train Missing: ',train_raw_copy.isnull().sum())
# Check Missing
print('Test Missing: ',test_raw.isnull().sum())
##### C4 - Convert
data = []
for feature in train_raw_copy.columns:
    # Defining the role of each variable
    if feature == 'target':
        use = 'target'
    elif feature == 'id':
        use = 'id'
    else:
        use = 'input'

    # Defining the statistical data type
    if 'bin' in feature or feature == 'target':
        type = 'binary'
    elif 'cat' in feature or feature == 'id':
        type = 'categorical'
    elif train_raw_copy[feature].dtype == float or isinstance(train_raw_copy[feature].dtype, float):
        type = 'real'
    elif train_raw_copy[feature].dtype == int:
        type = 'integer'

    # Initialize preserve to True for all variables except for id.
    # Since ONLY id is not in use
    preserve = True
    if feature == 'id':
        preserve = False

    # Defining the data type
    dtype = train_raw_copy[feature].dtype
    
    # Set default
    category = 'none'
    # Defining the category
    if 'ind' in feature:
        category = 'individual'
    elif 'reg' in feature:
        category = 'registration'
    elif 'car' in feature:
        category = 'car'
    elif 'calc' in feature:
        category = 'calculated'

    # Define UniqueValue Count
    NUnique = train_raw_copy[feature].nunique()

    # Creating a Dictionary that contains all the metadata for the variable to allocate/append above derivations
    feature_dictionary = {
        'varname': feature,
        'use': use,
        'type': type,
        'preserve': preserve,
        'dtype': dtype,
        'category': category,
        'NUnique': NUnique
    }
    data.append(feature_dictionary)

# Adjust & Define DataFrame
metadata = pd.DataFrame(data, columns=['varname', 'use', 'type', 'preserve', 'dtype', 'category', 'NUnique'])
# How many of each Feature types do we have?
print(metadata.groupby(['category'])['category'].count())
# How many of each Statistical data-types do we have?
print(metadata.groupby(['use', 'type'])['use'].count())
# Combining both of the above
print(metadata.groupby(['use', 'type', 'category'])['category'].count())
# Cat_Categorical
#BinaryLevel_cat_col = [col for col in train_raw_copy.columns if '_cat' in col] #Alternative List Comprehension Approach
BinaryLevel_cat_col = metadata.loc[metadata['type'] == 'categorical']['varname'] # Uses the metadata we made earlier
BinaryLevel_cat_col = list(BinaryLevel_cat_col)
BinaryLevel_cat_col.remove('id')

for c in BinaryLevel_cat_col:
    train_raw_copy[c] = train_raw_copy[c].astype('uint8')
    test_raw[c] = test_raw[c].astype('uint8')
# Bin_Binary
# NominalLevel_bin_col = [col for col in train_raw_copy.columns if 'bin' in col] #Alternative List Comprehension Approach
NominalLevel_bin_col = metadata.loc[metadata['type'] == 'binary']['varname'] # Uses the metadata we made earlier
NominalLevel_bin_col = list(NominalLevel_bin_col)
NominalLevel_bin_col.remove('target')

for c in NominalLevel_bin_col:
    train_raw_copy[c] = train_raw_copy[c].astype('uint8')
    test_raw[c] = test_raw[c].astype('uint8')
# Other_Others / Numerical
# Shortcut list comprehension method
other_col = [c for c in train_raw_copy.columns if c not in BinaryLevel_cat_col + NominalLevel_bin_col]
other_col.remove('id')
other_col.remove('target')
OrdinalLevel_other_col = [c for c in other_col if train_raw_copy[c].dtypes == 'int64']
IntervalLevel_other_col = [c for c in other_col if train_raw_copy[c].dtypes == 'float64']
basic_details(train_raw_copy)
# Break-Down WITHOUT 'id' & 'target'
# Categorical_cat
Categorical = BinaryLevel_cat_col
# Binary_bin
Binary = NominalLevel_bin_col
# Integer_'int'_Ordinal
Integer = OrdinalLevel_other_col
# Real_'float'_Interval
Real = IntervalLevel_other_col


# Original
Original_All_w = train_raw_copy.columns.get_values().tolist()
# Original WITHOUT 'id' & 'target'
Original_All_wo = [c for c in train_raw.columns if c not in ['id', 'target']]


# Converted Dtypes WITHOUT FeatureEngineering
Converted_dtypes_All_wo = Categorical + Binary + Integer + Real


# For Graph Chart Plots
# W/O 'id' & 'target'
Categorical_Chart_wo = Categorical
Binary_Chart_wo = Binary
Integer_Chart_wo = Integer
Real_Chart_wo = Real


# For Feature Selection /OR Interaction Building /OR Pre- Model Benchmarks
Features_PreSelect_Original = Original_All_wo
Features_PreSelect = Converted_dtypes_All_wo
# Missing values
print(train_raw_copy.isnull().sum())
print(test_raw.isnull().sum())
# Stats
basic_details(train_raw_copy)
basic_details(test_raw)
"""target (i.e.Target Variable)"""
print("Exploring target (i.e.Target Variable)...")

# List Comprehension
class_0 = [c for c in train_raw_copy['target'] if c == 0]
class_1 = [c for c in train_raw_copy['target'] if c == 1]
# # Alternative Mask Method
# class_0 = train_raw_copy.SeriousDlqin2yrs.value_counts()[0]
# class_1 = train_raw_copy.SeriousDlqin2yrs.value_counts()[1]

class_0_count = len(class_0)
class_1_count = len(class_1)

print("Target Variable Balance...")
print("Total number of class_0: {}".format(class_0_count))
print("Total number of class_1: {}".format(class_1_count))
print("Event rate: {} %".format(round(class_1_count/(class_0_count+class_1_count) * 100, 3)))   # round 3.dp
print('-' * ZZ)

# Plot
sns.countplot("target", data=train_raw_copy)
plt.show()
# Bar Plot # N/A
# Density Plot  # Chosen as opposed to histogram since this doesnt need bins parameter
print("Plotting Density Plot...for Categorical")
i = 0

# Single out the 'target' & those that are not for easy reference
t1 = train_raw_copy.loc[train_raw_copy['target'] != 0]
t0 = train_raw_copy.loc[train_raw_copy['target'] == 0]

sns.set_style('whitegrid')
# plt.figure()
fig, ax = plt.subplots(4, 4, figsize=(8, 8))

for feature in BinaryLevel_cat_col:
    i += 1
    plt.subplot(4, 4, i)
    sns.kdeplot(t1[feature], bw=0.5, label="target = 1")
    sns.kdeplot(t0[feature], bw=0.5, label="target = 0")
    plt.ylabel('Density plot', fontsize=10)
    plt.xlabel(feature, fontsize=10)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()
# Bar Plot"""   # N/A
# Density Plot"""  # Chosen
print("Plotting Density Plot...for Nominal")
i = 0
t1 = train_raw_copy.loc[train_raw_copy['target'] != 0]
t0 = train_raw_copy.loc[train_raw_copy['target'] == 0]

sns.set_style('whitegrid')
# plt.figure()
fig, ax = plt.subplots(4, 4, figsize=(8, 8))

for feature in NominalLevel_bin_col:
    i += 1
    plt.subplot(4, 4, i)
    sns.kdeplot(t1[feature], bw=0.5, label="target = 1")
    sns.kdeplot(t0[feature], bw=0.5, label="target = 0")
    plt.ylabel('Density plot', fontsize=10)
    plt.xlabel(feature, fontsize=10)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()
# Bar Plot"""   # N/A
# Density Plot"""   # N/A
# Violin Plot"""   # Chosen
print("Plotting Violin Plot...for Ordinal_Int")
sns.set_style("whitegrid")  # Chosen
for col in OrdinalLevel_other_col:
    ax = sns.violinplot(x="target", y=col, data=train_raw_copy)
    plt.show()
# Bar Plot"""   # N/A
# Density Plot"""   # N/A
# Violin Plot"""   # Chosen
print("Plotting...for Interval_Float")
sns.set_style("whitegrid")  # Chosen
for col in IntervalLevel_other_col:
    ax = sns.violinplot(x="target", y=col, data=train_raw_copy)
    plt.show()
# Set sample size to reduce computational cost
sample_SIZE = 800
sample = train_raw_copy.sample(sample_SIZE)
BinaryLevel_cat_col.extend(['target'])  # Add 'target' into list
var = BinaryLevel_cat_col
sample = sample[var]
g = sns.pairplot(sample,  hue='target', palette='Set1', size=1, diag_kind='kde', plot_kws={"s": 8})
plt.show()
BinaryLevel_cat_col.remove('target')  # Remove 'target' into list
# Set sample size to reduce computational cost
sample_SIZE = 800
sample = train_raw_copy.sample(sample_SIZE)
NominalLevel_bin_col.extend(['target'])  # Add 'target' into list
var = NominalLevel_bin_col
sample = sample[var]
g = sns.pairplot(sample,  hue='target', palette='Set1', size=1, diag_kind='kde', plot_kws={"s": 8})
plt.show()
NominalLevel_bin_col.remove('target') # Remove to revert to original
cor = train_raw_copy[NominalLevel_bin_col].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(cor,)
plt.show()
# Set sample size to reduce computational cost
sample_SIZE = 800
sample = train_raw_copy.sample(sample_SIZE)
OrdinalLevel_other_col.extend(['target'])  # Add 'target' into list
var = OrdinalLevel_other_col
sample = sample[var]
g = sns.pairplot(sample,  hue='target', palette='Set1', size=1, diag_kind='kde', plot_kws={"s": 8})
plt.show()
OrdinalLevel_other_col.remove('target') # Remove to revert to original
# Set sample size to reduce computational cost
sample_SIZE = 800
sample = train_raw_copy.sample(sample_SIZE)
IntervalLevel_other_col.extend(['target'])  # Add 'target' into list
var = IntervalLevel_other_col
sample = sample[var]
g = sns.pairplot(sample,  hue='target', palette='Set1', size=1, diag_kind='kde', plot_kws={"s": 8})
plt.show()
IntervalLevel_other_col.remove('target') # Remove to revert to original
cor = train_raw_copy[Features_PreSelect].corr()
plt.figure(figsize=(12, 9))
sns.heatmap(cor)
plt.show()
##################Lasso Parameter C Tuning
# # COMMENT: Best Parameter was found as {'logisticregression__C': 0.1}
#
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import GridSearchCV
#
# X = train_raw_copy.drop(['id', 'target'], axis = 1)
# y = train_raw_copy['target']
#
# # # {'logisticregression__C': [1, 10, 100, 1000]
# param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
# pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1'))
# grid = GridSearchCV(pipe, param_grid, cv=10)
# grid.fit(X, y)
# print(grid.best_params_)
#
#
#
# #################XGB Classifier Tuning
# from sklearn.model_selection import GridSearchCV
# from xgboost.sklearn import XGBClassifier
# from sklearn.preprocessing import StandardScaler

# # Substitute this after exery run for new parameter grid to test
# param_test1 = {
#  'classifier__max_depth': range(3, 10, 2),
#  'classifier__min_child_weight': range(1, 6, 2)
# }
#
# param_test2 = {
#  'classifier__gamma': [i/10.0 for i in range(0, 5)]
# }
#
# param_test3 = {
#  'classifier__learning_rate': [0.1, 0.01, 0.001],
#  'classifier__n_estimators=100': [100, 140, 200]
# }
#
# #Log down the best parameters
# # 'classifier__gamma': 0,
# # 'classifier__max_depth': 7,
# # 'classifier__min_child_weight': 5
#       
# print("Tuning XGBClassifier Parameters")
# #
# from sklearn.pipeline import make_pipeline
# from sklearn.pipeline import Pipeline
# print("Making XGBClassifier-Pipeline...")
# pipeXGBC = Pipeline([('scaler', StandardScaler()),
#                       ('classifier', XGBClassifier(gamma=0, max_depth=7, min_child_weight=5))])
# print("Running XGBClassifier-Pipeline Parameters GridSearchCV...")
# gsearchXGBC2 = GridSearchCV(pipeXGBC, cv=5, param_grid=param_test3)
# print("Fitting XGBClassifier-Pipeline Parameters GridSearchCV...")
# gsearchXGBC2.fit(X_train, y_train)
# print("Running XGBClassifier-Pipeline GridSearchCV Scores...")
# print(gsearchXGBC2.cv_results_, gsearchXGBC2.best_params_, gsearchXGBC2.best_score_)
# print("Running XGBClassifier-Pipeline Best Estimator...")
# best_gridXGBC2 = gsearchXGBC2.best_estimator_
# print(best_gridXGBC2)
#
#
# #################Random Forest Classifier Tuning
# # Create the parameter grid based on the results of random search
# param_grid0 = {
#      'bootstrap': [True],
#      'max_depth': [80, 90, 100, 110],
#      'max_features': [2, 3],
#      'min_samples_leaf': [3, 4, 5],
#      'min_samples_split': [8, 10, 12],
#      'n_estimators': [100, 200, 300, 1000]
# }
#
# param_grid1 = {
#     'classifier__bootstrap': [True],
#     'classifier__max_depth': [80, 100],
#     'classifier__max_features': [2, 4],
#     'classifier__min_samples_leaf': [4],
#     'classifier__min_samples_split': [10],
#     'classifier__n_estimators': [100, 200]
# }
#
# param_grid2 = {
#     'classifier__min_samples_leaf': [3, 5],
#     'classifier__min_samples_split': [10],
#     'classifier__n_estimators': [100, 200]
# }
#
# #Log down the best parameters
# #    'classifier__bootstrap': [True],
# #    'classifier__max_depth': [80],
# #    'classifier__max_features': [2],
#
# from sklearn.pipeline import make_pipeline
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# print("Making RFClassifier-Pipeline...")
# pipeRFC2 = Pipeline([('scaler', StandardScaler()),
#                      ('classifier', RandomForestClassifier(bootstrap=True, max_depth=80, max_features=2,
#                                                            criterion='entropy'))])
# print("Running RFClassifier-Pipeline Parameters GridSearchCV...")
# gsearchRFC2 = GridSearchCV(pipeRFC2, cv=5, param_grid=param_grid2)
# print("Fitting RFClassifier-Pipeline Parameters GridSearchCV...")
# gsearchRFC2.fit(X_train, y_train)
# print("Running RFClassifier-Pipeline GridSearchCV Scores...")
# print(gsearchRFC2.cv_results_, gsearchRFC2.best_params_, gsearchRFC2.best_score_)
# print("Running RFClassifier-Pipeline Best Estimator...")
# best_gridXGBC2 = gsearchRFC2.best_estimator_
# print(best_gridXGBC2)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

############# PRE DROPPING FEATURES
##### Organizing to validate C1 Drop
# ONLY C2->Fillna step iterated. NO COLUMNS DROPPED
TempToBeFilled = [c for c in train_raw.columns if train_raw[c].isnull().sum() > 0]
for col in TempToBeFilled:
    train_raw[col].fillna(train_raw[col].mode()[0], inplace=True)

train_x1 = train_raw.drop(columns=['id', 'target'])      
Y1 = train_raw['target'].values

# Preparing train/test split of dataset            
X_train, X_validation, y_train, y_validation = train_test_split(train_x1, Y1, train_size=0.9, random_state=1234)

##### Instantiate Logistic Regression 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Transform data for LogRef fitting"""
scaler = StandardScaler()
std_data = scaler.fit_transform(X_train.values)

# Establish Model
RandomState=42
model_LogRegLASSO1 = LogisticRegression(penalty='l1', C=0.1, random_state=RandomState, solver='liblinear', n_jobs=1)
model_LogRegLASSO1.fit(std_data, y_train)

# Run Accuracy score without any dropping of features
print("PRE DROPPING FEATURES: Running LASSO Accuracy Score without features drop...")
# make predictions for test data and evaluate
y_pred = model_LogRegLASSO1.predict(X_validation)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_validation, predictions)
print("PRE Accuracy: %.2f%%" % (accuracy * 100.0))
############# POST DROPPING FEATURES
train_x2 = train_raw_copy[Features_PreSelect]   
Y2 = train_raw_copy['target'].values  

# Preparing train/test split of dataset            
X_train, X_validation, y_train, y_validation = train_test_split(train_x2, Y2, train_size=0.9, random_state=1234)

##### Instantiate Logistic Regression 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Transform data for LogRef fitting"""
scaler = StandardScaler()
std_data = scaler.fit_transform(X_train.values)

# Establish Model

model_LogRegLASSO1 = LogisticRegression(penalty='l1', C=0.1, random_state=RandomState, solver='liblinear', n_jobs=1)
model_LogRegLASSO1.fit(std_data, y_train)

# Run Accuracy score without any dropping of features
print("POST DROPPING FEATURES: Running LASSO Accuracy Score with features dropped...")
# make predictions for test data and evaluate
y_pred = model_LogRegLASSO1.predict(X_validation)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_validation, predictions)
print("POST Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Preparing train/test split of dataset
train_x = train_raw_copy[Features_PreSelect]   
Y = train_raw_copy['target'].values             
X_train, X_validation, y_train, y_validation = train_test_split(train_x, Y, train_size=0.9, random_state=1234)

# Preparing Side to Side Comparative Function
from sklearn.preprocessing import MinMaxScaler

# Generic Function to Normalize Rankings/Coefficients
def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    # Transposes array of 'ranks' into single column array, then applies Fit_Transforms with MinMax
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    # shortcut map & lambda function to round ranks to 2 precision dp
    # Altenatively, You can use a list comprehension here as well. 
    # *See Mean rounding code at Chapter 9.Features (Side To Side Comparison) for example*
    ranks = map(lambda x: round(x, 2), ranks)   
    # Returns names with each respective rounded ranks
    return dict(zip(names, ranks))

names = Features_PreSelect
ranks = {}

print('Prep done...')
# LASSO via LogisticRegression l1 penalty - WhiteBox Model
print('Running LASSO via LogisticRegression l1 penalty...')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Transform data for LogReg fitting
scaler = StandardScaler()
std_data = scaler.fit_transform(X_train.values)

# Establish Model
model_LogRegLASSO = LogisticRegression(penalty='l1', C=0.1, random_state=RandomState, solver='liblinear', n_jobs=1)
model_LogRegLASSO.fit(std_data, y_train)

# For Side To Side
ranks["LogRegLASSO"] = rank_to_dict(list(map(float, model_LogRegLASSO.coef_.reshape(len(Features_PreSelect), -1))),
                                    names, order=1)
print(ranks["LogRegLASSO"])


######Alternative Direct Methods:

### Method 1 without Coefficients shown
#from sklearn.feature_selection import SelectFromModel
#model = SelectFromModel(model_LogRegLASSO, prefit=True)
#X_new = model.transform(X_train)
#print("New Shape", X_new.shape)
#print("Old Shape", X_train.shape)


### Method 2 with Coefficients shown
# Set df to append
#zero_feat = []
#nonzero_feat = []

# Loop through feature coefficients & append accordingly
#num_features = len(X_train.columns)
#for i in range(num_features):
#    coef = model_LogRegLASSO.coef_[0, i]
#    if coef == 0:
#        zero_feat.append(X_train.columns[i])
#    else:
#        nonzero_feat.append((coef, X_train.columns[i]))
#print('Features that have coefficient of 0 are: ', zero_feat, '\n')
#print('Features that have non-zero coefficients are:')
#print(sorted(nonzero_feat, reverse=True))
# Plotting
import operator
listsLASSO = sorted(ranks["LogRegLASSO"].items(), key=operator.itemgetter(1))
# convert list>array>dataframe
dfLASSO = pd.DataFrame(np.array(listsLASSO).reshape(len(listsLASSO),2), columns = ['Features','Ranks']).sort_values('Ranks') 
dfLASSO['Ranks']=dfLASSO['Ranks'].astype(float)
#df.sort_values('Ranks', ascending=True)

dfLASSO.plot.bar(x='Features', y='Ranks', color='blue')
#plt.xticks(rotation='vertical')
plt.xticks(rotation=90)

from pylab import rcParams
rcParams['figure.figsize'] = 7, 10
plt.show()
# Ridge via LogisticRegression l2 penalty - WhiteBox Model
print('Running Ridge via LogisticRegression l2 penalty...')
# Establish Model
model_LogRegRidge = LogisticRegression(penalty='l2', C=0.1, random_state=RandomState, solver='liblinear', n_jobs=1)
model_LogRegRidge.fit(std_data, y_train)

# For Side To Side
ranks["LogRegRidge"] = rank_to_dict(list(map(float, model_LogRegRidge.coef_.reshape(len(Features_PreSelect), -1))),
                                    names, order=1)
print(ranks["LogRegRidge"])
# Plotting
import operator
listsRidge = sorted(ranks["LogRegRidge"].items(), key=operator.itemgetter(1))
dfRidge = pd.DataFrame(np.array(listsRidge).reshape(len(listsRidge),2), columns = ['Features','Ranks']).sort_values('Ranks') # convert list>array>dataframe
dfRidge['Ranks']=dfRidge['Ranks'].astype(float)
#df.sort_values('Ranks', ascending=True)

dfRidge.plot.bar(x='Features', y='Ranks', color='blue')
#plt.xticks(rotation='vertical')
plt.xticks(rotation=90)

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
plt.show()
# LogisticRegression Standard 'Balanced' weighted - WhiteBox Model
print('RunningLogisticRegression Balanced...')
# Establish Model
model_LogRegBalance = LogisticRegression(class_weight='balanced', C=0.1, random_state=RandomState, solver='liblinear',
                                         n_jobs=1)
model_LogRegBalance.fit(std_data, y_train)

# For Side To Side
ranks["LogRegBalance"] = rank_to_dict(list(map(float, model_LogRegBalance.coef_.reshape(len(Features_PreSelect), -1))),
                                      names, order=1)
print(ranks["LogRegBalance"])
#Plotting
import operator
listsBal = sorted(ranks["LogRegBalance"].items(), key=operator.itemgetter(1))
dfBal = pd.DataFrame(np.array(listsBal).reshape(len(listsBal),2), columns = ['Features','Ranks']).sort_values('Ranks') # convert list>array>dataframe
dfBal['Ranks']=dfBal['Ranks'].astype(float)
#df.sort_values('Ranks', ascending=True)

dfBal.plot.bar(x='Features', y='Ranks', color='blue')
#plt.xticks(rotation='vertical')
plt.xticks(rotation=90)

from pylab import rcParams
rcParams['figure.figsize'] = 7, 10
plt.show()
# Extreme Gradiant Boosting Classifier - BlackBox Model
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

print("Running XGBClassifier Feature Importance Part 1...")
model_XGBC = XGBClassifier(objective='binary:logistic',
                           max_depth=7, min_child_weight=5,
                           gamma=0,
                           learning_rate=0.1, n_estimators=100,)
model_XGBC.fit(X_train, y_train)
print("XGBClassifier Fitted")

# For Side To Side
print("Ranking Features with XGBClassifier...")
ranks["XGBC"] = rank_to_dict(model_XGBC.feature_importances_, names)
print(ranks["XGBC"])
#Plotting
# plot feature importance for feature selection using default inbuild function
print("Plotting XGBClassifier Feature Importance")
plot_importance(model_XGBC)

from pylab import rcParams
rcParams['figure.figsize'] = 5, 10
plt.show()
# Random Forest Classifier - BlackBox Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model_RFC = RandomForestClassifier(bootstrap=True, max_depth=80,
                                   criterion='entropy',
                                   min_samples_leaf=3, min_samples_split=10, n_estimators=100)
model_RFC.fit(X_train, y_train)

# For Side To Side
print("Ranking Features with RFClassifier...")
ranks["RFC"] = rank_to_dict(model_RFC.feature_importances_, names)
print(ranks["RFC"])
#Plotting
# For Chart
importance = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(model_RFC.feature_importances_, 3)})
importance_sorted = importance.sort_values('importance', ascending=False).set_index('feature')
# plot feature importance for feature selection using default inbuild function
#print(importance_sorted)
importance_sorted.plot.bar()

from pylab import rcParams
rcParams['figure.figsize'] = 10, 20
plt.show()
pd.options.display.max_columns = 100
##### Collate Feature Coefficients Side by Side
print("Collating Side To Side Feature Scores...")

######## Easy quick print Method
# Create empty dictionary to store the mean value calculated across all the scores
r = {}
for name in names:
    # This is the alternative rounding method from the earlier map & lambda combination
    r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print("\t%s" % "\t".join(methods))
for name in names:
    print("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))
######## Alternatively, set into Dataframe. Advantage is that we can plot here.
# Loop through dictionary of scores to append into a dataframe
row_index = 0
AllFeatures_columns = ['Feature', 'Scores']
AllFeats = pd.DataFrame(columns=AllFeatures_columns)
for name in names:
    AllFeats.loc[row_index, 'Feature'] = name
    AllFeats.loc[row_index, 'Scores'] = [ranks[method][name] for method in methods]
        
    row_index += 1

# Here the dataframe scores are a list in a list. 
# To split them, we convert the 'Scores' column from a dataframe into a list & back into a dataframe again
AllFeatures_only = pd.DataFrame(AllFeats.Scores.tolist(), )
# Now to rename the column headers
AllFeatures_only.rename(columns={0:'LogRegBalance',1:'LogRegLASSO',2:'LogRegRidge',
                                     3:'Random ForestClassifier',4:'XGB Classifier', 5:'Mean'},inplace=True)
AllFeatures_only = AllFeatures_only[['LogRegBalance','LogRegLASSO','LogRegRidge', 
                                           'Random ForestClassifier', 'XGB Classifier', 'Mean']]
# Now to join both dataframes
AllFeatures_compare = AllFeats.join(AllFeatures_only).drop(['Scores'],  axis=1)
display(AllFeatures_compare)
#Plotting
df = AllFeatures_compare.melt('Feature', var_name='cols',  value_name='vals')
g = sns.factorplot(x="Feature", y="vals", hue='cols', data=df, size=10, aspect=2)

plt.xticks(rotation=90)
plt.show()
AllFeatures_compare_sort = AllFeatures_compare.sort_values(by=['Mean'], ascending=True)
order_ascending = AllFeatures_compare_sort['Feature']
#Plotting
df2 = AllFeatures_compare_sort.melt('Feature', var_name='cols',  value_name='vals')
# ONLY Difference is that now we use row_order to sort based on the above ascending Ascending Mean Features
g2 = sns.factorplot(x="Feature", y="vals", hue='cols', data=df2, size=10, aspect=2, row_order=order_ascending)

plt.xticks(rotation=90)
plt.show()
##### Ensemble Comparison of ROC AUC 
from sklearn import model_selection
import matplotlib.pyplot as plt

# run model 10x with 60/30 split, but intentionally leaving out 10% avoiding overfitting
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)

print("Charting ROC AUC for Ensembles...")
from sklearn.metrics import roc_curve, auc

# Establish Models
models = [
    {
        'label': 'LASSO',
        'model': model_LogRegLASSO,
    },
    {
        'label': 'Ridge',
        'model': model_LogRegRidge,
    },
    {
        'label': 'LogReg Balance',
        'model': model_LogRegBalance,
    },
    {
        'label': 'XGBoost Classifier',
        'model': model_XGBC,
    },
    {
        'label': 'Random Forest Classifier',
        'model': model_RFC,
    }
]

# Models Plot-loop
for m in models:
    #scaler = StandardScaler()
    #std_data2 = scaler.fit_transform(X_validation)
    #fpr, tpr, thresholds = roc_curve(y_validation, m['model'].predict_proba(std_data2).T[0])
    fpr, tpr, thresholds = roc_curve(y_validation, m['model'].predict_proba(X_validation).T[1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], roc_auc))

# Set Plotting attributes
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()
##### Ensemble Comparison of Accuracy Scores 
# Set dataframe for appending``
pd.options.display.max_columns = 100
Scores_columns = ['Model Name', 'Model Parameters', 'Train Accuracy Mean', 'Test Accuracy Mean']
Scores_compare = pd.DataFrame(columns=Scores_columns)

# Models CV-loop
row_index = 0
for m in models:
    # Name of Model
    Scores_compare.loc[row_index, 'Model Name'] = m['label']
    # Model Parameters
    Scores_compare.loc[row_index, 'Model Parameters'] = str(m['model'].get_params())
    
    # Execute Cross Validation (CV)
    cv_results = model_selection.cross_validate(m['model'], X_train, y_train, cv=cv_split)
    # Model Train Accuracy
    Scores_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()
    # Model Test Accuracy
    Scores_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()

    row_index += 1
    
display(Scores_compare)