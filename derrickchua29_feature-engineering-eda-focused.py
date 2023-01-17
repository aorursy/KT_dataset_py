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
"""Read & Open CSV files"""
# Open Train & Test files
train_raw = pd.read_csv('../input/cs-training.csv', na_values=-1) #FYI na_values are defined in the original data page
test_raw = pd.read_csv('../input/cs-test.csv', na_values=-1)
# Copy Train file for workings
train_raw_copy = train_raw.copy(deep=True)
"""Shape"""
print('Train Shape: ', train_raw_copy.shape)
print('Test Shape: ', test_raw.shape)
display(train_raw_copy.head())
display(test_raw.head())
SampleN = 10
display(train_raw_copy.sample(SampleN))
SampleN = 10
display(test_raw.sample(SampleN))
print(train_raw_copy.info())
print(test_raw.info())
display(train_raw_copy.describe())
display(test_raw.describe())
"""Quick Stats: Data Types"""
# Function to output missing values & UniqueCounts & DataTypes
def basic_details(df):
    details = pd.DataFrame()
    details['Missing value'] = df.isnull().sum()
    details['N unique value'] = df.nunique()
    details['dtype'] = df.dtypes
    display(details)
basic_details(train_raw_copy)
basic_details(test_raw)
"""Age"""
# Reason due to Age having 0 as minimum which is not sensible
# Decided to input Median

# From above info we found these Age Stats:
# Train                 Test
# Mean - 52.30          Mean - 52.41
# Max - 109             Max - 104
# 75% - 63              75% - 63
# 50% - 52              50% - 52
# 25% - 41              25% - 41
# Min - 0               Min - 21

# Find those less than assumed legal age 18yrs old
LegalAge = 18
LessLegalAgeCount = len(train_raw_copy.loc[train_raw_copy["age"] < 18, "age"])
print("Total number of less than assumed legal age {} is {}".format(LegalAge, LessLegalAgeCount))
# Replace with Median
train_raw_copy.loc[train_raw_copy["age"] == 0, "age"] = train_raw_copy.age.median()

# Convert data-type
train_raw_copy["age"] = train_raw_copy["age"].astype('int64')
"""Check Again"""
print(train_raw_copy["age"].describe())
# Reason due to Monthly Income having 29731 Missing Values
# Decided to input Median per quartile Age range

# Determine/Set rows that fulfil these quartile range conditions
Age_Range_1 = train_raw_copy.loc[(train_raw_copy["age"] >= 18) & (train_raw_copy["age"] < 41)]
Age_Range_2 = train_raw_copy.loc[(train_raw_copy["age"] >= 41) & (train_raw_copy["age"] < 63)]
Age_Range_3 = train_raw_copy.loc[(train_raw_copy["age"] >= 63)]

# Per Determine/Set rows, Find that range mean MonthlyIncome
Age_R1_MonthlyIncome_impute = Age_Range_1.MonthlyIncome.mean()
Age_R2_MonthlyIncome_impute = Age_Range_2.MonthlyIncome.mean()
Age_R3_MonthlyIncome_impute = Age_Range_3.MonthlyIncome.mean()
# Fill Missing MonthlyIncome with 99999 for easy reference
train_raw_copy["MonthlyIncome"] = train_raw_copy["MonthlyIncome"].fillna(99999)
# Convert into integer dtype
train_raw_copy["MonthlyIncome"] = train_raw_copy["MonthlyIncome"].astype('int64')

# Now to fill them
train_raw_copy.loc[(train_raw_copy["age"] >= 18) & (train_raw_copy["age"] < 41) & (train_raw_copy["age"] == 99999)] = Age_R1_MonthlyIncome_impute

train_raw_copy.loc[(train_raw_copy["age"] >= 41) & (train_raw_copy["age"] < 63) & (train_raw_copy["age"] == 99999)] = Age_R2_MonthlyIncome_impute

train_raw_copy.loc[(train_raw_copy["age"] >= 63) & (train_raw_copy["age"] == 99999)] = Age_R3_MonthlyIncome_impute
"""Check Again"""
basic_details(train_raw_copy)
# Reason due to NumberOfDependents having 3924 Missing Values

# Fill missing with zero's
train_raw_copy["NumberOfDependents"] = train_raw_copy["NumberOfDependents"].fillna(0)
# Convert into integer dtype
train_raw_copy["NumberOfDependents"] = train_raw_copy["NumberOfDependents"].astype('int64')

# Check counts per category of 'NumberOfDependents'
#print(train_raw_copy.NumberOfDependents.value_counts())
"""Check Again"""
basic_details(train_raw_copy)
"""SeriousDlqin2yrs (i.e.Target Variable)"""
print("Exploring SeriousDlqin2yrs (i.e.Target Variable)...")

# List Comprehension
class_0 = [c for c in train_raw_copy['SeriousDlqin2yrs'] if c == 0]
class_1 = [c for c in train_raw_copy['SeriousDlqin2yrs'] if c == 1]
# # Alternative Mask Method
# class_0 = train_raw_copy.SeriousDlqin2yrs.value_counts()[0]
# class_1 = train_raw_copy.SeriousDlqin2yrs.value_counts()[1]

class_0_count = len(class_0)
class_1_count = len(class_1)

print("Target Variable Balance...")
print("Total number of class_0: {}".format(class_0_count))
print("Total number of class_1: {}".format(class_1_count))
print("Occurance event rate: {} %".format(round(class_1_count/(class_0_count+class_1_count) * 100, 3)))   # round 3.dp
# Plot
sns.countplot("SeriousDlqin2yrs", data=train_raw_copy)
plt.show()
"""Correlation Heat-Map"""
cor = train_raw_copy.corr()
plt.figure(figsize=(10, 8))
# sns.set(font_scale=0.7)
sns.heatmap(cor, annot=True, cmap='YlGn')
plt.show()
"""
Clean Data - 3 - Creating
"""

"""Joining No# Times past due: CombinedDefault"""
# Reason huge strong correlations i.e.0.98 & 0.99
# Decided to Sum all & Change into Binary Feature (1/0) (Y/0)
# Create Dummy Reference df
train_raw_copy['CD'] = (train_raw_copy['NumberOfTime30-59DaysPastDueNotWorse']
                                     + train_raw_copy['NumberOfTimes90DaysLate']
                                     + train_raw_copy['NumberOfTime60-89DaysPastDueNotWorse'])

# Set '1' for those more than zero indicating Yes there was a default before
train_raw_copy['CombinedDefault'] = 1
train_raw_copy.loc[(train_raw_copy['CD'] == 0), 'CombinedDefault'] = 0
# Remove Dummy Reference df
del train_raw_copy['CD']
"""New Feature: Net Worth"""
# Decided on general formula NetWorth = (MonthlyIncome x Age) / 10
# https://www.bogleheads.org/forum/viewtopic.php?t=195357
NetWorthDivisor = 10
train_raw_copy['NetWorth'] = train_raw_copy['MonthlyIncome'] * train_raw_copy['age'] / NetWorthDivisor
"""Join No# Loans: CombinedLoan"""
# Reason huge strong correlations i.e.0.43
# Decided to Sum all & WITH BUFFER of 5times
LoanLinesBuffer = 5
# Create Dummy Reference df
train_raw_copy['CL'] = train_raw_copy['NumberOfOpenCreditLinesAndLoans'] + train_raw_copy['NumberRealEstateLoansOrLines']

train_raw_copy['CombinedLoan'] = 1
train_raw_copy.loc[train_raw_copy['CL'] >= LoanLinesBuffer, 'CombinedLoan'] = 1
train_raw_copy.loc[train_raw_copy['CL'] < LoanLinesBuffer, 'CombinedLoan'] = 0
# Remove Dummy Reference df
del train_raw_copy['CL']
"""New Feature: Monthly debt payments"""
# Derivative formula MonthlyDebtPayments = (DebtRatio) x (MonthlyIncome)
train_raw_copy['MonthlyDebtPayments'] = train_raw_copy['DebtRatio'] * train_raw_copy['MonthlyIncome']
train_raw_copy['MonthlyDebtPayments'] = train_raw_copy['MonthlyDebtPayments'].astype('int64')
"""New Feature: Age Category"""
train_raw_copy["Age_Map"] = train_raw_copy["age"]
train_raw_copy.loc[(train_raw_copy["age"] >= 18) & (train_raw_copy["age"] < 41), "Age_Map"] = 1
train_raw_copy.loc[(train_raw_copy["age"] >= 41) & (train_raw_copy["age"] < 63), "Age_Map"] = 2
train_raw_copy.loc[(train_raw_copy["age"] >= 63), "Age_Map"] = 3

# replacing those numbers to categorical features then get the dummy variables
train_raw_copy["Age_Map"] = train_raw_copy["Age_Map"].replace(1, "Working")
train_raw_copy["Age_Map"] = train_raw_copy["Age_Map"].replace(2, "Senior")
train_raw_copy["Age_Map"] = train_raw_copy["Age_Map"].replace(3, "Retired")

train_raw_copy = pd.concat([train_raw_copy, pd.get_dummies(train_raw_copy.Age_Map, prefix='is')], axis=1)
"""New Feature: Income Category"""
train_raw_copy["Income_Map"] = train_raw_copy["MonthlyIncome"]
train_raw_copy.loc[(train_raw_copy["MonthlyIncome"] <= 3400), "Income_Map"] = 1
train_raw_copy.loc[(train_raw_copy["MonthlyIncome"] > 3400) & (train_raw_copy["MonthlyIncome"] <= 8200), "Income_Map"] = 2
train_raw_copy.loc[(train_raw_copy["MonthlyIncome"] > 8200), "Income_Map"] = 3

# replacing those numbers to categorical features then get the dummy variables
train_raw_copy["Income_Map"] = train_raw_copy["Income_Map"].replace(1, "LowY")
train_raw_copy["Income_Map"] = train_raw_copy["Income_Map"].replace(2, "MidY")
train_raw_copy["Income_Map"] = train_raw_copy["Income_Map"].replace(3, "HighY")

train_raw_copy = pd.concat([train_raw_copy, pd.get_dummies(train_raw_copy.Income_Map, prefix='is')], axis=1)

"""Check Again"""
basic_details(train_raw_copy)
SampleN = 15
display(train_raw_copy.sample(SampleN))
"""
Prepare Data for Charting
"""
Features_Preselect_All = train_raw_copy.columns
Features_Preselect_Original = ['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
                               'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                               'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                               'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                               'NumberOfDependents']
Features_Preselect_Engineered = ['CombinedDefault', 'NetWorth', 'CombinedLoan', 'MonthlyDebtPayments', 'is_Retired',
                                 'is_Senior', 'is_Working', 'is_HighY', 'is_LowY', 'is_MidY']

# Binary_bin
Binary = ['CombinedLoan', 'Age_Map', 'Income_Map']
print("Binary", "\n", Binary)

# Integer_'int'_Ordinal
Integer = ['age', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
           'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
           'CombinedDefault']
print("Integer_Ordinal", "\n", Integer)

# Real_'float'_Interval
Real = ['DebtRatio', 'MonthlyIncome', 'NetWorth', 'MonthlyDebtPayments']
print("Real_float", "\n", Real)
"""Binary"""
print("Plotting Bar Plot...for Binary")
# https://stackoverflow.com/questions/35692781/python-plotting-percentage-in-seaborn-bar-plot
sns.set_style("whitegrid")  # Chosen
for col in Binary:
    sns.barplot(x=train_raw_copy[col].value_counts().index, y=train_raw_copy[col].value_counts())
    plt.show()
"""Integer"""
print("Plotting Density Plot...for Integer")
# Used as opposed to histogram since this doesnt need bins parameter
i = 0
t1 = train_raw_copy.loc[train_raw_copy['SeriousDlqin2yrs'] != 0]
t0 = train_raw_copy.loc[train_raw_copy['SeriousDlqin2yrs'] == 0]

sns.set_style('whitegrid')
# plt.figure()
fig, ax = plt.subplots(2, 4, figsize=(15, 10))

for feature in Integer:
    i += 1
    plt.subplot(2, 4, i)
    sns.kdeplot(t1[feature], bw=0.5, label="SeriousDlqin2yrs = 1")
    sns.kdeplot(t0[feature], bw=0.5, label="SeriousDlqin2yrs = 0")
    plt.ylabel('Density plot', fontsize=10)
    plt.xlabel(feature, fontsize=10)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()
print("Plotting Scatter Plot...for Integer")
sns.set_style("whitegrid")  # Chosen
for col in Integer:
    sns.lmplot(y=col, x="Unnamed: 0", data=train_raw_copy, fit_reg=False, hue='SeriousDlqin2yrs', legend=True,
               size=5, aspect=1)
    plt.show()
"""Real"""
print("Plotting Density Plot...for Binary")
# Used as opposed to histogram since this doesnt need bins parameter
i = 0
t1 = train_raw_copy.loc[train_raw_copy['SeriousDlqin2yrs'] != 0]
t0 = train_raw_copy.loc[train_raw_copy['SeriousDlqin2yrs'] == 0]

sns.set_style('whitegrid')
fig, ax = plt.subplots(2, 2, figsize=(8, 6))

for feature in Real:
    i += 1
    plt.subplot(2, 2, i)
    sns.kdeplot(t1[feature], bw=0.5, label="SeriousDlqin2yrs = 1")
    sns.kdeplot(t0[feature], bw=0.5, label="SeriousDlqin2yrs = 0")
    plt.ylabel('Density plot', fontsize=10)
    plt.xlabel(feature, fontsize=10)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()
print("Plotting Scatter Plot...for Real")
sns.set_style("whitegrid")  # Chosen
for col in Real:
    sns.lmplot(y=col, x="Unnamed: 0", data=train_raw_copy, fit_reg=False, hue='SeriousDlqin2yrs', legend=True,
               size=5, aspect=1)
    plt.show()
"""Pair-plot"""     
print("Plotting Pair-Plots")
PairPlot = Binary + Integer + Real
# # sample = train_raw_copy.sample(frac=0.5)
sample_SIZE = 800
sample = train_raw_copy.sample(sample_SIZE)
PairPlot.extend(['SeriousDlqin2yrs'])  # Add 'target' into list
var = PairPlot
sample = sample[var]
g = sns.pairplot(sample,  hue='SeriousDlqin2yrs', palette='Set1', diag_kind='kde', plot_kws={"s": 8})
plt.show()
PairPlot.remove('SeriousDlqin2yrs')
"""Age_Map"""
print("Plotting Scatter Plot...for Bi-variate Focus_set_1")
EDA_BiVariate_1 = ['CombinedLoan',
                   'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfDependents', 'CombinedDefault',
                   'DebtRatio', 'MonthlyIncome',  'NetWorth']
sns.set_style("whitegrid")  # Chosen
for col in EDA_BiVariate_1:
    sns.lmplot(y=col, x='age', data=train_raw_copy, fit_reg=False, hue='Age_Map', legend=True,
               size=5, aspect=1)
    plt.show()
"""Income_Map"""
print("Plotting Scatter Plot...for Bi-variate Focus_set_2")
EDA_BiVariate_2 = ['CombinedLoan',
                   'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfDependents', 'CombinedDefault',
                   'DebtRatio', 'MonthlyIncome',  'NetWorth']
sns.set_style("whitegrid")  # Chosen
for col in EDA_BiVariate_1:
    sns.lmplot(y=col, x='MonthlyIncome', data=train_raw_copy, fit_reg=False, hue='Income_Map', legend=True,
               size=5, aspect=1)
    plt.show()
"""Others_A"""
print("Plotting Scatter Plot...for Bi-variate Focus_set_3")
sns.set_style("whitegrid")  # Chosen
sns.lmplot(y='DebtRatio', x='MonthlyIncome', data=train_raw_copy, fit_reg=False, hue='SeriousDlqin2yrs', legend=True,
           size=5, aspect=1)
plt.show()
"""Others_B"""
sns.lmplot(y='NumberOfTime30-59DaysPastDueNotWorse', x='MonthlyIncome', data=train_raw_copy, fit_reg=False,
           hue='SeriousDlqin2yrs', legend=True, size=5, aspect=1)
plt.show()
"""Others_C"""
sns.lmplot(y='NumberOfOpenCreditLinesAndLoans', x='MonthlyIncome', data=train_raw_copy, fit_reg=False,
           hue='SeriousDlqin2yrs', legend=True, size=5, aspect=1)
plt.show()
"""Others_D"""
sns.lmplot(y='NumberRealEstateLoansOrLines', x='MonthlyIncome', data=train_raw_copy, fit_reg=False,
           hue='SeriousDlqin2yrs', legend=True, size=5, aspect=1)
plt.show()
"""Others_E"""
sns.lmplot(y='DebtRatio', x='NumberRealEstateLoansOrLines', data=train_raw_copy, fit_reg=False, hue='SeriousDlqin2yrs',
           legend=True, size=5, aspect=1)
plt.show()
"""Others_F"""
sns.lmplot(y='NumberOfOpenCreditLinesAndLoans', x='NumberRealEstateLoansOrLines', data=train_raw_copy, fit_reg=False,
           hue='SeriousDlqin2yrs', legend=True, size=5, aspect=1)
plt.show()
cor = train_raw_copy.corr()
plt.figure(figsize=(17, 10))
# sns.set(font_scale=0.7)
sns.heatmap(cor, annot=True, cmap='YlGn')
plt.show()
"""Prepare Data"""

"""Dummy set to be used by Model to Validate above drop"""
train_raw_copy_DropValidate = train_raw_copy

"""Dropping"""
ColumnsToDrop = ['Unnamed: 0', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
                 'MonthlyIncome',
                 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines',
                 'MonthlyDebtPayments',
                 'Age_Map', 'is_Retired', 'is_Senior', 'is_Working',
                 'Income_Map', 'is_LowY', 'is_MidY', 'is_HighY']

train_raw_copy.drop(columns=ColumnsToDrop, inplace=True)
"""Check Again"""
basic_details(train_raw_copy)
"""Correlation Heat-Map"""
cor = train_raw_copy.corr()
plt.figure(figsize=(10, 8))
# sns.set(font_scale=0.7)
sns.heatmap(cor, annot=True, cmap='YlGn')
plt.show()
"""Repeat Feature Engineering for Test Set"""


def clean_dataset(dataset):

    """CombinedDefault"""
    dataset['CD'] = (dataset['NumberOfTime30-59DaysPastDueNotWorse']
                     + dataset['NumberOfTimes90DaysLate']
                     + dataset['NumberOfTime60-89DaysPastDueNotWorse'])
    dataset['CombinedDefault'] = 1
    dataset.loc[(dataset['CD'] == 0), 'CombinedDefault'] = 0
    del dataset['CD']

    """NetWorth"""
    dataset['NetWorth'] = dataset['MonthlyIncome'] * dataset['age'] / NetWorthDivisor

    """CombinedLoans"""
    dataset['CL'] = (dataset['NumberOfOpenCreditLinesAndLoans']
                     + dataset['NumberRealEstateLoansOrLines'])
    dataset['CombinedLoan'] = 1
    dataset.loc[dataset['CL'] >= LoanLinesBuffer, 'CombinedLoan'] = 1
    dataset.loc[dataset['CL'] < LoanLinesBuffer, 'CombinedLoan'] = 0
    del dataset['CL']

    """Dropping"""
    to_drop = ['Unnamed: 0', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
               'MonthlyIncome',
               'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']

    dataset.drop(columns=to_drop, inplace=True)


clean_dataset(test_raw)


"""Check Columns"""
print(train_raw_copy.columns)
print(test_raw.columns)
"""Check Shape"""
print(train_raw_copy.shape)
print(test_raw.shape)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
############# PRE DROPPING FEATURES
train_x1 = train_raw_copy_DropValidate.drop("SeriousDlqin2yrs", axis=1).copy()
#Y1 = train_raw_copy_DropValidate.SeriousDlqin2yrs
Y1 = train_raw['SeriousDlqin2yrs'].values

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
train_x2 = train_raw_copy.drop("SeriousDlqin2yrs", axis=1).copy()   
Y2 = train_raw_copy['SeriousDlqin2yrs'].values  

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
"""
Prepare Data
"""
# Split our predictors and the target variable in our data-sets
"""Train set"""
X = train_raw_copy.drop("SeriousDlqin2yrs", axis=1).copy()
y = train_raw_copy.SeriousDlqin2yrs
print(X.shape, '\n', y.shape)

"""Test set"""
X_test = test_raw.drop("SeriousDlqin2yrs", axis=1).copy()
y_test = test_raw.SeriousDlqin2yrs
print(X_test.shape, '\n', y_test.shape)


"""Train/test split of data-set"""
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state=1234)
"""
Model
"""
RandomState = 42

"""Preparing Side to Side Comparative Function"""
from sklearn.preprocessing import MinMaxScaler


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


names = X.columns
ranks = {}
print('Prep done...')
"""
LASSO via LogisticRegression l1 penalty - Feature Importance - PART 1
"""
print('Running LASSO via LogisticRegression l1 penalty...')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

""""Transform data for LogRef fitting"""
scaler = StandardScaler()
std_data = scaler.fit_transform(X_train.values)

"""Establish Model"""
model_LogRegLASSO = LogisticRegression(penalty='l1', C=0.1, random_state=RandomState, solver='liblinear', n_jobs=1)
model_LogRegLASSO.fit(std_data, y_train)

"""For Side To Side"""
ranks["LogRegLASSO"] = rank_to_dict(list(map(float, model_LogRegLASSO.coef_.reshape(len(names), -1))), names, order=1)
print(ranks["LogRegLASSO"])
"""Plotting"""
import operator
listsLASSO = sorted(ranks["LogRegLASSO"].items(), key=operator.itemgetter(1))
# convert list>array>dataframe
dfLASSO = pd.DataFrame(np.array(listsLASSO).reshape(len(listsLASSO), 2),
                       columns=['Features', 'Ranks']).sort_values('Ranks')
dfLASSO['Ranks'] = dfLASSO['Ranks'].astype(float)

dfLASSO.plot.bar(x='Features', y='Ranks', color='blue')
plt.xticks(rotation=90)
plt.show()
"""
Ridge via LogisticRegression l2 penalty - Feature Importance - PART 1
"""

print('Running Ridge via LogisticRegression l2 penalty...')
"""Establish Model"""
model_LogRegRidge = LogisticRegression(penalty='l2', C=0.1, random_state=RandomState, solver='liblinear', n_jobs=1)
model_LogRegRidge.fit(std_data, y_train)

"""For Side To Side"""
ranks["LogRegRidge"] = rank_to_dict(list(map(float, model_LogRegRidge.coef_.reshape(len(names), -1))),
                                    names, order=1)
print(ranks["LogRegRidge"])
"""Plotting"""
import operator
listsRidge = sorted(ranks["LogRegRidge"].items(), key=operator.itemgetter(1))
dfRidge = pd.DataFrame(np.array(listsRidge).reshape(len(listsRidge), 2),
                       columns=['Features', 'Ranks']).sort_values('Ranks')
dfRidge['Ranks'] = dfRidge['Ranks'].astype(float)

dfRidge.plot.bar(x='Features', y='Ranks', color='blue')
plt.xticks(rotation=90)
plt.show()
"""
LogisticRegression Balanced - Feature Importance - PART 1
"""
print('RunningLogisticRegression Balanced...')
"""Establish Model"""
model_LogRegBalance = LogisticRegression(class_weight='balanced', C=0.1, random_state=RandomState, solver='liblinear', n_jobs=1)
model_LogRegBalance.fit(std_data, y_train)

"""For Side To Side"""
ranks["LogRegBalance"] = rank_to_dict(list(map(float, model_LogRegBalance.coef_.reshape(len(names), -1))),
                                      names, order=1)
print(ranks["LogRegBalance"])
"""Plotting"""
import operator
listsBal = sorted(ranks["LogRegBalance"].items(), key=operator.itemgetter(1))
dfBal = pd.DataFrame(np.array(listsBal).reshape(len(listsBal), 2),
                     columns=['Features', 'Ranks']).sort_values('Ranks')
dfBal['Ranks'] = dfBal['Ranks'].astype(float)

dfBal.plot.bar(x='Features', y='Ranks', color='blue')
plt.xticks(rotation=90)
plt.show()
"""
XGBClassifier - Feature Importance - PART 1
"""
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
print("Running XGBClassifier Feature Importance Part 1...")
model_XGBC = XGBClassifier(objective='binary:logistic',
                           max_depth=7, min_child_weight=5,
                           gamma=0,
                           learning_rate=0.1, n_estimators=100,)
# model_XGBC.fit(X_train, y_train)
model_XGBC.fit(std_data, y_train)
print("XGBClassifier Fitted")

"""Side To Side"""
print("Ranking Features with XGBClassifier...")
ranks["XGBC"] = rank_to_dict(model_XGBC.feature_importances_, names)
print(ranks["XGBC"])
"""Plotting"""
# plot feature importance for feature selection using default inbuild function
plot_importance(model_XGBC)
plt.show()
"""
Random Forest Classifier - Feature Importance - PART 1
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model_RFC = RandomForestClassifier(bootstrap=True, max_depth=80,
                                   criterion='entropy',
                                   min_samples_leaf=3, min_samples_split=10, n_estimators=100)
# model_RFC.fit(X_train, y_train)
model_RFC.fit(std_data, y_train)
print("Random Forest Classifier Fitted")

"""Side To Side"""
print("Ranking Features with RFClassifier...")
ranks["RFC"] = rank_to_dict(model_RFC.feature_importances_, names)
print(ranks["RFC"])
"""Plotting"""
# For Chart
importance = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(model_RFC.feature_importances_, 3)})
importance_sorted = importance.sort_values('importance', ascending=False).set_index('feature')
# plot feature importance for feature selection using default inbuild function
#print(importance_sorted)
importance_sorted.plot.bar()
plt.show()
"""
Collate Side to Side df - Feature Importance
"""

"""Quick Print Method"""
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
"""Alternatively, Set into a df"""
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
AllFeatures_only.rename(columns={0: 'LogRegBalance', 1: 'LogRegLASSO', 2: 'LogRegRidge',
                                 3: 'Random ForestClassifier', 4: 'XGB Classifier', 5: 'Mean'}, inplace=True)
AllFeatures_only = AllFeatures_only[['LogRegBalance', 'LogRegLASSO', 'LogRegRidge',
                                     'Random ForestClassifier', 'XGB Classifier', 'Mean']]
# Now to join both dataframes
AllFeatures_compare = AllFeats.join(AllFeatures_only).drop(['Scores'], axis=1)
display(AllFeatures_compare)
"""Quick Plotting"""
# Plotting
df = AllFeatures_compare.melt('Feature', var_name='cols',  value_name='vals')
g = sns.factorplot(x="Feature", y="vals", hue='cols', data=df, size=10, aspect=2)

plt.xticks(rotation=90)
plt.show()
"""Sorted Plotting"""
AllFeatures_compare_sort = AllFeatures_compare.sort_values(by=['Mean'], ascending=True)
order_ascending = AllFeatures_compare_sort['Feature']
#Plotting
df2 = AllFeatures_compare_sort.melt('Feature', var_name='cols',  value_name='vals')
# ONLY Difference is that now we use row_order to sort based on the above ascending Ascending Mean Features
# g2 = sns.factorplot(x="Feature", y="vals", hue='cols', data=df2, size=10, aspect=2, row_order=order_ascending)
g2 = sns.factorplot(x="Feature", y="vals", hue='cols', data=df2, size=10, aspect=2, row_order=order_ascending)
plt.xticks(rotation=60)
plt.show()
"""
Collate Side to Side df - ROC AUC
"""
# Ensemble Comparison of ROC AUC
from sklearn import model_selection
import matplotlib.pyplot as plt


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
    #fpr, tpr, thresholds = roc_curve(y_validation, m['model'].predict_proba(X_validation).T[1])
    scaler = StandardScaler()
    std_data2 = scaler.fit_transform(X_validation.values)
    fpr, tpr, thresholds = roc_curve(y_validation, m['model'].predict_proba(std_data2).T[1])
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
"""Collate Side to Side df - Accuracy Scores"""
# run model 10x with 60/30 split, but intentionally leaving out 10% avoiding overfitting
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)

# Ensemble Comparison of Accuracy Scores
# Set dataframe for appending``
# pd.options.display.max_columns = 100
ACCScores_columns = ['Model Name', 'Train Accuracy Mean', 'Test Accuracy Mean']
ACCScores_compare = pd.DataFrame(columns=ACCScores_columns)

# Models CV-loop
row_index = 0
for m in models:
    # Name of Model
    ACCScores_compare.loc[row_index, 'Model Name'] = m['label']

    # Execute Cross Validation (CV)
    cv_results = model_selection.cross_validate(m['model'], X_train, y_train, cv=cv_split)
    # Model Train Accuracy
    ACCScores_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()
    # Model Test Accuracy
    ACCScores_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()

    row_index += 1

display(ACCScores_compare)