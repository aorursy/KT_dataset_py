%matplotlib inline



import os

import pandas as pd

import numpy as np

import seaborn as sns

import lightgbm



from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import StratifiedKFold

from sklearn.impute import SimpleImputer

from matplotlib import pyplot as plt
# To remove the limit on the number of rows displayed by pandas

pd.set_option("display.max_rows", None)



# Read csv files in pandas dataframe

testDf = pd.read_csv('../input/test.csv')

trainDf = pd.read_csv('../input/train.csv')

print("Training dataset basic information")

print("Rows: {}".format(len(trainDf)))

print("Columns: {}".format(len(trainDf.columns)))

trainDf.head()
print("Test dataset basic information")

print("Rows: {}".format(len(testDf)))

print("Columns: {}".format(len(testDf.columns)))

testDf.head()
# Add null Target column to test

testDf['Target'] = np.nan

data = trainDf.append(testDf, ignore_index = True)
# 1. Find missing values in training and test dataset

def findColumnsWithNan(df):

    cols = df.columns[df.isna().any()]

    print("Number of columns with Nan: {}".format(len(cols)))

    print("Column names: {}".format(cols))

    print("-" * 80)

    for col in cols:

        print("Column: [{}] missing {} values.".format(col, len(df[df[col].isna() == True])))



print("Analysis of training dataset...")

findColumnsWithNan(trainDf)
print()

print("Analysis of test dataset...")

findColumnsWithNan(testDf)
data.loc[(data['tipovivi1'] == 1) & (data['v2a1'].isna()), 'v2a1'] = 0

print("Missing values after replacing: {}".format(len(data.loc[data['v2a1'].isna()])))
data.loc[data['v18q1'].isna(), 'v18q1'] = 0

print("Missing values after replacing: {}".format(len(data.loc[data['v18q1'].isna()])))
data.loc[(data['age'] < 7) & (data['rez_esc'].isna()), 'rez_esc'] = 0

data.loc[(data['age'] > 19) & (data['rez_esc'].isna()), 'rez_esc'] = 0

print("Missing values after replacing: {}".format(len(data.loc[data['rez_esc'].isna()])))
data.loc[data['age'] < 19 & data['meaneduc'].isna(), 'meaneduc'] = 0

print("Missing values after replacing: {}".format(len(data.loc[data['meaneduc'].isna()])))
data.drop('SQBmeaned', inplace=True, axis=1)

print("Total number of columns left: {}".format(len(data.columns)))
for cols in data.columns[1:]:

    if cols in ['idhogar', 'dependency', 'edjefe', 'edjefa']:

        continue

    percentile75 = np.percentile(data[cols].fillna(0), 75)

    percentile25 = np.percentile(data[cols].fillna(0), 25)

    threshold = (percentile75 - percentile25) * 1.5

    lower, upper = (percentile25 - threshold), (percentile75 + threshold)

    # identify outliers

    outliers = data.loc[(data[cols] < lower) & (data[cols] > upper)]

    if len(outliers) > 0:

        print('Feature: {}. Identified outliers: {}'.format(cols, len(outliers)))
for col in ['dependency', 'edjefe', 'edjefa']:

    data.loc[data[col] == 'yes', col] = 1.0

    data.loc[data[col] == 'no', col] = 0.0

    data[col] = pd.to_numeric(data[col])
corrMat = data.corr()

plt.figure(figsize=(30, 10))

sns.heatmap(corrMat.iloc[:10, :10])
def featuresToDrop(corrMatrix):

    """

    To remove correlated features, used this gem of a code from here:

    https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features

    """

    # Select upper triangle of correlation matrix

    upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))



    # Find index of feature columns with correlation greater than 0.95

    return [column for column in upper.columns if any(upper[column] > 0.95)]



toDrop = featuresToDrop(corrMat)

data.drop(toDrop, inplace=True, axis=1)

print("Correlated features which are dropped: {}".format(toDrop))
features = list(data.drop(columns = ['Id', 'idhogar', 'Target']).columns)

aggDf = data.drop(columns='Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std'])

# Rename the columns

new_col = []

for c in aggDf.columns.levels[0]:

    for stat in aggDf.columns.levels[1]:

        new_col.append('{}-{}'.format(c, stat))

        

aggDf.columns = new_col

toDrop = featuresToDrop(aggDf.corr())

aggDf.drop(toDrop, inplace=True, axis=1)

data = data.merge(aggDf, on='idhogar', how ='left')

print('Training feature shape: ', data.shape)
data['phones-per-capita'] = data['qmobilephone'] / data['tamviv']

data['tablets-per-capita'] = data['v18q1'] / data['tamviv']

data['rooms-per-capita'] = data['rooms'] / data['tamviv']

data['rent-per-capita'] = data['v2a1'] / data['tamviv']
# Labels for training

trainTarget = np.array(list(data[data['Target'].notnull()]['Target'].astype(np.uint8)))

submission = data.loc[data['Target'].isnull(), 'Id'].to_frame()



# Extract the training data

trainData = data[data['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])

testData = data[data['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])



# Impute training and test data

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

trainData = imputer.fit_transform(trainData)

testData = imputer.transform(testData)



# Scale training and test data

scaler = MinMaxScaler()

trainData = scaler.fit_transform(trainData)

testData = scaler.transform(testData)
model = lightgbm.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',

                             random_state=None, silent=True, metric='None', 

                             n_jobs=4, n_estimators=5000, class_weight='balanced',

                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)

kfold = 5

kf = StratifiedKFold(n_splits=kfold, shuffle=True)



predicts_result = []

for idx, (train_index, test_index) in enumerate(kf.split(trainData, trainTarget)):

    print("Fold: {}".format(idx))

    X_train, X_val = trainData[train_index], trainData[test_index]

    y_train, y_val = trainTarget[train_index], trainTarget[test_index]

    model.fit(X_train, y_train, verbose=100)

    predicts_result.append(model.predict(testData))

submission['Target'] = np.array(predicts_result).mean(axis=0).round().astype(int)

submission.to_csv('submission.csv', index=False)

print("Completed!")