import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
trainFile = '../input/house-prices-advanced-regression-techniques/train.csv'

testFile = '../input/house-prices-advanced-regression-techniques/test.csv'

trainDf = pd.read_csv(trainFile, index_col='Id')

testDf = pd.read_csv(testFile, index_col='Id')
trainDf.head()
testDf.head()
testDf.columns.difference(trainDf.columns)
trainDf.info()
sns.heatmap(trainDf.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Too many NAs LotShape, Fence, GarageType
plt.figure(figsize=(20,20))

sns.heatmap(trainDf.corr(), annot=True)
# Create correlation matrix

corr_matrix = trainDf.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.80

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
# Drop features 

trainDf.drop(trainDf[to_drop], axis=1, inplace=True)

testDf.drop(testDf[to_drop], axis=1, inplace=True)
# distribution might normalize with log, but unnecessary with trees

sns.distplot(trainDf['SalePrice'])
sns.distplot(np.log(trainDf['SalePrice']))
# Clean up NAs

# How many NAs?

NAs = pd.concat([trainDf.isnull().sum()], axis=1)

NAs[NAs.sum(axis=1) > 0]

missingCols = list(NAs[NAs.sum(axis=1) > 0].index)

missingCols
missingCols = list(NAs[NAs.sum(axis=1) > 0].index)

missingCols

missingCols.remove('Electrical')

missingCols
# zero out everything except electrical
# Fill NaN depending on what they are. Impute vs 0 out

dataframes = [trainDf,testDf]



for df in dataframes:

    for col in missingCols:

        df[col].fillna(0, inplace=True)
NAs[NAs.sum(axis=1) > 0]
# Test NAs

testNAs = pd.concat([testDf.isnull().sum()], axis=1)

testNAs[testNAs.sum(axis=1) > 0]

testMissingCols = list(testNAs[testNAs.sum(axis=1) > 0].index)

testNAs[testNAs.sum(axis=1) > 0]
testFillList=['MSZoning','Utilities','Functional','SaleType']

testZeroList=[x for x in testMissingCols if x not in testFillList]

for col in testZeroList:

    testDf[col].fillna(0, inplace=True)
# Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(trainDf.drop(['SalePrice'],axis=1), trainDf['SalePrice'], test_size=0.3)
#Create Labeled Data

object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

good_label_cols = [col for col in object_cols if set(X_train[col])==set(X_valid[col])]

bad_label_cols = list(set(object_cols)-set(good_label_cols))

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in good_label_cols:

    label_X_train[col] = le.fit_transform(X_train[col].astype(str))

    label_X_valid[col] = le.transform(X_valid[col].astype(str))

label_X_train.info()
label_X_test = testDf.drop(bad_label_cols, axis=1)

for col in good_label_cols:

    label_X_test[col] = le.fit_transform(testDf[col].astype(str))

label_X_test.info()
# Impute on training Data

from sklearn.impute import SimpleImputer
imputed_X_train = label_X_train

imputed_X_valid = label_X_valid
imputed_X_test = pd.DataFrame(columns=label_X_test.columns, index=label_X_test.index)

imputer = SimpleImputer(strategy='most_frequent')
imputed_X_test['MSZoning'] = imputer.fit_transform(label_X_test[['MSZoning']])

imputed_X_test = pd.concat([label_X_test.drop('MSZoning', axis=1), imputed_X_test['MSZoning']], axis = 1)
NAs = pd.concat([imputed_X_valid.isnull().sum()], axis=1)

NAs[NAs.sum(axis=1) > 0]
testNAs = pd.concat([imputed_X_test.isnull().sum()], axis=1)

testNAs[testNAs.sum(axis=1) > 0]
imputed_X_test.columns.difference(imputed_X_train.columns)
# Create Model

from xgboost import XGBRegressor

model = XGBRegressor(early_stopping_rounds=5,

                     eval_set=[(imputed_X_valid, y_valid)],

                     objective='reg:squarederror',

                     verbose=False)

model.fit(imputed_X_train,y_train)
pred = model.predict(imputed_X_valid)
# metrics

from sklearn import metrics

np.sqrt(metrics.mean_squared_error(np.log(y_valid), np.log(pred)))
# Merge Training and Validation Sets

full_X = pd.concat([imputed_X_train,imputed_X_valid],axis=0)

full_X.sort_index(inplace=True)



full_y = pd.concat([y_train,y_valid],axis=0)

full_y.sort_index(inplace=True)
imputed_X_test
imputed_X_test = imputed_X_test[full_X.columns]
# Train on full data

model.fit(full_X,full_y)
fullPred = model.predict(imputed_X_test)
presubmissionDf = pd.DataFrame(fullPred, columns=['SalePrice'])

presubmissionDf.index = np.arange(1461, 1461+len(presubmissionDf))

presubmissionDf = pd.concat([imputed_X_test,presubmissionDf], axis=1)

presubmissionDf
submissionDf=pd.DataFrame(presubmissionDf['SalePrice'], columns=['SalePrice'])

submissionDf = submissionDf.rename_axis('Id')

submissionDf
submissionDf.to_csv('housingPricePredictions.csv')