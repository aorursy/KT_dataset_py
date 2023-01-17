# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Loading training data

AUTOPATH = "../input"



def load_training_data():

    csv_path = os.path.join(AUTOPATH, "train.csv")

    return csv_path



def load_testing_data():

    csv_path = os.path.join(AUTOPATH, "test.csv")

    return csv_path



training_path = load_training_data()
train = pd.read_csv(training_path)

train.head()
train.info()
sns.heatmap(train.isnull())
null_cols = train.columns[train.isnull().any()].tolist()

for col in null_cols:

    print(str(col),": ", sum(train[col].isnull()))
train["LotFrontage"].describe()
train["LotFrontage"].fillna(train["LotFrontage"].mean(), inplace=True)

sns.distplot(train["LotFrontage"])
drop_cols = ["PoolQC", "Id", "Alley", "Fence", "MiscFeature", "FireplaceQu"]
def getValsUniq(col):

    print(train[col].value_counts())

    print(train[col].unique())
def fillAlley(a):

    if pd.isnull(a):

        return "NA"

    else:

        return a

    

train["Alley"] = train["Alley"].apply(fillAlley)

train["Alley"].value_counts()
getValsUniq("MasVnrType")
train["MasVnrType"].fillna("TEST", inplace=True)
train[train["MasVnrType"] == "TEST"]["MasVnrArea"]
train["MasVnrType"].replace({"TEST": "None"})

train["MasVnrArea"].fillna(0, inplace=True)

print(train["MasVnrType"].isnull().value_counts())

print(train["MasVnrArea"].isnull().value_counts())
getValsUniq("BsmtQual")
getValsUniq("BsmtCond")
train["BsmtQual"].fillna("TEST", inplace=True)
train[train["BsmtQual"] == "TEST"]["BsmtCond"].unique()
train["BsmtQual"].replace({"TEST": "NA"}, inplace=True)

train["BsmtCond"].fillna("NA", inplace=True)

print(train["BsmtQual"].value_counts())

print(train["BsmtCond"].value_counts())
getValsUniq("BsmtExposure")
# Above are 5 unique values, so NaN likely represents No Basement

train["BsmtExposure"].fillna("NA", inplace=True)
getValsUniq("BsmtFinType1")
# There are 7 unique values, NaN likely represent No Basement

train["BsmtFinType1"].fillna("NA", inplace=True)
getValsUniq("BsmtFinType2")
# Same as Type1 above

train["BsmtFinType2"].fillna("NA", inplace=True)
getValsUniq("Electrical")
# Need to look further later to determine what value to fill

# For now, I will fill w/ mode

train["Electrical"].fillna("SBrkr", inplace=True)
getValsUniq("FireplaceQu")
train["FireplaceQu"].fillna("TEST", inplace=True)
train[train["FireplaceQu"] == "TEST"]["Fireplaces"].unique()
# Since all NaN values in FireplaceQu are associated w/ 0 fireplaces

# NaN values should be filled w/ No Fireplace

train["FireplaceQu"].replace({"TEST": "NA"})

train["FireplaceQu"].isnull().unique()
garageNullCols = ["GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"]

for col in garageNullCols:

    getValsUniq(col)
train["GarageType"].fillna("TEST", inplace=True)
train[train["GarageType"] == "TEST"]["GarageArea"].unique()
train["GarageType"].replace({"TEST": "NA"})

for col in garageNullCols:

    if col != "GarageYrBlt":

        train[col].fillna("NA", inplace=True)

    else:

        train[col].fillna(0, inplace=True)

        

for col in garageNullCols:

    print(train[col].isnull().value_counts())
extrasNullCols = ["PoolQC", "Fence", "MiscFeature"]

for col in extrasNullCols:

    getValsUniq(col)
train["PoolQC"].fillna("TEST", inplace=True)

train["PoolQC"].unique()
train[train["PoolQC"] == "TEST"]["PoolArea"].unique()
train["PoolQC"].replace({"TEST": "NA"})

for col in extrasNullCols:

    train[col].fillna("NA", inplace=True)
sum(train.isnull().any())
from sklearn.preprocessing import LabelEncoder

for col in train.columns:

    if train[col].dtypes == 'object':

        le = LabelEncoder()

        train[col] = le.fit_transform(train[col])
sns.set(rc={'figure.figsize': (120,120)})

sns.set(font_scale=5)

sns.heatmap(train.corr())
plt.figure(figsize=(15,30))

plt.barh(train.corr().columns, train.corr()["SalePrice"])

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)

plt.xticks(np.arange(-1,1,step=0.2))
y=train["SalePrice"]

X=train.drop(["Id", "SalePrice"], axis=1, inplace=False)
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(train_X, train_y)
preds = lr.predict(test_X)
from sklearn.metrics import mean_absolute_error, mean_squared_error



def getMetrics (test_ys, preds):

    acc = sum(1 - (abs(preds - test_ys) / test_ys)) / len(preds)

    mae = mean_absolute_error(test_ys, preds)

    mse = mean_squared_error(test_ys, preds)

    rmse = mse ** 0.5

    

    print("Model Accuracy:", acc)

    print("Mean Absolute Error:", mae)

    print("Mean Square Error:", mse)

    print("Root Mean Square Error:", rmse)

    

    plt.figure(figsize=(15,15))

    plt.rc('xtick', labelsize=16)

    plt.rc('ytick', labelsize=16)

    plt.scatter(test_ys, preds, s=45)

    plt.xlabel("Predicted Values")

    plt.ylabel("Actual Values")

    plt.title("Predicted vs. Actual Values")
getMetrics(test_y, preds)
test_y.describe()
testing_path = load_testing_data()

test = pd.read_csv(testing_path)

test.head()
test.info()
testNullCols = test.columns[test.isnull().any()].tolist()

print(len(null_cols))

print(len(testNullCols))

testNullCols == null_cols
len(test.columns) == len(train.columns) - 1
nullObjectCols = []

nullIntFloatCols = []

for col in test.columns:

    if test[col].dtypes == "object" and col in testNullCols:

        nullObjectCols.append(col)

    elif (test[col].dtypes == "int64" or test[col].dtypes == "float64") and col in testNullCols:

        nullIntFloatCols.append(col)

        

print("NULL OBJECT COLS:\n",str(nullObjectCols))

print("NULL INT/FLOAT COLS:\n",str(nullIntFloatCols))
for col in nullObjectCols:

    print(col, ":", test[col].unique())

    

# Compare the number of categories w/ given list. Some NaN may correspond with NA in the feature descs.
for col in nullObjectCols:

    print(col, ":", sum(test[col].isnull()))
colsWithoutNA = np.array(["MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "Functional", "SaleType"])

colsWithNA = np.array([col for col in nullObjectCols if col not in colsWithoutNA])

for col in colsWithoutNA:

    test[col].fillna(test[col].mode().tolist()[0], inplace=True)

    

for col in colsWithNA:

    test[col].fillna("NA", inplace=True)
for col in nullIntFloatCols:

    print(col, ":", sum(test[col].isnull()))
test[test["GarageType"] == "NA"]["GarageArea"].unique()
test[test["GarageType"] == "NA"]["GarageCars"].unique()
colsWithZero = [col for col in nullIntFloatCols if col != "LotFrontage"]
for col in colsWithZero:

    test[col].fillna(0, inplace=True)
test["LotFrontage"].describe()
test["LotFrontage"].fillna(test["LotFrontage"].median(), inplace=True)
sum(test.isnull().any())
for col in test.columns:

    if test[col].dtypes == "object":

        le = LabelEncoder()

        test[col] = le.fit_transform(test[col])
test.info()
testing_ids = test["Id"]

submission_preds = lr.predict(test.drop("Id", axis=1, inplace=False))
output = pd.DataFrame({'Id': testing_ids,

                       'SalePrice': submission_preds})

output.to_csv('sample_submission.csv', index=False)