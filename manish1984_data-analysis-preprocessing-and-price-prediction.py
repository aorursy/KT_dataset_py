import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 80)

# plot libs

import seaborn as sns

sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,8)

plt.rcParams["axes.labelsize"] = 16
train_data = pd.read_csv("../input/train.csv", encoding = "ISO-8859-1")

test_data = pd.read_csv("../input/test.csv", encoding = "ISO-8859-1")

print("The training set shape is : %s" % str(train_data.shape))

print("The test set shape is : %s" % str(test_data.shape))
train_data.head(n=10)
train_and_test_data = pd.concat([train_data, test_data])

sum_result = train_and_test_data.isna().sum(axis=0).sort_values(ascending=False)



"""

Pandas dataframe.isna() function is used to detect missing values. It return a boolean same-sized 

object indicating if the values are NA. NA values, such as None or numpy.NaN, gets mapped to True 

values. Everything else gets mapped to False values. Characters such as empty strings ” or numpy.inf

are not considered NA values (unless you set pandas.options.mode.use_inf_as_na = True).

Parameters :

arr : input array.

axis : axis along which we want to calculate the sum value. Otherwise, it will consider arr to be 

flattened(works on all the axis). axis = 0 means along the column and axis = 1 means working along the

row.

out : Different array in which we want to place the result. The array must have same dimensions as 

expected output. Default is None.

initial : [scalar, optional] Starting value of the sum.

Return : Sum of the array elements (a scalar value if axis is none) or array with sum values along 

the specified axis.





"""

print(sum_result)



missing_values_columns = sum_result[sum_result > 0]

print('A')

print(missing_values_columns)

missing_values_columns1 = sum_result.count() > 0

print('B')

print(missing_values_columns1)

print('They are %s columns with missing values : \n%s' % (missing_values_columns.count(), [(index, value) for (index, value) in missing_values_columns.iteritems()]))
def impute_missing_values(train_data):

    dataset = train_data

    dataset["PoolQC"].fillna("NA", inplace=True)

    dataset["MiscFeature"].fillna("NA", inplace=True)

    dataset["Alley"].fillna("NA", inplace=True)

    dataset["Fence"].fillna("NA", inplace=True)

    dataset["FireplaceQu"].fillna("NA", inplace=True)

    dataset["LotFrontage"].fillna(dataset["LotFrontage"].median(), inplace=True)

    dataset["GarageType"].fillna("NA", inplace=True)

    dataset["GarageQual"].fillna("NA", inplace=True)

    dataset["GarageCond"].fillna("NA", inplace=True)

    dataset["GarageFinish"].fillna("NA", inplace=True)

    dataset["GarageYrBlt"].fillna(dataset["GarageYrBlt"].median(), inplace=True)

    dataset["BsmtExposure"].fillna("NA", inplace=True)

    dataset["BsmtFinType2"].fillna("NA", inplace=True)

    dataset["BsmtQual"].fillna("NA", inplace=True)

    dataset["BsmtCond"].fillna("NA", inplace=True)

    dataset["BsmtFinType1"].fillna("NA", inplace=True)

    dataset["MasVnrArea"].fillna(dataset["MasVnrArea"].median(), inplace=True)

    dataset["MasVnrType"].fillna("None", inplace=True)

    dataset["Electrical"].fillna("SBrkr", inplace=True)  # SBrkr is the most common value for 1334 houses

    dataset["BsmtQual"].fillna("NA", inplace=True)

    dataset["MSZoning"].fillna("TA", inplace=True)

    dataset["BsmtFullBath"].fillna(0, inplace=True)

    dataset["BsmtHalfBath"].fillna(0, inplace=True)

    dataset["Utilities"].fillna("AllPub", inplace=True)

    dataset["Functional"].fillna("Typ", inplace=True)

    dataset["Electrical"].fillna("SBrkr", inplace=True)

    dataset["Exterior2nd"].fillna("VinylSd", inplace=True)

    dataset["KitchenQual"].fillna("TA", inplace=True)

    dataset["Exterior1st"].fillna("VinylSd", inplace=True)

    dataset["GarageCars"].fillna(0, inplace=True)

    dataset["GarageArea"].fillna(dataset["GarageArea"].median(), inplace=True)

    dataset["TotalBsmtSF"].fillna(dataset["TotalBsmtSF"].median(), inplace=True)

    dataset["BsmtUnfSF"].fillna(dataset["BsmtUnfSF"].median(), inplace=True)

    dataset["BsmtFinSF2"].fillna(dataset["BsmtFinSF2"].median(), inplace=True)

    dataset["BsmtFinSF1"].fillna(dataset["BsmtFinSF1"].median(), inplace=True)

    dataset["SaleType"].fillna("WD", inplace=True)    

    return dataset

   

train_data = impute_missing_values(train_data)

test_data = impute_missing_values(test_data)
#Sale_Price=train_data.groupby("SalePrice").value_count()

# Here 50 is bin size

# Histogram creation using Matplotlib

plt.hist(train_data["SalePrice"], 50)



plt.xlabel("SalePrice")

plt.ylabel("Count")

plt.grid(True)

plt.show()



# Histogram creation using SNS

sns.distplot(train_data["SalePrice"],kde = False)

plt.ylabel("Count")

plt.show()



# Histogram creation using SNS with Histogram as False

sns.distplot(train_data["SalePrice"],hist = False)

plt.ylabel("Count")

plt.show()



# Histogram creation using SNS with no parameter

sns.distplot(train_data["SalePrice"])

plt.ylabel("Count")

plt.show()
train_data["SalePrice"].describe()
"""

DataFrame.select_dtypes(self, include=None, exclude=None)

include, exclude : scalar or list-like

A selection of dtypes or strings to be included/excluded. At least one of these parameters must be 

supplied.



To select all numeric types, use np.number or 'number'

To select strings you must use the object dtype, but note that this will return all object dtype 

columns

See the numpy dtype hierarchy

To select datetimes, use np.datetime64, 'datetime' or 'datetime64'

To select timedeltas, use np.timedelta64, 'timedelta' or 'timedelta64'

To select Pandas categorical dtypes, use 'category'

"""



numeric_cols = list(train_data.select_dtypes(include=[np.number]))

print("Here are the %s numeric variables : \n %s" % (len(numeric_cols), numeric_cols))

numeric_cols.remove('Id')

numeric_cols.remove('MSSubClass')

print("Here are the %s numeric variables : \n %s" % (len(numeric_cols), numeric_cols))
numerical_values = train_data[list(train_data.select_dtypes(include=[np.number]))]

# Get the more correlated variables by sorting in descending order for the SalePrice column

ix = numerical_values.corr().sort_values('SalePrice', ascending=False).index

df_sorted_by_correlation = numerical_values.loc[:, ix]

# take only the first 15 more correlated variables

fifteen_more_correlated = df_sorted_by_correlation.iloc[:, :15]

corr = fifteen_more_correlated.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    # display a correlation heatmap

    ax = sns.heatmap(corr, mask=mask, annot=True)
sns.boxplot(x="OverallQual", y="SalePrice", data=train_data[['OverallQual', 'SalePrice']])

plt.show()
sns.pairplot(data=train_data, x_vars=['GrLivArea'], y_vars=['SalePrice'], size=9, kind='reg')

plt.show()
train_data.replace({'MSSubClass': {

    20: "1-STORY 1946 & NEWER ALL STYLES",

    30: "1-STORY 1945 & OLDER",

    40: "1-STORY W/FINISHED ATTIC ALL AGES",

    45: "1-1/2 STORY - UNFINISHED ALL AGES",

    50: "1-1/2 STORY FINISHED ALL AGES",

    60: "2-STORY 1946 & NEWER",

    70: "2-STORY 1945 & OLDER",

    75: "2-1/2 STORY ALL AGES",

    80: "SPLIT OR MULTI-LEVEL",

    85: "SPLIT FOYER",

    90: "DUPLEX - ALL STYLES AND AGES",

   120: "1-STORY PUD (Planned Unit Development) - 1946 & NEWER",

   150: "1-1/2 STORY PUD - ALL AGES",

   160: "2-STORY PUD - 1946 & NEWER",

   180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",

   190: "2 FAMILY CONVERSION - ALL STYLES AND AGES",

}}, inplace=True)
g = sns.boxplot(x="MSSubClass", y="SalePrice", data=train_data)

for item in g.get_xticklabels():

    item.set_rotation(75)
neighborhood_median_plot = train_data.groupby('Neighborhood')['SalePrice'].median().plot(kind='bar')

neighborhood_median_plot.set_ylabel('SalePrice')

h = neighborhood_median_plot.axhline(train_data['SalePrice'].mean())
count_neighborhood_plot = train_data.groupby('Neighborhood')['Neighborhood'].count().plot(kind='bar')

count_neighborhood_plot = count_neighborhood_plot.set_ylabel('Count')
fig, axes = plt.subplots(nrows=2, ncols=3, squeeze=True)

figsize = (15, 10)

train_data.groupby("GarageType")["GarageType"].count().plot(kind='bar', ax=axes[0][0], figsize=figsize).set_ylabel('Count')

train_data.groupby("GarageFinish")["GarageFinish"].count().plot(kind='bar', ax=axes[0][1], figsize=figsize).set_ylabel('Count')

train_data.groupby("GarageCars")["GarageCars"].count().plot(kind='bar', ax=axes[0][2], figsize=figsize).set_ylabel('Count')

train_data.groupby("GarageQual")["GarageQual"].count().plot(kind='bar', ax=axes[1][0], figsize=figsize).set_ylabel('Count')

train_data.groupby("GarageCond")["GarageCond"].count().plot(kind='bar', ax=axes[1][1], figsize=figsize).set_ylabel('Count')

train_data.groupby("GarageYrBlt")["GarageYrBlt"].count().plot(kind='line', ax=axes[1][2], figsize=figsize).set_ylabel('Count')

fig.tight_layout(pad=3, w_pad=3, h_pad=3)
garage_cars_median_plot = train_data.groupby('GarageCars')['SalePrice'].median().plot(kind='bar')

garage_cars_median_plot.set_ylabel('SalePrice')

h = garage_cars_median_plot.axhline(train_data['SalePrice'].mean())
g = sns.pairplot(data=train_data, x_vars=['GarageArea'], y_vars=['SalePrice'], size=6, kind='reg')
def transform_variables(dataset):

    copy = dataset

    copy.replace({'MSSubClass': {

        20: "1-STORY 1946 & NEWER ALL STYLES",

        30: "1-STORY 1945 & OLDER",

        40: "1-STORY W/FINISHED ATTIC ALL AGES",

        45: "1-1/2 STORY - UNFINISHED ALL AGES",

        50: "1-1/2 STORY FINISHED ALL AGES",

        60: "2-STORY 1946 & NEWER",

        70: "2-STORY 1945 & OLDER",

        75: "2-1/2 STORY ALL AGES",

        80: "SPLIT OR MULTI-LEVEL",

        85: "SPLIT FOYER",

        90: "DUPLEX - ALL STYLES AND AGES",

       120: "1-STORY PUD (Planned Unit Development) - 1946 & NEWER",

       150: "1-1/2 STORY PUD - ALL AGES",

       160: "2-STORY PUD - 1946 & NEWER",

       180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",

       190: "2 FAMILY CONVERSION - ALL STYLES AND AGES",

    }}, inplace=True)

    # one hot encoding

    one_hot_columns = [

        'Neighborhood', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating',

        'Electrical', 'Functional', 'GarageType', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'Foundation'

    ]

    for col_name in one_hot_columns:

        copy = pd.concat([copy, pd.get_dummies(copy[col_name], prefix=col_name)], axis=1)

        copy = copy.drop(col_name, axis=1)

        

    # ordinal variables transformation

    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}

    basement_map = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

    ordinal_maps = {

        "ExterCond": quality_map,

        "ExterQual": quality_map,

        "LandSlope": {'Gtl': 0, 'Mod': 1, 'Sev': 2},

        "MasVnrType": {'None': 0, 'BrkCmn': 0, 'BrkFace': 1, 'Stone': 2},

        "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0},

        "BsmtFinType1": basement_map,

        "BsmtFinType2": basement_map,

        "BsmtQual": quality_map,

        "BsmtCond": quality_map,

        "HeatingQC": quality_map,

        "CentralAir": {'N': 0, 'Y': 1},

        "KitchenQual": quality_map,

        "FireplaceQu": quality_map,

        "GarageQual": quality_map,

        "GarageCond": quality_map,

        "GarageFinish": {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},

        "PavedDrive": {'N': 0, 'P': 1, 'Y': 2},

        "PoolQC": quality_map

    }

    for col_name, matching_map in ordinal_maps.items():

        copy[col_name] = copy[col_name].replace(matching_map)

    

    # remove high correlated variables to other variables

    copy = copy.drop(['YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotRmsAbvGrd', 'BsmtFinSF1'], axis=1)

    return copy
result = pd.DataFrame()

result['ExterQual'] = train_data['ExterQual'].replace({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0})

result.head()
garage_cars_median_plot = train_data.groupby('MasVnrType')['SalePrice'].median().plot(kind='bar')

garage_cars_median_plot.set_ylabel('SalePrice')

h = garage_cars_median_plot.axhline(train_data['SalePrice'].mean())
result = pd.DataFrame()

result['MasVnrType'] = train_data['MasVnrType'].replace({'None': 0, 'BrkCmn': 0, 'BrkFace': 1, 'Stone': 2})

result.head(n=8)
result = pd.get_dummies(train_data['LotConfig'], prefix='LotConfig')

result.head()
y_train = train_data['SalePrice']

X_train = transform_variables(train_data)



X_test = test_data

impute_missing_values(X_test)

X_test = transform_variables(X_test)



for col_name in list(X_train.columns):

    if col_name not in X_test.columns:

        X_test[col_name] = 0



# need to investigate why X_test got extra columns compare to X_train

test_cols = list(X_test.columns)

train_cols = list(X_train.columns)

def Diff(li1, li2):

    return (list(set(li1) - set(li2)))

X_test = X_test.drop(Diff(test_cols, train_cols), axis=1)        



predictor_cols = [col for col in X_train 

                  if col != 'SalePrice'

                 ]



print(str(X_test.shape) + " should be similar to " + str(X_train.shape))
from math import floor

from sklearn.preprocessing import MinMaxScaler

        

# filter some variables under represented

# number_of_ones_by_cols = X_train.astype(bool).sum(axis=0)

# less_than_ten_ones_cols = number_of_ones_by_cols[number_of_ones_by_cols < 10].keys()

# X_train = X_train.drop(list(less_than_ten_ones_cols), axis=1)

# X_test = X_test.drop(list(less_than_ten_ones_cols), axis=1)

        

# cols_to_scale = ['GrLivArea', 'TotalBsmtSF', 'GarageArea']

# scaler = MinMaxScaler()

# X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])

# X_test[cols_to_scale] = scaler.fit_transform(X_test[cols_to_scale])



from sklearn import linear_model



clf = linear_model.Lasso(alpha=1, max_iter=10000)

clf.fit(X_train[predictor_cols], y_train)



y_pred = clf.predict(X_test[predictor_cols])



print(clf.intercept_)

print(y_pred)

my_submission = pd.DataFrame({'Id': X_test.Id, 'SalePrice': y_pred})

my_submission.to_csv('lasso.csv', index=False)