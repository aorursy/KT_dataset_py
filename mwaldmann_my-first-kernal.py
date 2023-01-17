# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

import math

from scipy.stats import norm, skew, pearsonr

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))



# Any results you write to the current directory are saved as output.
def scoreData(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators = 100, random_state =1)

    model.fit(X_train,y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid,preds)
## Load in Data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')



## check to make sure target field is fully populated

print("There are "+ str(train_data.SalePrice.isnull().sum()) + " entries missing for the target column SalePrice.")



## Store ID field then drop the ID field. We drop the ID field because 

## it is not useful in predicting "SalePrice". Set "y_XXX" as the target/dependent column("SalePrice")

train_ID = train_data["Id"]

test_ID = test_data["Id"]

train_data = train_data.drop("Id",axis = 1)

test_data = test_data.drop("Id", axis = 1)

X_train = train_data

X_test = test_data

y_train = train_data["SalePrice"]

X_train = X_train.drop("SalePrice",axis = 1)





## validate that there are not extra fields within train data that are not present within test data ()

unique_col_check = len(set(list(X_train.columns)+list(X_test.columns)))

print("There are " + str(unique_col_check) + " unique columns between the two data sets")

print("There are " + str(len(X_train.columns)) + " within the training set.")

print("There are " + str(len(X_test.columns)) + " within the test set.")
## By running "y_train.describe()" we can tell that this data is skewed and might need to be normalized



## Try different transformations to reducde skewness

print("Initial")

print("Skewness: %.2f" % y_train.skew())

print("Kurtosis: %.2f" % y_train.kurtosis())

y_train_sqrt = np.sqrt(y_train)

print("Sqrt")

print("Skewness: %.2f" % y_train_sqrt.skew())

print("Kurtosis: %.2f" % y_train_sqrt.kurtosis())

y_train_cbrt = np.cbrt(y_train)

print("Cbrt")

print("Skewness: %.2f" % y_train_cbrt.skew())

print("Kurtosis: %.2f" % y_train_cbrt.kurtosis())

y_train_log = np.log(y_train)

print("Log")

print("Skewness: %.2f" % y_train_log.skew())

print("Kurtosis: %.2f" % y_train_log.kurtosis())

## Log is most effective in reducing skewness to normal distribution levels





fig = plt.figure(figsize=(14,7))

plt.subplot(2,2,1)

sns.distplot(y_train, fit = norm)

plt.subplot(2,2,2)

res = scipy.stats.probplot(y_train, plot = plt)



plt.subplot(2,2,3)

sns.distplot(y_train_log, fit = norm)

plt.subplot(2,2,4)

res = scipy.stats.probplot(y_train_log, plot = plt)



All_data = pd.concat([X_train,X_test])

print(All_data.shape)

empty_count = All_data.isnull().sum().sort_values(ascending = False)

empty_percent = empty_count/len(All_data)

empty_keep = [key for key in empty_percent.keys() if empty_percent[key] < .3] 

All_data = All_data[empty_keep]

print(All_data.shape)



##num_cols = [name for name in All_data.columns if All_data[name].dtype in ['int64','float64']]

##print(len(num_cols))

##cat_cols = [name for name in All_data.columns if All_data[name].dtype not in ['int64','float64']]

##print(len(cat_cols))

All_data["Age"] = All_data["YrSold"] - All_data["YearBuilt"]

All_data["MSSubClass"] = All_data["MSSubClass"].apply(str)

All_data["RemodAge"] = All_data["YrSold"] - All_data["YearRemodAdd"] ## Change to was there remod yes=1 no=0 

All_data["GarageYrBlt"] = All_data["GarageYrBlt"].apply(str)

All_data["MoSold"] = All_data["MoSold"].apply(str)

All_data["YearBuilt"] = All_data["YearBuilt"].apply(str)

All_data["YearRemodAdd"] = All_data["YearRemodAdd"].apply(str)

All_data["YrSold"] = All_data["YrSold"].apply(str)

All_data["TotalSF"] = All_data["TotalBsmtSF"] - All_data["BsmtUnfSF"] + All_data["1stFlrSF"] + All_data["2ndFlrSF"]

All_data = All_data.drop(["TotalBsmtSF","BsmtUnfSF","1stFlrSF","2ndFlrSF"],axis = 1 )

All_data["TotalBath"] = All_data["BsmtFullBath"] + .5*All_data["BsmtHalfBath"] + All_data["FullBath"] + .5*All_data["HalfBath"]

All_data = All_data.drop(["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"],axis = 1 )

#All_data["RemodYN"] = All_data["YearRemodAdd"].apply(lambda x: 'Y' if x == All_data["YearBuilt"] else "N")   

All_data["RemodYN"] = "Y"



All_data.loc[All_data.YearRemodAdd == All_data.YearBuilt, 'RemodYN'] = 'N'  



## Check how many columns are numerical versus categorical

print(All_data.shape)

num_cols = [name for name in All_data.columns if All_data[name].dtype in ['int64','float64']]

print(len(num_cols))

cat_cols = [name for name in All_data.columns if All_data[name].dtype not in ['int64','float64']]

print(len(cat_cols))



All_data_cat = All_data[cat_cols]

All_data_num = All_data[num_cols]



cat_count = All_data_cat.isnull().sum().sort_values(ascending = False)

num_count = All_data_num.isnull().sum().sort_values(ascending = False)

print(cat_count)

print(" *********************************************************  ")

print(num_count)
## Categorical attributes to be replaced with "None"

cat_none_replace = ["GarageCond","GarageQual","GarageFinish","GarageType","BsmtCond","BsmtExposure","BsmtQual",

                    "BsmtFinType2","BsmtFinType1","MasVnrType"]



## Categorical attributes to be replaced with their mode set values

cat_mode_replace = ["MSZoning", "Utilities", "Functional", "Exterior2nd", "Exterior1st", "SaleType",

                    "Electrical", "KitchenQual"]

## Numerical attributes to be replaced with their mean set values

num_mean_replace = ["TotalBath", "BsmtFinSF2", "BsmtFinSF1", "GarageArea", "GarageCars", "TotalSF","LotFrontage"]



## Replace MasVnrArea with 0 

All_data["MasVnrArea"] = All_data["MasVnrArea"].fillna(0)



## Replace Categoricals with "None"

for cat in cat_none_replace:

    All_data[cat] = All_data[cat].fillna("None")

    

## Replace Categoricals with mode

for cat in cat_mode_replace:

    All_data[cat] = All_data[cat].fillna(All_data[cat].mode()[0])

    

## Replace Numericals with median

for num in num_mean_replace:

    All_data[num] = All_data[num].fillna(All_data[num].median())



All_data_cat = All_data[cat_cols]

All_data_num = All_data[num_cols]





## Check to make sure there is no more missing data

cat_count = All_data_cat.isnull().sum().sort_values(ascending = False)

num_count = All_data_num.isnull().sum().sort_values(ascending = False)

print(cat_count)

print("   ")

print(num_count)



final_train = All_data[:len(train_data)]

final_test = All_data[len(train_data):]



num_cols = [name for name in final_train.columns if final_train[name].dtype in ['int64','float64']]

print(len(num_cols))





fig = plt.figure(figsize=(15,75))



low_corr_drop = []



i = 1

for num_cols in num_cols:

    ax = fig.add_subplot(14,3,i)

    sns.regplot(final_train[num_cols],y_train)

    corr = scipy.stats.pearsonr(final_train[num_cols],y_train)

    ax.set_title("r = " + "{0:.2f}".format(corr[0]) + "   p = " + "{0:.2f}".format(corr[1]))

    if abs(corr[0]) <.35 or corr[1] > .05:

        low_corr_drop += [num_cols]

    i+=1

print(low_corr_drop)



final_train.drop(low_corr_drop, axis = 1, inplace= True)

final_test.drop(low_corr_drop, axis = 1, inplace= True)

num_cols = [name for name in final_train.columns if final_train[name].dtype in ['int64','float64']]



print(len(final_train.columns))
plt.figure(figsize=(7,7))





# calculate the correlation matrix

corr = final_train[num_cols].corr()



# plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns,annot=True)



cat_cols = [name for name in final_train.columns if final_train[name].dtype not in ['int64','float64']]

print(len(cat_cols))



fig = plt.figure(figsize=(15,75))



i = 1

for c in cat_cols:

    ax = fig.add_subplot(15,3,i)

    sns.stripplot(x=final_train[c],y=train_data["SalePrice"])

    i+=1

dispersion_drop = []

dispersion = []

for cols in cat_cols:

    total = final_train[cols].value_counts()

    dispersion += [total/1460]



score = 0

total_score = []

for cat in dispersion:

    score = 0

    total = len(cat)

    expected = 1 / total

  

    for j in range(len(cat)):

        if (cat[j]/expected) < .05:

            score +=1

        

    if score/total > .499:

        total_score += [cat.name] 

print(total_score)

final_train.drop(total_score, axis = 1, inplace= True)

final_test.drop(total_score, axis = 1, inplace= True)    



print(final_train.shape)



cat_cols = [name for name in final_train.columns if final_train[name].dtype not in ['int64','float64']]

print(len(cat_cols))



fig = plt.figure(figsize=(15,75))



i = 1

for c in cat_cols:

    ax = fig.add_subplot(15,3,i)

    sns.stripplot(x=final_train[c],y=train_data["SalePrice"])

    i+=1
## Figure out unique entries per attribute



One_hot_unique = [[cat, len(final_train[cat].unique())] for cat in cat_cols if len(final_train[cat].unique()) >16]

print(One_hot_unique)



for i,x in One_hot_unique:

    final_train = final_train.drop(i, axis = 1)

    final_test = final_test.drop(i, axis = 1)



num_cols = [name for name in final_train.columns if All_data[name].dtype in ['int64','float64']]

print(len(num_cols))

cat_cols = [name for name in final_train.columns if All_data[name].dtype not in ['int64','float64']]

print(len(cat_cols))



#final_train = All_data[:len(train_data)]

#final_test = All_data[len(train_data):]

## Create encoder

OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

OH_train = pd.DataFrame(OH_encoder.fit_transform(final_train[cat_cols]))

OH_test = pd.DataFrame(OH_encoder.transform(final_test[cat_cols]))



OH_train.index = final_train.index

OH_test.index = final_test.index



num_final_train = final_train.drop(cat_cols, axis = 1)

num_final_test = final_test.drop(cat_cols, axis = 1)



combined_final_train = pd.concat([num_final_train,OH_train],axis = 1)

combined_final_test = pd.concat([num_final_test,OH_test],axis = 1)

## Consider replacing neighborhood with avg sale price for each neighborhood



## Consider bucketing (for example YrBuilt 1900-1909 =1,1910-1919 =2 etc.)
X_train, X_valid, y_train, y_valid = train_test_split(combined_final_train,y_train,train_size = 0.8, 

                                                      test_size = 0.2, random_state = 1)

scoreData(X_train, X_valid, y_train, y_valid)
y_train = train_data["SalePrice"]

model = RandomForestRegressor(n_estimators = 100, random_state =1)

model.fit(combined_final_train,y_train)

final_preds = model.predict(combined_final_test)



final_preds = pd.DataFrame(final_preds)

final_preds.columns = ["SalePrice"]



## sub = pd.concat([test_ID,final_preds])

sample_submission.iloc[:,1] = final_preds

print(sample_submission)



sample_submission.to_csv("submission.csv", index=False)