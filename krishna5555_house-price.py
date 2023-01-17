# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import pandas as pd

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/train.csv")

df.head()
label=df["SalePrice"]

df=df.drop("SalePrice",axis=1)

print(df.head())

print(label.head())
missingno.matrix(df)
df.dtypes
df.describe()
df_test=pd.read_csv("/kaggle/input/test.csv")
df_test.head()
df_test.describe()
sns.distplot(label)

print("Skewness:",label.skew())

print("Kurtosis:",label.kurt())
label=np.log(label)

print("Skewness:",label.skew())

print("Kurtosis:",label.kurt())

sns.distplot(label)
numerical_features=df.dtypes[df.dtypes!="object"]

categorical_features=df.dtypes[df.dtypes=="object"]

print("number of numerical features:",len(numerical_features))

print("number of categorical features:",len(categorical_features))
print("numerical_features:\n",numerical_features.head(),"\n\ncategorical_features:\n",categorical_features.head())
df.isnull().sum().sort_values(ascending=False)
numerical_features=list(df.columns[df.dtypes!="object"])

categorical_features=list(df.columns[df.dtypes=="object"])
rows=12

cols=3

fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*3))

numerical_features_without_id=numerical_features[1:]

df["YrSold"]=df["YrSold"].astype("int")

for r in range(rows):

    for c in range(cols):

        m=r*cols+c

        sns.regplot(df[numerical_features_without_id[m]],label,ax=axs[r][c])

plt.tight_layout()

plt.show()
df_cor=pd.read_csv("/kaggle/input/train.csv")

df_cor["SalePrice"]=np.log(df_cor["SalePrice"])

df_cor.corr()["SalePrice"]
rows=15

cols=3

fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*3))

categorical_features_without_id=categorical_features

for r in range(rows):

    for c in range(cols):

        m=r*cols+c

        if(m<len(categorical_features)):

            sns.boxplot(df[categorical_features[m]],label,ax=axs[r][c])

plt.tight_layout()

plt.show()
strong_corelation_category_list=["MSZoning","Neighborhood","Exterior1st","MasVnrType","ExterQual","BsmtQual","PoolQC","MiscFeature","SaleType","SaleCondition","SalePrice"]
df_cor[strong_corelation_category_list].head()
le = preprocessing.LabelEncoder()

df_cor[strong_corelation_category_list]=df_cor[strong_corelation_category_list].fillna("None",inplace=False)

le.fit(df_cor["MSZoning"])

df_cor["MSZoning"]=le.transform(df_cor["MSZoning"])



le.fit(df_cor["Neighborhood"])

df_cor["Neighborhood"]=le.transform(df_cor["Neighborhood"])



le.fit(df_cor["Exterior1st"])

df_cor["Exterior1st"]=le.transform(df_cor["Exterior1st"])



le.fit(df_cor["MasVnrType"])

df_cor["MasVnrType"]=le.transform(df_cor["MasVnrType"])



le.fit(df_cor["ExterQual"])

df_cor["ExterQual"]=le.transform(df_cor["ExterQual"])



le.fit(df_cor["BsmtQual"])

df_cor["BsmtQual"]=le.transform(df_cor["BsmtQual"])



le.fit(df_cor["PoolQC"])

df_cor["PoolQC"]=le.transform(df_cor["PoolQC"])



le.fit(df_cor["MiscFeature"])

df_cor["MiscFeature"]=le.transform(df_cor["MiscFeature"])



le.fit(df_cor["SaleType"])

df_cor["SaleType"]=le.transform(df_cor["SaleType"])



le.fit(df_cor["SaleCondition"])

df_cor["SaleCondition"]=le.transform(df_cor["SaleCondition"])



df_cor[strong_corelation_category_list].head()
rows=4

cols=3

fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*3))

categorical_to_numeric_features=strong_corelation_category_list

for r in range(rows):

    for c in range(cols):

        m=r*cols+c

        if(m<len(categorical_to_numeric_features)):

            sns.regplot(df_cor[categorical_to_numeric_features[m]],label,ax=axs[r][c])

plt.tight_layout()

plt.show()
numerical_features_with_id=numerical_features_without_id

numerical_features_with_id.append("SalePrice")

df_cor[numerical_features_with_id].head()
abs(df_cor[numerical_features_with_id].corr()["SalePrice"])
df_cor[strong_corelation_category_list[:-1]].head()
abs(df_cor[strong_corelation_category_list].corr()["SalePrice"])
final_features_list=["OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","Fireplaces","GarageYrBlt","GarageCars","GarageArea","ExterQual","BsmtQual"]

final_numeric_features_list=["OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","Fireplaces","GarageYrBlt","GarageCars","GarageArea"]

final_categorical_features_list=["ExterQual","BsmtQual"]
df_train=pd.read_csv("/kaggle/input/train.csv")

df_test=pd.read_csv("/kaggle/input/test.csv")

traintest_df=df_train.append(df_test,ignore_index=False)
traintest_df.head()
print(len(df_train),len(traintest_df))
traintest_df=traintest_df.fillna(-9999,inplace=False)



print(len(list(set(df.columns)-set(final_features_list))))

print(traintest_df.columns)

traintest_df=traintest_df.drop(list(set(df.columns)-set(final_features_list)),axis=1)

traintest_label=traintest_df["SalePrice"]

traintest_df=traintest_df.drop(["SalePrice"],axis=1)

print(traintest_df.columns)

traintest_df=pd.get_dummies(traintest_df, columns=final_categorical_features_list, drop_first=True)



print(final_features_list)



traintest_df.head()
clf = LinearRegression()

clf.fit(traintest_df[:len(df)], traintest_label[:len(df)])

pred = clf.predict(traintest_df)
clf.score(traintest_df[:len(df)], traintest_label[:len(df)])

cross_val_score(clf,traintest_df[:len(df)],traintest_label[:len(df)],cv=5).mean()
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()

clf.fit(traintest_df[:len(df)], traintest_label[:len(df)])

pred = clf.predict(traintest_df)

clf.score(traintest_df[:len(df)], traintest_label[:len(df)])

cross_val_score(clf,traintest_df[:len(df)],traintest_label[:len(df)],cv=5).mean()
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor()

clf.fit(traintest_df[:len(df)], traintest_label[:len(df)])

pred = clf.predict(traintest_df)

clf.score(traintest_df[:len(df)], traintest_label[:len(df)])

cross_val_score(clf,traintest_df[:len(df)],traintest_label[:len(df)],cv=5).mean()
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=50, max_depth=8,random_state=0)

clf.fit(traintest_df[:len(df)], traintest_label[:len(df)])

pred = clf.predict(traintest_df)

clf.score(traintest_df[:len(df)], traintest_label[:len(df)])

cross_val_score(clf,traintest_df[:len(df)],traintest_label[:len(df)],cv=5).mean()
#Ada Boost classifier

from sklearn.ensemble import AdaBoostRegressor

clf=AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=18, max_depth=8),n_estimators=16)

clf.fit(traintest_df[:len(df)], traintest_label[:len(df)])

pred = clf.predict(traintest_df)

clf.score(traintest_df[:len(df)], traintest_label[:len(df)])

cross_val_score(clf,traintest_df[:len(df)],traintest_label[:len(df)],cv=5).mean()
df_test["Id"].head()
test_df=pd.read_csv("/kaggle/input/test.csv")

test_df.head()
test_dataframe=pd.DataFrame()

test_dataframe["Id"]=test_df["Id"]

test_dataframe["SalePrice"]=np.array(clf.predict(traintest_df[len(df):]))

test_dataframe.head()
#export to csv file

test_dataframe.to_csv("random_forest_using_adaboost.csv",index=False)