# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.metrics import r2_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.info()
train.describe()
# Object olan verilerinde istatistiksel özetini alalım

# Let's take the statistical summary of "Object" data



train.describe(include=['O'])
train.Utilities.value_counts()
sns.distplot(train.SalePrice)
train.plot(kind = 'scatter',x="Id", y="SalePrice", color = 'r',label = 'Price',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df_inlier = train[train['SalePrice'] < 400000]

df_inlier.shape
train2 = df_inlier.drop(labels = ["Id", "SalePrice"], axis = 1)

train_sale = df_inlier.loc[:, "SalePrice"]
test_id = test.Id

test = test.drop(labels = ["Id"], axis = 1)
# np.concatenate([a, b])



df = pd.concat([train2, test])
df.shape
df.isnull().sum().sort_values(ascending=False).head(15)
df.dropna(axis=1, how="any", thresh=2480, inplace = True)

df.shape
df.drop(labels = ["2ndFlrSF", "MiscVal", "WoodDeckSF", "OpenPorchSF","EnclosedPorch","3SsnPorch",

                  "PoolArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","LowQualFinSF",

                  "ScreenPorch"], axis = 1, inplace = True)
df.shape
df.info()
missingValue = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

missingValue = missingValue.fit(df.iloc[:, 0:61])

df.iloc[:, 0:61] = missingValue.transform(df.iloc[:, 0:61])
missingValue = SimpleImputer(missing_values = 0, strategy = 'mean')

missingValue = missingValue.fit(df.iloc[:,31:33])

df.iloc[:, 31:33] = missingValue.transform(df.iloc[:, 31:33])
missingValue = SimpleImputer(missing_values = 0, strategy = 'mean')

missingValue = missingValue.fit(df.iloc[:,53:54])

df.iloc[:, 53:54] = missingValue.transform(df.iloc[:, 53:54])
df.isna().sum().sum()
df.loc[:,["MSSubClass","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFullBath",

             "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",

             "Fireplaces", "GarageCars", "GarageYrBlt", "MoSold",

             "YrSold"]] = df.loc[:,["MSSubClass", "OverallQual", "OverallCond", "YearBuilt",

                                       "YearRemodAdd", "BsmtFullBath", "BsmtHalfBath", "FullBath",

                                       "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",

                                       "Fireplaces", "GarageCars", "GarageYrBlt", "MoSold",

                                       "YrSold"]].astype("object")
df_obj = df.describe(include=["O"])

df_obj = df_obj.columns

df = pd.get_dummies(df, columns = df_obj, drop_first = True)



df.shape
x_train = df[:1432].values

x_test = df[1432:].values

y_train = train_sale.values
linear = LinearRegression()

linear.fit(x_train, y_train)
y_pred_train = linear.predict(x_train)
print("Root mean square error train = " + str(np.sqrt(mean_squared_error(y_train, y_pred_train))))

print("R2 score = " + str(r2_score(y_train, y_pred_train)))
y_test = linear.predict(x_test)
submission = pd.DataFrame({"Id": test_id})

submission["SalePrice"] = y_test

submission.to_csv("submission.csv", index=False)