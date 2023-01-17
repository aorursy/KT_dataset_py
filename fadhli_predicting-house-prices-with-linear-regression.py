# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.info()
test_df.info()
y = train_df['SalePrice'].values
combine_df = pd.concat([train_df.drop(['SalePrice'], axis=1), test_df], axis=0)
# fill up MSZoning with the mode value

combine_df['MSZoning'] = combine_df['MSZoning'].fillna(combine_df['MSZoning'].mode()[0])
# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

combine_df["LotFrontage"] = combine_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# from the data description file, NA = No Alley Access

combine_df["Alley"] = combine_df["Alley"].fillna("None")
# fill up NA values with mode

combine_df['Utilities'] = combine_df['Utilities'].fillna(combine_df['Utilities'].mode()[0])
# since both Exterior1st and 2nd only has 2 missing value, substitute with mode

combine_df['Exterior1st'] = combine_df['Exterior1st'].fillna(combine_df['Exterior1st'].mode()[0])

combine_df['Exterior2nd'] = combine_df['Exterior2nd'].fillna(combine_df['Exterior2nd'].mode()[0])
# fill up MasVnrType with the mode value

combine_df["MasVnrType"] = combine_df["MasVnrType"].fillna(combine_df['MasVnrType'].mode()[0])

combine_df["MasVnrArea"] = combine_df["MasVnrArea"].fillna(combine_df['MasVnrArea'].mode()[0])
# for these columns, NA = No Basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    combine_df[col] = combine_df[col].fillna('None')
# for these columns, NA is likely to be 0 due to no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    combine_df[col] = combine_df[col].fillna(0)
# substitue NA value here with mode

combine_df['Electrical'] = combine_df['Electrical'].fillna(combine_df['Electrical'].mode()[0])
# substitute NA value with mode

combine_df['KitchenQual'] = combine_df['KitchenQual'].fillna(combine_df['KitchenQual'].mode()[0])
# if no value, assume Typ, typical is also mode value

combine_df['Functional'] = combine_df['Functional'].fillna(combine_df['Functional'].mode()[0])
# NA = No Fireplace

combine_df['FireplaceQu'] = combine_df['FireplaceQu'].fillna('None')
# for these columns, NA = No Garage

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    combine_df[col] = combine_df[col].fillna('None')
# as there is no garage, NA value for this column is set to zero

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    combine_df[col] = combine_df[col].fillna(0)
# NA = no pool

combine_df['PoolQC'] = combine_df['PoolQC'].fillna('None')
# NA = no fence

combine_df['Fence'] = combine_df['Fence'].fillna('None')
#Misc Feature, NA = None

combine_df['MiscFeature'] = combine_df['MiscFeature'].fillna('None')
#sale type, only have 1 NA value. substitute it with mode value

combine_df['SaleType'] = combine_df['SaleType'].fillna(combine_df['SaleType'].mode()[0])
# checking for any null value left

combine_df.isnull().sum().sum()
combine_df.info()
# looking at the documentation there are several column in numerical that should be considered

# as categorical. Let's straighten that out



# MSSubClass = The building class

combine_df['MSSubClass'] = combine_df['MSSubClass'].astype(str)





#Changing OverallCond into a categorical variable

combine_df['OverallCond'] = combine_df['OverallCond'].astype(str)

combine_df['OverallQual'] = combine_df['OverallQual'].astype(str)

combine_df = combine_df.drop(['Id'], axis=1)

combine_dummies = pd.get_dummies(combine_df)



# from sklearn.preprocessing import LabelEncoder

# combine_df = combine_df.apply(LabelEncoder().fit_transform)
result = combine_dummies.values
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

result = scaler.fit_transform(result)
#creating matrices for sklearn:

X = result[:train_df.shape[0]]

test_values = result[train_df.shape[0]:]
# import train test split

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso





X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



#clf = LinearRegression()

clf = Lasso()



clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

y_train_pred = clf.predict(X_train)
from sklearn.metrics import r2_score



print("Train acc: " , r2_score(y_train, y_train_pred))

print("Test acc: ", r2_score(y_test, y_pred))
from sklearn.metrics import mean_squared_error



print("Train acc: " , clf.score(X_train, y_train))

print("Test acc: ", clf.score(X_test, y_test))
final_labels = clf.predict(test_values)
final_labels
final_result = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': final_labels})
#final_result.to_csv('house_price.csv', index=False)