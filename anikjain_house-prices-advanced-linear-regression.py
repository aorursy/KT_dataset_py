import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
print("shape of train :", train.shape)
print("shape of test :", test.shape)
df = pd.concat((train,test))
temp_df = df
("shape of df:", df.shape)
pd.options.display.max_columns = 2000
pd.options.display.max_rows = 85
df.info()
df.describe()
df.isnull().sum()
df.notnull().sum()
df.select_dtypes(include ='float64')
df.select_dtypes(include ='int64') 
df.select_dtypes(include ='float64').columns
plt.figure(figsize = (16,10))
sns.heatmap(df.isnull())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df["BsmtExposure"].fillna( method ='ffill', inplace = True)
df["BsmtFinSF1"].fillna( method ='ffill', inplace = True)
df["BsmtFinSF2"].fillna( method ='ffill', inplace = True)
df["BsmtFinType1"].fillna( method ='ffill', inplace = True)
df["BsmtFinType2"].fillna( method ='ffill', inplace = True)
df["BsmtFullBath"].fillna( method ='ffill', inplace = True)
df["BsmtHalfBath"].fillna( method ='ffill', inplace = True)
df["Functional"].fillna( method ='ffill', inplace = True)
df["FireplaceQu"].fillna( method ='ffill', inplace = True)
df["Fence"].fillna( method ='ffill', inplace = True)
df["Exterior2nd"].fillna( method ='ffill', inplace = True)
df["Exterior1st"].fillna( method ='ffill', inplace = True)
df["Electrical"].fillna( method ='ffill', inplace = True)
df["BsmtUnfSF"].fillna( method ='ffill', inplace = True)
df["BsmtQual"].fillna( method ='ffill', inplace = True)
df["GarageArea"].fillna( method ='ffill', inplace = True)
df["GarageCars"].fillna( method ='ffill', inplace = True)
df["GarageCond"].fillna( method ='ffill', inplace = True)
df["GarageFinish"].fillna( method ='ffill', inplace = True)
df["GarageQual"].fillna( method ='ffill', inplace = True)
df["GarageType"].fillna( method ='ffill', inplace = True)
df["GarageYrBlt"].fillna( method ='ffill', inplace = True)
df["LotFrontage"].fillna( method ='ffill', inplace = True)
df["MSZoning"].fillna( method ='ffill', inplace = True)
df["MasVnrArea"].fillna( method ='ffill', inplace = True)
df["MasVnrType"].fillna( method ='ffill', inplace = True)
df["MiscFeature"].fillna( method ='ffill', inplace = True)
df["PoolQC"].fillna( method ='ffill', inplace = True)
df["SalePrice"].fillna( method ='ffill', inplace = True)
df["TotalBsmtSF"].fillna( method ='ffill', inplace = True)
df["Utilities"].fillna( method ='ffill', inplace = True)
df["Alley"].interpolate(method ='linear', limit_direction ='forward')
df.drop(["Alley"], axis = 1, inplace = True)
df
df.head(10)
df.isnull().sum()
df["BsmtCond"].fillna( method ='ffill', inplace = True)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF']
sns.pairplot(df[cols], size = 4.5)
plt.show()
x = df.iloc[:,33:37].values #GarageFinish	GarageQual	GarageType	GarageYrBlt
x
df.head()
y = df.iloc[:,-11]#SalePrice
y
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,2] = labelencoder_x.fit_transform(x[:,2])
x
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x= onehotencoder.fit_transform(x).toarray()
x
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=100, test_size=0.35)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit (x_test,y_test)
y_pred = reg.predict(x_test)
y_pred
y_test
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import accuracy_score
accuracy = accuracy_score (y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification = classification_report(y_test,y_pred)
print(classification)