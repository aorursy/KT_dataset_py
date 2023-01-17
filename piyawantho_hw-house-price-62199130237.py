# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')
print(dataset.shape)

print(dataset.info())
dataset['Method'].value_counts(dropna=False)

len(dataset['Method'].value_counts(dropna=False))
dataset.describe(include='all')
df = dataset.replace(to_replace=[r'^\s*$', r'[?]', r'\'\s*\'', 'N/A', 'None'],value=np.nan, regex=True)
# df_dataset.isnull().any()
# df_dataset_drop_null = df_dataset.dropna()

# df_dataset_drop_null.shape
df = df[['Type','Method','Distance','Landsize','BuildingArea','YearBuilt','Rooms','Car','Bathroom','Propertycount','Price']]

df.isnull().any()



df = df.dropna()

# df['Rooms'] = df['Rooms'].astype(float)

# df = df['Type'].fillna(df['Type'].median(skipna=True), inplace=True)

# df = df['Method'].fillna(df['Method'].median(skipna=True), inplace=True)

# df = df['Distance'].fillna(df['Distance'].median(skipna=True), inplace=True)

# df = df['Landsize'].fillna(df['Landsize'].median(skipna=True), inplace=True)

# df = df['BuildingArea'].fillna(df['BuildingArea'].median(skipna=True), inplace=True)

# df = df['YearBuilt'].fillna(df['YearBuilt'].median(skipna=True), inplace=True)

# df = df['Rooms'].fillna(df['Rooms'].median(skipna=True), inplace=True)

# df = df['Price'].fillna(df['Price'].median(skipna=True), inplace=True)

df['Price']
df['Price'].max()
df['Price'].min()
# df['Distance'].fillna(df['Distance'].mean(skipna=True), inplace=True)

# df['Landsize'].fillna(df['Landsize'].mean(skipna=True), inplace=True)

# df['BuildingArea'].fillna(df['BuildingArea'].mean(skipna=True), inplace=True)

# df['YearBuilt'].fillna(df['YearBuilt'].mean(skipna=True), inplace=True)

# df['Car'].fillna(df['Car'].mean(skipna=True), inplace=True)

# df['Bathroom'].fillna(df['Bathroom'].mean(skipna=True), inplace=True)

# df['Propertycount'].fillna(df['Propertycount'].mean(skipna=True), inplace=True)

# df['Price'].fillna(df['YearBuilt'].mean(skipna=True), inplace=True)



df.isnull().any()
X = df.iloc[:, :-1].values

y = df.iloc[:, 10].values

# y[10]

#print(X[0,:])

#X[:,0].value_counts()
# # Try running this code, you'll get a warning message. 



# # Encoding categorical data

# # Encoding the Independent Variable

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# labelencoder_X = LabelEncoder()

# X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #Change text into number

# onehotencoder = OneHotEncoder(categorical_features = [0])

# X = onehotencoder.fit_transform(X).toarray()

# print(X.shape)

# print(X[0,:])
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



ct = ColumnTransformer(

    [('one_hot_encoder', OneHotEncoder(), [0,1])],    # The column numbers to be transformed (here is [3] but can be [0, 1, 3])

    remainder='passthrough'                         # Leave the rest of the columns untouched

)



X = np.array(ct.fit_transform(X), dtype=np.float)





##print##

print(X)

X = X[:, 1:] 

print(X)

print(X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1000)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

import statsmodels.api as sm

X = np.append(arr=np.ones((8895,1)).astype(int), values = X, axis=1)

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14]]

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()
# X_opt = X[:,:] # Initialize the metrix

# regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

# regressor_OLS.summary()
import seaborn as sns

g = sns.jointplot(x="Price", y="YearBuilt", height=5,data=df, kind="reg",ratio=3, color="salmon")
sns.pairplot(df, kind="reg")

plt.show()
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(df.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)

plt.show()