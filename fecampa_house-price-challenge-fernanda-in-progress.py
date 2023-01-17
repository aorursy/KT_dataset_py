import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))
df_test = pd.read_csv('../input/test.csv')
df_train = pd.read_csv('../input/train.csv')
df_train.describe()
plt.figure(figsize=(15,8))
sns.boxplot(df_train.YearBuilt, df_train.SalePrice)
sns.distplot(df_train['SalePrice']);
plt.figure(figsize=(12,6))
plt.scatter(x=df_train.GrLivArea, y=df_train.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
f, ax = plt.subplots(figsize=(11, 9))
corr = df_train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                 square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr.sort_values(by=['SalePrice'], ascending=False).SalePrice.head(12)
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']   
X = df_train[features]
y = df_train['SalePrice']
xtrain, xtest, ytrain, ytest = train_test_split(X, 
                                                y, 
                                                test_size = 0.3, 
                                                random_state = 42)
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
regressor.score(xtest, ytest)
df_test_fil = df_test[features].fillna(method='pad')
df_test_fil.head()
pred = regressor.predict(df_test_fil)
df_result = pd.DataFrame({'Id':df_test.Id, 'SalePrice':pred})

df_result.head()
df_result.to_csv('submission.csv',index=False)
