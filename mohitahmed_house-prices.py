# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_train.columns
df_test.columns
df_test.head()
df_train.info()
sns.distplot(df_train['SalePrice'], fit=norm);

plt.figure(figsize=(15,8))
sns.boxplot(df_train.YearBuilt, df_train.SalePrice)
print(df_train.shape)
print(df_test.shape)
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values
df = pd.concat((df_train, df_test)).reset_index(drop=True)
df.drop(['SalePrice'], axis=1, inplace=True)
total_test = df.isnull().sum().sort_values(ascending=False)
percent_test = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)
df.drop(['Utilities'], axis = 1) 
df.shape
#dropping columns with null values greater than 400
df = df.drop((missing_data[missing_data['Total'] > 400]).index,1)
df.shape
total_test = df.isnull().sum().sort_values(ascending=False)
percent_test = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)
#the null values are not actually missing values, they indicate that the house does not have that feature. For example if a house does not have a pool the data shows it as null.
#replacing null values with none and modal values of the column.
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df[col] = df[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df[col] = df[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df[col] = df[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[col] = df[col].fillna('None')

df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df["Functional"] = df["Functional"].fillna("Typ")
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['MSSubClass'] = df['MSSubClass'].fillna("None")

#checking if null values have been filled
total_test = df.isnull().sum().sort_values(ascending=False)
percent_test = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data.head(31)
#convert categorical variable into dummy
df = pd.get_dummies(df)

#bringing the old shape of test and train before concatenation
train = df[:ntrain]
test = df[ntrain:]
train.head(5)

#splitting
from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test = train_test_split(train,y_train,test_size = 0.2, random_state = 50)
#gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
G = GradientBoostingRegressor(n_estimators=3500, learning_rate=0.02,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=20, min_samples_split=15, 
                                   loss='huber', random_state =10)
G.fit(x_train,Y_train)
GBpred=G.predict(x_test)
import sklearn.metrics as metrics
print("R2 Gradient Boosting:", metrics.r2_score(Y_test,GBpred))
GBprediction = G.predict(test)
from sklearn.linear_model import  Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#L2 regularization
R = Ridge()
R.fit(x_train,Y_train)
RIDpred=R.predict(x_test)

print("R2 Ridge:", metrics.r2_score(Y_test,RIDpred))
RIDprediction = R.predict(test)
#L1 Regularization
L = Lasso()
L.fit(x_train,Y_train)
LApred=L.predict(x_test)
print("R2 Lasso:", metrics.r2_score(Y_test,LApred))
LAprediction = L.predict(test)
#Random Forest
R = RandomForestRegressor(n_estimators = 150)
R.fit(x_train,Y_train)
RFpred=R.predict(x_test)
print("R2 Random Forest:", metrics.r2_score(Y_test,RFpred))
RFprediction = R.predict(test)
#decision tree
DT = DecisionTreeRegressor(max_depth = 50)

DT.fit(x_train,Y_train)
DTpred=DT.predict(x_test)
print("R2 Decision Tree:", metrics.r2_score(Y_test,DTpred))
DTprediction = DT.predict(test)
#using the model with the highest R2 score for submission
submission = pd.DataFrame({'Id':test.Id,'SalePrice':GBprediction})
submission.head()
submission.to_csv('submission.csv', index=False)
submission = pd.DataFrame({'Id':test.Id,'SalePrice':DTprediction})
submission.head()
submission.to_csv('submissionDT.csv', index=False)
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values
df_pca = pd.concat((df_train, df_test)).reset_index(drop=True)
df_pca.drop(['SalePrice'], axis=1, inplace=True)
df_pca.info()
df_pca.drop(['Utilities'], axis = 1) 
total_test = df_pca.isnull().sum().sort_values(ascending=False)
percent_test = (df_pca.isnull().sum()/df_pca.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)
df_pca = df_pca.drop((missing_data[missing_data['Total'] > 400]).index,1)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_pca[col] = df_pca[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_pca[col] = df_pca[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df_pca[col] = df_pca[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_pca[col] = df_pca[col].fillna('None')

df_pca["MasVnrType"] = df_pca["MasVnrType"].fillna("None")
df_pca["MasVnrArea"] = df_pca["MasVnrArea"].fillna(0)
df_pca['MSZoning'] = df_pca['MSZoning'].fillna(df_pca['MSZoning'].mode()[0])
df_pca["Functional"] = df_pca["Functional"].fillna("Typ")
df_pca['Electrical'] = df_pca['Electrical'].fillna(df_pca['Electrical'].mode()[0])
df_pca['KitchenQual'] = df_pca['KitchenQual'].fillna(df_pca['KitchenQual'].mode()[0])
df_pca['Exterior1st'] = df_pca['Exterior1st'].fillna(df_pca['Exterior1st'].mode()[0])
df_pca['Exterior2nd'] = df_pca['Exterior2nd'].fillna(df_pca['Exterior2nd'].mode()[0])
df_pca['SaleType'] = df_pca['SaleType'].fillna(df_pca['SaleType'].mode()[0])
df_pca['MSSubClass'] = df_pca['MSSubClass'].fillna("None")
df_pca = pd.get_dummies(df_pca)

train_pca = df[:ntrain]
test_pca = df[ntrain:]
x_train,x_test,Y_train,Y_test = train_test_split(train,y_train,test_size = 0.2, random_state = 50)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)
scaled_data = scaler.transform(x_train)
scaled_data.shape
scaled_test_data = scaler.transform(x_test)
scaled_test_data.shape
from sklearn.decomposition import PCA
pca = PCA(n_components=7)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
x_pca.shape
per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
labels = ['PC'+str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=Y_train,cmap='rainbow')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
DTP = DecisionTreeRegressor(max_depth = 50)
DTP.fit(x_pca,Y_train)
x_test_pca = pca.transform(scaled_test_data)
#x_pca.shape
y_pred = DTP.predict(x_test_pca)

print("R2:", metrics.r2_score(Y_test,y_pred))