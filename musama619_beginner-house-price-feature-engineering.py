import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
pd.pandas.set_option('display.max_columns', None)

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()
data.shape
data.isnull().sum()
data = data.drop(['Id','Alley','FireplaceQu', 'PoolQC', 'Fence','MiscFeature' ], axis=1)
missNum = [f for f in data if data[f].isnull().sum()>0 and data[f].dtype!='O']
missNum_mean = data[missNum].mean()
missNum_mean
data[missNum] = data[missNum].fillna(missNum_mean)
data
missCat = [f for f in data if data[f].isnull().sum()>0 and data[f].dtype == 'O' ]
missCat
missCat_mode = data[missCat].mode().sum()
missCat_mode
data[missCat] = data[missCat].fillna(missCat_mode)
data[missCat].isnull().sum()
plt.figure(figsize=(10,8))
sns.heatmap(data.isnull())
numerical_r = [f for f in data if data[f].dtype !='O']
for f in numerical_r:
    dataNr = data.copy()
    plt.scatter(dataNr[f], data['SalePrice'])
    plt.xlabel(f)
    plt.show()
    
data.describe()
categorical =[f for f in data if data[f].dtype == 'O']
data[categorical].shape
for f in categorical:
    dataCat = data.copy()
    dataCat.groupby(f)['SalePrice'].mean().plot.bar()
    plt.show()
sale_price = data['SalePrice']
data = data.drop(['SalePrice'], axis=1)
numerical = [f for f in data if data[f].dtype !='O']
numerical
year = [f for f in numerical if 'Year' in f or 'Yr' in f]
data[year].head()
data = data.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'], axis=1)
numerical = [f for f in data if data[f].dtype !='O']
for f in numerical:
    dataC = data.copy()
    data[f].hist()
    plt.xlabel(f)
    plt.show()
dataT = np.log(data[numerical]+1)
dataT.describe()
for f in numerical:
    dataC = dataT.copy()
    dataC[f].hist()
    plt.xlabel(f)
    plt.show()
maxTh = dataT.quantile(0.95)
maxTh[1:32]
minTh = dataT.quantile(0.05)
minTh[1:32]
df2  = dataT[(dataT<maxTh) & (dataT>minTh)]
df2
df2.isnull().sum()
df3 = df2.drop(['BsmtFinSF2', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'Fireplaces',
               'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
               'MiscVal'], axis=1)
df3.isnull().sum()
df3 = df3.drop(['MasVnrArea', 'BedroomAbvGr'], axis=1)
numMissN = [f for f in df3 if df3[f].isnull().sum()>1]
numMissN = df3[numMissN].mean()
numMissN
df4 = df3.fillna(numMissN)
df4.isnull().sum()
df4
for f in df4:
    dataL = df4.copy()
    dataL[f].hist()
    plt.show()
df_cat = data[categorical]
df6 = pd.concat([df_cat, df4], axis=1)
df6.shape
df6
cat_features = [f for f in df6 if df6[f].dtype == 'O']
cat_features
from sklearn.preprocessing import LabelEncoder
df7= df6[cat_features].apply(LabelEncoder().fit_transform)
df7
df8 = pd.concat([df7, df4], axis=1)
df8
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
#independent features

X = df8
X
#dependent feature

y = sale_price
y
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
xtrain.shape
ytrain.shape
model1 = LinearRegression()
model1.fit(xtrain, ytrain)
print("Train Accuracy:",model1.score(xtrain, ytrain))
print("Test Accuracy:",model1.score(xtest, ytest))
model1.predict([[3, 1,3,3,0,4,0,5,2,2,0,5,1,1,12,13,1,2,4,2,2,3,3,2,5,1,0,1,4,2,6,1,1,4,4,2,8,4,4.110874,4.189655,9.042040,2.079442,1.791759,6.561031,5.017280,6.753438,6.753438,7.444833,2.197225,1.098612,6.308098,1.934685]])
model2 = RandomForestRegressor(n_estimators=250)
model2.fit(xtrain, ytrain)
print("Train Accuracy:",model2.score(xtrain, ytrain))
print("Test Accuracy:",model2.score(xtest, ytest))
model2.predict([[3, 1,3,3,0,4,0,5,2,2,0,5,1,1,12,13,1,2,4,2,2,3,3,2,5,1,0,1,4,2,6,1,1,4,4,2,8,4,4.110874,4.189655,9.042040,2.079442,1.791759,6.561031,5.017280,6.753438,6.753438,7.444833,2.197225,1.098612,6.308098,1.934685]])
model3 = Ridge(max_iter=100)
model3.fit(xtrain, ytrain)
print("Train Accuracy:",model3.score(xtrain, ytrain))
print("Test Accuracy:",model3.score(xtest, ytest))
model3.predict([[3, 1,3,3,0,4,0,5,2,2,0,5,1,1,12,13,1,2,4,2,2,3,3,2,5,1,0,1,4,2,6,1,1,4,4,2,8,4,4.110874,4.189655,9.042040,2.079442,1.791759,6.561031,5.017280,6.753438,6.753438,7.444833,2.197225,1.098612,6.308098,1.934685]])
plt.figure(figsize=(12,8))
sns.heatmap(df8.corr())