import numpy as np 

import pandas as pd 

from IPython.display import Image

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns

import scipy.stats as stats

from sklearn.preprocessing import StandardScaler

import os

Image("../input/outputpng/housesbanner.png")
datatrain = pd.read_csv('../input/houses-prices/train.csv')

datatest = pd.read_csv('../input/houses-prices/test.csv')
datatrain.head()

datatest.head()
prices=datatrain["SalePrice"]
data=pd.concat([datatrain,datatest], sort=False)
data = data.drop("SalePrice", 1)
prices.describe()
plt.hist(prices, bins=30)

plt.title("Price distributions")
plt.figure(1); plt.title('Johnson SU')

sns.distplot(datatrain['SalePrice'], kde=False, fit=stats.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(datatrain['SalePrice'], kde=False, fit=stats.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(datatrain['SalePrice'], kde=False, fit=stats.lognorm)
display(data.describe().transpose())
sns.set_style("whitegrid")

missing = data.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
# I delete same variables that contain NA 

data=data.drop("Id", 1)

data=data.drop("Alley", 1)

data=data.drop("Fence", 1)

data=data.drop("MiscFeature", 1)

data=data.drop("PoolQC", 1)

#data=data.drop("FireplaceQu", 1)
#FireplaceQu

data[['Fireplaces','FireplaceQu']].head(10)

data['FireplaceQu'].isnull().sum()
data['Fireplaces'].value_counts()
data['FireplaceQu']=data['FireplaceQu'].fillna('NO')

data['FireplaceQu'].unique()
#LotFrontage

data["LotFrontage"] = data["LotFrontage"].fillna(value=data['LotFrontage'].mean())
#Garage

data['GarageYrBlt'].isnull().sum()
data['GarageType'].isnull().sum()
data['GarageFinish'].isnull().sum()
data['GarageQual'].isnull().sum()
data['GarageCond'].isnull().sum()
data["GarageArea"].value_counts()
data['GarageType']=data['GarageType'].fillna('NO')

data['GarageCond']=data['GarageCond'].fillna('NO')

data['GarageFinish']=data['GarageFinish'].fillna('NO')

data['GarageYrBlt']=data['GarageYrBlt'].fillna('NO')

data['GarageQual']=data['GarageQual'].fillna('NO')
#Bsmt

data.BsmtFinType2.isnull().sum()
data.BsmtExposure.isnull().sum()
data.BsmtFinType1.isnull().sum()
data.BsmtCond.isnull().sum() 
data.BsmtQual.isnull().sum()
data.TotalBsmtSF.value_counts().head()
data['BsmtFinType2']=data['BsmtFinType2'].fillna('NO')

data['BsmtExposure']=data['BsmtExposure'].fillna('NO')

data['BsmtFinType1']=data['BsmtFinType1'].fillna('NO')

data['BsmtCond']=data['BsmtCond'].fillna('NO')

data['BsmtQual']=data['BsmtQual'].fillna('NO')
#Masvmr

data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean())
data['MasVnrType'].unique()
data['MasVnrType']=data['MasVnrType'].fillna('None')
#Electrical

data['Electrical']=data['Electrical'].fillna('Mix')
datatrain=datatrain.drop("Id", 1)

datatrain=datatrain.drop("Alley", 1)

datatrain=datatrain.drop("Fence", 1)

datatrain=datatrain.drop("MiscFeature", 1)

datatrain=datatrain.drop("PoolQC", 1)

#data=data.drop("FireplaceQu", 1)
quantitative = datatrain._get_numeric_data()
quantitative = datatrain._get_numeric_data()
sns.boxplot([data.MSSubClass])
sns.boxplot([data.LotFrontage])
data['LotFrontage']= data['LotFrontage'].clip(data['LotFrontage'].quantile(0.01),data['LotFrontage'].quantile(0.99))
sns.boxplot(data.LotFrontage)
sns.boxplot([data.LotArea])
data['LotArea']= data['LotArea'].clip_upper(data['LotArea'].quantile(0.99)) 
sns.boxplot([data.LotArea])
sns.boxplot([data.MasVnrArea])
data['MasVnrArea']= data['MasVnrArea'].clip_upper(data['MasVnrArea'].quantile(0.99)) 
sns.boxplot([data.MasVnrArea])
sns.boxplot([data.BsmtFinSF1])
data['BsmtFinSF1']= data['BsmtFinSF1'].clip_upper(data['BsmtFinSF1'].quantile(0.99))
sns.boxplot([data.BsmtFinSF1])
sns.boxplot([data.BsmtFinSF2])
data['BsmtFinSF2']= data['BsmtFinSF2'].clip_upper(data['BsmtFinSF2'].quantile(0.99))
sns.boxplot([data.BsmtFinSF2])
sns.boxplot([data.BsmtUnfSF])
data['BsmtUnfSF']= data['BsmtUnfSF'].clip_upper(data['BsmtUnfSF'].quantile(0.99))
sns.boxplot([data.BsmtUnfSF])
sns.boxplot([data.TotalBsmtSF])
data['TotalBsmtSF']= data['TotalBsmtSF'].clip_upper(data['TotalBsmtSF'].quantile(0.99))
sns.boxplot([data.TotalBsmtSF])
sns.boxplot(data["1stFlrSF"])
data['1stFlrSF']= data['1stFlrSF'].clip_upper(data['1stFlrSF'].quantile(0.99))
sns.boxplot(data["1stFlrSF"])
sns.boxplot(data["2ndFlrSF"])
data['2ndFlrSF']= data['2ndFlrSF'].clip_upper(data['2ndFlrSF'].quantile(0.99))
sns.boxplot(data["2ndFlrSF"])
sns.boxplot(data.LowQualFinSF)
sns.boxplot(data.GrLivArea)
data['GrLivArea']= data['GrLivArea'].clip_upper(data['GrLivArea'].quantile(0.99))
sns.boxplot(data.GrLivArea)
sns.boxplot(data.BsmtFullBath)
sns.boxplot(data.BsmtHalfBath)
sns.boxplot(data.BedroomAbvGr)
data['BedroomAbvGr']= data['BedroomAbvGr'].clip_upper(data['BedroomAbvGr'].quantile(0.99))
sns.boxplot(data.BedroomAbvGr)
sns.boxplot(data.KitchenAbvGr)
sns.boxplot(data.TotRmsAbvGrd)
sns.boxplot(data.Fireplaces)
sns.boxplot(data.GarageCars)
sns.boxplot(data.GarageArea)
data['GarageArea']= data['GarageArea'].clip_upper(data['GarageArea'].quantile(0.99))
sns.boxplot(data.GarageArea)
sns.boxplot(data.WoodDeckSF)
data['WoodDeckSF']= data['WoodDeckSF'].clip_upper(data['WoodDeckSF'].quantile(0.99))
sns.boxplot(data.WoodDeckSF)
sns.boxplot(data.OpenPorchSF)
data['OpenPorchSF']= data['OpenPorchSF'].clip_upper(data['OpenPorchSF'].quantile(0.99))
sns.boxplot(data.OpenPorchSF)
sns.boxplot(data.EnclosedPorch)
data['EnclosedPorch']= data['EnclosedPorch'].clip_upper(data['EnclosedPorch'].quantile(0.99))
sns.boxplot(data.EnclosedPorch)
sns.boxplot(data['3SsnPorch'])
data['3SsnPorch']= data['3SsnPorch'].clip_upper(data['3SsnPorch'].quantile(0.99))
sns.boxplot(data['3SsnPorch'])
sns.boxplot(data['ScreenPorch'])
data['ScreenPorch']= data['ScreenPorch'].clip_upper(data['ScreenPorch'].quantile(0.99))
sns.boxplot(data['ScreenPorch'])
sns.boxplot(data['PoolArea'])
data['PoolArea']= data['PoolArea'].clip_upper(data['PoolArea'].quantile(0.99))
sns.boxplot(data['PoolArea'])
sns.boxplot(data['MiscVal'])
data['MiscVal']= data['MiscVal'].clip_upper(data['MiscVal'].quantile(0.99))
sns.boxplot(data['MiscVal'])
sns.boxplot(data['MoSold'])
sns.boxplot(data['YrSold'])
sns.boxplot(prices)
prices=prices.clip(prices.quantile(0.01),prices.quantile(0.99))
sns.boxplot(prices)
num_corr=quantitative.corr()

plt.subplots(figsize=(30,15))

sns.heatmap(num_corr,annot = True,robust=True)

plt.text(15,0, "Heat Map", fontsize = 70, color='Black', fontstyle='italic')
data = pd.get_dummies(data)

data.head()
data_ = np.log1p(data)# convert distributed data to log 1p 

prices = np.log1p(prices)# convert distributed data to log 1p
from sklearn.impute import SimpleImputer # cualitative missing values (cualitative and cuantitative)

imp = SimpleImputer(missing_values = np.nan, strategy='most_frequent')

data = imp.fit_transform(data)
scaler = StandardScaler()

scaler.fit(data)                

p_data = scaler.transform(data)

p_data
pca = PCA(whiten=True)

pca.fit(p_data)
np.exp(pca.explained_variance_ratio_)
pca = PCA(n_components=45,whiten=True)

pca = pca.fit(p_data)

dataPCA = pca.transform(p_data)

dataPCA
dataPCA.shape
dataPCA1 = dataPCA[:1460,:]#row 

dataPCA1.shape

datatest1 = dataPCA[1460:,:]#row

datatest1.shape
linear = LinearRegression()

linear.fit(dataPCA1,prices)
X_train , X_test, Y_train, Y_test = train_test_split(dataPCA1,prices,test_size=0.20,random_state=2019)
y_pred = linear.predict(X_test)
r2_score(Y_test, y_pred)
np.expm1(linear.intercept_)
np.expm1(linear.coef_)
np.expm1(Y_test)

np.expm1(y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

rmse
y_predtestint = linear.predict(datatest1)

np.exp(y_predtestint)
prices1=prices[:1459]

prices1.shape
%matplotlib inline

plt.plot(np.expm1(prices1),np.expm1(y_predtestint), "ro")

plt.title("Valor Actual vs Predicción")
plt.hist( np.expm1(prices), bins=30)

plt.title("Distribución de precios")
plt.hist( np.exp(y_predtestint), bins=50)

plt.title("Distribución de predición de precios")
Pricespred= np.exp(y_predtestint)
output= {"Id": datatest.iloc[:,0], "SalePrice": Pricespred}

output
submission = pd.DataFrame(output)

submission
submission.set_index( "Id",inplace=True) # Delete index
pd.DataFrame.to_csv(submission,index=False,sep=",")
filename = "submission.csv"
submission.head()