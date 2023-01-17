import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df = df.drop(columns=['MoSold','YrSold'])

df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
def FillNA(df):

    df['Alley']=df['Alley'].fillna('NoAccess')

    df[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]=df[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].fillna('NoBa')

    df['MiscFeature']=df['MiscFeature'].fillna('NoMF')

    df['Fence']=df['Fence'].fillna('NoF')

    df['FireplaceQu']=df['FireplaceQu'].fillna('NoFire')

    df[['GarageType','GarageFinish','GarageQual','GarageCond']]=df[['GarageType','GarageFinish','GarageQual','GarageCond']].fillna('NoG')

    df['PoolQC']=df['PoolQC'].fillna('NoP')

    df['MasVnrType']=df['MasVnrType'].fillna('NoM')

    df['Electrical']=df['Electrical'].fillna('NoKnow')

    return(df)

df=FillNA(df)

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=(100))

df[['LotFrontage']]= imputer.fit_transform(df[['LotFrontage']])

df[['GarageYrBlt']]= imputer.fit_transform(df[['GarageYrBlt']])

df[['MasVnrArea']]= imputer.fit_transform(df[['MasVnrArea']])
df['BsmtBath']=df['BsmtFullBath']+df['BsmtHalfBath']

df['FBath']=df['FullBath']+df['HalfBath']

df['TotalBath']=df['BsmtBath']+df['FBath']
df_numbers=df.select_dtypes(include=['int64','float64'])

df_string=df.select_dtypes(include='object')

print("String Shape ",df_string.shape," Numbers Shape",df_numbers.shape)
from sklearn.preprocessing import OneHotEncoder



x=df[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']]



x_name=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']

Encoder = OneHotEncoder(handle_unknown='error',sparse=False)

OH_Encoder=Encoder.fit(x)

OH_Encoded=OH_Encoder.transform(x)

columns_names=OH_Encoder.get_feature_names(x_name)

OH_Encoded=pd.DataFrame(OH_Encoded,columns=columns_names)
df
df_numbers.isna().sum().sum()
OH_Encoded['Id']=OH_Encoded.index+1

OH_Encoded.head()

grouped=df.groupby('MSZoning')

ax=grouped['SalePrice'].agg(np.mean).plot(kind='bar',figsize=(6,6),color='#eff3c6')

ax.set_facecolor('#588da8')
grouped=df.groupby('Condition1')

ax=grouped['SalePrice'].agg(np.mean).plot(kind='bar',figsize=(6,6),color='#eff3c6')

ax.set_facecolor('#588da8')
grouped=df.groupby('Condition2')

ax=grouped['SalePrice'].agg(np.mean).plot(kind='bar',figsize=(6,6),color='#eff3c6')

ax.set_facecolor('#588da8')
grouped=df.groupby('HouseStyle')

ax=grouped['SalePrice'].agg(np.mean).plot(kind='bar',figsize=(6,6),color='#eff3c6')

ax.set_facecolor('#588da8')
grouped=df.groupby('RoofStyle')

ax=grouped['SalePrice'].agg(np.mean).plot(kind='bar',figsize=(6,6),color='#eff3c6')

ax.set_facecolor('#588da8')
grouped=df.groupby('Exterior1st')

ax=grouped['SalePrice'].agg(np.mean).plot(kind='bar',figsize=(6,6),color='#eff3c6')

ax.set_facecolor('#588da8')
grouped=df.groupby('Exterior2nd')

ax=grouped['SalePrice'].agg(np.mean).plot(kind='bar',figsize=(6,6),color='#eff3c6')

ax.set_facecolor('#588da8')
df['logSalePrice']=df['SalePrice'].apply(np.log)
df_numbers.columns
df_numbers.isna().sum()
df['FlrSf']=df['1stFlrSF']+df['2ndFlrSF']

df['BsmtSf1']=df['BsmtFinSF1']+df['BsmtFinSF2']

df['BsmtSf2']=df['BsmtFinSF1']+df['BsmtFinSF2']+df['BsmtUnfSF']

df['Porch']=df['OpenPorchSF']+df['ScreenPorch']

df['SfCars']=df['GarageArea']/df['GarageCars']

df['SfCars2']=df['GarageCars']/df['GarageArea']
corr=df.corr()

cor_target = abs(corr["logSalePrice"])

#Selecting highly correlated features

plt.rcParams['figure.figsize'] = (30.0,20.0)

plt.rcParams['font.family'] = "serif"

sns.heatmap(data=corr,annot=True,cmap='coolwarm')

plt.title('Correlation Matrix', fontsize=18)
relevant_features = cor_target[cor_target>=0.5]

relevant_features 
rel_ft=list(relevant_features.index)

rel_ft.append('Id')
selected = df[rel_ft]

data=OH_Encoded.merge(selected)

data=data.drop(columns=['Id','SalePrice'])

horizon = pd.concat([selected, OH_Encoded], axis=1)
def my_linear_regression(df, variable_a_predire, test_size=0.2):

    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import r2_score



    X = df.drop(columns=[variable_a_predire])

    y = df[variable_a_predire]

    p = X.shape[1] #ordre du modèle

    

    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=test_size,random_state=42)

    Ntrain = Xtrain.shape[0] # taille de l'ensemble d'apprentissage

   

    model = LinearRegression()

    model = model.fit(Xtrain,ytrain)

    ytest_predict = model.predict(Xtest)

    

    MSE = np.mean((ytest-ytest_predict)**2)

    RSS=sum((ytest-ytest_predict)**2)  

    RSE = np.sqrt(RSS)/(Ntrain-p-1)

    R2app = r2_score(ytrain,model.predict(Xtrain))

    R2test = r2_score(ytest,model.predict(Xtest))

    model.fit(X, y) # fit du modèle retourné sur l'ensemble des données



    return model, RSS, RSE,MSE, R2app, R2test
selected=selected.drop(columns=['Id','SalePrice'])

model,RSS,RSE,MSE,R2App,R2test = my_linear_regression(selected,'logSalePrice')

print("RSS : ",RSS, "RSE :",RSE)
MSE
selected.head()
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_data['FlrSf']=test_data['1stFlrSF']+test_data['2ndFlrSF']

test_data['BsmtSf1']=test_data['BsmtFinSF1']+test_data['BsmtFinSF2']

test_data['BsmtSf2']=test_data['BsmtFinSF1']+test_data['BsmtFinSF2']+test_data['BsmtUnfSF']

test_data['Porch']=test_data['OpenPorchSF']+test_data['ScreenPorch']

test_data['SfCars']=test_data['GarageArea']/test_data['GarageCars']

test_data['SfCars2']=test_data['GarageCars']/test_data['GarageArea']

test_data['BsmtBath']=test_data['BsmtFullBath']+test_data['BsmtHalfBath']

test_data['FBath']=test_data['FullBath']+test_data['HalfBath']

test_data['TotalBath']=test_data['BsmtBath']+test_data['FBath']
liste = ['OverallQual',

 'YearBuilt',

 'YearRemodAdd',

 'TotalBsmtSF',

 '1stFlrSF',

 'GrLivArea',

 'FullBath',

 'TotRmsAbvGrd',

 'GarageYrBlt',

 'GarageCars',

 'GarageArea',

 'FBath',

 'TotalBath',

 'FlrSf',

 'BsmtSf2']
test=test_data[liste]

columns_list=test.columns[test.isna().any()]
imputer = KNNImputer(n_neighbors=(100))

for col in columns_list:

    test[[col]]=imputer.fit_transform(test[[col]])
test.shape
sp_predict=np.expm1(model.predict(test))

output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': sp_predict})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")