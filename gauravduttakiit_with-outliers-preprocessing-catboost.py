import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
house=pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

house.head()
house.shape
house.info()
# Missing values are there
house.describe()
house.Id.dtype
house.Id.nunique()
house.Id.isnull().sum()
# They are all unique 
house.MSSubClass.dtype
house.MSSubClass.isnull().sum()
house.MSSubClass.value_counts(ascending=False)
house['MSSubClass']=house['MSSubClass'].astype('str').astype('object')
house.MSSubClass.dtype
house.MSZoning.dtype
house.MSZoning.isnull().sum()
house.MSZoning.value_counts(ascending=False)
house.LotFrontage.dtype
house.LotFrontage.isnull().sum()
# Missing value treatment will be needed 
house.LotFrontage.describe()
house.LotArea.dtype
house.LotArea.isnull().sum()
house.LotArea.describe()
house.Street.dtype
house.Street.isnull().sum()
house.Street.value_counts(ascending=False)
house.Alley.dtype
house.Alley.value_counts(ascending=False)
house.Alley.isnull().sum(),house.Alley.notnull().sum(),len(house)
house['Alley'].fillna('No Alley', inplace=True)
house.Alley.isnull().sum()
house.Alley.value_counts(ascending=False)
house.LotShape.dtype
house.LotShape.value_counts(ascending=False)
house.LotShape.isnull().sum()
house.LandContour.dtype
house.LandContour.value_counts(ascending=False)
house.LandContour.isnull().sum()
house.Utilities.dtype
house.Utilities.value_counts(ascending=False)
house.Utilities.isnull().sum()
house.LotConfig.dtype
house.LotConfig.value_counts(ascending=False)
house.LotConfig.isnull().sum()
house.LandSlope.dtype
house.LandSlope.value_counts(ascending=False)
house.LandSlope.isnull().sum()
house.Neighborhood.dtype
house.Neighborhood.value_counts(ascending=False)
house.Neighborhood.isnull().sum()
house.Condition1.dtype
house.Condition1.value_counts(ascending=False)
house.Condition1.isnull().sum()
house.Condition2.dtype
house.Condition2.value_counts(ascending=False)
house.Condition2.isnull().sum()
house.BldgType.dtype
house.BldgType.value_counts(ascending=False)
house.BldgType.isnull().sum()
house.HouseStyle.dtype
house.HouseStyle.value_counts(ascending=False)
house.HouseStyle.isnull().sum()
house.OverallQual.dtype
house['OverallQual']=house['OverallQual'].astype('object')
house.OverallQual.dtype
house.OverallQual.value_counts(ascending=False)
house.OverallQual.isnull().sum()
house.OverallCond.dtype
house['OverallCond']=house['OverallCond'].astype('object')
house.OverallCond.dtype
house.OverallCond.value_counts(ascending=False)
house.OverallCond.isnull().sum()
house.YearBuilt.dtype
house.YearBuilt.value_counts(ascending=False)
house.YearBuilt.isnull().sum()
house.YearRemodAdd.dtype
house.YearRemodAdd.value_counts(ascending=False)
house.YearRemodAdd.isnull().sum()
house.RoofStyle.dtype
house.RoofStyle.value_counts(ascending=False)
house.RoofStyle.isnull().sum()
house.RoofMatl.dtype
house.RoofMatl.value_counts(ascending=False)
house.RoofMatl.isnull().sum()
house.Exterior1st.dtype
house.Exterior1st.value_counts(ascending=False)
house.Exterior1st.isnull().sum()
house.Exterior2nd.dtype
house.Exterior2nd.value_counts(ascending=False)
house.Exterior2nd.isnull().sum()
house.MasVnrType.dtype
house.MasVnrType.value_counts(ascending=False)
house.MasVnrType.isnull().sum()
# Missing value found
house.MasVnrArea.dtype
house.MasVnrArea.describe()
house.MasVnrArea.isnull().sum()
# Missing values are there
house.ExterQual.dtype
house.ExterQual.value_counts(ascending=False)
house.ExterQual.isnull().sum()
house.ExterCond.dtype
house.ExterCond.value_counts(ascending=False)
house.ExterCond.isnull().sum()
house.Foundation.dtype
house.Foundation.value_counts(ascending=False)
house.Foundation.isnull().sum()
house.BsmtQual.dtype
house.BsmtQual.value_counts(ascending=False)
house.BsmtQual.isnull().sum(),house.BsmtQual.notnull().sum(),len(house)
# Here, NA is No Basement. But dataframe consider it as missing value. So, replace with a new value,'No Basement' instead of NA
house['BsmtQual'].fillna('No Basement', inplace=True)
house.BsmtQual.dtype
house.BsmtQual.value_counts(ascending=False)
house.BsmtCond.dtype
house.BsmtCond.value_counts(ascending=False)
house.BsmtCond.isnull().sum(),house.BsmtCond.notnull().sum(),len(house)
house['BsmtCond'].fillna('No Basement', inplace=True)
house.BsmtCond.value_counts(ascending=False)
house.BsmtCond.isnull().sum()
house.BsmtExposure.dtype
house.BsmtExposure.value_counts(ascending=False)
house.BsmtExposure.isnull().sum(),house.BsmtExposure.notnull().sum(),len(house)
house['BsmtExposure'].fillna('No Basement', inplace=True)
house.BsmtExposure.value_counts(ascending=False)
house.BsmtExposure.isnull().sum()
house.BsmtFinType1.dtype
house.BsmtFinType1.value_counts(ascending=False)
house.BsmtFinType1.isnull().sum(),house.BsmtFinType1.notnull().sum(),len(house)
house['BsmtFinType1'].fillna('No Basement', inplace=True)
house.BsmtFinType1.value_counts(ascending=False)
house.BsmtFinType1.isnull().sum()
house.BsmtFinSF1.dtype
house.BsmtFinSF1.describe()
house.BsmtFinSF1.isnull().sum()
house.BsmtFinType2.dtype
house.BsmtFinType2.value_counts(ascending=False)
house.BsmtFinType2.isnull().sum(),house.BsmtFinType2.notnull().sum(),len(house)
house['BsmtFinType2'].fillna('No Basement', inplace=True)
house.BsmtFinType2.value_counts(ascending=False)
house.BsmtFinType2.isnull().sum()
house.BsmtFinSF2.dtype
house.BsmtFinSF2.describe()
house.BsmtFinSF2.isnull().sum()
house.BsmtUnfSF.dtype
house.BsmtUnfSF.describe()
house.BsmtUnfSF.isnull().sum()
house.TotalBsmtSF.dtype
house.TotalBsmtSF.describe()
house.TotalBsmtSF.isnull().sum()
house.Heating.dtype
house.Heating.value_counts(ascending=False)
house.Heating.isnull().sum()
house.HeatingQC.dtype
house.HeatingQC.value_counts(ascending=False)
house.HeatingQC.isnull().sum()
house.CentralAir.dtype
house.CentralAir.value_counts(ascending=False)
house.CentralAir.isnull().sum()
house.Electrical.dtype
house.Electrical.value_counts(ascending=False)
house.Electrical.isnull().sum()
# Missing value treatment will be needed 
house['1stFlrSF'].dtype
house['1stFlrSF'].describe()
house['1stFlrSF'].isnull().sum()
house['2ndFlrSF'].dtype
house['2ndFlrSF'].describe()
house['2ndFlrSF'].isnull().sum()
house['LowQualFinSF'].dtype
house['LowQualFinSF'].describe()
house['LowQualFinSF'].isnull().sum()
house.GrLivArea.dtype
house.GrLivArea.describe()
house.GrLivArea.isnull().sum()
house.BsmtFullBath.dtype
house.BsmtFullBath.value_counts()
house['BsmtFullBath']=house['BsmtFullBath'].astype('object')
house.BsmtFullBath.dtype
house.BsmtFullBath.value_counts()
house.BsmtFullBath.isnull().sum()
house.BsmtHalfBath.dtype
house.BsmtHalfBath.value_counts()
house['BsmtHalfBath']=house['BsmtHalfBath'].astype('object')
house.BsmtHalfBath.dtype
house.BsmtHalfBath.value_counts()
house.BsmtHalfBath.isnull().sum()
house.FullBath.dtype
house.FullBath.value_counts()
house['FullBath']=house['FullBath'].astype('object')
house.FullBath.dtype
house.FullBath.value_counts()
house.FullBath.isnull().sum()
house.HalfBath.dtype
house['HalfBath'].value_counts()
house['HalfBath']=house['HalfBath'].astype('object')
house.HalfBath.dtype
house['HalfBath'].value_counts()
house['HalfBath'].isnull().sum()
house.BedroomAbvGr.dtype
house.BedroomAbvGr.value_counts()
house['BedroomAbvGr']=house['BedroomAbvGr'].astype('object')
house.BedroomAbvGr.dtype
house.BedroomAbvGr.value_counts()
house.BedroomAbvGr.isnull().sum()
house.KitchenAbvGr.dtype
house['KitchenAbvGr'].value_counts()
house['KitchenAbvGr']=house['KitchenAbvGr'].astype('object')
house.KitchenAbvGr.dtype
house['KitchenAbvGr'].value_counts()
house.KitchenAbvGr.isnull().sum()
house.KitchenQual.dtype
house.KitchenQual.value_counts()
house.KitchenQual.isnull().sum()
house.TotRmsAbvGrd.dtype
house.TotRmsAbvGrd.value_counts()
house['TotRmsAbvGrd']=house['TotRmsAbvGrd'].astype('object')
house.TotRmsAbvGrd.dtype
house.TotRmsAbvGrd.value_counts()
house.TotRmsAbvGrd.isnull().sum()
house.Functional.dtype
house.Functional.value_counts()
house.Functional.isnull().sum()
house.Fireplaces.dtype
house.Fireplaces.describe()
house.Fireplaces.isnull().sum()
house.FireplaceQu.dtype
house.FireplaceQu.value_counts()
house.FireplaceQu.isnull().sum(),house.FireplaceQu.notnull().sum(),len(house)
house.FireplaceQu.fillna('No Fireplace', inplace=True)
house.FireplaceQu.value_counts()
house.FireplaceQu.isnull().sum()
house.GarageType.dtype
house.GarageType.value_counts()
house.GarageType.isnull().sum(),house.GarageType.notnull().sum(),len(house)
house.GarageType.fillna('No Garage', inplace=True)
house.GarageType.value_counts()
house.GarageType.isnull().sum()
house.GarageYrBlt.dtype
house.GarageYrBlt.describe()
house.GarageYrBlt.isnull().sum()
# missing value found
house.GarageFinish.dtype
house.GarageFinish.value_counts()
house.GarageFinish.isnull().sum(),house.GarageFinish.notnull().sum(),len(house)
house['GarageFinish'].fillna('No Garage', inplace=True)
house.GarageFinish.value_counts()
house.GarageFinish.isnull().sum()
house.GarageCars.dtype
house.GarageCars.value_counts()
house['GarageCars']=house['GarageCars'].astype('object')
house.GarageCars.dtype
house.GarageCars.value_counts()
house.GarageCars.isnull().sum()
house.GarageArea.dtype
house.GarageArea.describe()
house.GarageArea.isnull().sum()
house.GarageQual.dtype
house.GarageQual.value_counts()
house.GarageQual.isnull().sum(),house.GarageQual.notnull().sum(),len(house)
house['GarageQual'].fillna('No Garage', inplace=True)
house.GarageQual.value_counts()
house.GarageQual.isnull().sum()
house.GarageCond.dtype
house.GarageCond.value_counts()
house.GarageCond.isnull().sum(),house.GarageCond.notnull().sum(),len(house)
house['GarageCond'].fillna('No Garage', inplace=True)
house.GarageCond.value_counts()
house.GarageCond.isnull().sum()
house.PavedDrive.dtype
house.PavedDrive.value_counts()
house.PavedDrive.isnull().sum()
house.WoodDeckSF.dtype
house.GrLivArea.describe()
house.WoodDeckSF.isnull().sum()
house.OpenPorchSF.dtype
house.OpenPorchSF.describe()
house.OpenPorchSF.isnull().sum()
house.EnclosedPorch.dtype
house.EnclosedPorch.describe()
house.EnclosedPorch.isnull().sum()
house['3SsnPorch'].dtype
house['3SsnPorch'].describe()
house['3SsnPorch'].isnull().sum()
house.ScreenPorch.dtype
house.ScreenPorch.describe()
house.ScreenPorch.isnull().sum()
house.PoolArea.dtype
house.PoolArea.describe()
house.PoolArea.isnull().sum()
house.PoolQC.dtype
house.PoolQC.value_counts()
house.PoolQC.isnull().sum(),house.PoolQC.notnull().sum(),len(house)
house['PoolQC'].fillna('No Pool', inplace=True)
house.PoolQC.value_counts()
house.PoolQC.isnull().sum()
house.Fence.dtype
house.Fence.value_counts()
house.Fence.isnull().sum(),house.Fence.notnull().sum(),len(house)
house['Fence'].fillna('No Fence', inplace=True)
house.Fence.value_counts()
house.Fence.isnull().sum()
house.MiscFeature.dtype
house.MiscFeature.value_counts()
house.MiscFeature.isnull().sum(),house.MiscFeature.notnull().sum(),len(house)
house['MiscFeature'].fillna('None', inplace=True)
house.MiscFeature.value_counts()
house.GarageQual.isnull().sum()
house.MiscVal.dtype
house.MiscVal.describe()
house.MiscVal.isnull().sum()
house.MoSold.dtype
house.MoSold.value_counts()
house['MoSold']=house['MoSold'].astype('object')
house.MoSold.dtype
house.MoSold.value_counts()
house.MoSold.isnull().sum()
house.YrSold.dtype
house.YrSold.describe()
house.YrSold.isnull().sum()
house.SaleType.dtype
house.SaleType.value_counts()
house.SaleType.isnull().sum()
house.SaleCondition.dtype
house.SaleCondition.value_counts()
house.SaleCondition.isnull().sum()
house.info()
import datetime

now = datetime.datetime.now()

now
house['Age of Building'] = now.year - house['YearBuilt']

house['Age of Building'].head()
house['Age of Building'].dtype
house['Age of Building'].describe()
house.drop('YearBuilt',axis=1,inplace=True)

house.columns
house['Last Sold'] = now.year - house['YrSold']

house['Last Sold'].head()
house['Last Sold'].dtype
house['Last Sold'].describe()
house.drop('YrSold',axis=1,inplace=True)

house.columns
house['Age of Garage'] = now.year - house['GarageYrBlt']

house['Age of Garage'].head()
house['Age of Garage'].dtype
house['Age of Garage'].describe()
house.drop('GarageYrBlt',axis=1,inplace=True)

house.columns
house['Last Remodelled'] = now.year - house['YearRemodAdd']

house['Last Remodelled'].head()
house['Last Remodelled'].dtype
house['Last Remodelled'].describe()
house.drop('YearRemodAdd',axis=1,inplace=True)

house.columns
house.info()
# Duplicate Check 
houseo=house.copy()

houseo.drop_duplicates(subset=None, inplace=True)

houseo.shape
house.shape
house.isnull().sum().sort_values(ascending=False)[:8]
(house.isnull().sum()*100 / len(house)).sort_values(ascending=False)[:8]
house.columns[house.isnull().any()]
house['LotFrontage'].isnull().sum()
house['LotFrontage'].fillna(0, inplace=True) 
house['LotFrontage'].isnull().sum()
house['MasVnrType'].isnull().sum()
house['MasVnrType'].fillna('None', inplace=True)
house['MasVnrType'].isnull().sum()
house['MasVnrArea'].isnull().sum()
house['MasVnrArea'].fillna(0, inplace=True) 
house['MasVnrArea'].isnull().sum()
house['Electrical'].isnull().sum()
# Since, there is 1 missing value, better to drop NA 
house = house[house['Electrical'].notna()]

house['Electrical'].isnull().sum()
house['Age of Garage'].isnull().sum()
house[house.isnull().any(axis=1)] 
house['GarageFinish'].nunique()
house['Age of Garage'].describe()
plt.figure(figsize = (20,10))

plt.grid()

ax=sns.distplot(house["Age of Garage"],bins=11,kde=False)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
minage=int(min(house['Age of Garage']))

maxage=int(max(house['Age of Garage']))

list1=list(range(minage,20))

list2=list(range(20,30))

list3=list(range(30,40))

list4=list(range(40,50))

list5=list(range(50,60))

list6=list(range(60,70))

list7=list(range(70,80))

list8=list(range(80,90))

list9=list(range(90,100))

list10=list(range(100,110))

list11=list(range(110,120))

if maxage == 120:

    list12=[120]

else:

    list12=list(range(120,maxage))

    
house['Age of Garage']=house['Age of Garage'].replace(list1,'Young < 20')

house['Age of Garage']=house['Age of Garage'].replace(list2,'in20s')

house['Age of Garage']=house['Age of Garage'].replace(list3,'in30s')

house['Age of Garage']=house['Age of Garage'].replace(list4,'in40s')

house['Age of Garage']=house['Age of Garage'].replace(list5,'in50s')

house['Age of Garage']=house['Age of Garage'].replace(list6,'in60s')

house['Age of Garage']=house['Age of Garage'].replace(list7,'in70s')

house['Age of Garage']=house['Age of Garage'].replace(list8,'in80s')

house['Age of Garage']=house['Age of Garage'].replace(list9,'in90s')

house['Age of Garage']=house['Age of Garage'].replace(list10,'in100s')

house['Age of Garage']=house['Age of Garage'].replace(list11,'in110s')

house['Age of Garage']=house['Age of Garage'].replace(list12,'Heritage >120 ')

house['Age of Garage'].value_counts()
house['Age of Garage'].dtype
plt.figure(figsize = (20,10))

ax=sns.countplot(data=house, x="Age of Garage")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')



plt.show()
house['Age of Garage'].fillna('N/A', inplace=True) 
plt.figure(figsize = (20,10))

ax=sns.countplot(data=house, x="Age of Garage")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')



plt.show()
house['Age of Garage']=house['Age of Garage'].astype('str')

house['Age of Garage']=house['Age of Garage'].astype('object')
house.isnull().sum().sort_values(ascending=False)[:8]
(house.isnull().sum()*100 / len(house)).sort_values(ascending=False)[:8]
#No Missing values in columns
house.isnull().sum(axis=1).sort_values(ascending=False)[:5]
(house.isnull().sum(axis=1)*100 / len(house)).sort_values(ascending=False)[:5]
#No Missing values in rows 
print('Available Rows % after Missing value treatment : ',round(len(house)*100/len(houseo),2))
house.drop('Id',axis=1,inplace=True)

house.head()
house_mm=house.copy()

house_no=house.copy()
features=list((house.dtypes[house.dtypes == np.object]).index)

len(features)
def histograms_plot(features, rows, cols):

    fig=plt.figure(figsize=(20,80))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        house[feature].hist(bins=20,ax=ax,facecolor='green')

        plt.xticks(rotation = 90)

        ax.set_yscale('log')

        ax.set_title(feature+" Distribution",color='red')

       

    fig.tight_layout()  

    plt.show()
histograms_plot(features,14, 4)
from sklearn.model_selection import train_test_split



# We should specify 'random_state' so that the train and test data set always have the same rows, respectively



np.random.seed(0)

train, test = train_test_split(house, train_size = 0.70, random_state = 42)
train.head()
X_train= train.drop(['SalePrice'],axis=1)

X_test= test.drop(['SalePrice'],axis=1)

y_train= train['SalePrice']

y_test=test['SalePrice']
final=list(set(list(house.columns))-set(list(features))-{'SalePrice'})

len(final)
house.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[final]= scaler.fit_transform(X_train[final])

X_train.head()

final
X_test[final]= scaler.transform(X_test[final])

X_test.head()
from catboost import CatBoostRegressor

model=CatBoostRegressor()
obj=list(np.where(X_train.dtypes == np.object)[0])
model.fit(X_train,y_train,cat_features=obj)
import sklearn.metrics

score_ss=model.score(X_test,y_test)

evs_ss=sklearn.metrics.explained_variance_score(y_test,model.predict(X_test))

me_ss=sklearn.metrics.max_error(y_test,model.predict(X_test))

mae_ss=sklearn.metrics.mean_absolute_error(y_test,model.predict(X_test))

mse_ss=sklearn.metrics.mean_squared_error(y_test,model.predict(X_test))

msle_ss=sklearn.metrics.mean_squared_log_error(y_test,model.predict(X_test))

Mae_ss=sklearn.metrics.median_absolute_error(y_test,model.predict(X_test))

r2_ss=sklearn.metrics.r2_score(y_test,model.predict(X_test))

mpd_ss=sklearn.metrics.mean_poisson_deviance(y_test,model.predict(X_test))

mgd_ss=sklearn.metrics.mean_gamma_deviance(y_test,model.predict(X_test))

mtd_ss=sklearn.metrics.mean_tweedie_deviance(y_test,model.predict(X_test))
print('Score                   :',score_ss)

print('Explained Variance Score:',evs_ss)

print('Max Error               :',me_ss)

print('Mean Absolute Error     :',mae_ss)

print('Mean Square Error       :',mse_ss)

print('Mean Squared Log Error  :',msle_ss)

print('Median Absolute Error   :',Mae_ss)

print('R2 Score                :',r2_ss)

print('Mean Poisson Deviance   :',mpd_ss)

print('Mean Gamma Deviance     :',mgd_ss)

print('Mean Tweedie Deviance   :',mtd_ss)
house_mm.head()
features=list((house_mm.dtypes[house_mm.dtypes == np.object]).index)

len(features)
def histograms_plot(features, rows, cols):

    fig=plt.figure(figsize=(20,80))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        house[feature].hist(bins=20,ax=ax,facecolor='green')

        plt.xticks(rotation = 90)

        ax.set_yscale('log')

        ax.set_title(feature+" Distribution",color='red')

       

    fig.tight_layout()  

    plt.show()
histograms_plot(features,14, 4)
from sklearn.model_selection import train_test_split



# We should specify 'random_state' so that the train and test data set always have the same rows, respectively



np.random.seed(0)

train, test = train_test_split(house_mm, train_size = 0.70, random_state = 42)



train.head()
X_train= train.drop(['SalePrice'],axis=1)

X_test= test.drop(['SalePrice'],axis=1)

y_train= train['SalePrice']

y_test=test['SalePrice']
final=list(set(list(house_mm.columns))-set(list(features))-{'SalePrice'})

len(final)
house_mm.info()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train[final]= scaler.fit_transform(X_train[final])

X_train.head()
final
X_test[final]= scaler.transform(X_test[final])

X_test.head()
from catboost import CatBoostRegressor

model1=CatBoostRegressor()
obj=list(np.where(X_train.dtypes == np.object)[0])

model1.fit(X_train,y_train,cat_features=obj)
import sklearn.metrics

score_mms=model1.score(X_test,y_test)

evs_mms=sklearn.metrics.explained_variance_score(y_test,model1.predict(X_test))

me_mms=sklearn.metrics.max_error(y_test,model1.predict(X_test))

mae_mms=sklearn.metrics.mean_absolute_error(y_test,model1.predict(X_test))

mse_mms=sklearn.metrics.mean_squared_error(y_test,model1.predict(X_test))

msle_mms=sklearn.metrics.mean_squared_log_error(y_test,model1.predict(X_test))

Mae_mms=sklearn.metrics.median_absolute_error(y_test,model1.predict(X_test))

r2_mms=sklearn.metrics.r2_score(y_test,model1.predict(X_test))

mpd_mms=sklearn.metrics.mean_poisson_deviance(y_test,model1.predict(X_test))

mgd_mms=sklearn.metrics.mean_gamma_deviance(y_test,model1.predict(X_test))

mtd_mms=sklearn.metrics.mean_tweedie_deviance(y_test,model1.predict(X_test))
print('Score                   :',score_mms)

print('Explained Variance Score:',evs_mms)

print('Max Error               :',me_mms)

print('Mean Absolute Error     :',mae_mms)

print('Mean Square Error       :',mse_mms)

print('Mean Squared Log Error  :',msle_mms)

print('Median Absolute Error   :',Mae_mms)

print('R2 Score                :',r2_mms)

print('Mean Poisson Deviance   :',mpd_mms)

print('Mean Gamma Deviance     :',mgd_mms)

print('Mean Tweedie Deviance   :',mtd_mms)
print('Score                   :',score_ss)

print('Score                   :',score_mms)
print('Explained Variance Score:',evs_ss)

print('Explained Variance Score:',evs_mms)
print('Max Error               :',me_ss)

print('Max Error               :',me_mms)
print('Mean Absolute Error     :',mae_ss)

print('Mean Absolute Error     :',mae_mms)
print('Mean Square Error       :',mse_ss)

print('Mean Square Error       :',mse_mms)
print('Mean Squared Log Error  :',msle_ss)

print('Mean Squared Log Error  :',msle_mms)
print('Median Absolute Error   :',Mae_ss)

print('Median Absolute Error   :',Mae_mms)
print('R2 Score                :',r2_ss)

print('R2 Score                :',r2_mms)
print('Mean Poisson Deviance   :',mpd_ss)

print('Mean Poisson Deviance   :',mpd_mms)
print('Mean Gamma Deviance     :',mgd_ss)

print('Mean Gamma Deviance     :',mgd_mms)
print('Mean Tweedie Deviance   :',mtd_ss)

print('Mean Tweedie Deviance   :',mtd_mms)
house_no.head()
features=list((house_no.dtypes[house_mm.dtypes == np.object]).index)

len(features)
def histograms_plot(features, rows, cols):

    fig=plt.figure(figsize=(20,80))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        house[feature].hist(bins=20,ax=ax,facecolor='green')

        plt.xticks(rotation = 90)

        ax.set_yscale('log')

        ax.set_title(feature+" Distribution",color='red')

       

    fig.tight_layout()  

    plt.show()
histograms_plot(features,14, 4)
from sklearn.model_selection import train_test_split



# We should specify 'random_state' so that the train and test data set always have the same rows, respectively



np.random.seed(0)

train, test = train_test_split(house_no, train_size = 0.70, random_state = 42)



train.head()
X_train= train.drop(['SalePrice'],axis=1)

X_test= test.drop(['SalePrice'],axis=1)

y_train= train['SalePrice']

y_test=test['SalePrice']
final=list(set(list(house_no.columns))-set(list(features))-{'SalePrice'})

len(final)
house_no.info()
final
from catboost import CatBoostRegressor

model2=CatBoostRegressor()
obj=list(np.where(X_train.dtypes == np.object)[0])

model2.fit(X_train,y_train,cat_features=obj)
import sklearn.metrics

score_no=model2.score(X_test,y_test)

evs_no=sklearn.metrics.explained_variance_score(y_test,model2.predict(X_test))

me_no=sklearn.metrics.max_error(y_test,model2.predict(X_test))

mae_no=sklearn.metrics.mean_absolute_error(y_test,model2.predict(X_test))

mse_no=sklearn.metrics.mean_squared_error(y_test,model2.predict(X_test))

msle_no=sklearn.metrics.mean_squared_log_error(y_test,model2.predict(X_test))

Mae_no=sklearn.metrics.median_absolute_error(y_test,model2.predict(X_test))

r2_no=sklearn.metrics.r2_score(y_test,model2.predict(X_test))

mpd_no=sklearn.metrics.mean_poisson_deviance(y_test,model2.predict(X_test))

mgd_no=sklearn.metrics.mean_gamma_deviance(y_test,model2.predict(X_test))

mtd_no=sklearn.metrics.mean_tweedie_deviance(y_test,model2.predict(X_test))
print('Score                   :',score_no)

print('Explained Variance Score:',evs_no)

print('Max Error               :',me_no)

print('Mean Absolute Error     :',mae_no)

print('Mean Square Error       :',mse_no)

print('Mean Squared Log Error  :',msle_no)

print('Median Absolute Error   :',Mae_no)

print('R2 Score                :',r2_no)

print('Mean Poisson Deviance   :',mpd_no)

print('Mean Gamma Deviance     :',mgd_no)

print('Mean Tweedie Deviance   :',mtd_no)
print('Score                   :',score_ss)

print('Score                   :',score_mms)

print('Score                   :',score_no)
print('Explained Variance Score:',evs_ss)

print('Explained Variance Score:',evs_mms)

print('Explained Variance Score:',evs_no)
print('Max Error               :',me_ss)

print('Max Error               :',me_mms)

print('Max Error               :',me_no)
print('Mean Absolute Error     :',mae_ss)

print('Mean Absolute Error     :',mae_mms)

print('Mean Absolute Error     :',mae_no)
print('Mean Square Error       :',mse_ss)

print('Mean Square Error       :',mse_mms)

print('Mean Square Error       :',mse_no)
print('Mean Squared Log Error  :',msle_ss)

print('Mean Squared Log Error  :',msle_mms)

print('Mean Squared Log Error  :',msle_no)
print('Median Absolute Error   :',Mae_ss)

print('Median Absolute Error   :',Mae_mms)

print('Median Absolute Error   :',Mae_no)
print('R2 Score                :',r2_ss)

print('R2 Score                :',r2_mms)

print('R2 Score                :',r2_no)
print('Mean Poisson Deviance   :',mpd_ss)

print('Mean Poisson Deviance   :',mpd_mms)

print('Mean Poisson Deviance   :',mpd_no)
print('Mean Gamma Deviance     :',mgd_ss)

print('Mean Gamma Deviance     :',mgd_mms)

print('Mean Gamma Deviance     :',mgd_no)
print('Mean Tweedie Deviance   :',mtd_ss)

print('Mean Tweedie Deviance   :',mtd_mms)

print('Mean Tweedie Deviance   :',mtd_no)