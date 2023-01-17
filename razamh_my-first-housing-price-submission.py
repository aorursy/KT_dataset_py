import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("darkgrid")

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.impute import SimpleImputer



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(font_scale=1)
#importing data from csv file using pandas

train=pd.read_csv('../input/home-data-for-ml-course/train.csv')

test=pd.read_csv('../input/home-data-for-ml-course/test.csv')



train.head()
print("Train",train.shape)

print("Test",test.shape)
X = pd.concat([train.drop("SalePrice", axis=1),test], axis=0)

y = train[['SalePrice']]
X.info()
numeric_ = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).copy()

numeric_.columns
cat_train = X.select_dtypes(include=['object']).copy()

cat_train['MSSubClass'] = X['MSSubClass']   #MSSubClass is nominal

cat_train.columns
#lets create scatterplot of GrLivArea and SalePrice

sns.scatterplot(x='GrLivArea',y='SalePrice',data=train)

plt.show()
#as per above plot we can see there are two outliers which can affect on out model,lets remove those outliers

train=train.drop(train.loc[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index,0)

train.reset_index(drop=True, inplace=True)
#lest we how its look after removing outliers

sns.scatterplot(x='GrLivArea',y='SalePrice',data=train)

plt.show()
#lets create heatmap first of all lest see on which feature SalePrice is dependent

corr=train.drop('Id',1).corr().sort_values(by='SalePrice',ascending=False).round(2)

print(corr['SalePrice'])
#here we can see SalePrice mostly dependent on this features OverallQual,GrLivArea,TotalBsmtSF,GarageCars,1stFlrSF,GarageArea 

plt.subplots(figsize=(12, 9))

sns.heatmap(corr, vmax=.8, square=True);
#now lets create heatmap for top 10 correlated features

cols =corr['SalePrice'].head(10).index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1)

hm = sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#lets see relation of 10 feature with SalePrice through Pairplot

sns.pairplot(train[corr['SalePrice'].head(10).index])

plt.show()
#lets store number of test and train rows

trainrow=train.shape[0]

testrow=test.shape[0]
#copying id data

testids=test['Id'].copy()
#copying sales priece

y_train=train['SalePrice'].copy()
#combining train and test data

data=pd.concat((train,test)).reset_index(drop=True)

data=data.drop('SalePrice',1)
#dropping id columns

data=data.drop('Id',axis=1)
#checking missing data

missing=data.isnull().sum().sort_values(ascending=False)

missing=missing.drop(missing[missing==0].index)

missing
#PoolQC is quality of pool but mostly house does not have pool so putting NA

data['PoolQC']=data['PoolQC'].fillna('NA')

data['PoolQC'].unique()
#MiscFeature: mostly house does not have it so putting NA

data['MiscFeature']=data['MiscFeature'].fillna('NA')

data['MiscFeature'].unique()
#Alley,Fence,FireplaceQu: mostly house does not have it so putting NA

data['Alley']=data['Alley'].fillna('NA')

data['Alley'].unique()



data['Fence']=data['Fence'].fillna('NA')

data['Fence'].unique()



data['FireplaceQu']=data['FireplaceQu'].fillna('NA')

data['FireplaceQu'].unique()
#LotFrontage: all house have linear connected feet so putting most mean value

data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].dropna().mean())
#GarageCond,GarageQual,GarageFinish

data['GarageCond']=data['GarageCond'].fillna('NA')

data['GarageCond'].unique()



data['GarageQual']=data['GarageQual'].fillna('NA')

data['GarageQual'].unique()



data['GarageFinish']=data['GarageFinish'].fillna('NA')

data['GarageFinish'].unique()
#GarageYrBlt,GarageType,GarageArea,GarageCars putting 0

data['GarageYrBlt']=data['GarageYrBlt'].fillna(0)

data['GarageType']=data['GarageType'].fillna(0)

data['GarageArea']=data['GarageArea'].fillna(0)

data['GarageCars']=data['GarageCars'].fillna(0)
#BsmtExposure,BsmtCond,BsmtQual,BsmtFinType2,BsmtFinType1 

data['BsmtExposure']=data['BsmtExposure'].fillna('NA')

data['BsmtCond']=data['BsmtCond'].fillna('NA')

data['BsmtQual']=data['BsmtQual'].fillna('NA')

data['BsmtFinType2']=data['BsmtFinType2'].fillna('NA')

data['BsmtFinType1']=data['BsmtFinType1'].fillna('NA')



#BsmtFinSF1,BsmtFinSF2 

data['BsmtFinSF1']=data['BsmtFinSF1'].fillna(0)

data['BsmtFinSF2']=data['BsmtFinSF2'].fillna(0)
#MasVnrType,MasVnrArea

data['MasVnrType']=data['MasVnrType'].fillna('NA')

data['MasVnrArea']=data['MasVnrArea'].fillna(0)
#MSZoning 

data['MSZoning']=data['MSZoning'].fillna(data['MSZoning'].dropna().sort_values().index[0])

#Utilities

data['Utilities']=data['Utilities'].fillna(data['Utilities'].dropna().sort_values().index[0])

#BsmtFullBath

data['BsmtFullBath']=data['BsmtFullBath'].fillna(0)



#Functional

data['Functional']=data['Functional'].fillna(data['Functional'].dropna().sort_values().index[0])



#BsmtHalfBath

data['BsmtHalfBath']=data['BsmtHalfBath'].fillna(0)



#BsmtUnfSF

data['BsmtUnfSF']=data['BsmtUnfSF'].fillna(0)

#Exterior2nd

data['Exterior2nd']=data['Exterior2nd'].fillna('NA')



#Exterior1st

data['Exterior1st']=data['Exterior1st'].fillna('NA')

#TotalBsmtSF

data['TotalBsmtSF']=data['TotalBsmtSF'].fillna(0)

#SaleType

data['SaleType']=data['SaleType'].fillna(data['SaleType'].dropna().sort_values().index[0])

#Electrical

data['Electrical']=data['Electrical'].fillna(data['Electrical'].dropna().sort_values().index[0])

#KitchenQual

data['KitchenQual']=data['KitchenQual'].fillna(data['KitchenQual'].dropna().sort_values().index[0])

#lets check any missing remain

missing=data.isnull().sum().sort_values(ascending=False)

missing=missing.drop(missing[missing==0].index)

missing
#as we know some feature are highly co-related with SalePrice so lets create some feature using these features

data['GrLivArea_2']=data['GrLivArea']**2

data['GrLivArea_3']=data['GrLivArea']**3

data['GrLivArea_4']=data['GrLivArea']**4



data['TotalBsmtSF_2']=data['TotalBsmtSF']**2

data['TotalBsmtSF_3']=data['TotalBsmtSF']**3

data['TotalBsmtSF_4']=data['TotalBsmtSF']**4



data['GarageCars_2']=data['GarageCars']**2

data['GarageCars_3']=data['GarageCars']**3

data['GarageCars_4']=data['GarageCars']**4



data['1stFlrSF_2']=data['1stFlrSF']**2

data['1stFlrSF_3']=data['1stFlrSF']**3

data['1stFlrSF_4']=data['1stFlrSF']**4



data['GarageArea_2']=data['GarageArea']**2

data['GarageArea_3']=data['GarageArea']**3

data['GarageArea_4']=data['GarageArea']**4
#lets add 1stFlrSF and 2ndFlrSF and create new feature floorfeet

data['Floorfeet']=data['1stFlrSF']+data['2ndFlrSF']

data=data.drop(['1stFlrSF','2ndFlrSF'],1)
#MSSubClass,MSZoning

data=pd.get_dummies(data=data,columns=['MSSubClass'],prefix='MSSubClass')

data=pd.get_dummies(data=data,columns=['MSZoning'],prefix='MSZoning')

data.head()
X['TotalLot'] = X['LotFrontage'] + X['LotArea']

X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']

X['TotalSF'] = X['TotalBsmtSF'] + X['2ndFlrSF']

X['TotalBath'] = X['FullBath'] + X['HalfBath']

X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['ScreenPorch']
colum = ['MasVnrArea','TotalBsmtFin','TotalBsmtSF','2ndFlrSF','WoodDeckSF','TotalPorch']



for col in colum:

    col_name = col+'_bin'

    X[col_name] = X[col].apply(lambda x: 1 if x > 0 else 0)
X = pd.get_dummies(X)
plt.figure(figsize=(10,6))

plt.title("Before transformation of SalePrice")

dist = sns.distplot(train['SalePrice'],norm_hist=False)

plt.figure(figsize=(10,6))

plt.title("After transformation of SalePrice")

dist = sns.distplot(np.log(train['SalePrice']),norm_hist=False)
y["SalePrice"] = np.log(y['SalePrice'])
x = X.loc[train.index]

y = y.loc[train.index]

test = X.loc[test.index]
#lets import StandardScaler from sklearn for feature scalling

from sklearn.preprocessing import StandardScaler

#lets split data using trainrow data and scale data

cols = x.select_dtypes(np.number).columns

transformer = RobustScaler().fit(x[cols])

x[cols] = transformer.transform(x[cols])

test[cols] = transformer.transform(test[cols])
num_correlation = train.select_dtypes(exclude='object').corr()

corr = num_correlation.corr()

print(corr['SalePrice'].sort_values(ascending=False))
# Create target object and call it y

y = train.SalePrice

# Create X

#features = ['OverallQual','LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF','FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GrLivArea','GarageCars', 'GarageArea']

featurestop=['OverallQual','TotalBsmtSF', 'YearBuilt','YearRemodAdd','GarageYrBlt','Fireplaces', '1stFlrSF', 'MasVnrArea','FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GrLivArea','GarageCars', 'GarageArea']

X = train[featurestop]

train[featurestop]

sns.heatmap(X.isnull(),yticklabels=False, cbar=False, cmap='viridis')
##Check TestData

# path to file you will use for predictions

test_data_path = '/kaggle/input/home-data-for-ml-course/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[featurestop]

#test_X.dropna(inplace=True)

test_X.info()
GarageYrBltmean=X.loc[:,"GarageYrBlt"].mean()

MasVnrAreamean=X.loc[:,"MasVnrArea"].mean()

print(GarageYrBltmean,MasVnrAreamean)
X['GarageYrBlt'].fillna(GarageYrBltmean,inplace = True)

X['MasVnrArea'].fillna(MasVnrAreamean,inplace = True)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
#working with missing Values

GarageCarsmean=test_X.loc[:,"GarageCars"].mean()

GarageAreamean=test_X.loc[:,"GarageArea"].mean()

GarageYrBltmean=test_X.loc[:,"GarageYrBlt"].mean()

MasVnrAreamean=test_X.loc[:,"MasVnrArea"].mean()

TotalBsmtSFmean=test_X.loc[:,"TotalBsmtSF"].mean()

print(GarageYrBltmean,MasVnrAreamean)

print(GarageCarsmean,GarageAreamean)
test_X['GarageArea'].fillna(GarageAreamean,inplace = True)

test_X['GarageYrBlt'].fillna(GarageYrBltmean,inplace = True)

test_X['MasVnrArea'].fillna(MasVnrAreamean,inplace = True)

test_X['GarageCars'].fillna(GarageCarsmean,inplace = True)

test_X['TotalBsmtSF'].fillna(TotalBsmtSFmean,inplace = True)

test_X.info()
rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X, y)
# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': test_preds})

#output.to_csv('submission.csv', index=False)

rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X, y)



# Then in last code cell





output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)