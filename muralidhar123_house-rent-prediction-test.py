# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df.head()
df.describe()
df.isnull().sum()
pd.pandas.set_option('display.max_columns', None)
df.head()
df.isnull().sum()
df.columns


nan=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtypes=='O']



for feature in nan:

    print("{}: {}% missing values".format(feature,np.round(df[feature].isnull().mean(),4)))
def replace(nan, df):

    data = df.copy()

    data[nan] = data[nan].fillna('Missing')

    return data

df = replace(nan,df)

df[nan].isnull().sum()
nan2=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtypes!='O']



for feature in nan2:

    print("{}: {}% missing values".format(feature,np.round(df[feature].isnull().mean(),4)))
df['LotFrontage'].unique()
df['MasVnrArea'].unique()
df['GarageYrBlt'].unique()
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
df.head()
df.isnull().sum()
df['LotFrontage'].isnull().sum()
df['MasVnrArea'].isnull().sum()
fig,ax = plt.subplots(figsize=(45,10),facecolor='white')

sns.boxplot(data = df , ax = ax ,width = 0.5 , fliersize = 3)
q = df["LotArea"].quantile(0.95)

data_cleaned = df[df['LotArea']<q]

#we are removing 3% of data from BloodPressure

q = df['SalePrice'].quantile(0.82)

data_cleaned = df[df['SalePrice']<q]
# X = df.drop(labels='SalePrice',axis=1)

# y=df['SalePrice']
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

X = df
X.head()
# label.fit(X.columns)

# for i,items in enumerate(label.classes_):

#     print(items , "-------->",i)
X.columns
X['MSSubClass']= label.fit_transform(X['MSSubClass'])

X['MSSubClass']
#df['Id'] = label.fit_transform(df['Id'])

df['MSSubClass'] = label.fit_transform(df['MSSubClass'])

df['MSZoning'] = label.fit_transform(df['MSZoning'])

df['LotFrontage'] = label.fit_transform(df['LotFrontage'])

df['LotArea'] = label.fit_transform(df['LotArea'])

df['Street'] = label.fit_transform(df['Street'])

df['Alley'] = label.fit_transform(df['Alley'])

df['LotShape'] = label.fit_transform(df['LotShape'])

df['LandContour'] = label.fit_transform(df['LandContour'])

df['Utilities'] = label.fit_transform(df['Utilities'])

df['LotConfig'] = label.fit_transform(df['LotConfig'])

df['LandSlope'] = label.fit_transform(df['LandSlope'])

df['Neighborhood'] = label.fit_transform(df['Neighborhood'])

df['Condition1'] = label.fit_transform(df['Condition1'])

df['Condition2'] = label.fit_transform(df['Condition2'])

df['BldgType'] = label.fit_transform(df['BldgType'])

df['HouseStyle'] = label.fit_transform(df['HouseStyle'])

df['OverallQual'] = label.fit_transform(df['OverallQual'])

df['OverallCond'] = label.fit_transform(df['OverallCond'])

df['YearBuilt'] = label.fit_transform(df['YearBuilt'])

df['YearRemodAdd'] = label.fit_transform(df['YearRemodAdd'])

df['RoofStyle'] = label.fit_transform(df['RoofStyle'])

df['RoofMatl'] = label.fit_transform(df['RoofMatl'])

df['Exterior1st'] = label.fit_transform(df['Exterior1st'])

df['Exterior2nd'] = label.fit_transform(df['Exterior2nd'])

df['MasVnrType'] = label.fit_transform(df['MasVnrType'])

df['MasVnrArea']=label.fit_transform(df['MasVnrArea'])

df['ExterQual']=label.fit_transform(df['ExterQual'])

df['ExterCond']=label.fit_transform(df['ExterCond'])

df['Foundation']=label.fit_transform(df['Foundation'])

df['BsmtQual']=label.fit_transform(df['BsmtQual'])

df['BsmtCond']=label.fit_transform(df['BsmtCond'])

df['Bsmtxposure']=label.fit_transform(df['BsmtExposure'].astype(str))

df['BsmtFinType1']=label.fit_transform(df['BsmtFinType1'])

df['BsmtFinSF1']=label.fit_transform(df['BsmtFinSF1'])

df['BsmtFinType2']=label.fit_transform(df['BsmtFinType2'])

df['BsmtFinSF2']=label.fit_transform(df['BsmtFinSF2'])

df['BsmtUnfSF']=label.fit_transform(df['BsmtUnfSF'])

df['TotalBsmtSF']=label.fit_transform(df['TotalBsmtSF'])

df['Heating']=label.fit_transform(df['Heating'])

df['HeatingQC']=label.fit_transform(df['HeatingQC'])

df['CentralAir']=label.fit_transform(df['CentralAir'])

df['1stFlrSF']=label.fit_transform(df['1stFlrSF'])

df['2ndFlrSF']=label.fit_transform(df['2ndFlrSF'])

df['LowQualFinSF']=label.fit_transform(df['LowQualFinSF'])

df['GrLivArea']=label.fit_transform(df['GrLivArea'])

df['BsmtFullBath']=label.fit_transform(df['BsmtFullBath'])

df['BsmtHalfBath']=label.fit_transform(df['BsmtHalfBath'])

df['FullBath']=label.fit_transform(df['FullBath'])

df['HalfBath']=label.fit_transform(df['HalfBath'])

df['BedroomAbvGr']=label.fit_transform(df['BedroomAbvGr'])

df['KitchenAbvGr']=label.fit_transform(df['KitchenAbvGr'])

df['KitchenQual']=label.fit_transform(df['KitchenQual'])

df['TotRmsAbvGrd']=label.fit_transform(df['TotRmsAbvGrd'])

df['Functional']=label.fit_transform(df['Functional'])

df['Fireplaces']=label.fit_transform(df['Fireplaces'])

df['FireplaceQu'] =label.fit_transform(df['FireplaceQu'])

df['GarageType']=label.fit_transform(df['GarageType'])

df['GarageYrBlt']=label.fit_transform(df['GarageYrBlt'])

df['GarageFinish']=label.fit_transform(df['GarageFinish'])

df['GarageCars']=label.fit_transform(df['GarageCars'])

df['GarageArea']=label.fit_transform(df['GarageArea'])

df['GarageQual']=label.fit_transform(df['GarageQual'])

df['GarageCond']=label.fit_transform(df['GarageCond'])

df['PavedDrive']=label.fit_transform(df['PavedDrive'])

df['WoodDeckSF']=label.fit_transform(df['WoodDeckSF'])

df['OpenPorchSF']=label.fit_transform(df['OpenPorchSF'])

df['EnclosedPorch'] =label.fit_transform(df['EnclosedPorch'])

df['3SsnPorch']=label.fit_transform(df['3SsnPorch'])

df['ScreenPorch'] =label.fit_transform(df['ScreenPorch'])

df['PoolArea'] =label.fit_transform(df['PoolArea'])

df['PoolQC']=label.fit_transform(df['PoolQC'])

df['Fence']=label.fit_transform(df['Fence'])

df['MiscFeature']=label.fit_transform(df['MiscFeature'])

df['MiscVal']=label.fit_transform(df['MiscVal'])

df['MoSold']=label.fit_transform(df['MoSold'])

df['YrSold']=label.fit_transform(df['YrSold'])

df['SaleType']=label.fit_transform(df['SaleType'])

df['SaleCondition'] = label.fit_transform(df['SaleCondition'])

df['Electrical']=label.fit_transform(df['Electrical'])

df['BsmtExposure'] = label.fit_transform(df['BsmtExposure'].astype(str))
X['Electrical'].unique()
X


X = df.drop(labels=["Id",'SalePrice'],axis=1)

y =df['SalePrice']
#np.log(y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.9,random_state=2000)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

accuracy_score(y_pred,y_test)


# from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(rf,X_test,y_test,cmap=plt.cm.Blues,normalize=None)

# #disp = plot_confusion_matrix(lg,X_test,y_test,cmap='viridis',normalize=None)

# disp.confusion_matrix
k = rf.fit(X, y)

k
random_grid = {'n_estimators': [1,2,3,4,5,6,7,15,20,25,260,600,700,800,900],

               'max_features': ['auto', 'sqrt'],

               'max_depth': [10, 20, 30, 40, 50, 60],

               'min_samples_split': [2, 5, 10],

               'min_samples_leaf': [1, 2, 4],

               'bootstrap': [True, False]}
from sklearn.model_selection import RandomizedSearchCV

randomcv= RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 260, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# randomcv.fit(X,y)

model = RandomForestClassifier(n_estimators=850,min_samples_split=30,min_samples_leaf=4,max_features='auto',max_depth=2,bootstrap=True)
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
nan3=[feature for feature in df_test.columns if df_test[feature].isnull().sum()>0 and df_test[feature].dtypes=='O']

for feature in nan3:

    print("{}: {}% missing values".format(feature,np.round(df_test[feature].isnull().mean(),4)))

    def replace(nan3, df):

        data2 = df_test.copy()

        data2[nan3] = data2[nan3].fillna('Missing')

        return data2

df_test = replace(nan3,df_test)

df_test[nan3].isnull().sum()
nan4=[feature for feature in df_test.columns if df_test[feature].isnull().sum()>0 and df_test[feature].dtypes!='O']



for feature in nan4:

    print("{}: {}% missing values".format(feature,np.round(df_test[feature].isnull().mean(),4)))
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())

df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mean())

df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mean())

df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mean())

df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean())

df['BsmtFullBath'] = df['BsmtFullBath'].fillna(df['BsmtFullBath'].mean())

df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mean())

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())

df['GarageCars'] = df['GarageCars'].fillna(df['GarageCars'].mean())

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
df_test
#df_test['Id'] = label.fit_transform(df_test['Id'])

df_test['MSSubClass'] = label.fit_transform(df_test['MSSubClass'])

df_test['MSZoning'] = label.fit_transform(df_test['MSZoning'])

df_test['LotFrontage'] = label.fit_transform(df_test['LotFrontage'])

df_test['LotArea'] = label.fit_transform(df_test['LotArea'])

df_test['Street'] = label.fit_transform(df_test['Street'])

df_test['Alley'] = label.fit_transform(df_test['Alley'])

df_test['LotShape'] = label.fit_transform(df_test['LotShape'])

df_test['LandContour'] = label.fit_transform(df_test['LandContour'])

df_test['Utilities'] = label.fit_transform(df_test['Utilities'])

df_test['LotConfig'] = label.fit_transform(df_test['LotConfig'])

df_test['LandSlope'] = label.fit_transform(df_test['LandSlope'])

df_test['Neighborhood'] = label.fit_transform(df_test['Neighborhood'])

df_test['Condition1'] = label.fit_transform(df_test['Condition1'])

df_test['Condition2'] = label.fit_transform(df_test['Condition2'])

df_test['BldgType'] = label.fit_transform(df_test['BldgType'])

df_test['HouseStyle'] = label.fit_transform(df_test['HouseStyle'])

df_test['OverallQual'] = label.fit_transform(df_test['OverallQual'])

df_test['OverallCond'] = label.fit_transform(df_test['OverallCond'])

df_test['YearBuilt'] = label.fit_transform(df_test['YearBuilt'])

df_test['YearRemodAdd'] = label.fit_transform(df_test['YearRemodAdd'])

df_test['RoofStyle'] = label.fit_transform(df_test['RoofStyle'])

df_test['RoofMatl'] = label.fit_transform(df_test['RoofMatl'])

df_test['Exterior1st'] = label.fit_transform(df_test['Exterior1st'])

df_test['Exterior2nd'] = label.fit_transform(df_test['Exterior2nd'])

df_test['MasVnrType'] = label.fit_transform(df_test['MasVnrType'])

df_test['MasVnrArea']=label.fit_transform(df_test['MasVnrArea'])

df_test['ExterQual']=label.fit_transform(df_test['ExterQual'])

df_test['ExterCond']=label.fit_transform(df_test['ExterCond'])

df_test['Foundation']=label.fit_transform(df_test['Foundation'])

df_test['BsmtQual']=label.fit_transform(df_test['BsmtQual'])

df_test['BsmtCond']=label.fit_transform(df_test['BsmtCond'])

df_test['Bsmtxposure']=label.fit_transform(df_test['BsmtExposure'].astype(str))

df_test['BsmtFinType1']=label.fit_transform(df_test['BsmtFinType1'])

df_test['BsmtFinSF1']=label.fit_transform(df_test['BsmtFinSF1'])

df_test['BsmtFinType2']=label.fit_transform(df_test['BsmtFinType2'])

df_test['BsmtFinSF2']=label.fit_transform(df_test['BsmtFinSF2'])

df_test['BsmtUnfSF']=label.fit_transform(df_test['BsmtUnfSF'])

df_test['TotalBsmtSF']=label.fit_transform(df_test['TotalBsmtSF'])

df_test['Heating']=label.fit_transform(df_test['Heating'])

df_test['HeatingQC']=label.fit_transform(df_test['HeatingQC'])

df_test['CentralAir']=label.fit_transform(df_test['CentralAir'])

df_test['1stFlrSF']=label.fit_transform(df_test['1stFlrSF'])

df_test['2ndFlrSF']=label.fit_transform(df_test['2ndFlrSF'])

df_test['LowQualFinSF']=label.fit_transform(df_test['LowQualFinSF'])

df_test['GrLivArea']=label.fit_transform(df_test['GrLivArea'])

df_test['BsmtFullBath']=label.fit_transform(df_test['BsmtFullBath'])

df_test['BsmtHalfBath']=label.fit_transform(df_test['BsmtHalfBath'])

df_test['FullBath']=label.fit_transform(df_test['FullBath'])

df_test['HalfBath']=label.fit_transform(df_test['HalfBath'])

df_test['BedroomAbvGr']=label.fit_transform(df_test['BedroomAbvGr'])

df_test['KitchenAbvGr']=label.fit_transform(df_test['KitchenAbvGr'])

df_test['KitchenQual']=label.fit_transform(df_test['KitchenQual'])

df_test['TotRmsAbvGrd']=label.fit_transform(df_test['TotRmsAbvGrd'])

df_test['Functional']=label.fit_transform(df_test['Functional'])

df_test['Fireplaces']=label.fit_transform(df_test['Fireplaces'])

df_test['FireplaceQu'] =label.fit_transform(df_test['FireplaceQu'])

df_test['GarageType']=label.fit_transform(df_test['GarageType'])

df_test['GarageYrBlt']=label.fit_transform(df_test['GarageYrBlt'])

df_test['GarageFinish']=label.fit_transform(df_test['GarageFinish'])

df_test['GarageCars']=label.fit_transform(df_test['GarageCars'])

df_test['GarageArea']=label.fit_transform(df_test['GarageArea'])

df_test['GarageQual']=label.fit_transform(df_test['GarageQual'])

df_test['GarageCond']=label.fit_transform(df_test['GarageCond'])

df_test['PavedDrive']=label.fit_transform(df_test['PavedDrive'])

df_test['WoodDeckSF']=label.fit_transform(df_test['WoodDeckSF'])

df_test['OpenPorchSF']=label.fit_transform(df_test['OpenPorchSF'])

df_test['EnclosedPorch'] =label.fit_transform(df_test['EnclosedPorch'])

df_test['3SsnPorch']=label.fit_transform(df_test['3SsnPorch'])

df_test['ScreenPorch'] =label.fit_transform(df_test['ScreenPorch'])

df_test['PoolArea'] =label.fit_transform(df_test['PoolArea'])

df_test['PoolQC']=label.fit_transform(df_test['PoolQC'])

df_test['Fence']=label.fit_transform(df_test['Fence'])

df_test['MiscFeature']=label.fit_transform(df_test['MiscFeature'])

df_test['MiscVal']=label.fit_transform(df_test['MiscVal'])

df_test['MoSold']=label.fit_transform(df_test['MoSold'])

df_test['YrSold']=label.fit_transform(df_test['YrSold'])

df_test['SaleType']=label.fit_transform(df_test['SaleType'])

df_test['SaleCondition'] = label.fit_transform(df_test['SaleCondition'])

df_test['Electrical']=label.fit_transform(df_test['Electrical'])

df_test['BsmtExposure'] = label.fit_transform(df_test['BsmtExposure'].astype(str))
df_test.head()
test_data = df_test.drop("Id", axis=1).copy()

#prediction = rf.predict(test_data)

test_data
df_test['BsmtExposure'] = label.fit_transform(df_test['BsmtExposure'])
model.fit(X,y)
prediction = model.predict(test_data)
df_test['Id']
# if (df_test.isnull() == True).all:

#     print(df_test.isnull())
submission = pd.DataFrame({

        "Id": df_test["Id"],

        "SalePrice": prediction

    })
submission.to_csv('sample_submission.csv')