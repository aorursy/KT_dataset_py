%%time

# Code you have previously used to load data

import time

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

#from sklearn.pipeline import make_pipeline

#from sklearn.ensemble import RandomForestClassifier

#from sklearn.model_selection import cross_val_score

#from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingRegressor

#from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

#from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.width', None)

# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

#iowa_file_path = '../input/train.csv'



#home_data = pd.read_csv(iowa_file_path)
#home_data=home_data.rename(columns={'1stFlrSF':'FirstFlrSF', '2ndFlrSF' : 'SecondFlrSF'} )
#pandas doesn't show us all the decimals

#pd.options.display.precision = 15
#home_data._get_numeric_data().head()
# Reading the file



iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Renaming the columns



home_data=home_data.rename(columns={'1stFlrSF':'FirstFlrSF', '2ndFlrSF' : 'SecondFlrSF'} )



# Filling the NAN values according to set metric , The Mean Absolute Error



home_data['MasVnrType'].unique()

home_data['MasVnrType']=home_data['MasVnrType'].fillna('BrkCmn')

home_data['MasVnrType']=home_data['MasVnrType'].fillna('BrkCmn')

home_data['MasVnrArea']=home_data['MasVnrArea'].fillna(0)

home_data['MiscFeature'].unique()

home_data['MiscFeature']=home_data['MiscFeature'].fillna('TenC')

home_data['Fence'].unique()

home_data['Fence']=home_data['Fence'].fillna('MnWw')

home_data['LotFrontage'].describe()

home_data['LotFrontage']=home_data['LotFrontage'].fillna(70)

home_data['PoolQC'].unique()

home_data['PoolQC']=home_data['PoolQC'].fillna('Gd')

home_data['GarageCond'].unique()

home_data['GarageCond']=home_data['GarageCond'].fillna('Ex')

home_data['GarageQual'].unique()

home_data['GarageQual']=home_data['GarageQual'].fillna('Po')

home_data['GarageFinish'].unique()

home_data['GarageFinish']=home_data['GarageFinish'].fillna('Fin')

home_data['GarageYrBlt'].describe()

home_data['GarageYrBlt']=home_data['GarageYrBlt'].fillna(1978)

home_data['GarageType'].unique()

home_data['GarageType']=home_data['GarageType'].fillna('2Types')

home_data['FireplaceQu'].unique()

home_data['FireplaceQu']=home_data['FireplaceQu'].fillna('Po')

home_data['Electrical'].unique()

home_data['Electrical']=home_data['Electrical'].fillna('FuseA')

home_data['BsmtFinType2'].unique()

home_data['BsmtFinType2']=home_data['BsmtFinType2'].fillna('BLQ')

home_data['BsmtFinType1'].unique()

home_data['BsmtFinType1']=home_data['BsmtFinType1'].fillna('LwQ')

home_data['BsmtExposure'].unique() 

home_data['BsmtExposure']=home_data['BsmtExposure'].fillna('Av')

home_data['BsmtCond'].unique()

home_data['BsmtCond']=home_data['BsmtCond'].fillna('Po')

home_data['Alley'].unique()

home_data['Alley']= home_data['Alley'].fillna('Pave')

home_data['BsmtQual'].unique()

home_data['BsmtQual']=home_data['BsmtQual'].fillna('Fa')





# Label Encoding (Transforming non-categorical values into tabular form)



from sklearn.preprocessing import LabelEncoder



for f in ['MSZoning','Street', 'Alley', 'LotShape','LandContour','Utilities','LotConfig','LandSlope',\

          'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',\

         'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',\

          'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',\

         'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',\

          'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',\

         'PoolQC','Fence','MiscFeature','SaleType','SaleCondition']:

    lbl = LabelEncoder()

    lbl.fit(list(home_data[f].values) + list(home_data[f].values))

    home_data[f] = lbl.transform(list(home_data[f].values))

    #test_data[f] = lbl.transform(list(test_data[f].values))

import seaborn as sns

import matplotlib.pyplot as plt

var = home_data['LotArea']

f, ax = plt.subplots(figsize=(13,8))

sns.scatterplot(y=home_data.SalePrice, x=var)

plt.show()
home_data['LotArea'].describe()
# Give some values to the data which is showing an anomaly



home_data.LotArea[home_data.LotArea >= 200000] =  11601.500000000000000

home_data.LotArea[home_data.LotArea >= 160000] = 45600.38

#home_data.MiscVal[home_data.MiscVal>= 70_000] = 250000

#home_data.LotArea[home_data.LotArea >= 150000] = 59600

#home_data.LotArea[home_data.LotArea >= 110000] = 60800

#home_data.LotFrontage[home_data.LotFrontage >= 300] =195

#home_data.MasVnrArea[home_data.MasVnrArea >= 1000] =500.25

#home_data.BsmtFinSF1[home_data.BsmtFinSF1 >= 5000] = 2200

#home_data.BsmtFinSF2[home_data.BsmtFinSF2 >= 1400] = 1150.32

#home_data.TotalBsmtSF[home_data.TotalBsmtSF >= 6000] = 2750.32
#home_data.loc[home_data['MiscVal'] >=4000]

#home_data['LotArea'].describe()

#home_data.Neighborhood[home_data.Neighborhood>= 70_000] = 250000
#home_data = home_data.drop([1230], axis=0)
#home_data['MiscVal'].describe()
#home_data.MiscVal[home_data.MiscVal >= 4_000] = 1000
### View of Correlation Matrix

import seaborn as sns

import matplotlib.pyplot as plt

corr_matrix=home_data._get_numeric_data().corr()

fig, ax = plt.subplots(figsize=(30,20))         # Sample figsize in inches

sns.heatmap(corr_matrix, annot=False, linewidths=5, ax=ax, xticklabels=corr_matrix.columns.values,yticklabels=corr_matrix.columns.values)

#sns.heatmap(corr, annot=True, fmt=".1f",linewidth=0.5 xticklabels=corr.columns.values,yticklabels=corr.columns.values)
home_data._get_numeric_data().corr()
from scipy import stats



#1

pearson_coef, p_value = stats.pearsonr(home_data['MSZoning'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient 'MSZoning' is", pearson_coef, " with a P-value of P =", p_value)



#2

pearson_coef, p_value = stats.pearsonr(home_data['Street'], home_data['SalePrice'])

print("#2 The Pearson Correlation Coefficient Street is", pearson_coef, " with a P-value of P =", p_value)



#3

pearson_coef, p_value = stats.pearsonr(home_data['Alley'], home_data['SalePrice'])

print("#3 The Pearson Correlation Coefficient 'Alley' is", pearson_coef, " with a P-value of P =", p_value)



#4

pearson_coef, p_value = stats.pearsonr(home_data['LotShape'], home_data['SalePrice'])

print("#4 The Pearson Correlation Coefficient 'LotShape' is", pearson_coef, " with a P-value of P =", p_value)



#5

pearson_coef, p_value = stats.pearsonr(home_data['LandContour'], home_data['SalePrice'])

print("#5 The Pearson Correlation Coefficient 'LandContour is", pearson_coef, " with a P-value of P =", p_value)



#6

pearson_coef, p_value = stats.pearsonr(home_data['Utilities'], home_data['SalePrice'])

print("#6 The Pearson Correlation Coefficient 'Utilities' is", pearson_coef, " with a P-value of P =", p_value)



#7

pearson_coef, p_value = stats.pearsonr(home_data['LotConfig'], home_data['SalePrice'])

print("#7 The Pearson Correlation Coefficient 'LotConfig' is", pearson_coef, " with a P-value of P =", p_value)



#8

pearson_coef, p_value = stats.pearsonr(home_data['LandSlope'], home_data['SalePrice'])

print("#8 The Pearson Correlation Coefficient 'LandSlope' is", pearson_coef, " with a P-value of P =", p_value)



#9

pearson_coef, p_value = stats.pearsonr(home_data['Neighborhood'], home_data['SalePrice'])

print("#9 The Pearson Correlation Coefficient 'Neighborhood' is", pearson_coef, " with a P-value of P =", p_value)



#10

pearson_coef, p_value = stats.pearsonr(home_data['Condition1'], home_data['SalePrice'])

print("#10 The Pearson Correlation Coefficient 'Condition1' is", pearson_coef, " with a P-value of P =", p_value)



#11

pearson_coef, p_value = stats.pearsonr(home_data['Condition2'], home_data['SalePrice'])

print("#11 The Pearson Correlation Coefficient'Condition2' is", pearson_coef, " with a P-value of P =", p_value)



#12

pearson_coef, p_value = stats.pearsonr(home_data['BldgType'], home_data['SalePrice'])

print("#12 The Pearson Correlation Coefficient 'BldgType' is", pearson_coef, " with a P-value of P =", p_value)



#13

pearson_coef, p_value = stats.pearsonr(home_data['HouseStyle'], home_data['SalePrice'])

print("#13 The Pearson Correlation Coefficient 'HouseStyle' is", pearson_coef, " with a P-value of P =", p_value)



#14

pearson_coef, p_value = stats.pearsonr(home_data['RoofStyle'], home_data['SalePrice'])

print("#14 The Pearson Correlation Coefficient 'RoofStyle' is", pearson_coef, " with a P-value of P =", p_value)



#15

pearson_coef, p_value = stats.pearsonr(home_data['RoofMatl'], home_data['SalePrice'])

print("#15 The Pearson Correlation Coefficient 'RoofMatl' is", pearson_coef, " with a P-value of P =", p_value)



#16

pearson_coef, p_value = stats.pearsonr(home_data['Exterior1st'], home_data['SalePrice'])

print("#16 The Pearson Correlation Coefficient 'Exterior1st' is", pearson_coef, " with a P-value of P =", p_value)

            

                                                 

#17

pearson_coef, p_value = stats.pearsonr(home_data['Exterior2nd'], home_data['SalePrice'])

print("#17 The Pearson Correlation Coefficient 'Exterior2nd'is", pearson_coef, " with a P-value of P =", p_value)



#18

pearson_coef, p_value = stats.pearsonr(home_data['MasVnrType'], home_data['SalePrice'])

print("#18 The Pearson Correlation Coefficient 'MasVnrType' is", pearson_coef, " with a P-value of P =", p_value)



#19

pearson_coef, p_value = stats.pearsonr(home_data['ExterQual'], home_data['SalePrice'])

print("#19 The Pearson Correlation Coefficient'ExterQual' is", pearson_coef, " with a P-value of P =", p_value)



#20

pearson_coef, p_value = stats.pearsonr(home_data['ExterCond'], home_data['SalePrice'])

print("#20 The Pearson Correlation Coefficient 'ExterCond' is", pearson_coef, " with a P-value of P =", p_value)



#21

pearson_coef, p_value = stats.pearsonr(home_data[ 'Foundation'], home_data['SalePrice'])

print("#21 The Pearson Correlation Coefficient  'Foundation' is", pearson_coef, " with a P-value of P =", p_value)



#22

pearson_coef, p_value = stats.pearsonr(home_data['BsmtQual'], home_data['SalePrice'])

print("#22 The Pearson Correlation Coefficient 'BsmtQual' is", pearson_coef, " with a P-value of P =", p_value)



#23

pearson_coef, p_value = stats.pearsonr(home_data['BsmtCond'], home_data['SalePrice'])

print("#23 The Pearson Correlation Coefficient 'Alley' is", pearson_coef, " with a P-value of P =", p_value)



#24

pearson_coef, p_value = stats.pearsonr(home_data['BsmtExposure'], home_data['SalePrice'])

print("#24 The Pearson Correlation Coefficient 'BsmtExposure' is", pearson_coef, " with a P-value of P =", p_value)



#25

pearson_coef, p_value = stats.pearsonr(home_data['BsmtFinType1'], home_data['SalePrice'])

print("#25 The Pearson Correlation Coefficient 'BsmtFinType1' is", pearson_coef, " with a P-value of P =", p_value)



#26

pearson_coef, p_value = stats.pearsonr(home_data['BsmtFinType2'], home_data['SalePrice'])

print("#26 The Pearson Correlation Coefficient 'BsmtFinType2' is", pearson_coef, " with a P-value of P =", p_value)



#27

pearson_coef, p_value = stats.pearsonr(home_data['Heating'], home_data['SalePrice'])

print("#27 The Pearson Correlation Coefficient 'Heating' is", pearson_coef, " with a P-value of P =", p_value)



#28

pearson_coef, p_value = stats.pearsonr(home_data['HeatingQC'], home_data['SalePrice'])

print("#28 The Pearson Correlation Coefficient 'HeatingQC' is", pearson_coef, " with a P-value of P =", p_value)



#29

pearson_coef, p_value = stats.pearsonr(home_data['CentralAir'], home_data['SalePrice'])

print("#29 The Pearson Correlation Coefficient 'CentralAir' is", pearson_coef, " with a P-value of P =", p_value)



#30

pearson_coef, p_value = stats.pearsonr(home_data['Electrical'], home_data['SalePrice'])

print("#30 The Pearson Correlation Coefficient 'Electrical' is", pearson_coef, " with a P-value of P =", p_value)



#31

pearson_coef, p_value = stats.pearsonr(home_data['KitchenQual'], home_data['SalePrice'])

print("#31 The Pearson Correlation Coefficient 'KitchenQual is", pearson_coef, " with a P-value of P =", p_value)



#32

pearson_coef, p_value = stats.pearsonr(home_data['Functional'], home_data['SalePrice'])

print("#32 The Pearson Correlation Coefficient 'Functional' is", pearson_coef, " with a P-value of P =", p_value)



#33

pearson_coef, p_value = stats.pearsonr(home_data['FireplaceQu'], home_data['SalePrice'])

print("#33 The Pearson Correlation Coefficient 'FireplaceQu' is", pearson_coef, " with a P-value of P =", p_value)



#34

pearson_coef, p_value = stats.pearsonr(home_data['GarageType'], home_data['SalePrice'])

print("#34 The Pearson Correlation Coefficient 'GarageType' is", pearson_coef, " with a P-value of P =", p_value)



#35

pearson_coef, p_value = stats.pearsonr(home_data['GarageFinish'], home_data['SalePrice'])

print("#35 The Pearson Correlation Coefficient 'GarageFinish' is", pearson_coef, " with a P-value of P =", p_value)



#36

pearson_coef, p_value = stats.pearsonr(home_data['GarageQual'], home_data['SalePrice'])

print("#36 The Pearson Correlation Coefficient 'GarageQual' is", pearson_coef, " with a P-value of P =", p_value)

                                            

#37

pearson_coef, p_value = stats.pearsonr(home_data['GarageCond'], home_data['SalePrice'])

print("#37 The Pearson Correlation Coefficient 'GarageCond'is", pearson_coef, " with a P-value of P =", p_value)



#38

pearson_coef, p_value = stats.pearsonr(home_data['PavedDrive'], home_data['SalePrice'])

print("#38 The Pearson Correlation Coefficient 'PavedDrive' is", pearson_coef, " with a P-value of P =", p_value)



#39

pearson_coef, p_value = stats.pearsonr(home_data[ 'PoolQC'], home_data['SalePrice'])

print("#39 The Pearson Correlation Coefficient'PoolQC' is", pearson_coef, " with a P-value of P =", p_value)



#40

pearson_coef, p_value = stats.pearsonr(home_data['Fence'], home_data['SalePrice'])

print("#40 The Pearson Correlation Coefficient 'Fence' is", pearson_coef, " with a P-value of P =", p_value)



#41

pearson_coef, p_value = stats.pearsonr(home_data['MiscFeature'], home_data['SalePrice'])

print("#41 The Pearson Correlation Coefficient 'MiscFeature' is", pearson_coef, " with a P-value of P =", p_value)



#42

pearson_coef, p_value = stats.pearsonr(home_data['SaleType'], home_data['SalePrice'])

print("#42 The Pearson Correlation Coefficient 'SaleType' is", pearson_coef, " with a P-value of P =", p_value)



#43

pearson_coef, p_value = stats.pearsonr(home_data['SaleCondition'], home_data['SalePrice'])

print("#43 The Pearson Correlation Coefficient'SaleCondition' is", pearson_coef, " with a P-value of P =", p_value)



from scipy import stats

#1

pearson_coef, p_value = stats.pearsonr(home_data['MSSubClass'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient MSSubClass is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['LotFrontage'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient LotFrontage is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['LotArea'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient LotArea is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['OverallQual'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient OverallQual is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['OverallCond'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient OverallCond is", pearson_coef, " with a P-value of P =", p_value)



#1 

pearson_coef, p_value = stats.pearsonr(home_data['YearBuilt'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient YearBuilt is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['YearRemodAdd'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient YearRemodAdd is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['MasVnrArea'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient MasVnrArea is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['BsmtFinSF1'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient BsmtFinSF1 is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['BsmtFinSF2'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient BsmtFinSF2 is", pearson_coef, " with a P-value of P =", p_value)





#1

pearson_coef, p_value = stats.pearsonr(home_data['EnclosedPorch'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient EnclosedPorch is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['3SsnPorch'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient BedroomAbvGr is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['ScreenPorch'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient ScreenPorch is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['PoolArea'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient PoolArea is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['MiscVal'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient MiscVal is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['MoSold'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient MoSold is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['YrSold'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient YrSold is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['HalfBath'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient HalfBath is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['BedroomAbvGr'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient BedroomAbvGr is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['KitchenAbvGr'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient KitchenAbvGr is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['TotRmsAbvGrd'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient TotRmsAbvGrd is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['Fireplaces'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient Fireplaces is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['GarageYrBlt'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient GarageYrBlt is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['GarageCars'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient GarageCars is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['GarageArea'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient GarageArea is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['WoodDeckSF'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient WoodDeckSF is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['OpenPorchSF'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient OpenPorchSF is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['BsmtUnfSF'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient BsmtUnfSF is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['BsmtFinSF2'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient BsmtFinSF2 is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['TotalBsmtSF'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient TotalBsmtSF is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['FirstFlrSF'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient 1stFlrSF is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['SecondFlrSF'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient 2ndFlrSF is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['LowQualFinSF'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient LowQualFinSF is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['GrLivArea'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient GrLivArea is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['BsmtFullBath'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient BsmtFullBath is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['BsmtHalfBath'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient BsmtHalfBath is", pearson_coef, " with a P-value of P =", p_value)



#1

pearson_coef, p_value = stats.pearsonr(home_data['FullBath'], home_data['SalePrice'])

print("#1 The Pearson Correlation Coefficient FullBath is", pearson_coef, " with a P-value of P =", p_value)

# Selected Features



X=home_data[['MSZoning','Street', 'Alley', 'LotShape','LandContour','Utilities','LandSlope',\

          'Condition1','Condition2','BldgType','HouseStyle','LotConfig', 'Neighborhood',\

         'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual','ExterCond',\

          'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',\

         'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',\

          'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',\

         'PoolQC','Fence','MiscFeature','SaleType','SaleCondition','LotFrontage',\

             'BsmtFinSF2','BsmtFinSF1','LotArea', 'OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea',\

            'BsmtUnfSF','TotalBsmtSF','FirstFlrSF','SecondFlrSF','LowQualFinSF',\

             'BsmtFullBath','GrLivArea','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',\

             'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea',\

             'WoodDeckSF','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','PoolArea',\

             'MiscVal','MoSold','YrSold',

           ]]



# Target Variable



y=home_data.SalePrice



# Split into validation and training data



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

#iowa_model = RandomForestRegressor(random_state=0,n_estimators=110,max_depth=None,max_leaf_nodes=324,

#                                              n_jobs=-1,min_samples_leaf=1) #110,325

                                           

#iowa_model = XGBRegressor(random_state=1,max_depth=6,max_leaf_nodes=None

                         # , learning_rate=0.3,n_estimators=500,n_jobs=4,objective="reg:linear")

                                       



# Rough Model to Start With       



iowa_model = GradientBoostingRegressor(random_state=0,max_depth=3,max_leaf_nodes=None,n_estimators=100,

                                       learning_rate=0.08,min_samples_leaf=1,

                                       )

                                           

# Fit Model



iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error



val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(max_leaf_nodes=max_leaf_nodes,random_state=1,max_depth=3,

                                      learning_rate=0.08)

                                     # n_estimators=110, random_state=1,n_jobs=-1)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y,preds_val)

    return(mae)



candidate_max_leaf_nodes = [5, 25, 50,75, 100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,600,700,800,900,1000]#[5, 25, 50,75, 100,125,150,175,200, 250, 500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for max_leaf_nodes in candidate_max_leaf_nodes:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))



#rf_model_on_full_data = RandomForestRegressor(random_state=1,max_leaf_nodes=100,n_estimators=150)

                                             # n_jobs=-1)



# fit rf_model_on_full_data on all data from the training data

#rf_model_on_full_data.fit(X,y)

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,

                                      max_leaf_nodes=5,learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [5,25,50,75,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,\

                     175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,max_leaf_nodes=5,\

                                      learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,

                     460,470,480,490,500,510,520,530,645,650,655,660,665,670,675,680,685,690,695,700]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,max_leaf_nodes=5,\

                                      learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [500,505,510,515,520,525,530,535,540,545,550,555,560,565,570,575,580,585,590,595,600,\

                    605,610,615,620,625,630,635,640,645,650,655,660,665,670,675,680,685,690,695,700]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,

                                      max_leaf_nodes=5,learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [700,705,710,715,720,725,730,735,740,745,750,755,760,765,770,775,780,785,790,\

                     795,800,805,810,815,820,825,830,835,840,845,850,855,860,865,870,875,880,885,\

                     890,895,900]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,

                                      max_leaf_nodes=5,learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [900,905,910,915,920,925,930,935,940,945,950,955,960,965,970,975,980,985,990,\

                     995,1000,1005,1010,1015,1020,1025,1030,1035,1040,1045,1050,1055,1060,1065,\

                     1070,1075,1080,1085,\

                     1090,1095,1100]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,

                                      max_leaf_nodes=5,learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [1100,1105,1110,1115,1120,1125,1130,1135,1140,1145,1150,1155,1160,1165,1170,\

                     1175,1180,1185,1190,\

                     1195,1200,1205,1210,1215,1220,1225,1230,1235,1240,1245,1250,1255,1260,1265,\

                     1270,1275,1280,1285,\

                     1290,1295,1200]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,

                                      max_leaf_nodes=5,learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [1200,1205,1210,1215,1220,1225,1230,1235,1240,1245,1250,1255,1260,1265,1270,\

                     1275,1280,1285,1290,\

                     1295,1300,1305,1310,1315,1320,1325,1330,1335,1340,1345,1350,1355,1360,1365,\

                     1370,1375,1380,1385,\

                     1390,1395,1400]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,

                                      max_leaf_nodes=5,learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,2400,2500,2600,2700,2800,

                     2900,3000]

                    

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=0,max_depth=3,

                                      max_leaf_nodes=5,learning_rate=0.08) 

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



find_n_estimators = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,14000,15000,16000,17000,

                     18000,1900,2000,30000]

                    

# Write loop to find the ideal tree size from candidate_max_leaf_nodes

for n_estimators in find_n_estimators:

    my_mae = get_mae(n_estimators, train_X, val_X, train_y, val_y)

    print("n_estimators: %d  \t\t Mean Absolute Error:  %d" %(n_estimators, my_mae))

rf_model_on_full_data = GradientBoostingRegressor(random_state=0,max_leaf_nodes=5,n_estimators=625,

                                              max_depth=3, learning_rate=0.08)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)
# Reading the file



test_data_path = '../input/test.csv'

test_data = pd.read_csv(test_data_path)

#test_data.head()

#test_data['TotalBsmtSF'].unique()



# Renaming Columns



test_data=test_data.rename(columns={'1stFlrSF':'FirstFlrSF', '2ndFlrSF' : 'SecondFlrSF'} )



# Filling NAN values



test_data['SaleType']=test_data['SaleType'].fillna('WD')

test_data['PoolQC']=test_data['PoolQC'].fillna('Gd')

test_data['Fence']=test_data['Fence'].fillna('MnWw')

test_data['MiscFeature']=test_data['MiscFeature'].fillna('TenC')

test_data['GarageQual']=test_data['GarageQual'].fillna('Po')

test_data['GarageCond']=test_data['GarageCond'].fillna('Ex')

test_data['GarageArea']=test_data['GarageArea'].fillna(472.768861)

test_data['GarageCars']=test_data['GarageCars'].fillna(1.766118)

test_data['GarageYrBlt']=test_data['GarageYrBlt'].fillna(1978)

test_data['FireplaceQu']=test_data['FireplaceQu'].fillna('Po')

test_data['GarageType']=test_data['GarageType'].fillna('2Types')

test_data['GarageFinish']=test_data['GarageFinish'].fillna('Fin')

test_data['Functional']=test_data['Functional'].fillna('Min2') 

test_data['KitchenQual']=test_data['KitchenQual'].fillna('TA')

test_data['BsmtFinSF1']=test_data['BsmtFinSF1'].fillna(439.203704)

test_data['BsmtFinSF2']=test_data['BsmtFinSF2'].fillna(52.619342)

test_data['TotalBsmtSF']=test_data['TotalBsmtSF'].fillna(1046.117970)

test_data['BsmtFullBath']=test_data['BsmtFullBath'].fillna( 0.434454)

test_data['BsmtHalfBath']=test_data['BsmtHalfBath'].fillna(0.065202)

test_data['LotFrontage']=test_data['LotFrontage'].fillna(70)

test_data['Alley']= test_data['Alley'].fillna('Pave')

test_data['Utilities']=test_data['Utilities'].fillna('AllPub')

test_data['Exterior1st']=test_data['Exterior1st'].fillna('CemntBd') 

test_data['Exterior1st']=test_data['Exterior2nd'].fillna('HdBoard')

test_data['MasVnrType']=test_data['MasVnrType'].fillna('BrkCmn')

test_data['MasVnrArea']=test_data['MasVnrArea'].fillna(0)

test_data['BsmtQual']=test_data['BsmtQual'].fillna('Fa')

test_data['BsmtCond']=test_data['BsmtCond'].fillna('Po')

test_data['BsmtExposure']=test_data['BsmtExposure'].fillna('Av')

test_data['BsmtFinType1']=test_data['BsmtFinType1'].fillna('LwQ')

test_data['BsmtFinType2']=test_data['BsmtFinType2'].fillna('BLQ')

test_data['MSZoning']=test_data['MSZoning'].fillna('RH')

test_data['Exterior2nd']=test_data['Exterior2nd'].fillna('Plywood')

test_data['BsmtUnfSF']=test_data['BsmtUnfSF'].fillna(554.294925)



#  Label Encoding



for f in ['MSZoning','Street', 'Alley', 'LotShape','LandContour','Utilities','LotConfig','LandSlope',\

          'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',\

         'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',\

          'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',\

         'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',\

          'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',\

         'PoolQC','Fence','MiscFeature','SaleType','SaleCondition']:

    lbl = LabelEncoder()

    lbl.fit(list(test_data[f].values) + list(test_data[f].values))

    #home_data[f] = lbl.transform(list(home_data[f].values))

    test_data[f] = lbl.transform(list(test_data[f].values))

    

# Features



test_X=test_data[['MSZoning','Street', 'Alley', 'LotShape','LandContour','Utilities','LandSlope',\

         'Condition1','Condition2','BldgType','HouseStyle','LotConfig', 'Neighborhood',\

         'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual','ExterCond',\

          'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',\

         'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',\

          'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',\

         'PoolQC','Fence','MiscFeature','SaleType','SaleCondition','LotFrontage',\

             'BsmtFinSF2','BsmtFinSF1','LotArea', 'OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea',\

            'BsmtUnfSF','TotalBsmtSF','FirstFlrSF','SecondFlrSF','LowQualFinSF',\

             'BsmtFullBath','GrLivArea','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',\

             'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea',\

             'WoodDeckSF','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','PoolArea',\

             'MiscVal','MoSold','YrSold',

           ]]

                 





# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring



output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)