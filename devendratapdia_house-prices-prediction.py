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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Metrics used for measuring the accuracy and performance of the models

from sklearn import metrics

from sklearn.metrics import mean_squared_error





# In[465]:





# Algorithms used for modeling

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb



# Pipeline and scaling preprocessing will be used for models that are sensitive

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



# Set visualisation colours

mycols = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]

sns.set_palette(palette = mycols, n_colors = 4)



# To ignore annoying warning

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

warnings.filterwarnings("ignore", category=DeprecationWarning)





train = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.shape, test.shape
# Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



# Now drop the  'Id' column as it's redundant for modeling

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)





# In[469]:





train.shape, test.shape





# In[470]:





print ("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()





# In[471]:





# Now we use np.log() to transform train.SalePrice and calculate the skewness a second time, 

target = np.log(train.SalePrice)



# as well as re-plot the data. A value closer to 0 means that we have improved the skewness of the data.

print ("Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()
# # Clear out Outliers



# In[472]:





# Lets take GrLivArea

#plt.subplots_adjust(left=2, right=4)

plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)

g = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=False).set_title("Before")



# Delete outliers

plt.subplot(1, 2, 2)                                                                                



train = train.drop(train[(train['GrLivArea']>4000)].index)



g = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=False).set_title("After")





# # Lets create All data merging train and test



# In[473]:





ntrain = train.shape[0]

ntest = test.shape[0]



y_train = train.SalePrice.values



# concatenate training and test data into all_data

all_data = pd.concat((train, test)).reset_index(drop=True)



#all_data.drop(['SalePrice'], axis=1, inplace=True)



print("all_data shape: {}".format(all_data.shape))





# # Replace Missing or nan or null values



# In[474]:





all_data.isna().sum().sort_values(ascending=False)[:35]

#all_data_na = all_data.isnull().sum()

#all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
# In[475]:





# We will seperate into Numeric and Non-numeric cols for replacing na values





# In[476]:





# NA Columns where we can replace with None/0

na_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",

           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MSSubClass", "MasVnrType", "MSSubClass"]

na_col_types = [{col: all_data[col].dtype} for col in na_cols]

na_col_types





# In[477]:





# Except MSSubClass all are object, But MSSubClass values are also categorical, so we will add One Hot encoding later to this

# So we can replace with 'None' for now

for col in na_cols:

    all_data[col] = all_data[col].fillna("None")





# In[478]:





# MSSubClass, fill with 0

# all_data['MSSubClass'] = all_data[col].fillna(0)





# In[479]:





# Numeric nan Cols

num_na_cols = ["GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 

               "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea", "BsmtFullBath", "BsmtHalfBath"]

na_col_types2 = [{col: all_data[col].dtype} for col in num_na_cols]

na_col_types2





# In[480]:





# Using data description, fill these missing values with 0 

for col in num_na_cols:

    all_data[col] = all_data[col].fillna(0)





# # LotFrontage



# In[481]:





# The area of the lot out front is likely to be similar to the houses in the local neighbourhood

# let's use the median value of the houses in the neighbourhood to fill this feature

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))





# # Other Cols: MSZoning, Electrical, KitchenQual, Exterior1st, Exterior2nd, SaleType, Functional



# In[482]:





# Fill these features with their mode, the most commonly occuring value. This is okay since there are a low number of missing values for these features



all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna(all_data['Functional'].mode()[0])

print("'mode' - treated...")



all_data_na = all_data.isnull().sum()

print("Features with missing values: ", all_data_na.drop(all_data_na[all_data_na == 0].index))





# # Drop Utilities Column



# In[483]:





plt.subplots(figsize =(15, 5))



plt.subplot(1, 2, 1)

g = sns.countplot(x = "Utilities", data = train).set_title("Utilities - Training")



plt.subplot(1, 2, 2)

g = sns.countplot(x = "Utilities", data = test).set_title("Utilities - Test")





# In[484]:





# From inspection, we can remove Utilities as it is applicable in only 1 case, others have same value

all_data = all_data.drop(['Utilities'], axis=1)





# In[485]:





# Now our data is in all_data, not train and its clean

sns.heatmap(all_data.isna(), yticklabels=False, cmap='viridis')





# In[486]:





train.isna().sum().sort_values(ascending=False)





# # Exploratory Data Analysis



# # Correlation matrix



# In[487]:





# Correlation is on numeric cols only

corr = train.corr()





# In[488]:





plt.subplots(figsize=(25, 25))

cmap = sns.diverging_palette(150, 250, as_cmap=True)

sns.heatmap(corr, cmap="RdYlBu", vmax=1, vmin=-0.6, center=0.2, square=True, linewidths=0, cbar_kws={"shrink": .5}, annot = True);





# In[489]:





#train.select_dtypes(include=[np.number])





# In[490]:





# First 15: most postively correlated no year ccolumns and 1st is SalePrice

print (corr['SalePrice'].sort_values(ascending=False)[:19], '\n')





# # Using the correlation matrix, the top influencing factors that I will use to create polynomials are:

# 1. **OverallQual**

# 2. **GrLivArea**

# 3. **GarageCars**

# 4. **GarageArea**

# 5. **TotalBsmtSF**

# 6. **1stFlrSF**

# 7. **FullBath**

# 8. **TotRmsAbvGrd**

# 9. **Fireplaces**

# 10. **MasVnrArea**

# 11. **BsmtFinSF1**

# 12. **LotFrontage**

# 13. **WoodDeckSF**

# 14. **OpenPorchSF**

# 15. **2ndFlrSF**



# # 1stFlrSF and 2ndFlrSF



# In[491]:





all_data['1stFlrSF'] = pd.cut(all_data["1stFlrSF"], np.arange(0, 3000, 500), labels=[0, 500, 1000, 1500, 2000])

all_data['2ndFlrSF'] = pd.cut(all_data["2ndFlrSF"], np.arange(0, 2500, 500), labels=[0, 500, 1000, 1500])





# # TotalBsmtSF



# In[492]:





all_data['TotalBsmtSF'] = pd.cut(all_data["TotalBsmtSF"], np.arange(0, 3000, 500), labels=[0, 500, 1000, 1500, 2000])





# In[493]:





all_data['BsmtQual'] = all_data['BsmtQual'].map({"None":0, "Fa":1, "TA":2, "Gd":3, "Ex":4})

all_data['BsmtCond'] = all_data['BsmtCond'].map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

all_data['BsmtExposure'] = all_data['BsmtExposure'].map({"None":0, "No":1, "Mn":2, "Av":3, "Gd":4})

all_data = pd.get_dummies(all_data, columns = ["BsmtFinType1"], prefix="BsmtFinType1")

all_data = pd.get_dummies(all_data, columns = ["BsmtFinType2"], prefix="BsmtFinType2")





# In[494]:





all_data['BsmtFinSf2_Flag'] = all_data['BsmtFinSF2'].map(lambda x:0 if x==0 else 1)

all_data.drop('BsmtFinSF2', axis=1, inplace=True)





# In[495]:





all_data['BsmtFinSF1_Band'] = pd.cut(all_data['BsmtFinSF1'], 4)

all_data.loc[all_data['BsmtFinSF1']<=1002.5, 'BsmtFinSF1'] = 1

all_data.loc[(all_data['BsmtFinSF1']>1002.5) & (all_data['BsmtFinSF1']<=2005), 'BsmtFinSF1'] = 2

all_data.loc[(all_data['BsmtFinSF1']>2005) & (all_data['BsmtFinSF1']<=3007.5), 'BsmtFinSF1'] = 3

all_data.loc[all_data['BsmtFinSF1']>3007.5, 'BsmtFinSF1'] = 4

all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].astype(int)



all_data.drop('BsmtFinSF1_Band', axis=1, inplace=True)



all_data = pd.get_dummies(all_data, columns = ["BsmtFinSF1"], prefix="BsmtFinSF1")





# In[496]:





all_data['BsmtUnfSF_Band'] = pd.cut(all_data['BsmtUnfSF'], 3)

all_data.loc[all_data['BsmtUnfSF']<=778.667, 'BsmtUnfSF'] = 1

all_data.loc[(all_data['BsmtUnfSF']>778.667) & (all_data['BsmtUnfSF']<=1557.333), 'BsmtUnfSF'] = 2

all_data.loc[all_data['BsmtUnfSF']>1557.333, 'BsmtUnfSF'] = 3

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].astype(int)



all_data.drop('BsmtUnfSF_Band', axis=1, inplace=True)



all_data = pd.get_dummies(all_data, columns = ["BsmtUnfSF"], prefix="BsmtUnfSF")





# In[497]:





all_data = pd.get_dummies(all_data, columns = ["1stFlrSF"], prefix="1stFlrSF")

all_data = pd.get_dummies(all_data, columns = ["2ndFlrSF"], prefix="2ndFlrSF")

all_data = pd.get_dummies(all_data, columns = ["TotalBsmtSF"], prefix="TotalBsmtSF")





# In[498]:





all_data['LotArea'] = pd.cut(all_data["LotArea"], np.arange(0, 30000, 5000), labels=[0, 500, 1000, 1500, 2000])

#all_data['MasVnrArea'] = pd.cut(all_data["MasVnrArea"], np.arange(0, 1000, 200), labels=[0, 200, 400, 600])

#all_data['LotFrontage'] = pd.cut(all_data["LotFrontage"], np.arange(0, 200, 25), labels=[0, 25, 50, 75, 100, 125, 150])

all_data['GarageYrBlt'] = pd.cut(all_data["GarageYrBlt"], np.arange(1900, 2040, 20), labels=[1900, 1920, 1940, 1960, 1980, 2000])





# In[499]:





all_data = pd.get_dummies(all_data, columns = ["LotArea"], prefix="LotArea")

#all_data = pd.get_dummies(all_data, columns = ["MasVnrArea"], prefix="MasVnrArea")

#all_data = pd.get_dummies(all_data, columns = ["LotFrontage"], prefix="LotFrontage")

all_data = pd.get_dummies(all_data, columns = ["GarageYrBlt"], prefix="GarageYrBlt")





# In[500]:





all_data = pd.get_dummies(all_data, columns = ["LotShape"], prefix="LotShape")

all_data = pd.get_dummies(all_data, columns = ["LandContour"], prefix="LandContour")





# In[501]:





all_data['LotConfig'] = all_data['LotConfig'].map({"Inside":"Inside", "FR2":"FR", "Corner":"Corner", "CulDSac":"CulDSac", "FR3":"FR"})



all_data = pd.get_dummies(all_data, columns = ["LotConfig"], prefix="LotConfig")





# In[502]:





all_data['LandSlope'] = all_data['LandSlope'].map({"Gtl":1, "Mod":2, "Sev":2})

def Slope(col):

    if col['LandSlope'] == 1:

        return 1

    else:

        return 0

    

all_data['GentleSlope_Flag'] = all_data.apply(Slope, axis=1)

all_data.drop('LandSlope', axis=1, inplace=True)





# In[503]:





# all_data['LowQualFinSF'].value_counts()





# In[504]:





# MasVnrArea has no correlation to SalePrice

all_data.drop('MasVnrArea', axis=1, inplace=True)

# As the samples are very few, we can drop this column

all_data.drop('LowQualFinSF', axis=1, inplace=True)





# In[505]:





all_data['GrLivArea'] = pd.cut(all_data["GrLivArea"], np.arange(0, 6000, 1000), labels=[0, 1000, 2000, 3000, 4000])

all_data = pd.get_dummies(all_data, columns = ["GrLivArea"], prefix="GrLivArea")



all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data = pd.get_dummies(all_data, columns = ["MSSubClass"], prefix="MSSubClass")



all_data['BldgType'] = all_data['BldgType'].astype(str)

all_data = pd.get_dummies(all_data, columns = ["BldgType"], prefix="BldgType")



all_data['HouseStyle'] = all_data['HouseStyle'].map({"2Story":"2Story", "1Story":"1Story", "1.5Fin":"1.5Story", "1.5Unf":"1.5Story", 

                                                     "SFoyer":"SFoyer", "SLvl":"SLvl", "2.5Unf":"2.5Story", "2.5Fin":"2.5Story"})

all_data = pd.get_dummies(all_data, columns = ["HouseStyle"], prefix="HouseStyle")



         

# all_data = pd.get_dummies(all_data, columns = ["LotArea"], prefix="LotArea")

# all_data = pd.get_dummies(all_data, columns = ["LotArea"], prefix="LotArea")

# all_data = pd.get_dummies(all_data, columns = ["LotArea"], prefix="LotArea")





# # Leave some columns as no relation can be drawn between them and SalePrice

# 

# # Bedroom, Kitchen, TotRmsAbvGrd, OverallCond, Remod_diff, LotFrontage, OverallQual, OverallCond, 

# 



# In[506]:





#all_data['Remod_Diff'] = all_data['YearRemodAdd'] - all_data['YearBuilt']

#all_data.drop('YearRemodAdd', axis=1, inplace=True)





# In[507]:





all_data['YearBuilt'] = pd.cut(all_data["YearBuilt"], np.arange(1800, 2100, 100), labels=[1800, 1900])



all_data = pd.get_dummies(all_data, columns = ["YearBuilt"], prefix="YearBuilt")

all_data = pd.get_dummies(all_data, columns = ["Foundation"], prefix="Foundation")





# In[508]:





all_data['Functional'] = all_data['Functional'].map({"Sev":1, "Maj2":2, "Maj1":3, "Mod":4, "Min2":5, "Min1":6, "Typ":7})

all_data = pd.get_dummies(all_data, columns = ["RoofStyle"], prefix="RoofStyle")

all_data = pd.get_dummies(all_data, columns = ["RoofMatl"], prefix="RoofMatl")





# In[509]:





def Exter2(col):

    if col['Exterior2nd'] == col['Exterior1st']:

        return 1

    else:

        return 0

    

all_data['ExteriorMatch_Flag'] = all_data.apply(Exter2, axis=1)

all_data.drop('Exterior2nd', axis=1, inplace=True)



all_data = pd.get_dummies(all_data, columns = ["Exterior1st"], prefix="Exterior1st")





# In[510]:





all_data = pd.get_dummies(all_data, columns = ["MasVnrType"], prefix="MasVnrType")



all_data['ExterQual'] = all_data['ExterQual'].map({"Fa":1, "TA":2, "Gd":3, "Ex":4})

all_data = pd.get_dummies(all_data, columns = ["ExterCond"], prefix="ExterCond")

all_data = pd.get_dummies(all_data, columns = ["GarageType"], prefix="GarageType")

#all_data = pd.get_dummies(all_data, columns = ["GarageYrBlt"], prefix="GarageYrBlt")

all_data = pd.get_dummies(all_data, columns = ["GarageFinish"], prefix="GarageFinish")



#all_data['YearBuilt'] = pd.cut(all_data["YearBuilt"], np.arange(1800, 2100, 100), labels=[1800, 1900])





# In[511]:





all_data['GarageArea_Band'] = pd.cut(all_data['GarageArea'], 3)

all_data.loc[all_data['GarageArea']<=496, 'GarageArea'] = 1

all_data.loc[(all_data['GarageArea']>496) & (all_data['GarageArea']<=992), 'GarageArea'] = 2

all_data.loc[all_data['GarageArea']>992, 'GarageArea'] = 3

all_data['GarageArea'] = all_data['GarageArea'].astype(int)



all_data.drop('GarageArea_Band', axis=1, inplace=True)



all_data = pd.get_dummies(all_data, columns = ["GarageArea"], prefix="GarageArea")





# In[512]:





def WoodDeckFlag(col):

    if col['WoodDeckSF'] == 0:

        return 1

    else:

        return 0

    

all_data['NoWoodDeck_Flag'] = all_data.apply(WoodDeckFlag, axis=1)



all_data['WoodDeckSF_Band'] = pd.cut(all_data['WoodDeckSF'], 4)



all_data.loc[all_data['WoodDeckSF']<=356, 'WoodDeckSF'] = 1

all_data.loc[(all_data['WoodDeckSF']>356) & (all_data['WoodDeckSF']<=712), 'WoodDeckSF'] = 2

all_data.loc[(all_data['WoodDeckSF']>712) & (all_data['WoodDeckSF']<=1068), 'WoodDeckSF'] = 3

all_data.loc[all_data['WoodDeckSF']>1068, 'WoodDeckSF'] = 4

all_data['WoodDeckSF'] = all_data['WoodDeckSF'].astype(int)



all_data.drop('WoodDeckSF_Band', axis=1, inplace=True)



all_data = pd.get_dummies(all_data, columns = ["WoodDeckSF"], prefix="WoodDeckSF")





# In[513]:





all_data['GarageQual'] = all_data['GarageQual'].map({"None":"None", "Po":"Low", "Fa":"Low", "TA":"TA", "Gd":"High", "Ex":"High"})

all_data = pd.get_dummies(all_data, columns = ["GarageQual"], prefix="GarageQual")



all_data['GarageCond'] = all_data['GarageCond'].map({"None":"None", "Po":"Low", "Fa":"Low", "TA":"TA", "Gd":"High", "Ex":"High"})

all_data = pd.get_dummies(all_data, columns = ["GarageCond"], prefix="GarageCond")



#all_data = pd.get_dummies(all_data, columns = ["WoodDeckSF"], prefix="WoodDeckSF")





# In[514]:





all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch'] 

all_data['OpenPorchSF'] = all_data['OpenPorchSF'] + all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']





# In[515]:





def PorchFlag(col):

    if col['TotalPorchSF'] == 0:

        return 1

    else:

        return 0

    

all_data['NoPorch_Flag'] = all_data.apply(PorchFlag, axis=1)



all_data['TotalPorchSF_Band'] = pd.cut(all_data['TotalPorchSF'], 4)

all_data['TotalPorchSF_Band'].unique()

all_data.loc[all_data['TotalPorchSF']<=431, 'TotalPorchSF'] = 1

all_data.loc[(all_data['TotalPorchSF']>431) & (all_data['TotalPorchSF']<=862), 'TotalPorchSF'] = 2

all_data.loc[(all_data['TotalPorchSF']>862) & (all_data['TotalPorchSF']<=1293), 'TotalPorchSF'] = 3

all_data.loc[all_data['TotalPorchSF']>1293, 'TotalPorchSF'] = 4

all_data['TotalPorchSF'] = all_data['TotalPorchSF'].astype(int)



all_data.drop('TotalPorchSF_Band', axis=1, inplace=True)



all_data = pd.get_dummies(all_data, columns = ["TotalPorchSF"], prefix="TotalPorchSF")

all_data.head(3)





# In[516]:

#all_data = pd.get_dummies(all_data, columns = ["TotalPorchSF"], prefix="TotalPorchSF")





# In[517]:





def PoolFlag(col):

    if col['PoolArea'] == 0:

        return 0

    else:

        return 1

    

all_data['HasPool_Flag'] = all_data.apply(PoolFlag, axis=1)

all_data.drop('PoolArea', axis=1, inplace=True)





# In[518]:

all_data.drop('PoolQC', axis=1, inplace=True)





# In[519]:

all_data = pd.get_dummies(all_data, columns = ["MSZoning"], prefix="MSZoning")

all_data = pd.get_dummies(all_data, columns = ["Neighborhood"], prefix="Neighborhood")





# In[520]:





all_data['Condition1'] = all_data['Condition1'].map({"Norm":"Norm", "Feedr":"Street", "PosN":"Pos", "Artery":"Street", "RRAe":"Train",

                                                    "RRNn":"Train", "RRAn":"Train", "PosA":"Pos", "RRNe":"Train"})

all_data['Condition2'] = all_data['Condition2'].map({"Norm":"Norm", "Feedr":"Street", "PosN":"Pos", "Artery":"Street", "RRAe":"Train",

                                                    "RRNn":"Train", "RRAn":"Train", "PosA":"Pos", "RRNe":"Train"})





# In[521]:





def ConditionMatch(col):

    if col['Condition1'] == col['Condition2']:

        return 0

    else:

        return 1

    

all_data['Diff2ndCondition_Flag'] = all_data.apply(ConditionMatch, axis=1)

all_data.drop('Condition2', axis=1, inplace=True)



all_data = pd.get_dummies(all_data, columns = ["Condition1"], prefix="Condition1")

all_data.head(3)





# In[522]:





all_data['TotalBathrooms'] = all_data['BsmtHalfBath'] + all_data['BsmtFullBath'] + all_data['HalfBath'] + all_data['FullBath']



columns = ['BsmtHalfBath', 'BsmtFullBath', 'HalfBath', 'FullBath']

all_data.drop(columns, axis=1, inplace=True)





# In[523]:





all_data['KitchenQual'] = all_data['KitchenQual'].map({"Fa":1, "TA":2, "Gd":3, "Ex":4})

all_data['FireplaceQu'] = all_data['FireplaceQu'].map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})





# In[524]:





# all_data['GrLivArea_Band'] = pd.cut(all_data['GrLivArea'], 6)

# all_data.loc[all_data['GrLivArea']<=1127.5, 'GrLivArea'] = 1

# all_data.loc[(all_data['GrLivArea']>1127.5) & (all_data['GrLivArea']<=1921), 'GrLivArea'] = 2

# all_data.loc[(all_data['GrLivArea']>1921) & (all_data['GrLivArea']<=2714.5), 'GrLivArea'] = 3

# all_data.loc[(all_data['GrLivArea']>2714.5) & (all_data['GrLivArea']<=3508), 'GrLivArea'] = 4

# all_data.loc[(all_data['GrLivArea']>3508) & (all_data['GrLivArea']<=4301.5), 'GrLivArea'] = 5

# all_data.loc[all_data['GrLivArea']>4301.5, 'GrLivArea'] = 6

# all_data['GrLivArea'] = all_data['GrLivArea'].astype(int)



# all_data.drop('GrLivArea_Band', axis=1, inplace=True)



# all_data = pd.get_dummies(all_data, columns = ["GrLivArea"], prefix="GrLivArea")





# In[525]:





#all_data.drop('Street', axis=1, inplace=True)

all_data = pd.get_dummies(all_data, columns = ["Alley"], prefix="Alley")

all_data = pd.get_dummies(all_data, columns = ["PavedDrive"], prefix="PavedDrive")





# In[527]:





all_data['GasA_Flag'] = all_data['Heating'].map({"GasA":1, "GasW":0, "Grav":0, "Wall":0, "OthW":0, "Floor":0})

#all_data.drop('Heating', axis=1, inplace=True)

all_data['HeatingQC'] = all_data['HeatingQC'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

all_data['CentralAir'] = all_data['CentralAir'].map({"Y":1, "N":0})





# In[528]:





all_data['Electrical'] = all_data['Electrical'].map({"SBrkr":"SBrkr", "FuseF":"Fuse", "FuseA":"Fuse", "FuseP":"Fuse", "Mix":"Mix"})



all_data = pd.get_dummies(all_data, columns = ["Electrical"], prefix="Electrical")





# In[529]:





columns=['MiscFeature', 'MiscVal']

all_data.drop(columns, axis=1, inplace=True)





# In[530]:





all_data = pd.get_dummies(all_data, columns = ["MoSold"], prefix="MoSold")

all_data = pd.get_dummies(all_data, columns = ["YrSold"], prefix="YrSold")

all_data['SaleType'] = all_data['SaleType'].map({"WD":"WD", "New":"New", "COD":"COD", "CWD":"CWD", "ConLD":"Oth", "ConLI":"Oth", 

                                                 "ConLw":"Oth", "Con":"Oth", "Oth":"Oth"})



all_data = pd.get_dummies(all_data, columns = ["SaleType"], prefix="SaleType")

all_data = pd.get_dummies(all_data, columns = ["SaleCondition"], prefix="SaleCondition")





# In[559]:





all_data.replace('None', 0, inplace=True)

all_data.fillna(method='ffill', inplace=True)





# In[560]:





all_data = all_data.select_dtypes(include=[np.number])

#testdf = testdf.select_dtypes(include=[np.number])
# First, re-create the training and test datasets

train = all_data[:ntrain]

test = all_data[ntrain:]



print(train.shape)

print(test.shape)



# # Skewness

X = train.iloc[:, train.columns != 'SalePrice']

y = np.log(train['SalePrice'])





from sklearn.model_selection import train_test_split

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)





from sklearn.linear_model import LinearRegression

#model=Lasso(alpha =0.001, random_state=1)

model = LinearRegression()

model.fit(X_train, y_train)



preds = model.predict(X_test)

preds = np.exp(preds)



# from sklearn.metrics import mean_squared_error

# print ('RMSE is: \n', mean_squared_error(y_test, preds))
test.shape, X_train.shape
test.head()
test.drop('SalePrice', axis=1, inplace=True)
test.shape
testpreds = model.predict(test)

testpreds = np.exp(testpreds)
len(testpreds)
finalpreds = pd.DataFrame(testpreds, columns=['SalePrice'])

finalpreds.index = np.arange(1461, len(testpreds)+1461)

finalpreds.index.name = 'Id'

finalpreds.to_csv('submission.csv')
len(finalpreds)