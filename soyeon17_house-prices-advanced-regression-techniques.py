# Import librairies

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt  # plotting

import seaborn as sns  # plotting

from statistics import mode

from scipy import stats

from scipy.stats import norm, skew # for some statistics

from sklearn.preprocessing import StandardScaler # scaling using mean and sd.

from pandas.plotting import scatter_matrix



# Set environments

%matplotlib inline

import warnings

warnings.filterwarnings('ignore') # ignore warning from sklearn and seaborn

pd.set_option('display.max_columns', None) # https://thispointer.com/python-pandas-how-to-display-full-dataframe-i-e-print-all-rows-columns-without-truncation/

pd.set_option('display.max_rows', None)

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) # set up print format, Limiting floats to 3 decimal points
# Import the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape)

print(test.shape) # response variable(SalePrice) not in the test dataset
#Getting rid of the IDs but keeping the test IDs in a vector. These are needed to compose the submission file

test_labels = test['Id']

test.drop(['Id'], axis=1, inplace=True)

train.drop(['Id'], axis=1, inplace=True)



all_data = pd. concat([train, test], ignore_index=True)

all_data0 = pd. concat([train, test], ignore_index=True)

print(all_data.shape) # without the Id, there're 79 predictors and 1 response variable (SalePrice)
# Check normality

all_dataSP = all_data.dropna(subset=['SalePrice'])

sns.distplot(all_dataSP['SalePrice'], fit=norm); # positive skewness(right skewed), curtosis(peakedness)

# this shape was guessed that few people can afford very expensive houses



# Check the normal probability plot

fig = plt.figure()

res = stats.probplot(all_dataSP['SalePrice'], plot=plt) # non normality

plt.show()
# correlations of variables

top_10_index = all_data.corr()['SalePrice'].sort_values(ascending=False)[:11].index  # select top 11 high correlations

plt.figure(figsize=(15, 6))

sns.heatmap(all_data[top_10_index].corr(), 

            annot=True, linewidths = 3, cbar=True, fmt=".2f")

plt.tick_params(labelsize=13); plt.gca().xaxis.tick_top() ; plt.xticks(rotation=45); plt.yticks(rotation=0); plt.show()
# the correlation between Overall Quality and SalePrice

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=all_data)

# the candidate of outlier : 1 expensive house in grade 4
# Scatter plot

sns.regplot(x=all_data['GrLivArea'], y=all_data['SalePrice'], fit_reg=True, line_kws={"color": "grey"})

plt.xlabel('GrLivArea', fontsize=12)

plt.ylabel('SalePrice', fontsize=12)

plt.show()

# the candidates of outlier : 2 cheap and big houses when Area size is over 4500 ft in section 7.2.
# missing data

total = all_data.isnull().sum().sort_values(ascending=False) # descending order

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print("There are ", len(total[total > 0]), 'columns with missning values')

missing_data[total > 0]

# 1495 Nas in SalePrice is from the test set

# the candidates to fix NAs are 34 predictor variables without SalePrice
### Pool Quality and PoolArea variable 

''' PoolQC (Pool quality)

    values: (Ex: Excellent); (Gd: Good); (TA: Average/Typical); (Fa: Fair); (NA: No Pool)

    values seem to be ordinal and need to put label encode

    2907 NAs could mean "No Pool" because the majority of houses have no pool in general

'''

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")  # NaN to None

all_data = all_data.replace({ "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4}})

all_data.PoolQC.value_counts()
''' PoolArea (Pool quality in square feet)

'''

# Is there any pools without PoolQC but physically presence?

all_data[(all_data['PoolQC'] == 0) & (all_data['PoolArea'] > 0)][['PoolArea','PoolQC','OverallQual']]
# the correlation among PoolQC, Overall Quality and PoolArea

plt.subplot(121)

fig = sns.boxplot(x='PoolQC', y="OverallQual", data=all_data) # a little bit of relation

plt.subplot(122)

fig = sns.boxplot(x='PoolQC', y="PoolArea", data=all_data) # no relation

plt.show()
# Imputting 3 values in the PoolQC based on OverallQual

all_data.PoolQC[2420] = 2  # impute 2 (OverallQual=4) into house #2420

all_data.PoolQC[2503] = 3  # impute 3 (OverallQual=6) into house #2503  

all_data.PoolQC[2599] = 2  # impute 2 (OverallQual=3) into house #2599 

all_data[(all_data['PoolArea'] > 0)][['PoolArea','PoolQC','OverallQual']]
''' MiscFeature (Miscellaneous feature not covered in other categories)

    values: (Elev: Elevator); (Gar2: 2nd Garage); (Othr: Other); (Shed: Shed); (TenC: Tennis Court); (NA: None)

    2814 NAs could mean "no misc feature"

'''

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")  # NaN to None

all_data.MiscFeature.value_counts()
sns.barplot(data=all_data, x="MiscFeature", y="SalePrice", estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.ylabel('Median SalePrice');

# Having shed means 'no Garage', which would explain lower SalePrice

# There is one expensive house with a tennis court in the training set
''' Alley (Type of alley)

    values: (Grvl: Gravel); (Pave: Paved); (NA: No alley access)

    2721 NAs could mean "no alley access", a categorical variable

'''

all_data["Alley"] = all_data["Alley"].fillna("None")  # NaN to None

all_data.Alley.value_counts()
sns.barplot(data=all_data, x="Alley", y="SalePrice", estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.ylabel('Median SalePrice');
''' Fence (Fence quality)

    values: (GdPrv: Good Privacy); (MnPrv: Minimum Privacy); (GdWo: Good Wood); (MnWw: Minimum Wood/Wire); (NA: No Fence)

    values seem to be ordinal but need to check the relationship with SalePrice before label encoding

    2348 NAs seem to mean "no fence", an ordinal variable

'''

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data = all_data.replace({ "Fence" : {"None" : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv" : 3, "GdPrv" : 4}})

all_data.Fence.value_counts()
sns.barplot(data=all_data, x="Fence", y="SalePrice", estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--'); 

plt.ylabel('Median SalePrice');

# the values do not seem ordinal so remain Fence into categorcal variable
''' FireplaceQu (Fireplace quality)

    values: (Ex: Excellent); (Gd: Good); (TA: Average); (Fa: Fair); (Po: Poor); (NA: No Fireplace)

    1420 NAs in FireplaceQu matches the number of houses with 0 fireplaces, which NAs mean "no fireplace"

    the values are ordinal

'''

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data = all_data.replace({ "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

all_data.FireplaceQu.value_counts()
'''Fireplaces is an integer variable, and there are no missing values'''

all_data.Fireplaces.value_counts()
''' LotFrontage (Linear feet of street connected to property)

    Since the area of each street connected to the house could have a similar area to other houses in its neighborhood,

    486 NAs seems to take the median of LotFrontage by each of the neighborhood

'''

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

 

# check the median of LotFrontage by each of Neighborhood

f, ax = plt.subplots(figsize=(10, 6))

sns.barplot(data=all_data, x="Neighborhood", y="LotFrontage", estimator=np.median);

plt.ylabel('Median LotFrontage');

plt.axhline(y=np.median(all_data.LotFrontage), color="red", linestyle = '--')

plt.xticks(rotation=45);
''' LotShape (general shape of property)

    values: (Reg : Regular); (IR1: Slightly irregular); (IR2: Moderately Irregular); (IR3: Irregular)

    0 NAs, values seem ordinal (Reg = best)

'''

all_data = all_data.replace({ "LotShape" : {"IR3" : 0, "IR2" : 1, "IR1" : 2, "Reg" : 3}})

all_data.LotShape.value_counts()
''' LotConfig (Lot configuration)

    values: (Inside: Inside lot); (Corner: Corner lot); (CulDSac: Cul-de-sac); 

    (FR2: Frontage on 2 sides of property); (FR3: Frontage on 3 sides of property)

    0 NAs, a categorcal variable

'''

sns.barplot(data=all_data, x="LotConfig", y="SalePrice", estimator=np.median);

plt.ylabel('Median SalePrice');

all_data.LotConfig.value_counts()
''' 1 NA:

    GarageArea (Size of garage in square feet) 

    GarageCars (Size of garage in car capacity)



    157 NAs:

    GarageType (Garage location)



    159 NAs:

    GarageQual (Garage quality)

    GarageYrBlt (Year garage was built)

    GarageFinish (Interior finish of the garage)

    GarageCond (Garage condition) 

'''



''' GarageYrBlt'''

all_data[(all_data['GarageYrBlt'].isnull())][['GarageYrBlt','YearBuilt','YearRemodAdd','GarageArea']]

# GarageYrBlt values are similar with the values in YearBuilt 

# also, those are similar to the values of YearRemodAdd when no remodeling or additions

# I am going to replace GarageYrBlt missing valeus with the values in YearBuilt (which is house #666)
# However, some Garages does not seem present when GarageArea = 0, so impute 0 into them

all_data[['GarageYrBlt']] = all_data[['GarageYrBlt']].fillna(0)
# check differences between the 157 NAs GarageType and the other 3 character variables with 159 NAs

print(len(all_data[(all_data['GarageType'].isnull()) & (all_data['GarageFinish'].isnull()) & (all_data['GarageCond'].isnull()) & (all_data['GarageQual'].isnull())]))



# find the 2 additional NAs

all_data[(all_data['GarageType'].notnull()) & (all_data['GarageFinish'].isnull())][['GarageType','GarageQual', 'GarageYrBlt', 'GarageFinish', 'GarageCond', 'GarageCars', 'GarageArea']]

# house #1116 doesn not seem to have a garage

# house #666 seems to have a garage. 
# impute the mode values (high frequancy) of GarageCond, GarageQual, GarageFinish into house #666

all_data.GarageCond[2126] = mode(all_data.GarageCond)

all_data.GarageQual[2126] = mode(all_data.GarageQual)

all_data.GarageFinish[2126] = mode(all_data.GarageFinish)

all_data.GarageYrBlt[2126] = all_data.YearBuilt[2126]

print(len(all_data[(all_data['GarageFinish'].isnull()) | (all_data['GarageCond'].isnull()) | (all_data['GarageQual'].isnull())]))



# impute no garage for house #1116

all_data.GarageCars[2576] = 0

all_data.GarageArea[2576] = 0

print(len(all_data[(all_data['GarageCars'].isnull()) | (all_data['GarageArea'].isnull())]))
''' GarageType

    values: (2Types: more than one type); (Attchd: attached); (Basment: in basement); ... (NA: No Garage)

    157 NAs, a categorical variable '''

all_data.GarageType = all_data.GarageType.fillna("None")

all_data.GarageType.value_counts()
''' GarageFinish (Interior finish of the garage)

    values: (Fin: Finished); (RFn: Rough Finished); (Unf: Unfinished); (NA: No Garage)

    158 NAs, an ordinal variable

'''

all_data.GarageFinish = all_data.GarageFinish.fillna("None")

all_data = all_data.replace({ "GarageFinish" : {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3}})

all_data.GarageFinish.value_counts()
''' GarageQual (Garage quality) 

    GarageCond (Garage condition)

    values: (Ex: Excellent); (Gd: Good); (TA: Average); (Fa: Fair); (Po: Poor); (NA: No Garage)

    158 NAs, ordinal variables

'''

all_data.GarageQual = all_data.GarageQual.fillna("None")

all_data = all_data.replace({ "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

print(all_data.GarageQual.value_counts())

print("----------")

all_data.GarageCond = all_data.GarageCond.fillna("None")

all_data = all_data.replace({ "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

print(all_data.GarageCond.value_counts())
''' BsmtCond (the general condition of the basement) : 82 NAs

    BsmtQual (the height of the basement) : 81 NAs

    values: (Ex: Excellent); (Gd: Good); (TA: Average); (Fa: Fair); (Po: Poor); (NA: No Basement)

    BsmtExposure (Refers to walkout or garden level walls) : 82 NAs

    values: (Gd: Good); (TA: Average); (Mn: Minimum); (No: No exposure); (NA: No Basement)



    BsmtFinType2 (Rating of basement finished area (if multiple types)) : 80 NAs

    BsmtFinType1 (Rating of basement finished area) : 79 NAs 

    values: (GLQ: Good); (ALQ: Average); (BLQ: Below Average); (Rec: Average); (LwQ: Low); (Unf: Unfinshed); (NA: No Basement)

'''

''' BsmtFinSF1 (Type 1 finished square feet) : 1 NA

    BsmtFinSF2 (Type 2 finished square feet) : 1 NA

    BsmtUnfSF (Unfinished square feet of basement area) : 1 NA

    TotalBsmtSF (Total square feet of basement area) : 1 NA

    BsmtFullBath (Basement full bathrooms) : 2 NA

    BsmtHalfBath (Basement half bathrooms) : 2 NA

'''

# check all 79 NAs are the same observations among other variables with 80+ NAs

print(len(all_data[(all_data['BsmtQual'].isnull()) & (all_data['BsmtCond'].isnull()) & (all_data['BsmtExposure'].isnull()) & (all_data['BsmtFinType1'].isnull()) & (all_data['BsmtFinType2'].isnull())]))



# find the additional NAs

all_data[(all_data['BsmtFinType1'].notnull()) & ((all_data['BsmtCond'].isnull()) | (all_data['BsmtExposure'].isnull()) | (all_data['BsmtQual'].isnull()) | (all_data['BsmtFinType2'].isnull()))][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]

# there are 79 houses without any basement

# impute mode (most frequant) values to these 9 houses



all_data.BsmtFinType2[332] = mode(all_data.BsmtFinType2)

all_data.BsmtExposure[948,1487,2348] = mode(all_data.BsmtExposure)

all_data.BsmtCond[2040,2185,2524] = mode(all_data.BsmtCond)

all_data.BsmtQual[2348,2524] = mode(all_data.BsmtQual)

print(all_data.BsmtFinType2.value_counts())

print(all_data.BsmtExposure.value_counts())

print(all_data.BsmtCond.value_counts())

print(all_data.BsmtQual.value_counts())
''' Ordinal variables

    BsmtCond (the general condition of the basement) : 82 NAs -> 79 NAs

    BsmtQual (the height of the basement) : 81 NAs -> 79 NAs

    values: (Ex: Excellent); (Gd: Good); (TA: Average); (Fa: Fair); (Po: Poor); (NA: No Basement)



    BsmtExposure (Refers to walkout or garden level walls) : 82 NAs -> 79 NAs

    values: (Gd: Good); (TA: Average); (Mn: Minimum); (No: No exposure); (NA: No Basement)



    BsmtFinType2 (Rating of basement finished area (if multiple types)) : 80 NAs -> 79 NAs

    BsmtFinType1 (Rating of basement finished area) : 79 NAs 

    values: (GLQ: Good); (ALQ: Average); (BLQ: Below Average); (Rec: Average); (LwQ: Low); (Unf: Unfinshed); (NA: No Basement)

''' 



all_data.BsmtCond = all_data.BsmtCond.fillna("None")

all_data = all_data.replace({ "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

print(all_data.BsmtCond.value_counts())

all_data.BsmtQual = all_data.BsmtQual.fillna("None")

all_data = all_data.replace({ "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

print(all_data.BsmtQual.value_counts())



all_data.BsmtExposure = all_data.BsmtExposure.fillna("None")

all_data = all_data.replace({ "BsmtExposure" : {"None" : 0, "No" : 1, "Mn" : 2, "Av" : 3, "Gd" : 4}})

print(all_data.BsmtExposure.value_counts())



all_data.BsmtFinType1 = all_data.BsmtFinType1.fillna("None")

all_data = all_data.replace({ "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ" : 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}})

print(all_data.BsmtFinType1.value_counts())

all_data.BsmtFinType2 = all_data.BsmtFinType2.fillna("None")

all_data = all_data.replace({ "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ" : 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}})

print(all_data.BsmtFinType2.value_counts())
''' BsmtFinSF1 (Type 1 finished square feet) : 1 NA

    BsmtFinSF2 (Type 2 finished square feet) : 1 NA

    BsmtUnfSF (Unfinished square feet of basement area) : 1 NA

    TotalBsmtSF (Total square feet of basement area) : 1 NA

    BsmtFullBath (Basement full bathrooms) : 2 NA

    BsmtHalfBath (Basement half bathrooms) : 2 NA

'''

# check NAs

all_data[(all_data['BsmtFinSF1'].isnull()) | ((all_data['BsmtFinSF2'].isnull()) | (all_data['BsmtUnfSF'].isnull()) | (all_data['TotalBsmtSF'].isnull()) 

        | (all_data['BsmtFullBath'].isnull()) | (all_data['BsmtHalfBath'].isnull()))

        ][['BsmtQual','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]

# these NAs mean 'no basement'



for i in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF'):

    all_data[i] = all_data[i].fillna(0)

    

for i in ('BsmtFullBath', 'BsmtHalfBath'):

    all_data[i] = all_data[i].fillna(0)

    print(all_data[i].value_counts())

    print("-----------")
''' MasVnrType (Masonry veneer type)

    values: (BrkCmn: Brick Common); (BrkFace: Brick Face); (CBlock: Cinder Block); (None: None); (Stone: Stone)

    24 NAs, a categorical variable

'''

# check if the 23 NAs are the same observation

print(len(all_data[(all_data['MasVnrArea'].isnull()) & (all_data['MasVnrArea'].isnull())]))



# find the one NA in MasVnrType

all_data[(all_data['MasVnrType'].isnull()) & ((all_data['MasVnrArea'].notnull()))][['MasVnrType','MasVnrArea']]
# impute the mode into the 1 missing in MasVnrType

all_data.ix[2610]['MasVnrType'] = "BrkFace" # the second mode "BrkFace"  cf. the 1st mode : "None"



# leave 23 houses as no masonry

all_data.MasVnrType = all_data.MasVnrType.fillna("None")

all_data.MasVnrType.value_counts()
# check MasVnrType variable with SalePrice

sns.barplot(data=all_data, x="MasVnrType", y="SalePrice", estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.ylabel('Median SalePrice');

# common brick and no masonry (wooden houses) seems cheaper than others

# change categorical variable to ordinal variable

all_data = all_data.replace({ "MasVnrType" : {"None" : 0, "BrkCmn" : 0, "BrkFace" : 1, "Stone" : 2}})

print(all_data.MasVnrType.value_counts())
''' MasVnrArea (Masonry veneer area in square feet)

    23 NAs, an integer variable

'''

all_data.MasVnrArea = all_data.MasVnrArea.fillna(0)

print(all_data.MasVnrArea.isnull().sum())
''' MSZoning (The general zoning classification)

    values: (A: Agriculture); (C: Commercial); (FV: Floating Village Residential); (I:Industrial); (RH: Residential High Density); 

    (RL: Residential Low Density); (RP: Residential Low Density Park); (RM: Residential Medium Density)

    4 NAs, a categorical variable

'''

# impute the mode

all_data.MSZoning = all_data.MSZoning.fillna(mode(all_data['MSZoning']))

all_data.MSZoning.value_counts()
''' KitchenQual (Kitchen quality)

    values: (Ex: Excellent); (Gd: Good); (TA: Average); (Fa: Fair); (Po: Poor)

    1 NA, an ordinal variable

'''

# impute mode into a missing value in KitchenQual

all_data.KitchenQual = all_data.KitchenQual.fillna(mode(all_data.KitchenQual))

all_data = all_data.replace({ "KitchenQual" : {"Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

all_data.KitchenQual.value_counts()
''' Utilities (Type of utilities available)

    values: (AllPub: All public Utilities (E,G,W,& S)); (NoSewr: Electricity, Gas, and Water (Septic Tank)); 

    (NoSeWa: Electricity and Gas Only); (ELO: Electricity only)

    2 NAs, an ordinal variable (more utilities is better)

'''

all_data.Utilities.value_counts()

# For this categorical feature all records are "AllPub", except for a "NoSeWa" and 2 NAs

# Since the house with 'NoSewa' is in the training set, (this feature won't help in predictive modelling)

# delete the 'Utilities' variable

all_data = all_data.drop(['Utilities'], axis=1)
''' Functional (Home functionality - Assuming typical unless deductions are warranted) : # of NA is 2

    values: (Typ: Typical Functionality); (Min1: Minor Deductions 1); (Min2: Minor Deductions 2); (Mod: Moderate Deductions);

    (Maj1: Major Deductions 1); (Maj2: Major Deductions 2); (Sev: Severely Damaged); (Sal: Salvage only)

    2 NAs, an ordinal variable (salvage only is worst, and typical is best)

'''

# impute the mode for the 1 NA

all_data.Functional = all_data.Functional.fillna(mode(all_data.Functional))

all_data = all_data.replace({ "Functional" : {"Sal" : 0, "Sev" : 1, "Maj2" : 2, "Maj1" : 3, "Mod" : 4, "Min1" : 5, "Min2" : 6, "Typ" : 7}})

all_data.Functional.value_counts()
''' Exterior1st (Exterior covering on house)

    Exterior2nd (Exterior covering on house (mixed material))

    1 NA, categorical variables about the types of materials

    

    ExterQual (the quality of the material on the exterior)

    ExterCond (the condition of the material on the exterior)

    values: (Ex: Excellent); (Gd: Good); (TA: Average); (Fa: Fair); (Po: Poor)

    0 NA, an ordinal variable

'''

# impute mode into missing values

all_data.Exterior1st = all_data.Exterior1st.fillna(mode(all_data.Exterior1st))

print(all_data.Exterior1st.value_counts())

print("----------")

all_data.Exterior2nd = all_data.Exterior2nd.fillna(mode(all_data.Exterior2nd))

print(all_data.Exterior2nd.value_counts())

print("----------")

# transform to an ordinal variable

all_data = all_data.replace({ "ExterQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

print(all_data.ExterQual.value_counts())

print("----------")

all_data = all_data.replace({ "ExterCond" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

print(all_data.ExterCond.value_counts())
''' Electrical (Electrical system)

    values: (SBrkr: Standard Circuit); (FuseA: Average wiring); (FuseF: Fair wiring); (FuseP: poor wiring); (Mix: Mixed)

    1 NA, a categorical variable

'''

# impute mode

all_data.Electrical = all_data.Electrical.fillna(mode(all_data.Electrical))

all_data.Electrical.value_counts()
''' SaleType (Type of sale)

    1 NA, a categorical variable about the types of Sale

    

    SaleCondition (Condition of sale)

    0 NA, a categorical variable about the condition types of Sale

'''

# impute mode

all_data.SaleType = all_data.SaleType.fillna(mode(all_data.SaleType))

print(all_data.SaleType.value_counts())

print(all_data.SaleCondition.describe())
all_data.info() # check missing values, numeric(int64/float) or categorical(object)

# no NAs
# index vector numeric / categorical variables

cols = all_data.columns

numeric = []

nonnumeric = []

list([ numeric.append(c) if all_data[c].dtype in ['int64', 'float64'] else nonnumeric.append(c) for c in cols])

print('There are', len(nonnumeric), 'remaining categorical variables (object types)')

print("------------")

print("categorical variables : ", nonnumeric)

# But I already handled with Alley, Electrcial, Exterior1st, Exterior2nd, GarageType, LotConfig, MSZoning, MiscFeature, SaleType, SaleCondition

# Let's focuse on Foundation, Heating, CentralAir, Roofs, LandSlope/Contour, BldgType, Neighborhood, Conditions, Street, PavedDrive
''' Foundation (Type of foundation)

    values: (BrkTil: Brick & Tile); (CBlock: Cinder Block); (PConc: Poured Contrete); (Slab: Slab); (Stone: Stone); (Wood: Wood)

    a categorical variable



    Heating (Type of heating)

    values: (Floor: Floor Furnace); (GasA: Gas - air furnace); (GasW: Gas - water or steam); (Grav: Gravity furnace);

    (OthW: Hot water or steam heat); (Wall: Wall furnace)

    a categorical variable

    

    HeatingQC (Heating quality and condition)

    values: (Ex: Excellent); (Gd: Good); (TA: Average); (Fa: Fair); (Po: Poor)

    an ordinal variable



    CentralAir (Central air conditioning)

    values: (N: No); (Y: Yes)

    a categorical variable

'''

print(all_data.Foundation.value_counts())



print(all_data.Heating.value_counts())



all_data = all_data.replace({ "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

print(all_data.HeatingQC.value_counts())



all_data = all_data.replace({ "CentralAir" : {"N" : 0, "Y" : 1}})

print(all_data.CentralAir.value_counts())
''' RoofStyle (Type of roof)

    values: (Flat: Flat); (Gable: Gable); (Gambrel: Gabrel; (Hip: Hip); (Mansard: Mansard); (Shed: Shed)

    a categorical variable

    

    RoofMatl (Roof material)

    values: (ClyTile: Clay or Tile); (CompShg: Standard (Composite) Shingle); (Membran: Membrane); (Metal: Metal);

    (Roll: Roll); (Tar&Grv: Gravel & Tar); (WdShake: Wood Shakes); (WdShngl: Wood Shingles);

    a categorical variable

'''

print(all_data.RoofStyle.value_counts())

print(all_data.RoofMatl.value_counts())
''' LandContour (Flatness of the property)

    values: (Lvl: Near Level); (Bnk: Banked - Quick and significant rise from street grade to building);

    (HLS: Hillside - Significant slope from side to side); (Low: Depression)

    a categorical variable



    LandSlope (Slope of property)

    values: (Gtl: Gentle slope); (Mod: Moderate Slope); (Sev: Severe Slope)

    an ordinal variable

'''

print(all_data.LandContour.value_counts())



all_data = all_data.replace({ "LandSlope" : {"Sev" : 0, "Mod" : 1, "Gtl":2}})

print(all_data.LandSlope.value_counts())
''' BldgType (Type of dwelling)

    values: (1Fam: Single-family Detached); (2FmCon: Two-family Conversion; originally built as one-family dwelling);

    (Duplx: Duplex); (TwnhsE: Townhouse End Unit); (TwnhsI: Townhouse Inside Unit)

    a categorical variable

'''

sns.barplot(data=all_data, x="BldgType", y="SalePrice", estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.ylabel('Median SalePrice');

print(all_data.BldgType.value_counts())



''' HouseStyle (Style of dwelling)

    values: (1Story: One story); (1.5Fin: One and one-half story: 2nd level finished);

    (1.5Unf: One and one-half story: 2nd level unfinished); (2Story: Two story);

    (2.5Fin: Two and one-half story: 2nd level finished); (2.5Unf: Two and one-half story: 2nd level unfinished)

    (SFoyer: Split Foyer); (SLvl: Split Level)

    a categorical variable

'''

print(all_data.HouseStyle.value_counts())
''' Neighborhood (Physical locations within Ames city limits)

    values: (Blmngtn: Bloomington Heights); (Blueste: Bluestem); (BrDale: Briardale) ...

    a categorical variable

    

    Condition1 (Proximity to various conditions)

    values: (Artery: Adjacent to arterial street); (Feedr: Adjacent to feeder street) ...

    a categorical variable

    

    Condition2 (Proximity to various conditions (if more than one is present))

    values: (Artery: Adjacent to arterial street); (Feedr: Adjacent to feeder street) ...

    a categorical variable

'''

# those are categorical variables. see in 5.2.2. section
''' Street (Type of road access to property)

    values: (Grvl: Gravel); (Pave: Paved)

    an ordinal variable

    

    PavedDrive (Paved driveway)

    values: (Y: Paved); (P: Partial Pavement); (N: Dirt/Gravel)

    an ordinal variable

'''

all_data = all_data.replace({ "Street" : {"Grvl" : 0, "Pave" : 1}})

print(all_data.Street.value_counts())



all_data = all_data.replace({ "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2}})

print(all_data.PavedDrive.value_counts())
# Check the tendency of the time data with SalePrice

plt.scatter(all_data['YearBuilt'], all_data['SalePrice'])

plt.show()  # old houses are worth less

sns.barplot(data=all_data, x="YrSold", y="SalePrice", estimator=np.median); 

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--')

plt.ylim(bottom=130000, top=180000); plt.ylabel('Median SalePrice');

plt.show()  # a little bit Banking crises after late 2007

sns.barplot(data=all_data, x="MoSold", y="SalePrice", estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--')

plt.ylim(bottom=75000); plt.ylabel('Median SalePrice');

plt.show()  # a little bit seasonal effect



# Change YearSold and MonthSold into categorical variables

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
''' MSSubClass (The building class) 

    values: '20' to '190'

    a categorical variable

'''

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data  =  all_data.replace({ "MSSubClass" : {'20' : '1 story 1946+', '30' : '1 story 1945-', '40' : '1 story unf attic', '45' : '1,5 story unf', '50' : '1,5 story fin',

                                                '60' : '2 story 1946+', '70' : '2 story 1945-', '75' : '2,5 story all ages', '80' : 'split/multi level', '85' : 'split foyer',

                                                '90' : 'duplex all style/age', '120' : '1 story PUD 1946+', '150' : '1,5 story PUD all', '160' : '2 story PUD 1946+',

                                                '180' : 'PUD multilevel', '190' : '2 family conversion'}})

all_data.MSSubClass.value_counts()
# index vector numeric / categorical variables

cols = all_data.columns

numeric = []

nonnumeric = []

list([ numeric.append(c) if all_data[c].dtype in ['int64', 'float64'] else nonnumeric.append(c) for c in cols])

print('There are', len(nonnumeric), 'remaining categorical variables (object types)')

print("------------")

print("categorical variables : ", nonnumeric)

# But I already handled with Alley, Electrcial, Exterior1st, Exterior2nd, GarageType, LotConfig, MSZoning, MiscFeature, SaleType, SaleCondition

# Let's focuse on Foundation, Heating, CentralAir, Roofs, LandSlope/Contour, BldgType, Neighborhood, Conditions, Street, PavedDrive
# correlations of some numeric variables

top_15_index = all_data.corr()['SalePrice'].sort_values(ascending=False)[:16].index  # select top 15 high correlations

plt.figure(figsize=(15, 6))

sns.heatmap(all_data[top_15_index].corr(), 

            annot=True, linewidths = 3, cbar=True, fmt=".2f")

plt.tick_params(labelsize=13); plt.gca().xaxis.tick_top() ; plt.xticks(rotation=90); plt.yticks(rotation=0); plt.show()
# Define features and target

X = all_data.iloc[0:1460, :] # train set

X = X.drop(['SalePrice'], axis=1)

y = all_data[['SalePrice']][0:1460]

#X = pd.get_dummies(X)

X = X.select_dtypes(exclude=['object']) # remove categorical variable

# X.info()

# y.info()



# Quick RandomForest to figure out variable importance

# https://stackoverflow.com/questions/50201913/using-scikit-learn-to-determine-feature-importances-per-class-in-a-rf-model

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor

from sklearn.metrics import mean_squared_error



#forest = ExtraTreesClassifier(n_estimators=100, random_state=1)

forest = RandomForestRegressor(n_estimators=100, random_state=1, criterion='mse')

forest.fit(X, y)



# Features importances

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1] # indices of features

feature_list = [X.columns[indices[f]] for f in range(X.shape[1])]  #names of features

ff = np.array(feature_list) # transforming into array



# Print the feature ranking

print("Feature ranking (importance values):")

for f in range(X.shape[1]):

    print("%d. feature %d (%f) name: %s" % (f + 1, indices[f], importances[indices[f]], ff[indices[f]]))



# Plot the feature importances of the forest unter ranking 20

plt.figure(figsize=(16, 6))

plt.title("Feature importances")

plt.bar(range(0,20), importances[indices][1:21],

       color="r", yerr=std[indices][0:20], align="center")

plt.xticks(range(0,20), ff[indices][0:20], rotation=90)

plt.show();

# the order of importance is GarageQual, BedroomAbvGr, BsmtFinType2, GrLivArea, GarageCars, OverallQual, ExterQual, GarageFinish, 

# HalfBath, MiscVal, HeatingQC, Fireplaces, Street, KitchenQual, BsmtFinishF2, Fullbath, LandSlope, BsmtCond, LowQualFinSF, FireplaceQu

# there might be, however, important categorical variables.



# Top 20 RF important variable list:

# there are 2 important Area variabbles: GrLivArea, LowQualFinSF

# Neighborhood is an important variable when including categorical variables

# there are 5 important Quality variables: (GarageQual), OverallQual, ExterQual, HeatingQC, KitchenQual, FireplaceQu

# MSSubClass is an important variable when including categorical variables

# there are 3 important Garage variables: GarageQual, GarageCars, GarageFinish, + (when including categorical variables) GarageType, GarageYrBlt

# there are 4 important Basement variables: BsmtFinType2, GarageFinish, BsmtFinishF2, BsmtCond

# Check these variables again.
plt.figure(figsize=(12, 20))



plt.subplot(4, 2, 1)

sns.distplot(all_data['GrLivArea'], hist=True, kde=True, bins=int(100/5), hist_kws={'edgecolor':'black'}) # seaborn histogram

plt.title('Histogram of Ground Living Area'); plt.xlabel('GrLivArea (square feet)'); plt.ylabel('density'); # Add labels



plt.subplot(4, 2, 2)

sns.distplot(all_data['TotRmsAbvGrd'], hist=True, kde=True, bins=int(100/5), hist_kws={'edgecolor':'black'})

plt.title('Histogram of Total Rooms Above Ground'); plt.xlabel('TotRmsAbvGrd (square feet)'); plt.ylabel('density'); 



plt.subplot(4, 2, 3)

sns.distplot(all_data['TotalBsmtSF'], hist=True, kde=True, bins=int(100/5), hist_kws={'edgecolor':'black'}) 

plt.title('Histogram of Total Basement Square Feet'); plt.xlabel('TotalBsmtSF (square feet)'); plt.ylabel('density'); 



plt.subplot(4, 2, 4)

sns.distplot(all_data['1stFlrSF'], hist=True, kde=True, bins=int(100/5), hist_kws={'edgecolor':'black'}) 

plt.title('Histogram of 1st Floor Square Feet'); plt.xlabel('1stFlrSF (square feet)'); plt.ylabel('density'); 



plt.subplot(4, 2, 5)

sns.distplot(all_data['2ndFlrSF'], hist=True, kde=True, bins=int(100/5), hist_kws={'edgecolor':'black'})

plt.title('Histogram of 2nd Floor Square Feet'); plt.xlabel('2ndFlrSF (square feet)'); plt.ylabel('density');



plt.subplot(4, 2, 6)

sns.distplot(all_data['LowQualFinSF'], hist=True, kde=True, bins=int(100/5), hist_kws={'edgecolor':'black'}) 

plt.title('Histogram of Low Quality Finished Square Feet'); plt.xlabel('LowQualFinSF (square feet)'); plt.ylabel('density');



plt.subplot(4, 2, 7)

sns.distplot(all_data[all_data.LotArea < 100000]['LotArea'], hist=True, kde=True, bins=int(100/5), hist_kws={'edgecolor':'black'})

plt.title('Histogram of Lot Area'); plt.xlabel('LotArea (square feet)'); plt.ylabel('density'); 

# 4 houses seems to be outliers, take out the lots above 100,000 square feet



plt.subplot(4, 2, 8)

sns.distplot(all_data['LotFrontage'], hist=True, kde=True, bins=int(100/5), hist_kws={'edgecolor':'black'})

plt.title('Histogram of Lot Frontage Area'); plt.xlabel('LotFrontage (square feet)'); plt.ylabel('density'); 

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show() 



#  There is a need to bundle them into some important variable

# cf. GarageArea is already taken care of in the Garage variable section
print(np.corrcoef(all_data['GrLivArea'], all_data['1stFlrSF'])[0,1]) # 0.5625

print(np.corrcoef(all_data['GrLivArea'], all_data['1stFlrSF']+all_data['2ndFlrSF'])[0,1]) # 0.9957

print(np.corrcoef(all_data['GrLivArea'], all_data['1stFlrSF']+all_data['2ndFlrSF']+all_data['LowQualFinSF'])[0,1]) # 1



# LowQualFinSF: Low quality finished square feet (all floors)

# LowQualFinSF area is included in the GrLivArea

# It seems to be able to create the new variable of "total ground area of house",

# which means the sum of GrLivArea (all of the above ground area) + TotalBsmtSF (all of the basement area)

# It is dealing with "total ground area of house" in the section 6.4.
plt.figure(figsize=(10, 5));

sns.barplot(data=all_data, x="Neighborhood", y="SalePrice", estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.xticks(rotation=90); plt.ylabel('Median SalePrice');

plt.show();



plt.figure(figsize=(10, 5));

sns.factorplot(data=all_data, x="Neighborhood", kind="count");

plt.xticks(rotation=90);

plt.show();
plt.figure(figsize=(10, 12))

plt.subplot(4, 2, 1)

sns.distplot(all_data['OverallQual'], hist=True, kde=False, hist_kws={'edgecolor':'black'}) # seaborn histogram

plt.xlabel('OverallQual'); plt.ylabel('count'); # Add labels



plt.subplot(4, 2, 2)

sns.distplot(all_data['ExterQual'], hist=True, kde=False, hist_kws={'edgecolor':'black'})

plt.xlabel('ExterQual'); plt.ylabel('count'); 



plt.subplot(4, 2, 3)

sns.distplot(all_data['BsmtQual'], hist=True, kde=False, hist_kws={'edgecolor':'black'}) 

plt.xlabel('BsmtQual'); plt.ylabel('count'); 



plt.subplot(4, 2, 4)

sns.distplot(all_data['KitchenQual'], hist=True, kde=False, hist_kws={'edgecolor':'black'}) 

plt.xlabel('KitchenQual'); plt.ylabel('count'); 



plt.subplot(4, 2, 5)

sns.distplot(all_data['GarageQual'], hist=True, kde=False, hist_kws={'edgecolor':'black'})

plt.xlabel('GarageQual'); plt.ylabel('count');



plt.subplot(4, 2, 6)

sns.distplot(all_data['FireplaceQu'], hist=True, kde=False, hist_kws={'edgecolor':'black'}) 

plt.xlabel('FireplaceQu'); plt.ylabel('count');



plt.subplot(4, 2, 7)

sns.distplot(all_data['PoolQC'], hist=True, kde=False, hist_kws={'edgecolor':'black'})

plt.xlabel('PoolQC'); plt.ylabel('count'); 

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show() 



# Overall Quality is more granular than the others

# External Quality is a high correlation with Overall Quality

# PoolQC seems to be able to be separated into 'has pool' and 'no pool'
plt.figure(figsize=(10, 5));

sns.barplot(data=all_data, x="MSSubClass", y="SalePrice", estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.xticks(rotation=90); plt.ylabel('Median SalePrice');

plt.show();



plt.figure(figsize=(10, 5));

#sns.catplot(data=all_data, x="MSSubClass", kind="count");

sns.factorplot(data=all_data, x="MSSubClass", kind="count");

plt.xticks(rotation=90);

plt.show();
# the value '2207' in GarageYrBlt seems to be the typo for '2007'

# because YearBuilt=2006, YearRomodAdd=2007 in the house

all_data["GarageYrBlt"] = np.where(all_data["GarageYrBlt"] == 2207, 2007, all_data["GarageYrBlt"])



plt.figure(figsize=(10, 12))

plt.subplot(4, 2, 1)

sns.distplot(all_data[all_data.GarageCars != 0]['GarageYrBlt'], hist=True, kde=False, color="grey", hist_kws={'edgecolor':'black'}) # seaborn histogram

plt.xlabel('GarageYrBlt', fontsize=12); plt.ylabel('count', fontsize=12); # Add labels



plt.subplot(4, 2, 5)

sns.distplot(all_data['GarageCars'], hist=True, kde=False, color="grey", hist_kws={'edgecolor':'black'})

plt.xlabel('GarageCars', fontsize=12); plt.ylabel('count', fontsize=12); 



plt.subplot(4, 2, 3)

sns.distplot(all_data['GarageArea'], hist=True, kde=False, color="grey", hist_kws={'edgecolor':'black'}) 

plt.xlabel('GarageArea', fontsize=12); plt.ylabel('count', fontsize=12); 



plt.subplot(4, 2, 4)

sns.distplot(all_data['GarageCond'], hist=True, kde=False, color="grey", hist_kws={'edgecolor':'black'}) 

plt.xlabel('GarageCond', fontsize=12); plt.ylabel('count', fontsize=12); 



plt.subplot(4, 2, 2)

#sns.distplot(all_data['GarageType'], hist=True, kde=False, hist_kws={'edgecolor':'black'})

sns.countplot(all_data['GarageType'], color="grey")

plt.xlabel('GarageType', fontsize=12); plt.ylabel('count', fontsize=12);



plt.subplot(4, 2, 6)

sns.distplot(all_data['GarageQual'], hist=True, kde=False, color="grey", hist_kws={'edgecolor':'black'}) 

plt.xlabel('GarageQual', fontsize=12); plt.ylabel('count', fontsize=12);



plt.subplot(4, 2, 7)

sns.distplot(all_data['GarageFinish'], hist=True, kde=False, color="grey", hist_kws={'edgecolor':'black'})

plt.xlabel('GarageFinish', fontsize=12); plt.ylabel('count', fontsize=12); 

plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show() 



# GarageCars and GarageArea are highly correlated

# GarageQual and GarageCond also seem highly correlated
plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)

sns.distplot(all_data['BsmtFinSF1'], hist=True, kde=False, hist_kws={'edgecolor':'black'}) # seaborn histogram

plt.xlabel('Type 1 finished square feet', fontsize=12); plt.ylabel('count', fontsize=12); # Add labels



plt.subplot(3, 3, 2)

sns.distplot(all_data['BsmtFinSF2'], hist=True, kde=False, hist_kws={'edgecolor':'black'})

plt.xlabel('Type 2 finished square feet', fontsize=12); plt.ylabel('count', fontsize=12); 



plt.subplot(3, 3, 3)

sns.distplot(all_data['BsmtUnfSF'], hist=True, kde=False, hist_kws={'edgecolor':'black'}) 

plt.xlabel('Unfinished square feet', fontsize=12); plt.ylabel('count', fontsize=12); 



plt.subplot(3, 3, 4)

sns.distplot(all_data['BsmtFinType1'], hist=True, kde=False, hist_kws={'edgecolor':'black'}) 

plt.xlabel('Rating of Type 1 finished area', fontsize=12); plt.ylabel('count', fontsize=12); plt.xticks(all_data.BsmtFinType1);



plt.subplot(3, 3, 5)

sns.distplot(all_data['BsmtFinType2'], hist=True, kde=False, hist_kws={'edgecolor':'black'})

plt.xlabel('Rating of Type 2 finished area', fontsize=12); plt.ylabel('count', fontsize=12); plt.xticks(all_data.BsmtFinType2);



plt.subplot(3, 3, 7)

sns.distplot(all_data['BsmtQual'], hist=True, kde=False, hist_kws={'edgecolor':'black'}) 

plt.xlabel('Height of the basement', fontsize=12); plt.ylabel('count', fontsize=12);



plt.subplot(3, 3, 8)

sns.distplot(all_data['BsmtCond'], hist=True, kde=False, hist_kws={'edgecolor':'black'})

plt.xlabel('Rating of general condition', fontsize=12); plt.ylabel('count', fontsize=12); 



plt.subplot(3, 3, 9)

sns.distplot(all_data['BsmtExposure'], hist=True, kde=False, hist_kws={'edgecolor':'black'})

plt.xlabel('Walkout or garden level walls', fontsize=12); plt.ylabel('count', fontsize=12); 

plt.show() ;



# Total Basement Surface seems as the sum of Type 1 & 2 finished sqft and Unfinished sqft

print(np.corrcoef(all_data['TotalBsmtSF'], all_data['BsmtFinSF1'])[0,1]) # corr. coef. = 0.5366

print(np.corrcoef(all_data['TotalBsmtSF'], all_data['BsmtFinSF1']+all_data['BsmtFinSF2'])[0,1]) # 0.5441

print(np.corrcoef(all_data['TotalBsmtSF'], all_data['BsmtFinSF1']+all_data['BsmtFinSF2']+all_data['BsmtUnfSF'])[0,1]) # 1
''' BsmtFullBath (Basement full bathrooms) 

    BsmtHalfBath (Basement half bathrooms)

    FullBath (Full bathrooms above grade)

    HalfBath (Half baths above grade)

'''

# cf. Definition of half bath: a bathroom containing a sink and toilet but no bathtub or shower (by Merriam-Webster)

# cf. Definition of full bath: a bathroom with a sink, toilet, and a bathtub or shower (by Merriam-Webster)

# Thus, the half baths seem to be counted as half, functionally

all_data['TotBathrooms'] = all_data.FullBath + all_data.HalfBath*0.5 + all_data.BsmtFullBath + all_data.BsmtHalfBath*0.5

all_data.TotBathrooms.value_counts()
# Scatter plot with SalePrice

sns.regplot(x=all_data.TotBathrooms, y=all_data['SalePrice'], fit_reg=True, line_kws={"color": "grey"})

plt.xlabel('TotBathrooms', fontsize=12)

plt.ylabel('SalePrice', fontsize=12)

plt.show()



sns.countplot(all_data.TotBathrooms.astype(str), color="grey")

plt.xlabel('TotBathrooms', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.show() ;
# Remodel(Y/N) : 0=No Remodeling, 1=Remodeling

all_data.loc[all_data['YearBuilt'] == all_data['YearRemodAdd'], 'Remod'] = 0 

all_data.loc[all_data['YearBuilt'] != all_data['YearRemodAdd'], 'Remod'] = 1



# house age

all_data['Age'] = all_data.YrSold.astype(int) - all_data.YearRemodAdd.astype(int) # create the age of house

all_data['YrSold'] = all_data['YrSold'].astype('str')



# is new at that saled time? (Old/New) : 0=Old Houses, 1=New Houses

all_data.loc[all_data['YrSold'].astype(int) == all_data['YearBuilt'].astype(int), 'IsNew'] = 1 

all_data.loc[all_data['YrSold'].astype(int) != all_data['YearBuilt'].astype(int), 'IsNew'] = 0

all_data['IsNew'] = all_data['IsNew'].astype(str)

all_data.IsNew.value_counts() # 116 new houses



all_data.loc[:,['YearBuilt','YearRemodAdd','YrSold','Age','Remod', 'IsNew']]

# there is no new house at that saled time
# Scatter plot of Age with SalePrice

sns.regplot(x=all_data.Age, y=all_data['SalePrice'], fit_reg=True, line_kws={"color": "black"})

plt.xlabel('Age', fontsize=12)

plt.ylabel('SalePrice', fontsize=12)

plt.show();

# negative corrleation with Age (old houses are worth less)

all_dataSP = all_data.loc[all_data['SalePrice'].notnull(), ['SalePrice','Age']]

print(np.corrcoef(all_dataSP.Age, all_dataSP.SalePrice)[0,1]) # -0.5090
# barplot of Remodeled houses with the SalePrice

sns.barplot(data=all_data, x='Remod', y='SalePrice', estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.ylabel('Median SalePrice'); plt.ylim(bottom=100000)

plt.show();

# remodeled houses are cheaper



# barplot of newness with the SalePrice

sns.barplot(data=all_data, x='IsNew', y='SalePrice', estimator=np.median);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.ylabel('Median SalePrice'); 

plt.show();

# remodeled houses are cheaper
order = all_data.groupby('Neighborhood')['SalePrice'].agg(['mean', 'median'])



plt.figure(figsize=(10, 5));

sns.barplot(data=all_data, x="Neighborhood", y="SalePrice", estimator=np.median, order = order.sort_values(by='median', ascending=True).index);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.xticks(rotation=45); plt.ylabel('Median SalePrice');

plt.show();



plt.figure(figsize=(10, 5));

sns.barplot(data=all_data, x="Neighborhood", y="SalePrice", estimator=np.mean, order = order.sort_values(by='mean', ascending=True).index);

plt.axhline(y=np.median(train.SalePrice), color="red", linestyle = '--');

plt.xticks(rotation=45); plt.ylabel('Mean SalePrice');

plt.show();



# top 3 neighborhoods are able to merge to a rich group in both the median and mean SalePrice

# bottom 3 neighborhoods are also able to merge to a poor group

# but others keep remaining to prevent from the overbinning

all_data.loc[all_data.Neighborhood.isin(["StoneBr", "NridgHt" ,"NoRidge"]), 'NeighRich'] = 2

all_data.loc[all_data.Neighborhood.isin(["StoneBr", "NridgHt" ,"NoRidge", "MeadowV", "IDOTRR" ,"BrDale"]) == False, 'NeighRich'] = 1

all_data.loc[all_data.Neighborhood.isin(["MeadowV", "IDOTRR" ,"BrDale"]), 'NeighRich'] = 0

all_data['NeighRich'] = all_data['NeighRich'].astype(str)

print(all_data.NeighRich.value_counts())
all_data['TotalSqFeet'] = all_data['TotalBsmtSF'].astype(float) + all_data['GrLivArea'].astype(float)



# Scatter plot

sns.regplot(x=all_data['TotalSqFeet'], y=all_data['SalePrice'], fit_reg=True, line_kws={"color": "grey"})

plt.xlabel('TotalSqFeet', fontsize=12)

plt.ylabel('SalePrice', fontsize=12)

plt.show()

# It seems to be 2 outliers here (TotalSqFeet > 7500)

all_data[(all_data.TotalSqFeet > 7500)][['TotalSqFeet','SalePrice']]

# the index number of outlier candidates : 523, 1298 (see section 7.2.)



# correlation coefficient

all_dataTSF = all_data.loc[all_data['SalePrice'].notnull(), ['SalePrice','TotalSqFeet']]

print(np.corrcoef(all_dataTSF.TotalSqFeet, all_dataTSF.SalePrice)[0,1]) # 0.7789
''' WoodDeckSF (Wood deck area in square feet) : is a type of unsheltered porch

    OpenPorchSF (Open porch area in square feet) : 

    EnclosedPorch (Enclosed porch area in square feet)

    3SsnPorch (Three season porch area in square feet)

    ScreenPorch (Screen porch area in square feet)

'''

# consolidating the 4 porch variables except WoodDeck porch

all_data['TotalPorchSF'] = all_data.OpenPorchSF + all_data.EnclosedPorch + all_data['3SsnPorch'] + all_data.ScreenPorch



# Scatter plot

sns.regplot(x=all_data['TotalPorchSF'], y=all_data['SalePrice'], fit_reg=True, line_kws={"color": "grey"})

plt.xlabel('TotalPorchSF', fontsize=12)

plt.ylabel('SalePrice', fontsize=12)

plt.show()



# correlation coefficient

all_dataTP = all_data.loc[all_data['SalePrice'].notnull(), ['SalePrice','TotalPorchSF']]

print(np.corrcoef(all_dataTP.TotalPorchSF, all_dataTP.SalePrice)[0,1]) # 0.1957
# correlations of some numeric variables

top_15_index = all_data.corr()['SalePrice'].sort_values(ascending=False)[:16].index  # select top 11 high correlations

plt.figure(figsize=(15, 6))

sns.heatmap(all_data[top_15_index].corr(), 

            annot=True, linewidths = 3, cbar=True, fmt=".2f")

plt.tick_params(labelsize=13); plt.gca().xaxis.tick_top() ; plt.xticks(rotation=90); plt.yticks(rotation=0); plt.show()



# dropping a variable if two or more variables are highly correlated

# GarageArea and GarageCars have a correlation of 0.89 (section 5.1.)

# GarageArea has a SalePrice correlation of 0.62. GarageCars has a SalePrice correlation of 0.64. Thus, choose the GarageCars.

# YearRemodAdd is dealt with in the the age of house (section 6.2.)

# TotalBsmtSF, TotRmsAbvGrd, BsmtFinSF1, 1stFlrSF, GrLivArea is dealt with in 'TotalSqFeet' (in the section 5.2.1.; 5.2.6.; 6.1.; 6.4.)

# ExterQual and KitchenQual have a high correlation of 0.87, but those are meaningly distinguishable.

# FullBath and TotBathrooms are highly correlated to 0.71, and FullBath are included to TotBathrooms.

# FireplaceQu and Fireplaces have a high correlation of 0.86

# FireplaceQu has a SalePrice correlation of 0.52, Fireplaces has a SalePrice correlation of 0.47. Thus, choose the FireplaceQu.

# MasVnrType and MasVnrArea have a high correlation of 0.62, and MasVnrArea is more correlated with SalePrice



all_data.drop(['YearRemodAdd','GarageYrBlt', 'GarageArea', 'GarageCond','TotalBsmtSF', 'FullBath', 

               'TotRmsAbvGrd', 'BsmtFinSF1', '1stFlrSF', 'Fireplaces', 'MasVnrType'], axis=1, inplace=True)
# drop two outliers in TotalSqFeet

all_data.drop([523, 1298], inplace=True)



# Check the plot again

plt.scatter(all_data['TotalSqFeet'], all_data['SalePrice']);
# Define features and target of train set

X1 = all_data.iloc[0:1458, :]

X1 = X1.drop(['SalePrice'], axis=1)

y1 = all_data.iloc[0:1458, :]['SalePrice'] # save target variable

X1_num = X1.select_dtypes(exclude=['object']) # remove categorical 

X1_cat = X1.select_dtypes(include=['object']) # save categorical



# Define features and target of test set

X2 = all_data.iloc[1458:, :]

X2 = X2.drop(['SalePrice'], axis=1)

y2 = all_data.iloc[1458:, :]['SalePrice'] # save target variable

X2_num = X2.select_dtypes(exclude=['object']) # remove categorical 

X2_cat = X2.select_dtypes(include=['object']) # save categorical



# save the indexes and columns

X1_index = X1_num.index

X1_columns = X1_num.columns

X2_index = X2_num.index

X2_columns = X2_num.columns



# distinguishing (true) numeric predictors and ordinal factors

print('There are', len(X1_num.columns), 'numeric variables before having done anything')

print("numeric variables : ", X1_num.columns)

print("------------")



print('There are', len(X1_cat.columns), 'categorical variables before having done anything')

print("categorical variables : ", X1_cat.columns)
X1X2_num = pd.concat([X1_num,X2_num], ignore_index=True)



# log+1 transformation some numerical variables whose skewness > 0.8

skewness = X1X2_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) >0.8]

print(str(skewness.shape[0]) + " skewed numerical features to log+1 transform")

skewed_features = skewness.index

X1X2_num[skewed_features] = np.log1p(X1X2_num[skewed_features])



X1_num = X1X2_num.iloc[0:1458, :]

X2_num = X1X2_num.iloc[1458:, :]
#np.isnan(X1_num.any())  # False = no NaN

#np.isfinite(X1_num.all()) # True = no infinite

#np.isnan(X2_num.any())  # False = no NaN

#np.isfinite(X2_num.all()) # True = no infinite



# Scale features for improved regularisation performance

from sklearn.preprocessing import StandardScaler



# Fit scaler (using MinMaxScaler as not all variables are normal distributions) to training set mean and variance

scaler = StandardScaler()

scaler.fit(X1_num)

scaler.fit(X2_num)



# Transform both the training and testing sets

X1_scaled = scaler.transform(X1_num)

X2_scaled = scaler.transform(X2_num)



# Put scaled data back into a pandas dataframe

X1_num = pd.DataFrame(X1_scaled, index = [i for i in range(X1_scaled.shape[0])], 

                  columns = ['f'+str(k) for k in range(X1_scaled.shape[1])])

X2_num = pd.DataFrame(X2_scaled, index = [i for i in range(X2_scaled.shape[0])], 

                  columns = ['f'+str(k) for k in range(X2_scaled.shape[1])])



# rename the rows and columns of the dataframes

X1_num.index = X1_index

X2_num.index = X2_index

X1_num.columns = X1_columns

X2_num.columns = X2_columns

print("X1 :", X1.shape, ", X2 :", X2.shape)



# merge the numerical, categorical variables

all_data_num = pd.concat([X1_num,X2_num], ignore_index=True) # (2917, 49)

all_data_cat = pd.concat([X1_cat,X2_cat], ignore_index=True) # (2917, 25)
'''# https://scikit-learn.org/stable/modules/preprocessing.html#

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html

# (in Korean) https://mkjjo.github.io/python/2019/01/10/scaler.html

# (in Korean) https://datascienceschool.net/view-notebook/afb99de8cc0d407ba32079590b25180d/ 

# (in Korean) http://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221293217463&categoryNo=49&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=search



plt.figure(figsize=(10, 5));

sns.pairplot(all_data, y_vars="SalePrice", x_vars=all_data_num.columns[0:9], diag_kind="kde");

sns.pairplot(all_data, y_vars="SalePrice", x_vars=all_data_num.columns[10:19], diag_kind="kde");

sns.pairplot(all_data, y_vars="SalePrice", x_vars=all_data_num.columns[20:29], diag_kind="kde");

sns.pairplot(all_data, y_vars="SalePrice", x_vars=all_data_num.columns[30:39], diag_kind="kde");

sns.pairplot(all_data, y_vars="SalePrice", x_vars=all_data_num.columns[40:49], diag_kind="kde");'''
print('check the data size before making dummies: {}'.format(all_data_cat.shape)) # (2917, 25)

all_data_cat = pd.get_dummies(all_data_cat) 

print('check the data size after making dummies: {}'.format(all_data_cat.shape)) # (2917, 198)



#np.isnan(all_data2_cat.any())  # False = no NaN

#np.isfinite(all_data2_cat.all()) # True = no infinite
# check if some values are absent or few observation

cols = all_data_cat.columns

DropCat1 = []

RemainCatList = []

list([ DropCat1.append(c) if all_data_cat[c].mean() < 0.01  else RemainCatList.append(c) for c in cols])

DropCatList = []

list([ DropCatList.append(c) if  all_data_cat[c].std() < 0.1 else RemainCatList.append(c) for c in DropCat1])

len(DropCatList) # predictor lists to drop 



# drop predictors from dummies

all_data_cat.drop(DropCatList, axis=1, inplace=True)

all_data_cat.info() # from 198 variables to 129 variables
# Check normality

y1.skew(axis = 0, skipna = True) # too high skewness = 1.8812

sns.distplot(y1, fit=norm); # positive skewness(right skewed), curtosis(peakedness)



# Check the normal probability plot (test normality of the RESIDUALS not of the data)

fig = plt.figure()

res = stats.probplot(y1, plot=plt) # non normality

plt.show()





# Log+1 transformation of the DV(SalePrice)

y1 = np.log1p(y1) # log1p: appling log(1+x), because of x=0



# Check normality again

y1.skew(axis = 0, skipna = True) # too high skewness = 1.8812



# Check again the distribution

sns.distplot(y1, fit=norm);



# Check again the normal probability plot

fig = plt.figure()

res = stats.probplot(y1, plot=plt) # got normality

plt.show();
# merge the terget variables

all_data = pd.concat([all_data_num, all_data_cat], axis=1) # (2917, 178)

print(all_data.shape)



# merge the terget variables

all_data_y = pd.concat([y1,y2], ignore_index=True) # (2917, )

all_data_y = all_data_y.to_frame()

all_data = pd.concat([all_data, all_data_y], axis=1)

print(all_data.shape)



# composing train and test sets

train1 = all_data.loc[all_data[all_data['SalePrice'].notnull()].index,:]

test1 = all_data.loc[all_data[all_data['SalePrice'].isnull()].index,:]

train1.info() # (1458, 179)

test1.info() # (1459, 179)
# Import librairies

from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNet, Lasso, BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error, make_scorer, r2_score

import xgboost as xgb

import lightgbm as lgb





# Split the train and test sets

X_train, X_test, y_train, y_test = train_test_split(train1[[column for column in all_data.columns if column != "SalePrice"]],

                                                             train1[["SalePrice"]], test_size=0.3, shuffle=True, random_state=42)

print(X_train.shape) #(1020, 178)

print(y_train.shape) #(1020, 1)

print(X_test.shape) #(438, 178)

print(y_test.shape) #(438, 1)



test1.drop(['SalePrice'], axis=1, inplace=True)
# Define Validation function

scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 50))

    return(rmse)

def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 50))

    return(rmse)



# LASSO Regression

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)) # RobustScaler(): scailing with median to 0, IQR to 1



# Elastic Net Regression

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



# Kernel Ridge Regression (non-parametric form of ridge regression)

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



# Gradient Boosting Regression

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5) # loss='huber' : makes it robust to outliers

# XGBoost

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,random_state =7, nthread = -1)

# LightGBM

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5, learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



# scores by evaluating the cross-validation rmse error

score_lasso_train = rmse_cv_train(lasso)

score_lasso_test = rmse_cv_test(lasso)

print("RMSE on Train set in Lasso score: {:.4f} ({:.4f})".format(score_lasso_train.mean(), score_lasso_train.std()))

print("RMSE on Test set in Lasso score: {:.4f} ({:.4f})\n".format(score_lasso_test.mean(), score_lasso_test.std()))

#RMSE on Train set in Lasso score: 0.1036 (0.0545)

#RMSE on Test set in Lasso score: 0.1005 (0.0753)



score_ENet_train = rmse_cv_train(ENet)

score_ENet_test = rmse_cv_test(ENet)

print("RMSE on Train set in ENet score: {:.4f} ({:.4f})".format(score_ENet_train.mean(), score_ENet_train.std()))

print("RMSE on Test set in ENet score: {:.4f} ({:.4f})\n".format(score_ENet_test.mean(), score_ENet_test.std()))

#RMSE on Train set in ENet score: 0.1037 (0.0545)

#RMSE on Test set in ENet score: 0.1008 (0.0756)



score_KRR_train = rmse_cv_train(KRR)

score_KRR_test = rmse_cv_test(KRR)

print("RMSE on Train set in KRR score: {:.4f} ({:.4f})".format(score_KRR_train.mean(), score_KRR_train.std()))

print("RMSE on Test set in KRR score: {:.4f} ({:.4f})\n".format(score_KRR_test.mean(), score_KRR_test.std()))

#RMSE on Train set in KRR score: 0.1000 (0.0553)

#RMSE on Test set in KRR score: 0.0986 (0.0723)



score_GBoost_train = rmse_cv_train(GBoost)

score_GBoost_test = rmse_cv_test(GBoost)

print("RMSE on Train set in GBoost score: {:.4f} ({:.4f})".format(score_GBoost_train.mean(), score_GBoost_train.std()))

print("RMSE on Test set in GBoost score: {:.4f} ({:.4f})\n".format(score_GBoost_test.mean(), score_GBoost_test.std()))

#RMSE on Train set in GBoost score: 0.1165 (0.0253)

#RMSE on Test set in GBoost score: 0.1287 (0.0420)



score_xgb_train = rmse_cv_train(model_xgb)

score_xgb_test = rmse_cv_test(model_xgb)

print("RMSE on Train set in xgb score: {:.4f} ({:.4f})".format(score_xgb_train.mean(), score_xgb_train.std()))

print("RMSE on Test set in xgb score: {:.4f} ({:.4f})".format(score_xgb_test.mean(), score_xgb_test.std()))

#RMSE on Train set in xgb score: 0.1152 (0.0272)

#RMSE on Test set in xgb score: 0.1277 (0.0410)
# figure out important variable with GBRegressior

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.ensemble import GradientBoostingRegressor

# Define a model and calculate permutation importance of all numeric columns

simpleGBR = GradientBoostingRegressor(max_depth=4,random_state=0)

simpleGBR.fit(X_train,y_train)

perm = PermutationImportance(simpleGBR, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist(),top=30)

# source: https://www.kaggle.com/nischaydnk/beginners-top-4-with-simple-xgboost
'''# import warnings

# warnings.filterwarnings('ignore')

# conda install -c anaconda py-xgboost

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer

#from sklearn.preprocessing import CategoricalEncoder

#import category_encoders as ce

from xgboost import XGBRegressor

# install in cmd : pip install https://github.com/scikit-learn/scikit-learn/archive/master.zip 



# Preprocessing for numerical data

numerical_transformer = Pipeline(verbose=False,steps=[

    ('imputer_num', SimpleImputer(strategy='median')),

#     ('remove_outlier', OutlierExtractor())

])



# Preprocessing for categorical data

categorical_onehot_transformer = Pipeline(verbose=False,steps=[

    ('imputer_onehot', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



categorical_label_transformer = Pipeline(verbose=False,steps=[

    ('imputer_label', SimpleImputer(strategy='most_frequent')),

    ('label', ce.OrdinalEncoder())

])



categorical_count_transformer = Pipeline(verbose=False,steps=[

    ('imputer_count', SimpleImputer(strategy='most_frequent')),

    ('count', ce.TargetEncoder(handle_missing='count'))

#     ('count', ce.CountEncoder(min_group_size = 1,handle_unknown=0,handle_missing='count'))

])



numerical_cols = all_data_num.columns



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(verbose=False,

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cox_box', PowerTransformer(method='yeo-johnson', standardize=False),skew_cols),

        ('cat_label', categorical_label_transformer, categorical_label_cols),

#        ('cat_onehot', categorical_onehot_transformer, categorical_small_variety_cols),

#        ('cat_count', categorical_count_transformer, categorical_large_variety_cols),

    ])

train_pipeline = Pipeline(verbose=False,steps=[

                    ('preprocessor', preprocessor),   

                    ('scale', StandardScaler(with_mean=True,with_std=True)),

                    ('model', XGBRegressor(random_state=0))

                    ])



param_grid = {'model__nthread':[2], #when use hyperthread, xgboost may become slower

              'model__learning_rate': [0.04, 0.05], #so called `eta` value

              'model__max_depth': range(3,5,1),

              "model__min_child_weight" : [ 1 ],

              "model__gamma"            : [ 0.0],

              "model__colsample_bytree" : [ 0.2 ],

              'model__silent': [1],

              'model__n_estimators': [500], #number of trees

             }



searched_model = GridSearchCV(estimator=train_pipeline, param_grid = param_grid, scoring="neg_mean_absolute_error", cv=20, error_score='raise', verbose = 1)

searched_model.fit(X_train,y_train)

preds_test = searched_model.predict(test1)

print(searched_model.best_estimator_)

print(searched_model.best_score_)



# Save test predictions to file

result = pd.DataFrame({'Id':test_labels,'SalePrice':lasso_pred})

result.to_csv('submission_lasso.csv',index=False)

result.head(5)'''
# defininge a rmse evaluation function

def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



# Prediction

lasso.fit(X_train, y_train)

lasso_train_pred = lasso.predict(X_train)

lasso_test_pred = lasso.predict(X_test)

lasso_pred = np.expm1(lasso.predict(test1))  # np.expm1() : the inverse of log(1 + x) transformation

print(rmse(y_train, lasso_train_pred)) # 0.101888672306

print(r2_score(y_train, lasso_train_pred)) # 0.935403976687

print(rmse(y_test, lasso_test_pred)) # 0.113172780464

print(r2_score(y_test, lasso_test_pred), "\n") # 0.918469439086 

# public score : 0.12483



ENet.fit(X_train, y_train)

ENet_train_pred = ENet.predict(X_train)

ENet_test_pred = ENet.predict(X_test)

ENet_pred = np.expm1(ENet.predict(test1))

print(rmse(y_train, ENet_train_pred)) # 0.10154062783

print(r2_score(y_train, ENet_train_pred)) # 0.935844533808

print(rmse(y_test, ENet_test_pred)) # 0.11338085492

print(r2_score(y_test, ENet_test_pred), "\n") # 0.918169366541 

# public score : 0.12490



KRR.fit(X_train, y_train)

KRR_train_pred = KRR.predict(X_train)

KRR_test_pred = KRR.predict(X_test)

KRR_pred = np.expm1(KRR.predict(test1))

print(rmse(y_train, KRR_train_pred)) # 0.0855966565867

print(r2_score(y_train, KRR_train_pred)) # 0.954410211299

print(rmse(y_test, KRR_test_pred), "\n") # 0.108868250814

#print(r2_score(y_test, KRR_test_pred, "\n"))

# public score : 0.12206



GBoost.fit(X_train, y_train)

GBoost_train_pred = GBoost.predict(X_train)

GBoost_test_pred = GBoost.predict(X_test)

GBoost_pred = np.expm1(GBoost.predict(test1))

print(rmse(y_train, GBoost_train_pred)) # 0.0442762384138

print(r2_score(y_train, GBoost_train_pred)) # 0.98780180951

print(rmse(y_test, GBoost_test_pred)) # 0.119147642387

print(r2_score(y_test, GBoost_test_pred), "\n") # 0.909633520183 

# public score : 0.12857



model_xgb.fit(X_train, y_train)

model_xgb_train_pred = model_xgb.predict(X_train)

model_xgb_test_pred = model_xgb.predict(X_test)

model_xgb_pred = np.expm1(model_xgb.predict(test1))

print(rmse(y_train, model_xgb_train_pred)) # 0.0803106859358

print(r2_score(y_train, GBoost_train_pred)) # 0.98780180951

print(rmse(y_test, model_xgb_test_pred)) # 0.119701106331

print(r2_score(y_test, model_xgb_test_pred), "\n") # 0.908792030566

# public score : 0.12282
# averaging predictions while choosing Lasso and XGB model

# Averaged = lasso_pred*0.60 + model_xgb_pred*0.40



KRR_pred = np.squeeze(KRR_pred)



# Save test predictions to file

result = pd.DataFrame({'Id':test_labels,'SalePrice':KRR_pred})

result.to_csv('submission_gbr50.csv',index=False)

result.head(5)