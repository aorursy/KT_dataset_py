# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

#%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# The error metric: RMSE on the log of the sale prices.

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))

# Any results you write to the current directory are saved as output.
# I.data pre-processing part

# Load the data.

dtrain = pd.read_csv('../input/train.csv')

dtest = pd.read_csv('../input/test.csv')

dtrain.head()
dtrain.describe()
# There are a few houses with more than 4000 sq ft living area that are

# outliers, so we drop them from the training data. (There is also one in

# the test set but we obviously can't drop that one.)

dtrain.drop(dtrain[dtrain["GrLivArea"] > 4000].index, inplace=True)



# The test example with ID 666 has GarageArea, GarageCars, and GarageType 

# but none of the other fields, so use the mode and median to fill them in.

dtest.loc[666, "GarageQual"] = "TA"

dtest.loc[666, "GarageCond"] = "TA"

dtest.loc[666, "GarageFinish"] = "Unf"

dtest.loc[666, "GarageYrBlt"] = "1980"

# The test example 1116 only has GarageType but no other information. We'll 

# assume it does not have a garage.

dtest.loc[1116, "GarageType"] = np.nan

# For imputing missing values: fill in missing LotFrontage values by the median

# LotFrontage of the neighborhood.

lot_frontage_by_neighborhood = dtrain["LotFrontage"].groupby(dtrain["Neighborhood"])
# Numerical Coding:

# convert categorical features into ordinal numbers.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



def factorize(df, factor_df, column, fill_na=None):

    factor_df[column] = df[column]

    if fill_na is not None:

        factor_df[column].fillna(fill_na, inplace=True)

    le.fit(factor_df[column].unique())

    factor_df[column] = le.transform(factor_df[column])

    return factor_df



# Combine all the (numerical) features into one big DataFrame. We don't add 

# the one-hot encoded variables here yet, that happens later on.

def munge(df):

    all_df = pd.DataFrame(index = df.index)

   

    all_df["LotFrontage"] = df["LotFrontage"]   

    for key, group in lot_frontage_by_neighborhood:

        idx = (df["Neighborhood"] == key) & (df["LotFrontage"].isnull())

        all_df.loc[idx, "LotFrontage"] = group.median()    



    all_df["LotArea"] = df["LotArea"]



    all_df["MasVnrArea"] = df["MasVnrArea"]

    all_df["MasVnrArea"].fillna(0, inplace=True)

   

    all_df["BsmtFinSF1"] = df["BsmtFinSF1"]

    all_df["BsmtFinSF1"].fillna(0, inplace=True)



    all_df["BsmtFinSF2"] = df["BsmtFinSF2"]

    all_df["BsmtFinSF2"].fillna(0, inplace=True)



    all_df["BsmtUnfSF"] = df["BsmtUnfSF"]

    all_df["BsmtUnfSF"].fillna(0, inplace=True)



    all_df["TotalBsmtSF"] = df["TotalBsmtSF"]

    all_df["TotalBsmtSF"].fillna(0, inplace=True)



    all_df["1stFlrSF"] = df["1stFlrSF"]

    all_df["2ndFlrSF"] = df["2ndFlrSF"]

    all_df["GrLivArea"] = df["GrLivArea"]

    

    all_df["GarageArea"] = df["GarageArea"]

    all_df["GarageArea"].fillna(0, inplace=True)



    all_df["WoodDeckSF"] = df["WoodDeckSF"]

    all_df["OpenPorchSF"] = df["OpenPorchSF"]

    all_df["EnclosedPorch"] = df["EnclosedPorch"]

    all_df["3SsnPorch"] = df["3SsnPorch"]

    all_df["ScreenPorch"] = df["ScreenPorch"]

    

    all_df["BsmtFullBath"] = df["BsmtFullBath"]

    all_df["BsmtFullBath"].fillna(0, inplace=True)



    all_df["BsmtHalfBath"] = df["BsmtHalfBath"]

    all_df["BsmtHalfBath"].fillna(0, inplace=True)



    all_df["FullBath"] = df["FullBath"] 

    all_df["HalfBath"] = df["HalfBath"] 

    all_df["BedroomAbvGr"] = df["BedroomAbvGr"] 

    all_df["KitchenAbvGr"] = df["KitchenAbvGr"] 

    all_df["TotRmsAbvGrd"] = df["TotRmsAbvGrd"] 

    all_df["Fireplaces"] = df["Fireplaces"] 



    all_df["GarageCars"] = df["GarageCars"]

    all_df["GarageCars"].fillna(0, inplace=True)



    all_df["CentralAir"] = (df["CentralAir"] == "Y") * 1.0

   

    all_df["OverallQual"] = df["OverallQual"]

    all_df["OverallCond"] = df["OverallCond"]



    # Quality measurements are stored as text but we can convert them to 

    # numbers where a higher number means higher quality.



    qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    all_df["ExterQual"] = df["ExterQual"].map(qual_dict).astype(int)

    all_df["ExterCond"] = df["ExterCond"].map(qual_dict).astype(int)

    all_df["BsmtQual"] = df["BsmtQual"].map(qual_dict).astype(int)

    all_df["BsmtCond"] = df["BsmtCond"].map(qual_dict).astype(int)

    all_df["HeatingQC"] = df["HeatingQC"].map(qual_dict).astype(int)

    all_df["KitchenQual"] = df["KitchenQual"].map(qual_dict).astype(int)

    all_df["FireplaceQu"] = df["FireplaceQu"].map(qual_dict).astype(int)

    all_df["GarageQual"] = df["GarageQual"].map(qual_dict).astype(int)

    all_df["GarageCond"] = df["GarageCond"].map(qual_dict).astype(int)



    all_df["BsmtExposure"] = df["BsmtExposure"].map(

        {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)



    bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

    all_df["BsmtFinType1"] = df["BsmtFinType1"].map(bsmt_fin_dict).astype(int)

    all_df["BsmtFinType2"] = df["BsmtFinType2"].map(bsmt_fin_dict).astype(int)



    all_df["Functional"] = df["Functional"].map(

        {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, 

         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)



    all_df["GarageFinish"] = df["GarageFinish"].map(

        {None: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)



    all_df["Fence"] = df["Fence"].map(

        {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)



    all_df["YearBuilt"] = df["YearBuilt"]

    all_df["YearRemodAdd"] = df["YearRemodAdd"]



    all_df["GarageYrBlt"] = df["GarageYrBlt"]

    all_df["GarageYrBlt"].fillna(0.0, inplace=True)



    all_df["MoSold"] = df["MoSold"]

    all_df["YrSold"] = df["YrSold"]

    

    all_df["LowQualFinSF"] = df["LowQualFinSF"]

    all_df["MiscVal"] = df["MiscVal"]



    all_df["PoolQC"] = df["PoolQC"].map(qual_dict).astype(int)



    all_df["PoolArea"] = df["PoolArea"]

    all_df["PoolArea"].fillna(0, inplace=True)

    

    # Add categorical features as numbers too. It seems to help a bit.

    all_df = factorize(df, all_df, "MSSubClass")

    all_df = factorize(df, all_df, "MSZoning", "RL")

    all_df = factorize(df, all_df, "LotConfig")

    all_df = factorize(df, all_df, "Neighborhood")

    all_df = factorize(df, all_df, "Condition1")

    all_df = factorize(df, all_df, "BldgType")

    all_df = factorize(df, all_df, "HouseStyle")

    all_df = factorize(df, all_df, "RoofStyle")

    all_df = factorize(df, all_df, "Exterior1st", "Other")

    all_df = factorize(df, all_df, "Exterior2nd", "Other")

    all_df = factorize(df, all_df, "MasVnrType", "None")

    all_df = factorize(df, all_df, "Foundation")

    all_df = factorize(df, all_df, "SaleType", "Oth")

    all_df = factorize(df, all_df, "SaleCondition")



    # IR2 and IR3 don't appear that often, so just make a distinction

    # between regular and irregular.

    all_df["IsRegularLotShape"] = (df["LotShape"] == "Reg") * 1



    # Most properties are level; bin the other possibilities together

    # as "not level".

    all_df["IsLandLevel"] = (df["LandContour"] == "Lvl") * 1



    # Most land slopes are gentle; treat the others as "not gentle".

    all_df["IsLandSlopeGentle"] = (df["LandSlope"] == "Gtl") * 1



    # Most properties use standard circuit breakers.

    all_df["IsElectricalSBrkr"] = (df["Electrical"] == "SBrkr") * 1



    # About 2/3rd have an attached garage.

    all_df["IsGarageDetached"] = (df["GarageType"] == "Detchd") * 1



    # Most have a paved drive. Treat dirt/gravel and partial pavement

    # as "not paved".

    all_df["IsPavedDrive"] = (df["PavedDrive"] == "Y") * 1



    # The only interesting "misc. feature" is the presence of a shed.

    all_df["HasShed"] = (df["MiscFeature"] == "Shed") * 1.  



    # If YearRemodAdd != YearBuilt, then a remodeling took place at some point.

    all_df["Remodeled"] = (all_df["YearRemodAdd"] != all_df["YearBuilt"]) * 1

    

    # Did a remodeling happen in the year the house was sold?

    all_df["RecentRemodel"] = (all_df["YearRemodAdd"] == all_df["YrSold"]) * 1

    

    # Was this house sold in the year it was built?

    all_df["VeryNewHouse"] = (all_df["YearBuilt"] == all_df["YrSold"]) * 1



    all_df["Has2ndFloor"] = (all_df["2ndFlrSF"] == 0) * 1

    all_df["HasMasVnr"] = (all_df["MasVnrArea"] == 0) * 1

    all_df["HasWoodDeck"] = (all_df["WoodDeckSF"] == 0) * 1

    all_df["HasOpenPorch"] = (all_df["OpenPorchSF"] == 0) * 1

    all_df["HasEnclosedPorch"] = (all_df["EnclosedPorch"] == 0) * 1

    all_df["Has3SsnPorch"] = (all_df["3SsnPorch"] == 0) * 1

    all_df["HasScreenPorch"] = (all_df["ScreenPorch"] == 0) * 1



    # These features actually lower the score a little.

    # all_df["HasBasement"] = df["BsmtQual"].isnull() * 1

    # all_df["HasGarage"] = df["GarageQual"].isnull() * 1

    # all_df["HasFireplace"] = df["FireplaceQu"].isnull() * 1

    # all_df["HasFence"] = df["Fence"].isnull() * 1



    # Months with the largest number of deals may be significant.

    all_df["HighSeason"] = df["MoSold"].replace( 

        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})



    all_df["NewerDwelling"] = df["MSSubClass"].replace(

        {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,

         90: 0, 120: 1, 150: 0, 160: 1, 180: 0, 190: 0})   

    

    all_df.loc[df.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1

    all_df.loc[df.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1

    all_df.loc[df.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1

    all_df.loc[df.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1

    all_df.loc[df.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1

    all_df["Neighborhood_Good"].fillna(0, inplace=True)



    all_df["SaleCondition_PriceDown"] = df.SaleCondition.replace(

        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})



    # House completed before sale or not

    all_df["BoughtOffPlan"] = df.SaleCondition.replace(

        {"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})

    

    all_df["BadHeating"] = df.HeatingQC.replace(

        {'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})



    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 

                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]

    all_df["TotalArea"] = all_df[area_cols].sum(axis=1)



    all_df["TotalArea1st2nd"] = all_df["1stFlrSF"] + all_df["2ndFlrSF"]



    all_df["Age"] = 2010 - all_df["YearBuilt"]

    all_df["TimeSinceSold"] = 2010 - all_df["YrSold"]



    all_df["SeasonSold"] = all_df["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 

                                                  6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)

    

    all_df["YearsSinceRemodel"] = all_df["YrSold"] - all_df["YearRemodAdd"]

    

    # Simplifications of existing features into bad/average/good.

    all_df["SimplOverallQual"] = all_df.OverallQual.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})

    all_df["SimplOverallCond"] = all_df.OverallCond.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})

    all_df["SimplPoolQC"] = all_df.PoolQC.replace(

        {1 : 1, 2 : 1, 3 : 2, 4 : 2})

    all_df["SimplGarageCond"] = all_df.GarageCond.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplGarageQual"] = all_df.GarageQual.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplFireplaceQu"] = all_df.FireplaceQu.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplFireplaceQu"] = all_df.FireplaceQu.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplFunctional"] = all_df.Functional.replace(

        {1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})

    all_df["SimplKitchenQual"] = all_df.KitchenQual.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplHeatingQC"] = all_df.HeatingQC.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplBsmtFinType1"] = all_df.BsmtFinType1.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})

    all_df["SimplBsmtFinType2"] = all_df.BsmtFinType2.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})

    all_df["SimplBsmtCond"] = all_df.BsmtCond.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplBsmtQual"] = all_df.BsmtQual.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplExterCond"] = all_df.ExterCond.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    all_df["SimplExterQual"] = all_df.ExterQual.replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

            

    # Bin by neighborhood (a little arbitrarily). Values were computed by: 

    # dtrain["SalePrice"].groupby(train_df["Neighborhood"]).median().sort_values()

    neighborhood_map = {

        "MeadowV" : 0,  #  88000

        "IDOTRR" : 1,   # 103000

        "BrDale" : 1,   # 106000

        "OldTown" : 1,  # 119000

        "Edwards" : 1,  # 119500

        "BrkSide" : 1,  # 124300

        "Sawyer" : 1,   # 135000

        "Blueste" : 1,  # 137500

        "SWISU" : 2,    # 139500

        "NAmes" : 2,    # 140000

        "NPkVill" : 2,  # 146000

        "Mitchel" : 2,  # 153500

        "SawyerW" : 2,  # 179900

        "Gilbert" : 2,  # 181000

        "NWAmes" : 2,   # 182900

        "Blmngtn" : 2,  # 191000

        "CollgCr" : 2,  # 197200

        "ClearCr" : 3,  # 200250

        "Crawfor" : 3,  # 200624

        "Veenker" : 3,  # 218000

        "Somerst" : 3,  # 225500

        "Timber" : 3,   # 228475

        "StoneBr" : 4,  # 278000

        "NoRidge" : 4,  # 290000

        "NridgHt" : 4,  # 315000

    }



    all_df["NeighborhoodBin"] = df["Neighborhood"].map(neighborhood_map)

    return all_df
dtrain_munged = munge(dtrain)

dtest_munged = munge(dtest)



print(dtrain_munged.shape)

print(dtest_munged.shape)
#dtrain_munged.describe()

dtrain_munged.head()
# One-Hot Coding:

# Copy NeighborhoodBin into a temporary DataFrame because we want to use the

# unscaled version later on (to one-hot encode it). 

neighborhood_bin_train = pd.DataFrame(index = dtrain.index)

neighborhood_bin_train["NeighborhoodBin"] = dtrain_munged["NeighborhoodBin"]

neighborhood_bin_test = pd.DataFrame(index = dtest.index)

neighborhood_bin_test["NeighborhoodBin"] = dtest_munged["NeighborhoodBin"]

################################################################################

#log transform the target:

dtrain["SalePrice"] = np.log1p(dtrain["SalePrice"])



#log transform skewed numeric features:

numeric_features = dtrain_munged.dtypes[dtrain_munged.dtypes != "object"].index



# Transform the skewed numeric features by taking log(feature + 1).

# This will make the features more normal.

from scipy.stats import skew



skewed = dtrain_munged[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))

skewed = skewed[skewed > 0.75]

skewed = skewed.index



dtrain_munged[skewed] = np.log1p(dtrain_munged[skewed])

dtest_munged[skewed] = np.log1p(dtest_munged[skewed])



# Additional processing: scale the data.   

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(dtrain_munged[numeric_features])



scaled = scaler.transform(dtrain_munged[numeric_features])

for i, col in enumerate(numeric_features):

    dtrain_munged[col] = scaled[:, i]



scaled = scaler.transform(dtest_munged[numeric_features])

for i, col in enumerate(numeric_features):

    dtest_munged[col] = scaled[:, i]



################################################################################
# Convert categorical features using one-hot encoding.

def onehot(onehot_df, df, column_name, fill_na, drop_name):

    onehot_df[column_name] = df[column_name]

    if fill_na is not None:

        onehot_df[column_name].fillna(fill_na, inplace=True)



    dummies = pd.get_dummies(onehot_df[column_name], prefix="_" + column_name)

    

    # Dropping one of the columns actually made the results slightly worse.

    # if drop_name is not None:

    #     dummies.drop(["_" + column_name + "_" + drop_name], axis=1, inplace=True)



    onehot_df = onehot_df.join(dummies)

    onehot_df = onehot_df.drop([column_name], axis=1)

    return onehot_df



def munge_onehot(df):

    onehot_df = pd.DataFrame(index = df.index)



    onehot_df = onehot(onehot_df, df, "MSSubClass", None, "40")

    onehot_df = onehot(onehot_df, df, "MSZoning", "RL", "RH")

    onehot_df = onehot(onehot_df, df, "LotConfig", None, "FR3")

    onehot_df = onehot(onehot_df, df, "Neighborhood", None, "OldTown")

    onehot_df = onehot(onehot_df, df, "Condition1", None, "RRNe")

    onehot_df = onehot(onehot_df, df, "BldgType", None, "2fmCon")

    onehot_df = onehot(onehot_df, df, "HouseStyle", None, "1.5Unf")

    onehot_df = onehot(onehot_df, df, "RoofStyle", None, "Shed")

    onehot_df = onehot(onehot_df, df, "Exterior1st", "VinylSd", "CBlock")

    onehot_df = onehot(onehot_df, df, "Exterior2nd", "VinylSd", "CBlock")

    onehot_df = onehot(onehot_df, df, "Foundation", None, "Wood")

    onehot_df = onehot(onehot_df, df, "SaleType", "WD", "Oth")

    onehot_df = onehot(onehot_df, df, "SaleCondition", "Normal", "AdjLand")



    # Fill in missing MasVnrType for rows that do have a MasVnrArea.

    temp_df = df[["MasVnrType", "MasVnrArea"]].copy()

    idx = (df["MasVnrArea"] != 0) & ((df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))

    temp_df.loc[idx, "MasVnrType"] = "BrkFace"

    onehot_df = onehot(onehot_df, temp_df, "MasVnrType", "None", "BrkCmn")



    # Also add the booleans from calc_df as dummy variables.

    onehot_df = onehot(onehot_df, df, "LotShape", None, "IR3")

    onehot_df = onehot(onehot_df, df, "LandContour", None, "Low")

    onehot_df = onehot(onehot_df, df, "LandSlope", None, "Sev")

    onehot_df = onehot(onehot_df, df, "Electrical", "SBrkr", "FuseP")

    onehot_df = onehot(onehot_df, df, "GarageType", "None", "CarPort")

    onehot_df = onehot(onehot_df, df, "PavedDrive", None, "P")

    onehot_df = onehot(onehot_df, df, "MiscFeature", "None", "Othr")



    # Features we can probably ignore (but want to include anyway to see

    # if they make any positive difference).

    # Definitely ignoring Utilities: all records are "AllPub", except for

    # one "NoSeWa" in the train set and 2 NA in the test set.

    onehot_df = onehot(onehot_df, df, "Street", None, "Grvl")

    onehot_df = onehot(onehot_df, df, "Alley", "None", "Grvl")

    onehot_df = onehot(onehot_df, df, "Condition2", None, "PosA")

    onehot_df = onehot(onehot_df, df, "RoofMatl", None, "WdShake")

    onehot_df = onehot(onehot_df, df, "Heating", None, "Wall")



    # I have these as numerical variables too.

    onehot_df = onehot(onehot_df, df, "ExterQual", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "ExterCond", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "BsmtQual", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "BsmtCond", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "HeatingQC", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "KitchenQual", "TA", "Ex")

    onehot_df = onehot(onehot_df, df, "FireplaceQu", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "GarageQual", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "GarageCond", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "PoolQC", "None", "Ex")

    onehot_df = onehot(onehot_df, df, "BsmtExposure", "None", "Gd")

    onehot_df = onehot(onehot_df, df, "BsmtFinType1", "None", "GLQ")

    onehot_df = onehot(onehot_df, df, "BsmtFinType2", "None", "GLQ")

    onehot_df = onehot(onehot_df, df, "Functional", "Typ", "Typ")

    onehot_df = onehot(onehot_df, df, "GarageFinish", "None", "Fin")

    onehot_df = onehot(onehot_df, df, "Fence", "None", "MnPrv")

    onehot_df = onehot(onehot_df, df, "MoSold", None, None)

    

    # Divide up the years between 1871 and 2010 in slices of 20 years.

    year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))



    yearbin_df = pd.DataFrame(index = df.index)

    yearbin_df["GarageYrBltBin"] = df.GarageYrBlt.map(year_map)

    yearbin_df["GarageYrBltBin"].fillna("NoGarage", inplace=True)



    yearbin_df["YearBuiltBin"] = df.YearBuilt.map(year_map)

    yearbin_df["YearRemodAddBin"] = df.YearRemodAdd.map(year_map)

    

    onehot_df = onehot(onehot_df, yearbin_df, "GarageYrBltBin", None, None)

    onehot_df = onehot(onehot_df, yearbin_df, "YearBuiltBin", None, None)

    onehot_df = onehot(onehot_df, yearbin_df, "YearRemodAddBin", None, None)



    return onehot_df
# Add the one-hot encoded categorical features.

onehot_df = munge_onehot(dtrain)

onehot_df = onehot(onehot_df, neighborhood_bin_train, "NeighborhoodBin", None, None)

dtrain_munged = dtrain_munged.join(onehot_df)



# These onehot columns are missing in the test data, so drop them from the

# training data or we might overfit on them.

drop_cols = [

                "_Exterior1st_ImStucc", "_Exterior1st_Stone",

                "_Exterior2nd_Other","_HouseStyle_2.5Fin", 

            

                "_RoofMatl_Membran", "_RoofMatl_Metal", "_RoofMatl_Roll",

                "_Condition2_RRAe", "_Condition2_RRAn", "_Condition2_RRNn",

                "_Heating_Floor", "_Heating_OthW",



                "_Electrical_Mix", 

                "_MiscFeature_TenC",

                "_GarageQual_Ex", "_PoolQC_Fa"

            ]

dtrain_munged.drop(drop_cols, axis=1, inplace=True)

#################################################################################

onehot_df = munge_onehot(dtest)

onehot_df = onehot(onehot_df, neighborhood_bin_test, "NeighborhoodBin", None, None)

dtest_munged = dtest_munged.join(onehot_df)



# This column is missing in the training data. There is only one example with

# this value in the test set. So just drop it.

dtest_munged.drop(["_MSSubClass_150"], axis=1, inplace=True)



# Drop these columns. They are either not very helpful or they cause overfitting.

drop_cols = [

    "_Condition2_PosN",    # only two are not zero

    "_MSZoning_C (all)",

    "_MSSubClass_160",

]

dtrain_munged.drop(drop_cols, axis=1, inplace=True)

dtest_munged.drop(drop_cols, axis=1, inplace=True)



# save the processed data for tuning papameters of XGBoost model

dtrain_munged.to_csv('train_modified.csv',index=False)

dtest_munged.to_csv('test_modified.csv',index=False)
################################################################################



# We take the log here because the error metric is between the log of the

# SalePrice and the log of the predicted price. That does mean we need to 

# exp() the prediction to get an actual sale price.

label_df = pd.DataFrame(index = dtrain_munged.index, columns=["SalePrice"])

label_df["SalePrice"] = np.log(dtrain["SalePrice"])

# save the processed data for tuning papameters of XGBoost model



label_df.to_csv('label_df.csv',index=False)

print("Training set size:", dtrain_munged.shape)

print("Test set size:", dtest_munged.shape)



################################################################################
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 
# show the data after log transform

print("Some Statistics of the Housing Price:\n")

print(np.exp(dtrain['SalePrice']).describe())

print("\nThe median of the Housing Price is: ", np.exp(dtrain['SalePrice']).median(axis = 0))
all_data = pd.concat((dtrain.loc[:,'MSSubClass':'SaleCondition'],

                      dtest.loc[:,'MSSubClass':'SaleCondition']))
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

prices = pd.DataFrame({"price":dtrain["SalePrice"]})

prices.hist()
#all_data_munged = pd.concat((dtrain_munged.loc[:,'MSSubClass':'SaleCondition'],

#                     dtest_munged.loc[:,'MSSubClass':'SaleCondition']))

#prices = pd.DataFrame({"price":dtrain_munged["SalePrice"], "log(price + 1)":np.log1p(dtrain_munged["SalePrice"])})

#prices.hist()
all_data = pd.get_dummies(all_data)



#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:

X_train = all_data[:dtrain.shape[0]]

X_test = all_data[dtrain.shape[0]:]

y = dtrain.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)



model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0009, 0.0007, 0.0005]).fit(X_train, y)

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
import numpy as np

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

from sklearn import cross_validation, metrics

from sklearn.grid_search import GridSearchCV



import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
train = xgb.DMatrix(X_train, label = y)

test = xgb.DMatrix(X_test)

# this is not a good way to remodefy params manually. 

# So I reference another solution presented down below.

params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, train,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) 

model_xgb.fit(X_train, y)
xgb_preds = np.exp(model_xgb.predict(X_test))

lasso_preds = np.exp(model_lasso.predict(X_test))

# compare the two models

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.7*lasso_preds + 0.3*xgb_preds

solution = pd.DataFrame({"id":dtest.Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)
train = pd.read_csv('train_modified.csv')

test = pd.read_csv('test_modified.csv')

#label_df = pd.read_csv('label_df.csv')



train.shape, test.shape,label_df.shape
xgb2 = xgb.XGBRegressor(

                 colsample_bytree=0.1,

                 gamma=0.0,

                 learning_rate=0.01,

                 max_depth=3,

                 min_child_weight=1.2,

                 n_estimators=7200,                                                                  

                 reg_alpha=0.01,

                 reg_lambda=1,

                 subsample=0.4,

                 silent=1)

xgb2.fit(train, label_df)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005],max_iter=50000).fit(train, label_df.SalePrice)

y_pred = model_lasso.predict(train)

y_test = label_df

print("Lasso score on training set: ", rmse(y_test, y_pred))
# Blend the results of the two regressors and save the prediction to a CSV file.

# Run prediction on the Kaggle test set.

y_pred_xgb = np.exp(xgb2.predict(test))

y_pred_lasso = np.exp(model_lasso.predict(test))



y_pred = (y_pred_xgb + y_pred_lasso) / 2.0

#y_pred = 0.3*y_pred_xgb + 0.7*y_pred_lasso

y_preds = np.exp(y_pred)



solution = pd.DataFrame({"id":dtest.Id, "SalePrice":y_preds})

solution.to_csv("xgb_lasso_final.csv", index = False)