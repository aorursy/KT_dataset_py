# Base 

# -----------------------------------

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



# Missing Values 

# -----------------------------------

# !pip install missingno

import missingno as msno



# Models 

# -----------------------------------

import lightgbm as lgb



# Metrics & Evaluation

# -----------------------------------

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,r2_score

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV



# MAPE

# -----------------------------------

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100





# Configuration

# -----------------------------------

import warnings

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)



pd.set_option('display.max_columns', None)

pd.options.display.float_format = '{:.2f}'.format
# Import

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



# Bind rows

df = train.append(test)







# Num of dtypes

print("Num of Object Variables:", df.select_dtypes(object).shape[1])

print("Num of Integer Variables:", df.select_dtypes("integer").shape[1])

print("Num of Float Variables:", df.select_dtypes("float").shape[1])



# Dimensions

df.shape, train.shape, test.shape
df.head()
msno.heatmap(df)

plt.show()
def missing_values(data, plot = False, target = "SalePrice"):

    

    mst = pd.DataFrame({"Num_Missing":df.isnull().sum(), "Missing_Ratio":df.isnull().sum() / df.shape[0]}).sort_values("Num_Missing", ascending = False)

    mst["DataTypes"] = df[mst.index].dtypes.values

    mst = mst[mst.Num_Missing > 0].reset_index().rename({"index":"Feature"}, axis = 1)

    mst = mst[mst.Feature != target]

    

    print("Number of Variables include Missing Values:", mst.shape[0], "\n")

    

    if mst[mst.Missing_Ratio > 0.99].shape[0] > 0:  

        print("Full Missing Variables:",mst[mst.Missing_Ratio > 0.99].Feature.tolist())

        data.drop(mst[mst.Missing_Ratio > 0.99].Feature.tolist(), axis = 1, inplace = True)



        print("Full missing variables are deleted!", "\n")



    if plot:

        plt.figure(figsize = (25, 8))    

        p = sns.barplot(mst.Feature, mst.Missing_Ratio)

        for rotate in p.get_xticklabels():

            rotate.set_rotation(90)

            

            

    print(mst, "\n")        

    

missing_values(df, plot = True, target = "SalePrice")
cat_missing = ["Alley", 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2','Fence', "MiscFeature",

               "FireplaceQu", "GarageFinish","GarageCond", "GarageQual", "GarageType", "MasVnrType"]

for i in cat_missing:

    df[i] = np.where(df[i].isnull(), "None", df[i])

    

missing_values(df, plot = False, target = "SalePrice")
# If observation is NA in the GarageYrBlt variable, assign YearBuilt value.

df["GarageYrBlt"]= np.where((df.GarageYrBlt.isnull() == True) & (df.GarageArea == 0), df.YearBuilt,df.GarageYrBlt)
def quick_missing_imp(data, num_method = "median", cat_length = 20, target = "SalePrice"):

    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]

    

    temp_target = data[target]

    

    print("# BEFORE")

    print(data[variables_with_na].isnull().sum(), "\n\n")

    

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

        

    if num_method == "mean":

        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

    elif num_method == "median":

        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

        

    data[target] = temp_target

    

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")

    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")

    print(data[variables_with_na].isnull().sum(), "\n\n")

        

    return data

        

df = quick_missing_imp(df, num_method = "median", cat_length = 17)
ordinal_vars = [

    "LotShape", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual",

    "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional",

    "FireplaceQu", "GarageQual", "GarageCond", "Fence", "Electrical"

]



def ordinal(serie, category):

    numeric = np.arange(1, len(category)+1, 1)

    zip_iterator = zip(category, numeric)

    mapping = dict(zip_iterator)

    serie = serie.map(mapping)

    return serie





def transform_ordinal(data, ordinal_vars, category):

    

    for i in ordinal_vars:

        data[i] = ordinal(data[i], category = category)

        



        

# Same Categories        

transform_ordinal(

    df, 

    ordinal_vars = ["ExterQual", "ExterCond", "HeatingQC","KitchenQual"],

    category = ["Po", "Fa", "TA", "Gd" ,"Ex"]

)



transform_ordinal(

    df, 

    ordinal_vars = ["BsmtQual", "BsmtCond", "FireplaceQu","GarageCond", "GarageQual"],

    category = ["None", "Po", "Fa", "TA", "Gd","Ex"]

)



transform_ordinal(

    df, 

    ordinal_vars = ["BsmtFinType1", "BsmtFinType2"],

    category = ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ","GLQ"]

)



# Different

df["LotShape"] = ordinal(df["LotShape"], category = ["IR3", "IR2", "IR1","Reg"])

df["BsmtExposure"] = ordinal(df["BsmtExposure"], category = ["None", "No", "Mn", "Av","Gd"])

df["Functional"] = ordinal(df["Functional"], category = ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1","Typ"])

df["Fence"] = ordinal(df["Fence"], category = ["None", "MnWw", "GdWo", "MnPrv","GdPrv"])



df["Electrical"] = ordinal(df["Electrical"], category = ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'])



print("Ordinal transformation completed!")
sns.set(font_scale=1.1)

c = df.corr()

mask = np.triu(c.corr())

plt.figure(figsize=(20, 20))

sns.heatmap(c,

            annot=True,

            fmt='.1f',

            cmap='coolwarm',

            square=True,

            mask=mask,

            linewidths=1,

            cbar=False)



plt.show()
def find_linear(data, target, plot = False):



    c = data.corr()

    temp = c[target].sort_values(ascending = False).reset_index()

    temp = temp[temp["index"] != target]

    temp.columns = ["Variables", "Cor"]

    

    highly = temp[(temp["Cor"] > 0.5) | (temp["Cor"] < -0.5)].reset_index(drop = True)

    

    print("##################################")

    print("###### CORRELATION ANALYSIS ######")

    print("##################################", "\n\n")

    

    print("DEPENDENT VARIABLES")

    print("----------------------------------")

    print(highly, "\n\n")

    

    print("Highly Correlated Variables: \n", highly["Variables"].tolist(), "\n\n")

    

    

    print("INDEPENDENT VARIABLES")

    print("----------------------------------")

    for i in highly.Variables:

        c2 = c[i].sort_values(ascending = False).reset_index()

        c2 = c2[~c2["index"].isin([i,target])]

        c2.columns = ["Variables", "Cor"]

    

        cwith = c2[(c2["Cor"] >= 0.8) | (c2["Cor"] <= -0.8)].reset_index(drop = True)

        if len(cwith) > 0:

            print(i,":")

            print(cwith, "\n\n")

            

    if plot:       

        plt.figure(figsize = (10, 6))  

        p = sns.barplot(highly.Cor, highly["Variables"])

        plt.suptitle("Target Correlation")

            

find_linear(df, target = "SalePrice", plot = True)
def num_plot(data, cat_length = 16, remove = ["Id"], hist_bins = 12, figsize = (20,4)):

    

    num_cols = [col for col in data.columns if data[col].dtypes != "O" 

                and len(data[col].unique()) >= cat_length]

    

    if len(remove) > 0:

        num_cols = list(set(num_cols).difference(remove))

            

    for i in num_cols:

        fig, axes = plt.subplots(1, 3, figsize = figsize)

        data.hist(str(i), bins = hist_bins, ax=axes[0])

        data.boxplot(str(i),  ax=axes[1], vert=False);

        try: 

            sns.kdeplot(np.array(data[str(i)]))

        except: ValueError

        

        axes[1].set_yticklabels([])

        axes[1].set_yticks([])

        axes[0].set_title(i + " | Histogram")

        axes[1].set_title(i + " | Boxplot")

        axes[2].set_title(i + " | Density")

        plt.show()

        

        

num_plot(df.drop(ordinal_vars, axis = 1), cat_length = 16, remove = ["Id"], hist_bins = 10, figsize = (20,4))
for i in ["Street", "Utilities", "LandSlope", "MiscFeature"]:

    print(df[i].value_counts())
def cat_eda(data, cat_length, target = "SalePrice"):  

    dataframe = data.copy()

    

    #if len(ordinal_variable) > 0:

    #    dataframe.drop(ordinal_variable, axis = 1, inplace = True)

        

    more_cat_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) < cat_length]

    

    for i in more_cat_cols: 

        print(i, ":", len(dataframe[i].value_counts()), "Unique Category -", str(dataframe[i].dtype))

        print(pd.DataFrame({"COUNT": dataframe[i].value_counts(),

                            "RATIO": dataframe[i].value_counts() / len(dataframe),

                            "TARGET_MEDIAN": dataframe.groupby(i)[target].median(),

                            "TARGET_COUNT": dataframe.groupby(i)[target].count(),

                            "TARGET_STD": dataframe.groupby(i)[target].std()}), end="\n\n\n")

    

    print("# DTYPES -----------------------------")

    print("Object Variables:",dataframe[more_cat_cols].select_dtypes("object").columns.tolist(), "\n")

    print("Integer Variables:",dataframe[more_cat_cols].select_dtypes("integer").columns.tolist(), "\n")

    print("Float Variables:",dataframe[more_cat_cols].select_dtypes("float").columns.tolist(), "\n")



cat_eda(df, cat_length=17)
# Remove Useless Variables

drop_list = ["Street", "Utilities", "LandSlope", "MiscFeature", "PoolArea"]





for col in drop_list:

    df.drop(col, axis=1, inplace=True)
df2 = df.copy()

df2.head()
df2["TotalQual"] = df2[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1", 

                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1)

df2["BsmtQual"] = df2[["BsmtCond", "BsmtFinType1", "BsmtFinType2"]].sum(axis = 1)

df2["TotalGarageQual"] = df2[["GarageQual", "GarageCond"]].sum(axis = 1)

df2["Overall"] = df2[["OverallQual", "OverallCond"]].sum(axis = 1)

df2["Exter"] = df2[["ExterQual", "ExterCond"]].sum(axis = 1)

df2["ExtraQual"] = df2[["Fence", "FireplaceQu", "Functional", "HeatingQC"]].sum(axis = 1)

df2["Qual"] = df2[["OverallQual", "ExterQual", "GarageQual", "Fence", "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu"]].sum(axis = 1)

df2["Cond"] = df2[["OverallCond", "ExterCond", "GarageCond", "BsmtCond", "HeatingQC", "Functional"]].sum(axis = 1)
cat_eda(df2[ordinal_vars+["SalePrice"]], cat_length=17)
df2["LotShape"] = np.where(df2.LotShape == 1, 2, df2["LotShape"])



df2["OverallQual"] = np.where(df2.OverallQual.isin([1,2]), 3, df2["OverallQual"])



df2["OverallCond"] = np.where(df2.OverallCond.isin([1,2]), 3, df2["OverallCond"])

df2["OverallCond"] = np.where(df2.OverallCond.isin([7,8]), 6, df2["OverallCond"])



df2["ExterCond"] = np.where(df2.ExterCond.isin([1,2]), 0, 1)



df2["BsmtQual"] = np.where(df2.BsmtQual.isin([3,6]), 7, df2["BsmtQual"])

df2["BsmtQual"] = np.where(df2.BsmtQual.isin([17,18]), 16, df2["BsmtQual"])

df2["BsmtQual"] = np.where(df2.BsmtQual.isin([9,10,11]), 12, df2["BsmtQual"])



df2["BsmtCond"] = np.where(df2.BsmtCond == 1, 2, df2["BsmtCond"])



df2["HeatingQC"] = np.where(df2.HeatingQC == 1, 2, df2["HeatingQC"])



df2["Functional"] = np.where(df2.Functional == 8, 1, 0)



df2["GarageQual"] = np.where(df2.GarageQual == 1, 2, df2["GarageQual"])

df2["GarageQual"] = np.where(df2.GarageQual == 6, 5, df2["GarageQual"])



df2["GarageCond"] = np.where(df2.GarageCond == 1, 2, df2["GarageCond"])

df2["GarageCond"] = np.where(df2.GarageCond == 6, 5, df2["GarageCond"])



df2["Fence"] = np.where(df2.Fence.isin([2,4]), 3, df2["Fence"])



df2["Electrical"] = np.where(df2["Electrical"].isin([5, 4]), 3, df2["Electrical"])
cat_eda(df2[ordinal_vars+["SalePrice"]], cat_length=17)
# Total Floor

df2["TotalFlrSF"] = df2["1stFlrSF"] + df2["2ndFlrSF"]



# Total Finished Basement Area

df2["TotalBsmtFin"] = df2.BsmtFinSF1+df2.BsmtFinSF2



df2["BsmtFinRatio"] = (df2.TotalBsmtFin / df2.TotalBsmtSF).fillna(0)



# Is there a basement?

df2["Basement"] = np.where(df2.TotalBsmtSF < 1 , 0, 1)



# Porch Area

df2["PorchArea"] = df2.OpenPorchSF + df2.EnclosedPorch + df2.ScreenPorch + df2["3SsnPorch"] + df2.WoodDeckSF



# Total House Area

df2["TotalHouseArea"] = df2.TotalFlrSF + df2.TotalBsmtSF



df2["TotalSqFeet"] = df2.GrLivArea + df2.TotalBsmtSF





# Basement Rooms

df2["BsmtRoom"] = np.where((df2.BsmtFinSF1 > 0) & (df2.BsmtFinSF2 < 1), 1, np.nan)

df2["BsmtRoom"] = np.where((df2.BsmtFinSF1 < 1) & (df2.BsmtFinSF2 > 0 ), 1, df2["BsmtRoom"])

df2["BsmtRoom"] = np.where((df2.BsmtFinSF1 < 1) & (df2.BsmtFinSF2 < 1 ), 0, df2["BsmtRoom"])

df2["BsmtRoom"] = np.where((df2.BsmtFinSF1 > 0) & (df2.BsmtFinSF2 > 0 ), 2, df2["BsmtRoom"])





# Floor

df2["Floor"] = np.where((df2["2ndFlrSF"] < 1), 1, 2)





# Bath Room

df2["FullBath"] = np.where(df2.FullBath == 4, 3, df2.FullBath)

df2["BsmtFullBath"] = np.where(df2.BsmtFullBath == 3, 2, df2.BsmtFullBath)



df2["TotalFullBath"] = df2.BsmtFullBath + df2.FullBath

df2["TotalHalfBath"] = df2.BsmtHalfBath + df2.HalfBath



df2["TotalBath"] = df2["TotalFullBath"] + (df2["TotalHalfBath"]*0.5)

df2["TotalBath"] = np.where(df2.TotalBath.isin([6,7]), 5, df2.TotalBath)



# Fireplace

df2["Fireplaces"] = np.where(df2.Fireplaces > 3, 3, df2.Fireplaces)



# Garage

df2["GarageAreaRatio"] = (df2.GarageArea / df2.GarageCars).fillna(0)

df2["GarageCars"] = np.where(df2.GarageCars > 3, 3, df2.GarageCars)



# Rooms

df2["TotRmsAbvGrd"] = np.where(df2.TotRmsAbvGrd > 12, 12, df2.TotRmsAbvGrd)

df2["TotRmsAbvGrd"] = np.where(df2.TotRmsAbvGrd == 2, 3, df2.TotRmsAbvGrd)



# Lot Ratio

df2["LotRatio"] = df2.GrLivArea / df2.LotArea



df2["RatioArea"] = df2.TotalHouseArea / df2.LotArea



df2["GarageLotRatio"] = df2.GarageArea / df2.LotArea



# MasVnrArea

df2["MasVnrRatio"] = df2.MasVnrArea / df2.TotalHouseArea



# Dif Area

df2["DifArea"] = (df2.LotArea - df2["1stFlrSF"] - df2.GarageArea - df2.PorchArea - df2.WoodDeckSF)



# LowQualFinSF

df2["LowQualFinSFRatio"] = df2.LowQualFinSF / df2.TotalHouseArea







df2["OverallGrade"] = df2["OverallQual"] * df2["OverallCond"]

# Overall quality of the garage

df2["GarageGrade"] = df2["GarageQual"] * df2["GarageCond"]

# Overall quality of the exterior

df2["ExterGrade"] = df2["ExterQual"] * df2["ExterCond"]

# Overall kitchen score

df2["KitchenScore"] = df2["KitchenAbvGr"] * df2["KitchenQual"]

# Overall fireplace score

df2["FireplaceScore"] = df2["Fireplaces"] * df2["FireplaceQu"]





df2["HasMasVnrType"] = np.where(df2.MasVnrType == "None", 0, 1)

df2["BoughtOffPlan"] = np.where(df2.SaleCondition == "Partial", 1, 0)
for i in ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]:

    plt.figure(figsize = (25,15))

    p = sns.boxplot(x = i, y = "SalePrice", data = df)

    for item in p.get_xticklabels():

        item.set_rotation(90)
df2.YearRemodAdd = np.where(df2.YearBuilt > df2.YearRemodAdd, df2.YearBuilt, df2.YearRemodAdd)

df2.YrSold = np.where(df2.YearBuilt > df2.YrSold, df2.YearBuilt, df2.YrSold)

df2.YrSold = np.where(df2.YearRemodAdd > df2.YrSold, df2.YearRemodAdd, df2.YrSold)



df2["Restoration"] = df2.YearRemodAdd - df2.YearBuilt

df2["HouseAge"] = df2.YrSold - df2.YearBuilt

df2["RestorationAge"] = df2.YrSold - df2.YearRemodAdd

df2["GarageAge"] = df2.GarageYrBlt - df2.YearBuilt

df2["GarageRestorationAge"] = np.abs(df2.GarageYrBlt - df2.YearRemodAdd)

df2["GarageSold"] = df2.YrSold - df2.GarageYrBlt



df2.GarageYrBlt = np.where(df2.GarageAge < 0, df2.YearBuilt, df2.GarageYrBlt)

df2.GarageYrBlt = np.where((df2.GarageYrBlt == 2207), df2.YearBuilt, df2.GarageYrBlt)

df2["GarageAge"] = df2.GarageYrBlt - df2.YearBuilt

df2["GarageSold"] = df2.YrSold - df2.GarageYrBlt





df2["Remodeled"] = np.where(df2.YearBuilt == df2.YearRemodAdd, 0 ,1)

df2["IsNewHouse"] = np.where(df2.YearBuilt == df2.YrSold, 1 ,0)







# YearBuiltCut

df2["YearBuiltCut"] = pd.cut(df2.YearBuilt, 15, labels = np.arange(1,16,1))

df2["GarageYearBuiltCut"] = pd.cut(df2.YearBuilt, 15, labels = np.arange(1,16,1))

df2["RemodAddCut"] = pd.cut(df2.YearRemodAdd, 10, labels = np.arange(1, 11, 1))



# Great Depression 1929-1939

df2["GreatDepression"] = np.where((df2.YearBuilt >=1929) & (df2.YearBuilt <= 1939), 1,0)

df2["BeforeGreatDepression"] = np.where((df2.YearBuilt <1929), 1,0)

# Mortgage Crisis

df2["Mortgage"] = np.where((df2.YearBuilt >=2007) & (df2.YearBuilt <= 2010), 1,0)

df2["MortgageRestoration"] = np.where((df2.YearRemodAdd >=2007) & (df2.YearRemodAdd <= 2010), 1,0)





for i in ["GreatDepression","BeforeGreatDepression", "Mortgage","MortgageRestoration"]:

    print(df2.groupby(i).SalePrice.agg(["count", "median"]), "\n")

    

    

df2.drop(["YearBuilt", "GarageYrBlt", "YearRemodAdd"], axis = 1, inplace = True)    
cat_eda(df2[set(df2.columns).difference(ordinal_vars)], cat_length=17)
ngb =df2.groupby("Neighborhood").SalePrice.median().reset_index()

ngb["Neighborhood_Clusters"] = pd.cut(df2.groupby("Neighborhood").SalePrice.median().values, 6, labels = np.arange(1,7,1))

df2 = pd.merge(df2, ngb.drop(["SalePrice"], axis = 1), how = "left", on = "Neighborhood").drop("Neighborhood", axis = 1)

df2["Neighborhood_Clusters"] = np.where(df2.Neighborhood_Clusters > 4, 5, df2.Neighborhood_Clusters)



del ngb



# Conditions

df2["Railroad"] = np.where((df2.Condition1.isin(["RRNn", "RRAn", "RRNe", "RRAe"])) | (df2.Condition2.isin(["RRNn", "RRAn", "RRNe", "RRAe"])), 1, 0)

df2["Park"] = np.where((df2.Condition1.isin(["PosN", "PosA"])) | (df2.Condition2.isin(["PosN", "PosA"])), 1, 0)

df2["Adjacent"] = np.where((df2.Condition1.isin(["PosA", "Artery", "Feedr", "RRAn", "RRAe"])) | (df2.Condition2.isin(["PosA", "Artery", "Feedr", "RRAn", "RRAe"])), 1, 0)

df2["Within"] = np.where((df2.Condition1.isin(["RRNn", "PosN", "RRNe"])) | (df2.Condition2.isin(["RRNn", "PosN", "RRNe"])), 1, 0)

df2["Norm"] = np.where((df2.Condition1.isin(["Norm"])) | (df2.Condition2.isin(["Norm"])), 1, 0)

df2["NorthSouth"] = np.where((df2.Condition1.isin(["RRNn", "RRAn"])) | (df2.Condition2.isin(["RRNn", "RRAn"])), 1, 0)

df2["EastWest"] = np.where((df2.Condition1.isin(["RRNe", "RRAe"])) | (df2.Condition2.isin(["RRNe", "RRAe"])), 1, 0)

df2["Road"] = np.where((df2.Condition1.isin(["Artery", "Feedr"])) | (df2.Condition2.isin(["Artery", "Feedr"])), 1, 0)



# Exterior

df2["Exterior1st"] = np.where((df2["Exterior1st"].isin(["Brk Cmn", "BrkComm"])) | (df2["Exterior2nd"].isin(["Brk Cmn", "BrkComm"])), "BrkComm",df2["Exterior1st"])

df2["Exterior2nd"] = np.where((df2["Exterior1st"].isin(["Brk Cmn", "BrkComm"])) | (df2["Exterior2nd"].isin(["Brk Cmn", "BrkComm"])), "BrkComm",df2["Exterior2nd"])

df2["Exterior1st"] = np.where((df2["Exterior1st"].isin(["CemntBd", "CmentBd"])) | (df2["Exterior2nd"].isin(["CemntBd", "CmentBd"])), "CmentBd",df2["Exterior1st"])

df2["Exterior2nd"] = np.where((df2["Exterior1st"].isin(["CemntBd", "CmentBd"])) | (df2["Exterior2nd"].isin(["CemntBd", "CmentBd"])), "CmentBd",df2["Exterior2nd"])

df2["Exterior1st"] = np.where((df2["Exterior1st"].isin(["Wd Shng", "WdShing"])) | (df2["Exterior2nd"].isin(["Wd Shng", "WdShing"])), "WdShing",df2["Exterior1st"])

df2["Exterior2nd"] = np.where((df2["Exterior1st"].isin(["Wd Shng", "WdShing"])) | (df2["Exterior2nd"].isin(["Wd Shng", "WdShing"])), "WdShing",df2["Exterior2nd"])



df2["SameExterior"] = np.where(df2.Exterior1st == df2.Exterior2nd, 1, 0)



ex = pd.merge(pd.DataFrame({"COUNT": df2["Exterior1st"].value_counts(),

                            "RATIO": df2["Exterior1st"].value_counts() / len(df2),

                            "TARGET_MEDIAN": df2.groupby("Exterior1st")["SalePrice"].median(),

                            "TARGET_COUNT": df2.groupby("Exterior1st")["SalePrice"].count()}).reset_index(),

           pd.DataFrame({"COUNT": df2["Exterior2nd"].value_counts(),

                            "RATIO2": df2["Exterior2nd"].value_counts() / len(df2),

                            "TARGET_MEDIAN2": df2.groupby("Exterior2nd")["SalePrice"].median(),

                            "TARGET_COUNT2": df2.groupby("Exterior2nd")["SalePrice"].count()}).reset_index(),

           how = "outer", on = "index").sort_values("index")

ex["ExteriorRank"] = pd.cut(ex.TARGET_MEDIAN, 4, labels = np.arange(1, 5, 1))

ex["ExteriorRank"] = np.where(ex["index"] == "Other", 4, ex["ExteriorRank"])



df2 = pd.merge(df2, ex[["index", "ExteriorRank"]].rename({"index":"Exterior1st", "ExteriorRank":"ExteriorRank1"}, axis = 1), how = "left", on = "Exterior1st")

df2 = pd.merge(df2, ex[["index", "ExteriorRank"]].rename({"index":"Exterior2nd", "ExteriorRank":"ExteriorRank2"}, axis = 1), how = "left", on = "Exterior2nd")

df2["TotalExteriorRank"] = df2["ExteriorRank1"]+df2["ExteriorRank2"]



del ex



# Heating

df2["Heating"] = np.where(df2.Heating.str.contains("Gas"), 1, 0)  



# MSZoning

df2["MSZoning_IsResidential"] = np.where(df2.MSZoning.isin(["RH", "RM", "RL"]), 1, 0)



# Sale Type

df2["SaleType_Cat"] = np.where(df2.SaleType.isin(["Con", "ConLD", "ConLI", "ConLw"]), "Con", df2.SaleType)

df2["SaleType_Cat"] = np.where(df2.SaleType.isin(["WD", "CWD", "VWD"]), "War", df2.SaleType_Cat)

df2["SaleType_Cat"] = np.where(df2.SaleType.isin(["COD"]), "Oth", df2.SaleType_Cat)



# MasVnrType

df2["MasVnrType"] = np.where(df2.MasVnrType.isin(["BrkCmn", "BrkFace"]), "Brk", df2.MasVnrType)

df2["HasMasVnr"] = np.where(df2.MasVnrType == "None", 0, 1)



# RoofMatl

df2["RoofMatl"] = np.where(df2.RoofMatl.isin(["CompShg"]), 1, 0)



df2["Foundation"] = np.where(df2["Foundation"] == "Stone", "BrkTil", df2["Foundation"])

df2["Foundation"] = np.where(df2["Foundation"] == "Wood", "PConc", df2["Foundation"])



# MSSubClass

df2["MSSubClass"] = np.where(df2.MSSubClass.isin([150]), 120, df2.MSSubClass)

df2["MSSubClass_AllAges"] = np.where(df2.MSSubClass.isin([40,45,50,75,90,150, 190]), 1, 0)

df2["MSSubClass_Newer"] = np.where(df2.MSSubClass.isin([20, 60, 120, 160]), 1, 0)

df2["MSSubClass_Older"] = np.where(df2.MSSubClass.isin([30, 70]), 1, 0)

df2["MSSubClass_AllStyles"] = np.where(df2.MSSubClass.isin([20, 90, 190]), 1, 0)

df2["MSSubClass_Pud"] = np.where(df2.MSSubClass.isin([150, 160, 180]), 1, 0)

df2["MSSubClass_SplitMultiLevel"] = np.where(df2.MSSubClass.isin([80,85, 180]), 1, 0)

df2["MSSubClass_1946"] = np.where(df2.MSSubClass.isin([20, 60, 120, 160]), 1, 0)

df2["MSSubClass_1945"] = np.where(df2.MSSubClass.isin([30, 70]), 1, 0)





# Building Type

df2["BldgType_Short"] = np.where(df2.BldgType.isin(['1Fam', '2fmCon']), "Fam", df2.BldgType)

df2["BldgType_Short"] = np.where(df2.BldgType.isin(['TwnhsE', 'Twnhs']), "Twnhs", df2.BldgType_Short)



# House Style

df2["HouseStyle_Floor"] = np.where(df2.HouseStyle.isin(["1Story"]), "1", "2")

df2["HouseStyle_Floor"] = np.where(df2.HouseStyle.isin(["1.5Fin", "1.5Unf"]), "1.5", df2["HouseStyle_Floor"])

df2["HouseStyle_Floor"] = np.where(df2.HouseStyle.isin(["2.5Fin", "2.5Unf"]), "2.5", df2["HouseStyle_Floor"])

df2["HouseStyle_Floor"] = np.where(df2.HouseStyle.isin(["SLvl", "SFoyer"]), "S", df2["HouseStyle_Floor"])



df2["HouseStyle_IsSplit"] = np.where(df2.HouseStyle.isin(["SLvl","SFoyer"]),1,0)





df2.drop(["MSSubClass", "Exterior1st", "Exterior2nd", "Condition1", "Condition2", "SaleType", "YrSold", "MoSold"], axis = 1, inplace = True)
cat_eda(df2[set(df2.columns).difference(ordinal_vars)], cat_length=17)
def label_encoder(dataframe):

    from sklearn import preprocessing

    labelencoder = preprocessing.LabelEncoder()



    label_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"

                  and len(dataframe[col].value_counts()) == 2]



    for col in label_cols:

        dataframe[col] = labelencoder.fit_transform(dataframe[col])

    return dataframe



df2 = label_encoder(df2)









cat_cols = [col for col in df2.columns if df2[col].dtypes == 'O']



def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):

    original_columns = list(dataframe.columns)

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)

    new_columns = [c for c in dataframe.columns if c not in original_columns]

    return dataframe, new_columns





df2, new_cols_ohe = one_hot_encoder(df2, cat_cols, nan_as_category=False)



df2.shape
df2.head()
### TRAIN - TEST

train_model = df2[df2.Id.isin(train.Id)]

test_model = df2[df2.Id.isin(test.Id)]

test_model.drop("SalePrice", axis = 1, inplace = True)





X = train_model.drop(['SalePrice', "Id"], axis=1)



y = train_model[["SalePrice"]]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)







######### MODEL: LGBM

import lightgbm as lgb

lgb_param = {

    # Configuration

    "nthread": -1,

    "objective": "regression",

    "metric": "rmse",

    "verbose": -1,

}



reg = lgb.LGBMRegressor(

    random_state=46, **lgb_param

)

reg.fit(X_train,y_train, 

        eval_set=[(X_train, y_train),(X_test, y_test)],

        eval_metric = ["rmse", "mae"],

        eval_names=["Train", "Valid"],

        early_stopping_rounds=10,

        verbose=10,

        categorical_feature='auto')



print("")

print("Train RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train, reg.predict(X_train)))))

print("Train RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test, reg.predict(X_test)))), "\n")

print("Train MAE:", "{:,.2f}".format(mean_absolute_error(y_train, reg.predict(X_train))))

print("Valid MAE:", "{:,.2f}".format(mean_absolute_error(y_test, reg.predict(X_test))), "\n")

print("Train MAPE:", "{:,.2f}".format(mean_absolute_percentage_error(y_train, reg.predict(X_train))))

print("Valid MAPE:", "{:,.2f}".format(mean_absolute_percentage_error(y_test, reg.predict(X_test))), "\n")

print("Train R^2:", "{:,.2f}".format(r2_score(y_train, reg.predict(X_train))))

print("Valid R^2:", "{:,.2f}".format(r2_score(y_test, reg.predict(X_test))))
lgb_param = {

    # Configuration

    "nthread": -1,

    "objective": "regression",

    "metric": "rmse",

    "verbose": -1,

}



reg = lgb.LGBMRegressor(

    random_state=46, **lgb_param

)

reg.fit(X,y)





print("Train RMSE:","{:,.2f}".format(np.sqrt(mean_squared_error(y, reg.predict(X)))))

print("Train MAE:", "{:,.2f}".format(mean_absolute_error(y, reg.predict(X))), "\n")



print("CV RMSE:", "{:,.2f}".format(np.sqrt(-cross_val_score(reg, X, y, cv=10, scoring="neg_mean_squared_error")).mean()))

print("CV MAE:", "{:,.2f}".format(-cross_val_score(reg, X, y, cv=10, scoring="neg_mean_absolute_error").mean()))
def plotImp(model, X , num = 30, figsize=(20, 8)):

    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})

    plt.figure(figsize=figsize)



    p = sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 

                                                                      ascending=False)[0:num])

    p.axes.set_title('LightGBM Features (avg over folds)',fontsize=20)

    plt.tight_layout()

    #plt.savefig('lgbm_importances-01.png')

    plt.show()

    

plotImp(reg, X)
# Benchmark

#------------------------------

lgb_param = {

    # Configuration

    "nthread": -1,

    "objective": "regression",

    "metric": "rmse",

    "verbose": -1

}



reg = lgb.LGBMRegressor(

    random_state=46, **lgb_param

)

reg.fit(X_train,y_train)





print("Train RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train, reg.predict(X_train)))))

print("Train MAE:", "{:,.2f}".format(mean_absolute_error(y_train, reg.predict(X_train))), "\n")



print("CV RMSE:", "{:,.2f}".format(np.sqrt(-cross_val_score(reg, X_train, y_train, cv=10, scoring="neg_mean_squared_error")).mean()))

print("CV MAE:", "{:,.2f}".format(-cross_val_score(reg, X_train, y_train, cv=10, scoring="neg_mean_absolute_error").mean()), "\n")



print("Valid RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test, reg.predict(X_test)))))

print("Valid MAE:", "{:,.2f}".format(mean_absolute_error(y_test, reg.predict(X_test))), "\n")
# Tuned

#------------------------------



params = {

    # Training 

    'learning_rate': [0.1, 0.09, 0.08, 0.07, 0.05],

    # Tree

    'max_depth': [-1] + np.arange(5, 36, 1),

    # Deal with Over-fitting

    'num_leaves':np.arange(20, 81, 1),

    'n_estimators':[50, 100, 500, 1000, 3000, 5000, 10000], 

    'max_bin':np.arange(150, 301, 1), # Default 255

    'min_data_in_leaf':np.arange(15, 61, 1), # Default 20

    "feature_fraction_seed":np.arange(2, 1001, 1),

    "feature_fraction":np.arange(0.1, 1.01, 0.01),#[0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 1], #0.41,# Default: 1 

    "bagging_freq":[0], # Default: 0 | Note: to enable bagging, bagging_freq should be set to a non zero value as well

    "bagging_fraction":[1], # Default: 1 

    "min_data_in_leaf":np.arange(20, 101, 1), # Default: 20 

    "min_sum_hessian_in_leaf":[1e-2,1e-3, 1e-4, 1e-5, 1e-6], # Default: 1e-3 

    # Regularization

    # - Try lambda_l1, lambda_l2 and min_gain_to_split for regularization

    "lambda_l1":np.arange(0.0, 1.1, 0.1),

    "lambda_l2":np.arange(0.0, 1.1, 0.1),

    "min_gain_to_split ": np.arange(0, 101, 1)  # Default: 0

}





reg = lgb.LGBMRegressor(

    random_state=46, objective = "rmse", metric = "rmse"

)



from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(reg, params, random_state=46, n_jobs=-1, cv = 10, scoring = "neg_mean_squared_error")

rs_lgbm = rs.fit(X_train, y_train,

                 eval_set=[(X_train, y_train),(X_test, y_test)],

                 eval_metric = ["rmse", "mae"],

                 eval_names=["Train"],

                 early_stopping_rounds=10,

                 verbose=10,

                 categorical_feature='auto')



print("")

print("Train RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train, rs_lgbm.predict(X_train)))))

print("Train MAE:", "{:,.2f}".format(mean_absolute_error(y_train, rs_lgbm.predict(X_train))), "\n")



print("CV RMSE:", "{:,.2f}".format(np.sqrt(-rs_lgbm.cv_results_["mean_test_score"]).mean()))

print("CV RMSE STD:", "{:,.2f}".format(np.sqrt(rs_lgbm.cv_results_["std_test_score"]).mean()), "\n")



print("Test RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test, rs_lgbm.predict(X_test)))))

print("Test MAE:", "{:,.2f}".format(mean_absolute_error(y_test, rs_lgbm.predict(X_test))), "\n\n")



print("--------------------------------------")

print("# BEST PARAMETERS")

print("--------------------------------------")

print(rs_lgbm.best_params_)
# Tuned

#------------------------------



params = {

    # Training 

    'learning_rate': [0.1, 0.09, 0.08, 0.07, 0.05],

    # Tree

    'max_depth': [-1] + np.arange(5, 16, 1),

    # Deal with Over-fitting

    'num_leaves':np.arange(50, 71, 1),

    'n_estimators':[50, 100, 500, 1000, 3000, 5000, 10000], 

    'max_bin':np.arange(140, 171, 1), # Default 255

    'min_data_in_leaf':np.arange(15, 61, 1), # Default 20

    "feature_fraction_seed":[334],

    "feature_fraction":np.arange(0.1, 1.01, 0.01),#[0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 1], #0.41,# Default: 1 

    "bagging_freq":[0], # Default: 0 | Note: to enable bagging, bagging_freq should be set to a non zero value as well

    "bagging_fraction":[1], # Default: 1 

    "min_data_in_leaf":np.arange(30, 51, 1), # Default: 20 

    "min_sum_hessian_in_leaf":[1e-2,1e-3, 1e-4, 1e-5, 1e-6], # Default: 1e-3 

    # Regularization

    # - Try lambda_l1, lambda_l2 and min_gain_to_split for regularization

    "lambda_l1":np.arange(0.0, 1.1, 0.1),

    "lambda_l2":np.arange(0.0, 1.1, 0.1),

    "min_gain_to_split ": np.arange(80, 111, 1)  # Default: 0

}





reg = lgb.LGBMRegressor(

    random_state=46, objective = "rmse", metric = "rmse"

)



from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(reg, params, random_state=46, n_jobs=-1, cv = 10, scoring = "neg_mean_squared_error")

rs_lgbm = rs.fit(X_train, y_train,

                 eval_set=[(X_train, y_train),(X_test, y_test)],

                 eval_metric = ["rmse", "mae"],

                 eval_names=["Train"],

                 early_stopping_rounds=10,

                 verbose=10,

                 categorical_feature='auto')



print("")

print("Train RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train, rs_lgbm.predict(X_train)))))

print("Train MAE:", "{:,.2f}".format(mean_absolute_error(y_train, rs_lgbm.predict(X_train))), "\n")



print("CV RMSE:", "{:,.2f}".format(np.sqrt(-rs_lgbm.cv_results_["mean_test_score"]).mean()))

print("CV RMSE STD:", "{:,.2f}".format(np.sqrt(rs_lgbm.cv_results_["std_test_score"]).mean()), "\n")



print("Test RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test, rs_lgbm.predict(X_test)))))

print("Test MAE:", "{:,.2f}".format(mean_absolute_error(y_test, rs_lgbm.predict(X_test))), "\n\n")



print("--------------------------------------")

print("# BEST PARAMETERS")

print("--------------------------------------")

print(rs_lgbm.best_params_)
res = pd.DataFrame({"Actual":y_test.SalePrice, "Pred":rs_lgbm.predict(X_test)})

res["Error"] = res.Actual - res.Pred

res["AbsoluteError"] = np.abs(res.Error)

res.sort_values("AbsoluteError", ascending = False)
res.sort_values("AbsoluteError", ascending = False).head(20)
res.describe([0.01, 0.05, 0.10, 0.20, 0.40, 0.70, 0.80, 0.90, 0.95, 0.99]).T
fig, axes = plt.subplots(4, 2, figsize = (20,20))



for axi in axes.flat:

    axi.ticklabel_format(style="sci", axis="y", scilimits=(0,10))

    axi.ticklabel_format(style="sci", axis="x", scilimits=(0,10))

    axi.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    axi.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    

res.hist("Error", ax = axes[0, 0], color = "steelblue", bins = 20)

res.hist("AbsoluteError", ax = axes[0,1], color = "steelblue", bins = 20)

sr = res.copy()

sr["StandardizedR"] = (sr.Error / sr.Error.std())

sr["StandardizedR2"] = ((sr.Error / sr.Error.std())**2)

sr.plot.scatter(x = "Pred",y = "StandardizedR", color = "red", ax = axes[1,0])

sr.plot.scatter(x = "Pred",y = "StandardizedR2", color = "red", ax = axes[1,1])

res.Actual.hist(ax = axes[2, 0], color = "purple", bins = 20)

res.Pred.hist(ax = axes[2, 1], color = "purple", bins = 20)

res.plot.scatter(x = "Actual",y = "Pred", color = "seagreen", ax = axes[3,0]);

# QQ Plot

import statsmodels.api as sm

import pylab

sm.qqplot(sr.Pred, ax = axes[3,1], c = "seagreen")

plt.suptitle("ERROR ANALYSIS", fontsize = 20)

axes[0,0].set_title("Error Histogram", fontsize = 15)

axes[0,1].set_title("Absolute Error Histogram", fontsize = 15)

axes[1,0].set_title("Standardized Residuals & Fitted Values", fontsize = 15)

axes[1,1].set_title("Standardized Residuals^2 & Fitted Values", fontsize = 15)

axes[2,0].set_title("Actual Histogram", fontsize = 15)

axes[2,1].set_title("Pred Histogram", fontsize = 15);

axes[3,0].set_title("Actual Pred Relationship", fontsize = 15);

axes[3,1].set_title("QQ Plot", fontsize = 15);

axes[1,0].set_xlabel("Fitted Values (Pred)", fontsize = 12)

axes[1,1].set_xlabel("Fitted Values (Pred)", fontsize = 12)

axes[3,0].set_xlabel("Actual", fontsize = 12)

axes[1,0].set_ylabel("Standardized Residuals", fontsize = 12)

axes[1,1].set_ylabel("Standardized Residuals^2", fontsize = 12)

axes[3,0].set_ylabel("Pred", fontsize = 12)

fig.tight_layout(pad=3.0)

plt.savefig("errors.png")

# Tüm Veri



### TRAIN - TEST

train_model = df2[df2.Id.isin(train.Id)]

test_model = df2[df2.Id.isin(test.Id)]

test_model.drop("SalePrice", axis = 1, inplace = True)





X = train_model.drop(['SalePrice', "Id"], axis=1)

y = train_model[["SalePrice"]]



final = lgb.LGBMRegressor(

    random_state=46, objective = "regression", metric = "rmse",

    **rs_lgbm.best_params_

)



final.fit(X,y, 

        eval_set=[(X,y)],

        eval_metric = ["rmse", "mae"],

        eval_names=["Train"],

        early_stopping_rounds=10,

        verbose=1000,

        categorical_feature='auto')





submission = pd.DataFrame({"Id": test_model.Id, "SalePrice":final.predict(test_model.drop("Id", axis = 1))})

submission.to_csv("submission.csv", index = None)
submission.SalePrice.hist();
submission