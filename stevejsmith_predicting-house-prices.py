# Import Functions

import numpy as np # linear algebra

# Stats

from scipy.stats import norm 

from scipy.stats import skew

from scipy import stats

# Model Assistants

from sklearn.preprocessing import StandardScaler

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Graphical Processing

import matplotlib.pyplot as plt

import seaborn as sns



# Warnings

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline
# Return Ordinals for categorical data

def getOrdinals(f1, f2, df):

    """

    Returns the medians as Ordinals

    """

    medians = df[[f1, f2]].groupby([f1], as_index=False).median().sort_values(by=f1, ascending=True)

    medians[f1+'Ords'] = round(medians[f2]/100000,1)

    medians = medians.drop([f2], axis=1)

    return medians



def getOrdinalMeans(f1, f2, df):

    means = df[[f1, f2]].groupby([f1], as_index=False).mean().sort_values(by=f1, ascending=True)

    means[f1+'Ords'] = round(means[f2]/100000,1)

    means = means.drop([f2], axis=1)

    return means





def applyOrdinals(ordinal_df, combined_df):

    """

    Adds a column to the combined dataframe

    """

    # creating a dict

    index1 = ordinal_df.columns.values[0]

    index2 = ordinal_df.columns.values[1]

    val_dict = ordinal_df.set_index(index1).to_dict()

    # creating a column with values from those in the dict

    

    for dataset in combined_df:

        dataset[index2] = dataset[index1].apply(lambda x: val_dict[index2][x])

    

    return combined





def replaceNans(feature, combined_df, replacement):

    for i in range(len(combined_df)):

        combined_df[i].loc[ combined_df[i][feature].isnull(), feature] = replacement

    return combined_df

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



combined = [df_train, df_test]
# Functions written based on the Tutorial



#========================================

# Descriptive Stats

def featureStats(feature, df, retdata=False):

    # Function for collecting desciptive Stats

    des = df[feature].describe()

    skew = df[feature].skew() 

    kurt = df[feature].kurt()

    if retdata is not False:

        return des, skew, kurt

    else:

        print("Descriptive Stats")

        print(des)

        print("Skewness: %f" % skew)

        print("Kurtosis: %f" % kurt)

    return

#=========================================

# Plots

def scPlot(indf, depf, df, retdata=False):

    # Scatter Plot

    data = pd.concat([df[depf], df[indf]], axis=1)

    fig = data.plot.scatter(x=indf, y=depf)

    if retdata is not False:

        return data

    else:

        pass

    return



def multiScatter(features, df):

    # grid of scatter plots

    sns.set()

    sns.pairplot(df[features], size=3.5)

    plt.show()

    return



def bxPlot(catf, depf, df, retdata=False, xrot=0):

    # box plot

    data = pd.concat([df[depf], df[catf]], axis=1)

    f, ax = plt.subplots(figsize=(12,10))

    fig = sns.boxplot(x=catf, y=depf, data=data)

    plt.xticks(rotation=xrot)

    if retdata is not False:

        return data

    else:

        pass 

    return

 

def corrHeat(df, feature=None, map_vars=10):

    # Correlation Heatmap (feature=None)

    # Correlation Heatmap with vals (feature=feature)

    corrmat = df.corr()

    if (feature is None):

        

        f, ax = plt.subplots(figsize=(12,9))

        sns.heatmap(corrmat, vmax=.8, square=True)

    else:

        cols = corrmat.nlargest(map_vars, feature)[feature].index

        cm = np.corrcoef(df[cols].values.T)

        sns.set(font_scale=1.25)

        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', \

                         annot_kws={'size': 10}, yticklabels=cols.values, \

                         xticklabels=cols.values)

        plt.show()

        

    return

 





def histNormPlot(feature, df, fit=norm):

    # Plots histogram and normal distribution

    sns.distplot(df[feature], fit=fit)

    fig = plt.figure()

    res = stats.probplot(df[feature], plot=plt)

    return



#=====================================================

# Data operations



def missingData(df):

    # Fetching the columns in a dataframe with missing data

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data.head(20)



def standardizeData(df, feature, retdata=True):

    # data standardization means converting data values 

    # to have mean of 0 and a standard deviation of 1

    feature_scaled = StandardScaler().fit_transform(df[feature][:,np.newaxis])

    low_range = feature_scaled[feature_scaled[:,0].argsort()][:10]

    high_range = feature_scaled[feature_scaled[:,0].argsort()][-10:]

    print('outer range (low) of the distribution:')

    print(low_range)

    print('\nouter range (high) of the distribution:')

    print(high_range)

    if retdata is True:

        return feature_scaled

    else:

        return

    

def fetchOutlierId(df, indf, depf, ifgt, dfgt):

    # Finds the outliers above specified values

    # Use after identifying outliers in a scatter

    # df = datframe to fetch values from

    # indf is independent feature

    # depf is dependent feature

    # ifgt, values of the ifeat greater than ifgt

    # dgft, values of the dfeat greater than ifgt

    vals = df.loc[(df[indf] > ifgt) & (df[depf] > dfgt)]

    data = pd.concat([vals['Id'], vals[indf], vals[depf]], axis=1)

    return data



#===========================================

# Delete Operations

def delPoints(id_list, df):

    for val in id_list:

        df = df.drop(df[df['Id'] == val].index)

    return df



def delColumn(feature, combined_df):

    train_df = combined_df[0]

    test_df = combined_df[1]

    train_df = train_df.drop([feature], axis=1)

    test_df = test_df.drop([feature], axis=1)

    combined = [train_df, test_df]

    return combined



#===========================================

# Restore Operations

def restoreFeature(list_of_features, df):

    """

    Used to restore a Feature to the combined dataframe

    if deleted in error

    """

    fetch_df = pd.read_csv('../input/train.csv')

    fetch_df1 = pd.read_csv('../input/test.csv')   

    for feature in list_of_features:

        df[0][feature] = fetch_df[feature]

        df[1][feature] = fetch_df1[feature]

    return df



def printFunctions():

    useful_functions = {

        'Stats' : {'featureStats(feature, df, retdata=False)'},

        'Plots' : {'scPlot(indf, depf, df, retdata=False)',\

                   'multiScatter(features, df)',\

                   'bxPlot(catf, depf, df, retdata=False, xrot=0)',\

                   'corrHeat(df, feature=None, map_vars=10)',\

                   'histNormPlot(feature, df, fit=norm)'},

        'Data operations' : {'missingData(df)', \

                             'standardizeData(df, feature, retdata=False',\

                             'fetchOutlierId(df, indf, depf, ifgt, dfgt)'},

        'Delete Operations' : {'delPoints(id_list, df)',\

                               'delColumn(feature, combined_df)'},

        'Restore Operations' : {'restoreFeature(list_of_features, df)'},

        'Ordinals' : {'getOrdinals(f1, f2, df)',\

                      'getOrdinalMeans(f1, f2, df)',\

                      'applyOrdinals(ordinal_df, combined_df)'},

        'NaNs' : {'replaceNans(feature, combined_df, replacement)'},

    }

    import pprint

    return pprint.pprint(useful_functions)

# Test

printFunctions()

#histNormPlot('SalePrice', combined[0])

#fetchOutlierId(combined[0], 'GrLivArea', 'SalePrice', 4000, 100000)

#id_list = [524, 1299]

#combined[0] = delPoints(id_list, combined[0])

#fetchOutlierId(combined[0], 'GrLivArea', 'SalePrice', 4000, 100000)



#featureStats('SalePrice', combined[0])

#scPlot('TotalBsmtSF', 'SalePrice', combined[0])

#bxPlot('OverallQual', 'SalePrice', combined[0])

#bxPlot('YearBuilt', 'SalePrice', combined[0], xrot=90)

#corrHeat(combined[0], feature='SalePrice', map_vars=9)

#corrHeat(combined[0])

#features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',\

#            'TotalBsmtSF', 'FullBath', 'YearBuilt']

#multiScatter(combined[0], features)

# missingData(combined[0])



#standardizeData(combined[0], 'SalePrice')
# Fetching the feature names



combined[0].columns
# Fetching feature data info

combined[0].info()
featureStats('SalePrice', combined[0])
histNormPlot('SalePrice', combined[0])
#applying log transformation

combined[0]['LogSalePrice'] = np.log(combined[0]['SalePrice'])



# Fetching the histogram for LogSalePrice

histNormPlot('LogSalePrice', combined[0])
combined[0] = combined[0].drop(['SalePrice'], axis=1)
combined[0].info()
# Handle missing values for features where median/mean or most common value doesn't make sense

for dataset in combined:

    # Alley : data description says NA means "no alley access"

    dataset.loc[:, "Alley"] = dataset.loc[:, "Alley"].fillna("None")

    # BedroomAbvGr : NA most likely means 0

    dataset.loc[:, "BedroomAbvGr"] = dataset.loc[:, "BedroomAbvGr"].fillna(0)

    # BsmtQual etc : data description says NA for basement features is "no basement"

    dataset.loc[:, "BsmtQual"] = dataset.loc[:, "BsmtQual"].fillna("No")

    dataset.loc[:, "BsmtCond"] = dataset.loc[:, "BsmtCond"].fillna("No")

    dataset.loc[:, "BsmtExposure"] = dataset.loc[:, "BsmtExposure"].fillna("No")

    dataset.loc[:, "BsmtFinType1"] = dataset.loc[:, "BsmtFinType1"].fillna("No")

    dataset.loc[:, "BsmtFinType2"] = dataset.loc[:, "BsmtFinType2"].fillna("No")

    dataset.loc[:, "BsmtFullBath"] = dataset.loc[:, "BsmtFullBath"].fillna(0)

    dataset.loc[:, "BsmtHalfBath"] = dataset.loc[:, "BsmtHalfBath"].fillna(0)

    dataset.loc[:, "BsmtUnfSF"] = dataset.loc[:, "BsmtUnfSF"].fillna(0)

    dataset.loc[:, "BsmtFinSF1"] = dataset.loc[:, "BsmtFinSF1"].fillna(0)

    dataset.loc[:, "BsmtFinSF2"] = dataset.loc[:, "BsmtFinSF2"].fillna(0)

    dataset.loc[:, "TotalBsmtSF"] = dataset.loc[:, "TotalBsmtSF"].fillna(0)

    



    # CentralAir : NA most likely means No

    dataset.loc[:, "CentralAir"] = dataset.loc[:, "CentralAir"].fillna("N")

    # Condition : NA most likely means Normal

    dataset.loc[:, "Condition1"] = dataset.loc[:, "Condition1"].fillna("Norm")

    dataset.loc[:, "Condition2"] = dataset.loc[:, "Condition2"].fillna("Norm")

    # EnclosedPorch : NA most likely means no enclosed porch

    dataset.loc[:, "EnclosedPorch"] = dataset.loc[:, "EnclosedPorch"].fillna(0)

    # External stuff : NA most likely means average

    dataset.loc[:, "ExterCond"] = dataset.loc[:, "ExterCond"].fillna("TA")

    dataset.loc[:, "ExterQual"] = dataset.loc[:, "ExterQual"].fillna("TA")

    # Electrical

    dataset.loc[:, "Electrical"] = dataset.loc[:, "Electrical"].fillna("SBrkr")

    dataset.loc[:, "Exterior1st"] = dataset.loc[:, "Exterior1st"].fillna("Wd Sdng")

    dataset.loc[:, "Exterior2nd"] = dataset.loc[:, "Exterior2nd"].fillna("Wd Sdng")

    

    # Fence : data description says NA means "no fence"

    dataset.loc[:, "Fence"] = dataset.loc[:, "Fence"].fillna("No")

    # FireplaceQu : data description says NA means "no fireplace"

    dataset.loc[:, "FireplaceQu"] = dataset.loc[:, "FireplaceQu"].fillna("No")

    dataset.loc[:, "Fireplaces"] = dataset.loc[:, "Fireplaces"].fillna(0)

    # Functional : data description says NA means typical

    dataset.loc[:, "Functional"] = dataset.loc[:, "Functional"].fillna("Typ")

    # GarageType etc : data description says NA for garage features is "no garage"

    dataset.loc[:, "GarageType"] = dataset.loc[:, "GarageType"].fillna("No")

    dataset.loc[:, "GarageFinish"] = dataset.loc[:, "GarageFinish"].fillna("No")

    dataset.loc[:, "GarageQual"] = dataset.loc[:, "GarageQual"].fillna("No")

    dataset.loc[:, "GarageCond"] = dataset.loc[:, "GarageCond"].fillna("No")

    dataset.loc[:, "GarageArea"] = dataset.loc[:, "GarageArea"].fillna(0)

    dataset.loc[:, "GarageCars"] = dataset.loc[:, "GarageCars"].fillna(0)

    dataset.loc[:, "GarageYrBlt"] = dataset.loc[:, "GarageYrBlt"].fillna(0)

    # HalfBath : NA most likely means no half baths above grade

    dataset.loc[:, "HalfBath"] = dataset.loc[:, "HalfBath"].fillna(0)

    # HeatingQC : NA most likely means typical

    dataset.loc[:, "HeatingQC"] = dataset.loc[:, "HeatingQC"].fillna("TA")

    # KitchenAbvGr : NA most likely means 0

    dataset.loc[:, "KitchenAbvGr"] = dataset.loc[:, "KitchenAbvGr"].fillna(0)

    # KitchenQual : NA most likely means typical

    dataset.loc[:, "KitchenQual"] = dataset.loc[:, "KitchenQual"].fillna("TA")

    # LotFrontage : NA most likely means no lot frontage

    dataset.loc[:, "LotFrontage"] = dataset.loc[:, "LotFrontage"].fillna(0)

    # LotShape : NA most likely means regular

    dataset.loc[:, "LotShape"] = dataset.loc[:, "LotShape"].fillna("Reg")

    # MasVnrType : NA most likely means no veneer

    dataset.loc[:, "MasVnrType"] = dataset.loc[:, "MasVnrType"].fillna("None")

    dataset.loc[:, "MasVnrArea"] = dataset.loc[:, "MasVnrArea"].fillna(0)

    # MiscFeature : data description says NA means "no misc feature"

    dataset.loc[:, "MiscFeature"] = dataset.loc[:, "MiscFeature"].fillna("No")

    dataset.loc[:, "MiscVal"] = dataset.loc[:, "MiscVal"].fillna(0)

    

    # OpenPorchSF : NA most likely means no open porch

    dataset.loc[:, "OpenPorchSF"] = dataset.loc[:, "OpenPorchSF"].fillna(0)

    # PavedDrive : NA most likely means not paved

    dataset.loc[:, "PavedDrive"] = dataset.loc[:, "PavedDrive"].fillna("N")

    # PoolQC : data description says NA means "no pool"

    dataset.loc[:, "PoolQC"] = dataset.loc[:, "PoolQC"].fillna("No")

    dataset.loc[:, "PoolArea"] = dataset.loc[:, "PoolArea"].fillna(0)

    # SaleCondition : NA most likely means normal sale

    dataset.loc[:, "SaleCondition"] = dataset.loc[:, "SaleCondition"].fillna("Normal")

    # SaleType: 

    dataset.loc[:, "SaleType"] = dataset.loc[:, "SaleType"].fillna("WD")

    # ScreenPorch : NA most likely means no screen porch

    dataset.loc[:, "ScreenPorch"] = dataset.loc[:, "ScreenPorch"].fillna(0)

    # TotRmsAbvGrd : NA most likely means 0

    dataset.loc[:, "TotRmsAbvGrd"] = dataset.loc[:, "TotRmsAbvGrd"].fillna(0)

    # Utilities : NA most likely means all public utilities

    dataset.loc[:, "Utilities"] = dataset.loc[:, "Utilities"].fillna("AllPub")

    # WoodDeckSF : NA most likely means no wood deck

    dataset.loc[:, "WoodDeckSF"] = dataset.loc[:, "WoodDeckSF"].fillna(0)
# Some numerical features are actually really categories

for dataset in combined:

    dataset = dataset.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                               "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

                              })
# Encode some categorical features as ordered numbers when there is information in the order

for dataset in combined:

    dataset = dataset.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                     )
combined[0].info()

combined[1].info()
corrHeat(combined[0])
corrHeat(combined[0], feature='LogSalePrice')
to_drop = ['1stFlrSF', 'GarageCars']

for feature in to_drop:

    combined = delColumn(feature, combined)

    

# Let's check the heatmap again



corrHeat(combined[0], feature='LogSalePrice')
to_drop = ['TotRmsAbvGrd']



for feature in to_drop:

    combined = delColumn(feature, combined)

    

# Checking the Heat Map again



corrHeat(combined[0], feature='LogSalePrice')


features = ['LogSalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

multiScatter(features, combined[0])
scPlot('GrLivArea', 'LogSalePrice', combined[0])
fetchOutlierId(combined[0], 'GrLivArea', 'LogSalePrice', 4000, 11.5)
id_list = [524, 1299]

combined[0] = delPoints(id_list, combined[0])



# Plotting again to see if they've been removed

scPlot('GrLivArea', 'LogSalePrice', combined[0])
# Checking Normality



histNormPlot('GrLivArea', combined[0])
#applying log transformation to data

combined[0]['LogGrLivArea'] = np.log(combined[0]['GrLivArea'])



# Checking Normality



histNormPlot('LogGrLivArea', combined[0])
#applying log transformation to test data

combined[1]['LogGrLivArea'] = np.log(combined[1]['GrLivArea'])



# dropping 'GrLivArea

combined = delColumn('GrLivArea', combined)
# Checking correlation

combined[0][['LogGrLivArea','LogSalePrice']].corr()
scPlot('OverallQual', 'LogSalePrice', combined[0])
histNormPlot('OverallQual', combined[0])
histNormPlot('GarageArea', combined[0])
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

for dataset in combined:

    dataset['HasGarage'] = dataset['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
#histogram and normal probability plot



histNormPlot('TotalBsmtSF', combined[0])
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

for dataset in combined:

    dataset['HasBsmt'] = dataset['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
#transform data

for dataset in combined:

    dataset.loc[dataset['HasBsmt']==1,'TotalBsmtSF'] = np.log(dataset['TotalBsmtSF'])

    
#histogram and normal probability plot

sns.distplot(combined[0][combined[0]['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(combined[0][combined[0]['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
combined = delColumn('HasBsmt', combined)
#log transform skewed numeric features:

numeric_feats = combined[0].dtypes[combined[0].dtypes != "object"].index



skewed_feats = combined[0][numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

skewed_feats

for dataset in combined:

    dataset[skewed_feats] = np.log1p(dataset[skewed_feats])
combined[0].head()
log_sale_price = combined[0]['LogSalePrice']



combined[0] = combined[0].drop('LogSalePrice', axis=1)

all_data = pd.concat((combined[0].loc[:,'MSSubClass':'HasGarage'],

                      combined[1].loc[:,'MSSubClass':'HasGarage']))
all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
# split the data



#creating matrices for sklearn:

X_train = all_data[:combined[0].shape[0]]

X_test = all_data[combined[0].shape[0]:]

y = log_sale_price
# https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models



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

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
print(cv_ridge)
cv_ridge.min()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.7*lasso_preds + 0.3*xgb_preds
solution = pd.DataFrame({"id":combined[1].Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)