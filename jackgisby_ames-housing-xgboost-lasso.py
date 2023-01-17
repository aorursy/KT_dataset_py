import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore scikit/sns warnings



from scipy import stats

from scipy.stats import norm, skew



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #limit float outputs to 3
test = pd.read_csv("../input/test.csv", index_col = 0)

train = pd.read_csv("../input/train.csv", index_col = 0)

train.head()
sns.lmplot(data = train, x = "GrLivArea", y = "SalePrice") #can see outliers that affect the model

plt.ylabel("Sale Price")

plt.xlabel("Living Area")
#remove outliers - very large

train = train.drop(train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index)

sns.lmplot(data = train, x = "GrLivArea", y = "SalePrice")

plt.ylabel("Sale Price")

plt.xlabel("Living Area")
#now to look at saleprice

plt.boxplot(train["SalePrice"])
sns.distplot(train['SalePrice'] , fit=norm);



(mu, sigma) = norm.fit(train['SalePrice']) #get normal dist parameters

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best') #plot distribution details

plt.ylabel('Frequency')

plt.title('Sale Price distribution')



fig = plt.figure() #qqplot

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show() #data is heavily right skewed, can correct
#apply log(1+x) to sale price

train["SalePrice"] = np.log1p(train["SalePrice"])



sns.distplot(train['SalePrice'] , fit=norm);



(mu, sigma) = norm.fit(train['SalePrice']) #get normal dist parameters

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best') #plot distribution details

plt.ylabel('Frequency')

plt.title('Sale Price distribution')



fig = plt.figure() #qqplot

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show() #skew is gone



y_train = train["SalePrice"] #assign normalised y
plt.boxplot(train["SalePrice"]) #skew has been removed
#combine test and train

all_data = pd.concat([train.drop(["SalePrice"], axis = 1), test])



#look at na values

all_data.describe()
na_ratios = all_data.isnull().sum() / len(all_data)

na_ratios = na_ratios.drop(na_ratios[na_ratios == 0].index).sort_values(ascending = False)

na_ratios = pd.DataFrame({"Missing Proportion": na_ratios})

na_ratios = na_ratios.drop(na_ratios[na_ratios["Missing Proportion"] <0.005].index)

na_ratios
#the amount of missing data for "Garage", "Bsmt" and "Mas" are each roughly the same, clearing up plot

na_ratios = na_ratios.drop(index=["GarageFinish", "GarageYrBlt", "GarageQual", "GarageCond"])

na_ratios = na_ratios.drop(index=["BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2"])

na_ratios = na_ratios.drop(index=["MasVnrType"])



plt.subplots(figsize=(12, 12))

plt.xticks(rotation = "90")

sns.barplot(x=na_ratios.index, y = na_ratios["Missing Proportion"])

plt.xlabel("Features", fontsize = 15)

plt.title("Proportion of Missing Data for Each Feature", fontsize = 18)

plt.ylabel("Missing Proportion", fontsize = 15)
plt.subplots(figsize=(20,15))

sns.heatmap(all_data.corr(), vmax = 0.9, square = True, annot = True) #correlation plot for various features
plt.subplots(figsize=(20,15))

sns.heatmap(train.corr(), vmax = 0.9, square = True, annot = True) #correlation plot with saleprice
na_ratios = all_data.isnull().sum()

na_ratios = na_ratios.drop(na_ratios[na_ratios == 0].index)

na_ratios = pd.DataFrame({"Missing Count": na_ratios})

print(na_ratios) #all missing values

na_ratios.to_csv("NA_Ticklist.csv")
all_data["MSZoning"] = all_data["MSZoning"].fillna("RL") #fill na with most common value

all_data["MSZoning"].value_counts()
#basing the lotfrontage on the building subclass - lotfrontage related to building type

lf_est = all_data[["LotFrontage", "MSSubClass"]].pivot_table("LotFrontage", "MSSubClass", aggfunc = "median")

lf_est = lf_est.fillna(np.nanmedian(all_data["LotFrontage"]))

lf_est = pd.Series(lf_est["LotFrontage"], index = lf_est.index)

lf_est
lf_na_index = all_data["LotFrontage"][all_data["LotFrontage"].isnull()].index

lf_na_ests = all_data["MSSubClass"][lf_na_index].map(lf_est)

all_data["LotFrontage"][lf_na_index] = lf_na_ests

all_data[["LotFrontage", "MSSubClass"]].ix[lf_na_index].head() #check for correct mapping

#could also use groupby on neighbourhood and use those medians

all_data["LotFrontage"] = all_data["LotFrontage"].fillna(np.nanmedian(all_data["LotFrontage"]))
#following NAs need to be convered to "none" or 0 according to data descriptions

na_means_none_features = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",

                         "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",

                         "GarageCond", "PoolQC", "Fence", "MiscFeature"]

all_data[na_means_none_features] = all_data[na_means_none_features].fillna("None")
all_data["Utilities"] = all_data["Utilities"].fillna("AllPub") #fill with most common value

all_data["Utilities"].value_counts()
all_data["Exterior1st"] = all_data["Exterior1st"].fillna("VinylSd")#fill with most common value

all_data["Exterior1st"].value_counts()
all_data["Exterior2nd"] = all_data["Exterior2nd"].fillna("VinylSd") #fill with most common value

all_data["Exterior2nd"].value_counts()
fill_feature = "MasVnrType"

all_data[fill_feature] = all_data[fill_feature].fillna("None") #missing value likely means none

all_data[fill_feature].value_counts()

fill_feature = "MasVnrArea"

all_data[fill_feature] = all_data[fill_feature].fillna(0) #as previous
#bsmtqual means no basement, so fill other variables appropriately

fill_feature = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath", "TotalBsmtSF"]

all_data[fill_feature] = all_data[fill_feature].fillna(0)

all_data[fill_feature].head()
fill_feature = "SaleType"

all_data[fill_feature] = all_data[fill_feature].fillna("WD") #replace with most common value

all_data[fill_feature].value_counts()
fill_feature = "Electrical"

all_data[fill_feature] = all_data[fill_feature].fillna("SBrkr") #replace with most common value

all_data[fill_feature].value_counts()
fill_feature = "KitchenQual"

all_data[fill_feature] = all_data[fill_feature].fillna("TA") #replace with most common value

all_data[fill_feature].value_counts()
fill_feature = "Functional"

all_data[fill_feature] = all_data[fill_feature].fillna("Typ") #replace with most common value

all_data[fill_feature].value_counts()
fill_feature = ["GarageCars", "GarageArea"]

all_data[fill_feature] = all_data[fill_feature].fillna(0) #replace with most common value

all_data[fill_feature].head()
#replace garageyrblt nas based on OverallQual

garageYrMap = all_data.pivot_table("GarageYrBlt", "OverallQual")

garageYrMap = pd.Series(garageYrMap["GarageYrBlt"],

                       index = garageYrMap.index)

qual_to_garage_map = all_data["OverallQual"].map(garageYrMap)

garageYrBlt_nas = all_data["GarageYrBlt"].isnull().index

all_data["GarageYrBlt"][all_data["GarageYrBlt"].isnull()] = qual_to_garage_map[all_data["GarageYrBlt"].isnull()]

garageYrMap
na_ratios = all_data.isnull().sum()

na_ratios = na_ratios.drop(na_ratios[na_ratios == 0].index)

na_ratios = pd.DataFrame({"Missing Count": na_ratios})

print(na_ratios) #all missing values filled
numerical_features = ['LotFrontage',

 'LotArea',

 'OverallQual',

 'YearBuilt',

 'MasVnrArea',

 'BsmtFinSF1',

 'BsmtFinSF2',

 'BsmtUnfSF',

 'TotalBsmtSF',

 '1stFlrSF',

 '2ndFlrSF',

 'GrLivArea',

 'FullBath',

 'TotRmsAbvGrd',

 'Fireplaces',

 'GarageYrBlt',

 'GarageCars',

 'GarageArea',

 'WoodDeckSF',

 'OpenPorchSF',

 'EnclosedPorch',

 'ScreenPorch',

 'TotalSF'] #list of all the features that will be considered numerical or ordered categorical



unordered_cat_features = ['MSSubClass',

 'MSZoning',

 'Street',

 'Alley',

 'LotShape',

 'LandContour',

 'LotConfig',

 'LandSlope',

 'Neighborhood',

 'Condition1',

 'Condition2',

 'BldgType',

 'HouseStyle',

 'OverallCond',

 'RoofStyle',

 'RoofMatl',

 'Exterior1st',

 'Exterior2nd',

 'MasVnrType',

 'ExterQual',

 'ExterCond',

 'Foundation',

 'BsmtQual',

 'BsmtCond',

 'BsmtExposure',

 'BsmtFinType1',

 'BsmtFinType2',

 'Heating',

 'HeatingQC',

 'CentralAir',

 'Electrical',

 'LowQualFinSF',

 'BsmtFullBath',

 'BsmtHalfBath',

 'HalfBath',

 'BedroomAbvGr',

 'KitchenAbvGr',

 'KitchenQual',

 'Functional',

 'FireplaceQu',

 'GarageType',

 'GarageFinish',

 'GarageQual',

 'GarageCond',

 'PavedDrive',

 '3SsnPorch',

 'PoolArea',

 'PoolQC',

 'Fence',

 'MiscFeature',

 'MiscVal',

 'MoSold',

 'YrSold',

 'SaleType',

 'SaleCondition'] #list of all the features that will be considered unordered categorical - see data_description.txt
all_data['MSZoning'] = all_data['MSZoning'].apply(str) #encode mssubclass as categorical
sns.distplot(all_data["LowQualFinSF"])

all_data["LowQualFinSF"][all_data["LowQualFinSF"] > 0] = 1

all_data["LowQualFinSF"] = all_data["LowQualFinSF"].fillna(0) #almost all of the data is at 0, so encode as categorical varialbe (0 or >0)
def cat_pivot(feature):

    """

    feature: a feature found in all_data

    

    returns: a pivot table based on the median value for saleprice for each category

    """

    

    ms_saleprice = pd.concat([all_data.ix[train.index][feature], y_train], axis = 1)

    return ms_saleprice.pivot_table("SalePrice", feature, aggfunc = "median").sort_values("SalePrice")



def lin_plot(feature):

    """

    feature: a feature found in all_data

    

    returns: a seaborn lmplot to assess the correlation of a variable with sale price

    """

        

    lf_saleprice = pd.concat([all_data.ix[train.index][feature], y_train], axis = 1)

    return sns.lmplot(data = lf_saleprice, x = feature, y = "SalePrice") 
cat_pivot("LowQualFinSF")
sns.distplot(all_data["3SsnPorch"])

all_data["3SsnPorch"][all_data["3SsnPorch"] > 0] = 1

all_data["3SsnPorch"] = all_data["3SsnPorch"].fillna(0) #almost all of the data is at 0, so encode as categorical varialbe (0 or >0)
cat_pivot("3SsnPorch")
sns.distplot(all_data["PoolArea"])

all_data["PoolArea"][all_data["PoolArea"] > 0] = 1

all_data["PoolArea"] = all_data["PoolArea"].fillna(0) #almost all of the data is at 0, so encode as categorical varialbe (0 or >0)
cat_pivot("PoolArea")
sns.distplot(all_data["MiscVal"])

all_data["MiscVal"][all_data["MiscVal"] > 0] = 1 #almost all of the data is at 0, so encode as categorical varialbe (0 or >0)
cat_pivot("MiscVal")
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str) #encode mssubclass as categorical



cat_pivot("MSSubClass") #demonstrating that this is a categorical variable - ordered by saleprice
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

lin_plot("TotalSF") #create new feature - the total surface area of all floors, is well correlated with saleprice
#check for skew in numerical or ordered categorical variables

skewed_features = all_data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewness
lin_plot("LotArea") #lot area before scaling
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)

    

#all_data[skewed_features] = np.log1p(all_data[skewed_features]) #can also log transform        
lin_plot("LotArea") #lot area post scaling
#can make plots of each variable



#for feature in numerical_features:

#    lin_plot(feature)



#for feature in unordered_cat_features:

#    cat_pivot(feature)
from sklearn.preprocessing import OneHotEncoder



all_numerical_features = all_data[numerical_features]



unordCatOHE = OneHotEncoder(sparse = False) #use onehotencoder to deal with categorical variables - drop = "first" not available in this kernel

all_categorical_features = pd.DataFrame(unordCatOHE.fit_transform(all_data[unordered_cat_features]), index = all_data.index)



all_data_features = pd.concat([all_numerical_features,

                               all_categorical_features],

                               axis = 1) #create a new dataframe only containing the useful features



X_train = all_data_features.ix[train.index]

X_test = all_data_features.ix[test.index] #create new train and test datasets for use in modelling below
def preds_to_output(preds):

    """

    preds: predictions of saleprice for the test data

    

    returns: nothing - writes submission.csv in the same format as sample_submission.csv

    """

    output = pd.DataFrame({

        "Id": X_test.index,

        "SalePrice": preds

    })



    output.to_csv("submission.csv", index = False)

    

    print("CSV written successfully")
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error



rtree = RandomForestRegressor()

fitted_model = rtree.fit(X_train, y_train)

preds = np.expm1(rtree.predict(X_test))



def cross_val(model):

    return -np.mean(cross_val_score(model, X_train, y_train, cv = 5, scoring = "neg_mean_absolute_error"))

    

cross_val(rtree) #demonstrating cross validation method with randomforestregressor
from xgboost import XGBRegressor



model = XGBRegressor(n_estimators = 3000, #100-1000

    learning_rate = 0.005, #increase while decreasing n_trees

    max_depth = 5, #increase incrementally by 1; default 6, increasing can lead to overfit

    colsample_bytree = 0.3, # 0.3 to 0.8

    gamma = 0) #0, 1 or 5



model.fit(X_train, y_train)

cross_val(model)

xgb_preds = np.expm1(model.predict(X_test)) #store the predictions for xgbregressor

preds_to_output(xgb_preds)
cross_val(model)
importances = pd.DataFrame({'Variable':X_test.columns,

              'Importance':model.feature_importances_}).sort_values('Importance', ascending=False)



plt.subplots(figsize=(15, 15))

plt.xticks(rotation = "90")

sns.barplot(data = importances[importances["Importance"] > 0.004], x = "Variable", y = "Importance") #plot the importance of variables according to the xgbregressor model
from sklearn.linear_model import Lasso

from sklearn.preprocessing import RobustScaler



robScale = RobustScaler()

robScaleXtrain = robScale.fit_transform(X_train)

robScaleXtest = robScale.transform(X_test)



model=Lasso(alpha = 0.00001, random_state=1)

model.fit(robScaleXtrain, y_train)



lasso_preds = np.expm1(model.predict(robScaleXtest)) #store the predictions from the lasso model

preds_to_output(lasso_preds)
cross_val(model)
preds_to_output(((lasso_preds + xgb_preds) / 2)) #take an average of the xgbregressor and lasso models for final submission (~0.)