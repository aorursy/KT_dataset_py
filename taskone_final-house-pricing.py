import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from datetime import datetime

from scipy import stats

from scipy.stats import norm, skew #for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from statsmodels.formula.api import ols

from statsmodels.stats.diagnostic import het_white

from statsmodels.stats.diagnostic import het_goldfeldquandt as GQ

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.stats.diagnostic import normal_ad

from statsmodels.stats.stattools import durbin_watson



from statsmodels.stats.diagnostic import linear_harvey_collier 



import statsmodels.stats.api as sms

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline  



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



train.rename(columns={

    "1stFlrSF": "FirstFlrSF", "2ndFlrSF": "secondFlrSF", 

    "3SsnPorch": "ThreeSsnPorch", "BsmtFinSF1": "BsmtFinSFOne", 

    "BsmtFinSF2": "BsmtFinSFTwo"  

}, inplace=True)



test.rename(columns={

    "1stFlrSF": "FirstFlrSF", "2ndFlrSF": "secondFlrSF", 

    "3SsnPorch": "ThreeSsnPorch", "BsmtFinSF1": "BsmtFinSFOne", 

    "BsmtFinSF2": "BsmtFinSFTwo"  

}, inplace=True)
train.shape
y=train["SalePrice"].copy()

y = np.log1p(y)

train.drop(["SalePrice"], axis=1, inplace=True)



all_data = pd.concat([train, test]).reset_index(drop=True)

all_data.drop(["Id"], axis = 1, inplace=True)

# Let's get some numbers about the NAs for each feature

def nulls_by_feature(df):

    # Finding those features who have missing values

    nulls_map = df.isnull()

    nulls_by_feature = nulls_map.sum().sort_values(ascending=False)[nulls_map.sum() != 0]

    features_with_nulls = nulls_by_feature.index.values



    # total number of observations per feature

    totals = nulls_map.count()[features_with_nulls].sort_values(ascending=False)

    percentage = (nulls_by_feature / totals).sort_values(ascending=False)

    table = pd.concat([nulls_by_feature, percentage], axis=1, keys=["Total", "Percentage"], sort=True)

    indexed_table = table.copy()

    # Transform the Index in a column

    table.index.name = "Feature"

    table.reset_index(inplace=True)

    

    # NULLS by Feature

    table = table.sort_values(by="Percentage", ascending=False)

    return (table, indexed_table)



features_nulls = nulls_by_feature(all_data)

print(features_nulls[0])
# Let's plot an histogram with this data

# We want to know which variables which have NaN values are categoricals and which ones are not



def is_categorical(array_like):

    return array_like.dtype.name == 'object'



def plot_features_by_nulls(nulls_table):

    if(len(nulls_table) == 0):

        print("Nulls Table is Empty")

        return

    nulls_table["isCategorical"] = [is_categorical(all_data[feature]) for feature in nulls_table["Feature"]]

    nulls_plot = sns.catplot(x="Feature", y="Percentage", kind="bar",hue="isCategorical", data=nulls_table)

    plt.gcf().set_size_inches(17,4)

    nulls_plot.set_xticklabels(rotation=90)

    

plot_features_by_nulls(features_nulls[0])
# Some auxiliary functions to get the features with NaNs

def extract_categoricals_with_nan(features_with_nan, df):

    return [feature for feature in features_with_nan if is_categorical(df[feature])]



def extract_numericals_with_nan(features_with_nan, df):

    return [feature for feature in features_with_nan if

            feature not in extract_categoricals_with_nan(features_with_nan, df)]



categorical_variables = extract_categoricals_with_nan(features_nulls[0]["Feature"].values, all_data)

numerical_variables = extract_numericals_with_nan(features_nulls[0]["Feature"].values, all_data)
# Let's start with categorical features

def features_with_nulls(data, categorical=True):

    features_nulls = nulls_by_feature(data)

    if categorical:

        chosen_variables = extract_categoricals_with_nan(features_nulls[0]["Feature"].values, all_data)

    else:

        chosen_variables = extract_numericals_with_nan(features_nulls[0]["Feature"].values, all_data)

        

    cat_feat_null = features_nulls[0][features_nulls[0]["Feature"].isin(chosen_variables)]

    return cat_feat_null



plot_features_by_nulls(features_with_nulls(all_data))
# For all these features, NA means "feature not present". So we can input "None" as category. 

to_fill_with_none = [

    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",

    "GarageFinish", "GarageQual", "GarageType", "GarageCond",

    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

    "MasVnrType"

]



def fill_na_with(features, df, value):

    for feature in features:

        df[feature] = df[feature].fillna(value)

        

fill_na_with(to_fill_with_none, all_data, "None")
# Let's verify that now those features does not have NA values

plot_features_by_nulls(features_with_nulls(all_data))
# Let's deal with MSZoning. This feature Identifies the general zoning classification of the sale.

# It makes sense to assume that houses with the same type of dwelling have the same zoning classification

# Another attempt could be grouping by Neighborhood

all_data["MSZoning"] = all_data.groupby("MSSubClass")["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))



# For all these other features NA means that the value is actually unknown. We'll replace it with the mode.

to_fill_with_mode = ["Utilities", "Functional", "Exterior2nd", "KitchenQual", "Exterior1st", "SaleType", "Electrical"]



def fill_with_mode(features, data):

    for feature in features:

        data[feature] = data[feature].fillna(data[feature].mode()[0])

        

fill_with_mode(to_fill_with_mode, all_data)



# Again let's verify our job

plot_features_by_nulls(features_with_nulls(all_data))
# We can now foucs on numerical features

plot_features_by_nulls(features_with_nulls(all_data, categorical=False))
# Let's focus on Lot Frontage. According to the description this variable indiates the Linear feet of street connected to property

# We can input the median grouping by neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



#Here we have a problem. 

all_data["GarageYrBlt"] = all_data.groupby("Neighborhood")["GarageYrBlt"].transform(lambda x: x.fillna(x.median()))



# The following features NaN mean that feature is not present. 0 is the best value we can input in this case. 

to_fill_with_0 = ["MasVnrArea", "BsmtFullBath", "BsmtHalfBath", "BsmtFinSFOne", "BsmtFinSFTwo", "BsmtUnfSF", "TotalBsmtSF", "GarageArea", "GarageCars", "GarageYrBlt"]

fill_na_with(to_fill_with_0, all_data, 0)



plot_features_by_nulls(features_with_nulls(all_data, categorical=False))
# These variables are numerical but should be categorical



#Identifies the type of dwelling involved in the sale.

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)



#Rates the overall condition of the house

all_data['OverallCond'] = all_data['OverallCond'].astype(str)
# It's time for plotting and checking for the assumptions of Linear Regression!

# Let's start with our target variable y
df_data = all_data.copy()

df_data["SalePrice"] = y



corrmat = df_data.corr().sort_values(by="SalePrice",ascending=False)

sns.heatmap(corrmat[["SalePrice"]], vmax=.8, square=True)
features = corrmat[["SalePrice"]].index.values

print(features)
# Let's start with Linearity. For each indipendent variable 

# we should check if there's a Linear Relationship with the target variable

# Let's start with some scatter plot 



def plot_fitted_scatter_plots(df_data, features):

    nrows = int(len(features)/3) + 1

    ncols = 3

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15, nrows*5))

    axs = axs.flat



    count = 0

    for feature in features:

        sns.regplot(x=feature, y="SalePrice", data=df_data, ax=axs[count])

        count = count + 1



plot_fitted_scatter_plots(df_data, features)





# We can clearly see that most of the relationship seems to be linear 

# Now we check for Heteroskedasticity or non-constant variance of the error

# We can do it by looking at residual plots first



def plot_resid_plots(df_data, features):

    nrows = int(len(features)/3) + 1

    ncols = 3

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15, nrows*5))

    axs = axs.flat

    count = 0

    for feature in features:

        sns.residplot(x=feature, y="SalePrice", lowess=True, line_kws={'color': 'red'}, data=df_data, ax=axs[count])

        count = count + 1

        

plot_resid_plots(df_data, features)



# It seems that simple regression models with one feature are prone to show hetero skedasticity in many cases

# We want to know more. What about a model containing all numerical features? Let's follow the analytical way. 
# Now we'll perform some statistical tests about the assumptions of linear regression

# Let's first generate our OLS Model



from sklearn import linear_model

import statsmodels.formula.api as smf

from statsmodels.stats.diagnostic import het_breuschpagan



def data_for_tests(all_data, y):

    tr = all_data.iloc[:1460].copy()

    tr["SalePrice"] = y

    return tr



t_data = data_for_tests(all_data, y)



def prepare_ols(data_for_tests, features):

    #MLR Function

    f = "SalePrice~"

    for feature in features:

        f = f + feature + "+"

    f= f[:-1]

    print("Regression function: ", f)

    result = ols(formula=f, data=data_for_tests).fit(cov_type='HC1')

    return result



ols_res = prepare_ols(t_data, features) 
# Now we can be more rigorous and use statistical tests to check for heteroskedasticity

def heteroskedasticity_detector(ols_res, train ,features):

    # The Breusch–Pagan test tests whether the variance of the errors from a regression 

    # is dependent on the values of the independent variables.

    bp_test = het_breuschpagan(ols_res.resid, train[features])

    

    # The Goldfeld Quandt Test compares variances of two subgroups; one set of high values and one set of low values. 

    # If the variances differ, the test rejects the null hypothesis that the variances of the errors are not constant.

    goldfeld_quandt = GQ(ols_res.resid, ols_res.model.exog)

    

    white_test = het_white(ols_res.resid,  ols_res.model.exog)

    

    labels = ["LM  Statistic", "LM - Test p - value", "F - Statistic", "F - Test p - value"]

    print("Heteroskedasticity tests: ")

    print("Breusch–Pagan test: ", dict(zip(labels, bp_test)))

    print("White Test: ", dict(zip(labels, white_test)))



    gq_labels = ["F-statistic", "p-value", "result"]

    print("Goldfeld Quandt Test: ", dict(zip(gq_labels, goldfeld_quandt)))



heteroskedasticity_detector(ols_res, t_data, features)



# Yes, our model seems heteroskedastic. p-values are really low
# Now we want to know if there's multicollinearity in our dataset. 

# We'll use the variance inflation factor. 

def calculate_vif(x):

    thresh = 5.0

    output = pd.DataFrame()

    k = x.shape[1]

    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]

    return vif



def multicollinearity_detector(data_for_tests, threshold):

    feat = features

    feat = np.delete(feat, 0)

    vifs = pd.Series(calculate_vif(all_data.iloc[:1460][feat]))

    vifs_by_feature = pd.concat([pd.Series(feat),round(vifs,2)], keys=["Feature", "Vif"], axis=1)

    multi = vifs_by_feature[vifs_by_feature["Vif"] > threshold].sort_values(by="Vif", ascending=False)

    print("Multicollinearity test with variance inference factor: ")

    print(multi)

    return multi



multicollinearity_detector(t_data, 5)



#Several Features Are Really Collinear
# This fails for some reasons

def nonlinearity_detector(ols_res):

    hc = linear_harvey_collier(ols_res)

    print(hc)

    

#nonlinearity_detector(ols_res)
# Normality of Residuals

import pylab 

import scipy.stats as stats 

stats.probplot(ols_res.resid, dist="norm", plot=pylab) 

pylab.title('QQ Plot: Test Gaussian Residuals') 

pylab.show()



# Graphically it doesn't seem too bad.

# Let's test for normality



def normality_test(residuals):

    labels = ["Test statistic", "p-value"]

    res1 = normal_ad(residuals, axis=0)

    res2 = stats.shapiro(residuals)

    print("Normality tests: ",dict(zip(labels,res1)), dict(zip(labels,res2)))



normality_test(ols_res.resid)

# One last check for autocorrelation of residuals



def autocorrelation_detector(residuals):

    #dw_statistics = pd.Series(np.round(durbin_watson(data[features]),4))

    #res = pd.concat([pd.Series(features), dw_statistics], keys=["features", "dw_statistic"], axis=1)

    #res = res.sort_values(by="dw_statistic", ascending=False)

    

    print("Durbin-watson statistic",durbin_watson(residuals))

    

#autocorrelation_detector(t_data, features)

autocorrelation_detector(ols_res.resid)

# his statistic will always be between 0 and 4. The closer to 0 the statistic, 

#the more evidence for positive serial correlation. The closer to 4, 

# the more evidence for negative serial correlation.



#So.. 1.52 doesn't seem so bad.
#Until know we checked for : heteroskedasticity, nonlinearity, multicollinearity, and normality and auto-correlation

# Let's see what we can do to clean this mess. 
#First let's put everything togheter for handy evaluation of the conditions. 

def assumptions_of_linear_regression(all_data, y, features):

    dt = data_for_tests(all_data, y)

    ols_res = prepare_ols(dt, features)

    heteroskedasticity_detector(ols_res, dt, features)

    multicollinearity_detector(dt, 5)

    normality_test(ols_res.resid)

    autocorrelation_detector(ols_res.resid)

    

#assumptions_of_linear_regression(all_data, y, features)
# First, let's detect skewed features to correct the skeweness and then see if the situation improves.



def detect_skewed_features(df):

    numeric_feats = df.dtypes[df.dtypes != "object"].index

    skeweness_by_feature = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    #print(skeweness_by_feature)

    return skeweness_by_feature[abs(skeweness_by_feature) > 0.75]





def fix_skeweness(df):

    skewed_features_skeweness = detect_skewed_features(df)

    skewed_features = skewed_features_skeweness.index.values



    for feature in skewed_features:

        df[feature] = boxcox1p(df[feature], boxcox_normmax(df[feature] + 1))



fix_skeweness(all_data)



#assumptions_of_linear_regression(all_data, y, features)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb





end_data = pd.get_dummies(all_data).reset_index(drop=True)

sub_indexes = end_data[1460:].index.values

end_data = RobustScaler().fit_transform(end_data)

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor





def rmsle_cv2(model, X, y):

    kf = KFold(10, shuffle=True, random_state=42).get_n_splits(X)

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



X_train_robust = end_data[:1460]

#X_train_robust = X_train_robust[:10]

X_sub = end_data[1460:]

#y = y[:10]



kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',

                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=42)

lightgbm = LGBMRegressor(objective='regression',

                         num_leaves=4,

                         learning_rate=0.01,

                         n_estimators=5000,

                         max_bin=200,

                         bagging_fraction=0.75,

                         bagging_freq=5,

                         bagging_seed=7,

                         feature_fraction=0.2,

                         feature_fraction_seed=7,

                         verbose=-1,

                         )



xgboost = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,

                             learning_rate=0.05, max_depth=3,

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread=-1

                       )



rf = RandomForestRegressor(n_estimators=100, random_state=1)

ridge = RidgeCV(alphas=alphas_alt, cv=kfolds)

lasso = LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds)

elasticnet = ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio)

svr = SVR(C=20, epsilon=0.008, gamma=0.0003,)

xgb = xgboost



stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, svr, xgboost, lightgbm, rf),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)

stack_gen.fit(X_train_robust, y)

predicted_prices = stack_gen.predict(X_sub)



my_submission = pd.DataFrame({"Id": sub_indexes, "SalePrice": predicted_prices})

my_submission.to_csv('submission.csv', index=False)
print(my_submission)