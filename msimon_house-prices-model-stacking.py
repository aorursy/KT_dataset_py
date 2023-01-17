# TODO:

# See whether some outliers should be removed

# See whether using a box-cox transform on non-normal numeric variables would help



import os.path

import time



import numpy as np

import scipy as sp

import pandas as pd



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV, 

                                  LinearRegression, Perceptron)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, RBF

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MaxAbsScaler

from sklearn.externals import joblib

from sklearn.model_selection import cross_val_predict



import matplotlib.pyplot as plt

import seaborn as sns





RNG_SEED = int(time.time())

print("Seed: %s" % RNG_SEED)



overwrite_models = True

add_was_missing_features = False



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))

# os.listdir("..")

train_df = pd.read_csv("../input/train.csv")

# Use the log price as it is the target for the evaluation. 

# Moreover, this suppresses the skewness.

outcome_df = np.log1p(train_df.loc[:, "SalePrice"])

train_ids = train_df["Id"]

feature_df = train_df.drop(["Id", "SalePrice"], axis=1)



test_df = pd.read_csv("../input/test.csv")

test_ids = test_df["Id"]

test_df = test_df.drop("Id", axis=1)



print("train set size: %s x %s" % feature_df.shape)

print("outcome size: %s" % outcome_df.shape)

print("test set size: %s x %s" % test_df.shape)



# Quick check for the outcome variable

plt.figure()

ax = sns.distplot(outcome_df)

# Handling missing data

# We merge the train and test set to compute the modes and medians 

# since the calculations do not rely on the outcome.

def get_replacement(data):

# def get_replacement(train_set, test_set):

    # data = pd.concat([train_set, test_set])

    nb_nan_per_col = data.shape[0] - data.count()

    print(nb_nan_per_col[nb_nan_per_col != 0])

    

    missing_val_replace = {}

    

    # Type of zone (residential, commercial etc.), cannot be guessed 

    # with current data. Set the mode.

    missing_val_replace["MSZoning"] = data["MSZoning"].mode()[0]

    

    # either 'AllPub' or 'NoSeWa'. Set the mode.

    missing_val_replace["Utilities"] = data["Utilities"].mode()[0]

        

    test_df.loc[np.any(pd.isnull(test_df), axis=1), nb_nan_per_col != 0]

    

    # Linear feet of street connected to property. 

    # No 0 so nan is likely to mean "property not connected to the street"

    missing_val_replace["LotFrontage"] = 0



    # Type of alley access: nan, 'Grvl' or 'Pave'

    # nan probably means "no alley", replace with 'None'

    missing_val_replace["Alley"] = "None"

    

    # Exterior covering material. Set to mode.

    missing_val_replace["Exterior1st"] = data["Exterior1st"].mode()[0]

    missing_val_replace["Exterior2nd"] = data["Exterior2nd"].mode()[0]



    # Masonry veneer type and area in square feet

    # 'None' is already a value and 0 exists as an area

    missing_val_replace["MasVnrType"] = "None"

    missing_val_replace["MasVnrArea"] = 0



    # When BsmtFinType1 is missing, BsmtFinSF1 equals 0. Also true when 

    # BsmtFinType1 is "Unf"(inished) which makes sense.

    missing_val_replace["BsmtFinType1"] = "Unf"

    # same as BsmtFinType1 except for one case where BsmtFinSF2 equals

    # 479. Still keeping 'Unf' as an approximation.

    missing_val_replace["BsmtFinType2"] = "Unf"



    # Same as above for BsmtQual and value 'Ta'

    missing_val_replace["BsmtQual"] = "TA"

    # same as BsmtQual

    missing_val_replace["BsmtCond"] = "TA"



    # Not as straightforward as above set missing values as 

    # the mode of the column

    missing_val_replace["BsmtExposure"] = data["BsmtExposure"].mode()[0]

    

    # Basement (un)finished/total square feet. Set to 0 because there is no basement

    missing_val_replace["BsmtFinSF1"] = 0

    missing_val_replace["BsmtFinSF2"] = 0

    missing_val_replace["BsmtUnfSF"] = 0

    missing_val_replace["TotalBsmtSF"] = 0

    

    # Basement bathrooms. Set to 0 because there is no basement

    missing_val_replace["BsmtFullBath"] = 0

    missing_val_replace["BsmtHalfBath"] = 0

    

    # only one missing value, set mode

    missing_val_replace["Electrical"] = data["Electrical"].mode()[0]



    # Kitchen quality, since in the only missing case, the number of 

    # kitchen (KitchenAbvGr) is 1, we set it to the mode

    missing_val_replace["KitchenQual"] = data["KitchenQual"].mode()[0]

    

    # Home functionality rating; cannot be guessed. Set the mode.

    missing_val_replace["Functional"] = data["Functional"].mode()[0]

    

    # missing if there is no fireplace (Fireplaces equals 0). set to "None"

    missing_val_replace["FireplaceQu"] = "None"



    # For the test set, when one of GarageType, GarageYrBlt, GarageFinish, GarageQual, 

    # GarageCond is missing, all the others are missing and GarageCars, GarageArea equal 0. 

    # Thus there is no garage, we set "None" for categorical variable and to the median for 

    # GarageYrBlt to avoid type issues.

    # For the test set, GarageType is set (so there is one) but the info is missing.

    # Since only one entry has this problem, we keep the same idea as for the train set

    # even though using the mode instead of "None" could be better.

    missing_val_replace["GarageType"] = "None"

    missing_val_replace["GarageYrBlt"]= np.round(data["GarageYrBlt"].median())

    missing_val_replace["GarageCars"]= np.round(data["GarageCars"].median())

    missing_val_replace["GarageArea"]= np.round(data["GarageArea"].median())

    missing_val_replace["GarageFinish"]= "None"

    missing_val_replace["GarageQual"]= "None"

    missing_val_replace["GarageCond"] = "None"



    # When PoolQc is missing, PoolArea is 0 because there is no pool. Set to "None".

    missing_val_replace["PoolQC"] = "None"



    # No fence, set to "None"

    missing_val_replace["Fence"] = "None"



    # Probably no special features, set to "None"

    missing_val_replace["MiscFeature"] = "None"



    missing_val_replace["SaleType"] = data["SaleType"].mode()[0]

    return missing_val_replace



# Add a "was missing" features in case the missing values

# were not random

if add_was_missing_features:

    for col in feature_df:

        if np.any(pd.isnull(feature_df[col])) or np.any(pd.isnull(test_df[col])):

            feature_df["Missing_" + col] = pd.isnull(feature_df[col])

            test_df["Missing_" + col] = pd.isnull(test_df[col])



# replace missing values

# replacement_dict = get_replacement(feature_df, test_df)

# feature_df.fillna(replacement_dict, inplace=True)

# test_df.fillna(replacement_dict, inplace=True)    

feature_df.fillna(get_replacement(feature_df), inplace=True)

test_df.fillna(get_replacement(test_df), inplace=True)    



# sanity check

print("Remaining missing values in train and test sets:")

print(np.sum((feature_df.shape[0] - feature_df.count()) != 0))

print(np.sum((test_df.shape[0] - test_df.count()) != 0))

# Getting the right column types

# reference: https://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt



def ordinal_object_to_str(df):

    df["LotShape"] = df["LotShape"].astype(str)

    df["Utilities"] = df["Utilities"].astype(str)

    df["LandSlope"] = df["LandSlope"].astype(str)

    df["ExterQual"] = df["ExterQual"].astype(str)

    df["ExterCond"] = df["ExterCond"].astype(str)

    df["BsmtQual"] = df["BsmtQual"].astype(str)

    df["BsmtCond"] = df["BsmtCond"].astype(str)

    df["BsmtExposure"] = df["BsmtExposure"].astype(str)

    df["BsmtFinType1"] = df["BsmtFinType1"].astype(str)

    df["BsmtFinType2"] = df["BsmtFinType2"].astype(str)

    df["HeatingQC"] = df["HeatingQC"].astype(str)

    df["Electrical"] = df["Electrical"].astype(str)

    df["KitchenQual"] = df["KitchenQual"].astype(str)

    df["Functional"] = df["Functional"].astype(str)

    df["FireplaceQu"] = df["FireplaceQu"].astype(str)

    df["GarageQual"] = df["GarageQual"].astype(str)

    df["GarageCond"] = df["GarageCond"].astype(str)

    df["PavedDrive"] = df["PavedDrive"].astype(str)

    df["PoolQC"] = df["PoolQC"].astype(str)

    df["Fence"] = df["Fence"].astype(str)

    return df



def fix_dtypes(df):

    df["MSSubClass"] = df["MSSubClass"].astype(object)

    

    df["LotShape"] = df["LotShape"].astype(int)

    df["Utilities"] = df["Utilities"].astype(int)

    df["LandSlope"] = df["LandSlope"].astype(int)

    df["ExterQual"] = df["ExterQual"].astype(int)

    df["ExterCond"] = df["ExterCond"].astype(int)

    df["BsmtQual"] = df["BsmtQual"].astype(int)

    df["BsmtCond"] = df["BsmtCond"].astype(int)

    df["BsmtExposure"] = df["BsmtExposure"].astype(int)

    df["BsmtFinType1"] = df["BsmtFinType1"].astype(int)

    df["BsmtFinType2"] = df["BsmtFinType2"].astype(int)

    df["HeatingQC"] = df["HeatingQC"].astype(int)

    df["Electrical"] = df["Electrical"].astype(int)

    df["KitchenQual"] = df["KitchenQual"].astype(int)

    df["Functional"] = df["Functional"].astype(int)

    df["FireplaceQu"] = df["FireplaceQu"].astype(int)

    df["GarageQual"] = df["GarageQual"].astype(int)

    df["GarageCond"] = df["GarageCond"].astype(int)

    df["PavedDrive"] = df["PavedDrive"].astype(int)

    df["PoolQC"] = df["PoolQC"].astype(int)

    df["Fence"] = df["Fence"].astype(int)

    df["GarageYrBlt"] = df["GarageYrBlt"].astype(int)



    df["LotArea"] = df["LotArea"].astype(float)

    df["BsmtFinSF1"] = df["BsmtFinSF1"].astype(float)

    df["BsmtFinSF2"] = df["BsmtFinSF2"].astype(float)

    df["BsmtUnfSF"] = df["BsmtUnfSF"].astype(float)

    df["TotalBsmtSF"] = df["TotalBsmtSF"].astype(float)

    df["1stFlrSF"] = df["1stFlrSF"].astype(float)

    df["2ndFlrSF"] = df["2ndFlrSF"].astype(float)

    df["LowQualFinSF"] = df["LowQualFinSF"].astype(float)

    df["GrLivArea"] = df["GrLivArea"].astype(float)

    df["GarageArea"] = df["GarageArea"].astype(float)

    df["WoodDeckSF"] = df["WoodDeckSF"].astype(float)

    df["OpenPorchSF"] = df["OpenPorchSF"].astype(float)

    df["EnclosedPorch"] = df["EnclosedPorch"].astype(float)

    df["3SsnPorch"] = df["3SsnPorch"].astype(float)

    df["ScreenPorch"] = df["ScreenPorch"].astype(float)

    df["PoolArea"] = df["PoolArea"].astype(float)

    df["MiscVal"] = df["MiscVal"].astype(float)

    

    return df



ordinal_replacements = {}

ordinal_replacements["LotShape"] = {"Reg": "0", "IR1": "1", "IR2": "2", "IR3": "3"}

ordinal_replacements["Utilities"] = {"AllPub": "0", "NoSewr": "1", "NoSeWa": "2", "ELO": "3"}

ordinal_replacements["LandSlope"] = {"Gtl": "0", "Mod": "1", "Sev": "2"}

ordinal_replacements["ExterQual"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}

ordinal_replacements["ExterCond"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}

ordinal_replacements["BsmtQual"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}

ordinal_replacements["BsmtCond"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}

ordinal_replacements["BsmtExposure"] = {"Gd": "0", "Av": "1", "Mn": "2", "No": "3"}

ordinal_replacements["BsmtFinType1"] = {"GLQ": "0", "ALQ": "1", "BLQ": "2", "Rec": "3", "LwQ": "4", "Unf": "5"}

ordinal_replacements["BsmtFinType2"] = {"GLQ": "0", "ALQ": "1", "BLQ": "2", "Rec": "3", "LwQ": "4", "Unf": "5"}

ordinal_replacements["HeatingQC"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}

ordinal_replacements["Electrical"] = {"SBrkr": "0", "FuseA": "1", "FuseF": "2", "FuseP": "3", "Mix": "4"}

ordinal_replacements["KitchenQual"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}

ordinal_replacements["Functional"] = {"Typ": "0", "Min1": "1", "Min2": "2", "Mod": "3", "Maj1": "4", 

                                      "Maj2": "5", "Sev": "6", "Sal": "7"}

ordinal_replacements["FireplaceQu"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4", "None": 5}

ordinal_replacements["GarageQual"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4", "None": 5}

ordinal_replacements["GarageCond"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4", "None": 5}

ordinal_replacements["PavedDrive"] = {"Y": "0", "P": "1", "N": "2"}

ordinal_replacements["PoolQC"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4", "None": 5}

ordinal_replacements["Fence"] = {"GdPrv": "0", "MnPrv": "1", "GdWo": "2", "MnWw": "3", "None": 5}



feature_df = ordinal_object_to_str(feature_df)

feature_df.replace(ordinal_replacements, inplace=True)

feature_df = fix_dtypes(feature_df)



test_df = ordinal_object_to_str(test_df)

test_df.replace(ordinal_replacements, inplace=True)

test_df = fix_dtypes(test_df)

# unskewing can be usefull for regression

# Props to https://www.kaggle.com/apapiu/regularized-linear-models

def unskew_dataset(data):

    numeric_features = data.dtypes[data.dtypes == float].index

    skewed_features = data[numeric_features].apply(lambda x: sp.stats.skew(x)) #compute skewness

    skewed_features = skewed_features[skewed_features > 0.75]

    skewed_features = skewed_features.index

    data[skewed_features] = np.log1p(data[skewed_features])

    return data
# Total surface.

# Propos to https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

feature_df['TotalSF'] = feature_df['TotalBsmtSF'] + feature_df['1stFlrSF'] + feature_df['2ndFlrSF']

test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']



# encode categorical variables as numeric values for sklearn

categorical_vars_indices = np.where((feature_df.dtypes == object))[0]

categorical_vars = feature_df.columns[categorical_vars_indices]



# ft_df = pd.concat([feature_df, test_df])

# unskew features

# ft_df = unskew_dataset(ft_df)

feature_df = unskew_dataset(feature_df)

test_df = unskew_dataset(test_df)

# encode categorical variables as dummies (one hot) for linear regressors

# ft_df_dummies = pd.get_dummies(ft_df, columns=categorical_vars, 

#                                     drop_first=True, sparse=False)

# feature_df_dummies = ft_df_dummies.iloc[range(feature_df.shape[0]), :]

# test_df_dummies = ft_df_dummies.iloc[range(feature_df.shape[0], ft_df_dummies.shape[0]), :]

feature_df_dummies = pd.get_dummies(feature_df, columns=categorical_vars, 

                                    drop_first=True, sparse=False)

test_df_dummies = pd.get_dummies(test_df, columns=categorical_vars, 

                                 drop_first=True, sparse=False)



# Only keep columns common to both the train and test sets

# so that they have the same features

common_cols = list(set(feature_df_dummies.columns) & set(test_df_dummies.columns))

feature_df_dummies = feature_df_dummies[common_cols]

test_df_dummies = test_df_dummies[common_cols]



label_enc = LabelEncoder()

# fit on the concatenated train and test sets to get the same encoding 

# for them both

for var in categorical_vars:

    var_all = pd.concat([feature_df.loc[:, var], test_df.loc[:, var]])

    label_enc.fit(var_all)

    feature_df.loc[:, var] = label_enc.transform(feature_df.loc[:, var])

    test_df.loc[:, var] = label_enc.transform(test_df.loc[:, var])



# sanity checks

assert np.all(feature_df_dummies["LotShape"] == feature_df["LotShape"])

assert np.all(test_df_dummies["LotShape"] == test_df["LotShape"])

print(feature_df.shape)

print(feature_df_dummies.shape)

print(test_df.shape)

print(test_df_dummies.shape)

# standardize the variables.

scaler = StandardScaler()



feature_df[:] = scaler.fit_transform(feature_df)

feature_df_dummies[:] = scaler.fit_transform(feature_df_dummies)

test_df[:] = scaler.fit_transform(test_df)

test_df_dummies[:] = scaler.fit_transform(test_df_dummies)

rkf_cv = KFold(n_splits=5, random_state=RNG_SEED)

stack_folds = list(KFold(n_splits=5, random_state=RNG_SEED).split(feature_df))
# L1 + L2 penalized linear regression



l1_ratios = [.1, .5, .7, .9, .95, .99, 1]

alphas = alphas=[1] + [10 ** -x for x in range(1, 8)] + [5 * 10 ** -x for x in range(1, 8)]



if not os.path.isfile("cv_opt_en.pkl") or overwrite_models:

    en_cv = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas,

                         normalize=True, selection ="random", random_state=RNG_SEED,

                         max_iter=10000, cv=RepeatedKFold(10, 3, random_state=RNG_SEED))

    cv_opt_en = en_cv.fit(feature_df_dummies, outcome_df)

    joblib.dump(cv_opt_en, "cv_opt_en.pkl")

else:

    cv_opt_en = joblib.load("cv_opt_en.pkl")



# cross-validated rmse for the best parameters

l1_ratio_index = np.where(l1_ratios == cv_opt_en.l1_ratio_)[0][0]

en_alpha_index = np.where(cv_opt_en.alphas_ == cv_opt_en.alpha_)[0][0]

en_rmse = np.sqrt(np.mean(cv_opt_en.mse_path_, axis=2)[l1_ratio_index, en_alpha_index])

print(en_rmse)

print(cv_opt_en)

    

# model using the best parameters so that the cross-validation 

# does not run for every fold when we cross_val_predict() 

# (to reduce the computation time)

cv_opt_en_model = ElasticNet(alpha=cv_opt_en.alpha_, l1_ratio=cv_opt_en.l1_ratio_, 

                         fit_intercept=True, normalize=True, 

                         precompute=False, max_iter=10000, copy_X=True, tol=0.0001, 

                         warm_start=False, positive=False, random_state=RNG_SEED, 

                         selection="random")

cv_opt_en_model = cv_opt_en_model.fit(feature_df_dummies, outcome_df)

"""

# From a previous run of what is above, for the submission

cv_opt_en_model = ElasticNet(alpha=0.0003, l1_ratio=0.3, fit_intercept=True, normalize=True, 

                   precompute=False, max_iter=10000, copy_X=True, tol=0.0001, 

                   warm_start=False, positive=False, random_state=RNG_SEED, 

                   selection="random")

cv_opt_en_model = cv_opt_en_model.fit(feature_df_dummies, outcome_df)

en_rmse = 0.136357599601

"""



en_preds = cv_opt_en_model.predict(test_df_dummies)

en_cv_preds = cross_val_predict(cv_opt_en_model, feature_df_dummies, outcome_df, 

                                cv=stack_folds)



fig = plt.figure()

ax = fig.add_subplot(111)

for i in range(cv_opt_en.mse_path_.shape[0]):

    ax.plot(np.log10(cv_opt_en.alphas_), np.mean(cv_opt_en.mse_path_[i, :, :], axis=1),

             label=l1_ratios[i])

ax.set_title(("Elastic net regularization path (L1 / alpha vs rmse)\n"

             "best params: %s, %s" % (cv_opt_en.l1_ratio_, cv_opt_en.alpha_)))

plt.legend()



fig = plt.figure(figsize=(8, 50))

ax = fig.add_subplot(111)

ax.barh(np.arange(len(cv_opt_en.coef_), 0, -1), cv_opt_en.coef_,

       tick_label=feature_df_dummies.columns,)

ax.set_title("Elastic network coefs")

plt.show()

# L1 penalized linear regression



alphas = [1] + [10 ** -x for x in range(1, 8)] + [5 * 10 ** -x for x in range(1, 8)]

if not os.path.isfile("cv_opt_ll.pkl") or overwrite_models:

    ll_cv = LassoCV(alphas=alphas,

                    max_iter=10000, normalize=False, 

                    cv=RepeatedKFold(10, 3, random_state=RNG_SEED),

                    random_state=RNG_SEED, selection="random")

    cv_opt_ll = ll_cv.fit(feature_df_dummies, outcome_df)

    joblib.dump(cv_opt_ll, "cv_opt_ll.pkl")

else: 

    cv_opt_ll = joblib.load("cv_opt_ll.pkl")

    

# cross-validated rmse for the best parameters

ll_alpha_index = np.where(cv_opt_ll.alphas_ == cv_opt_ll.alpha_)[0][0]

ll_rmse = np.sqrt(np.mean(cv_opt_ll.mse_path_, axis=1)[ll_alpha_index])

print(ll_rmse)

print(cv_opt_ll)



# model using the best parameters so that the cross-validation 

# does not run for every fold when we cross_val_predict() 

# (to reduce the computation time)

cv_opt_ll_model = Lasso(alpha=cv_opt_ll.alpha_, fit_intercept=True, normalize=False, 

                        precompute=False, copy_X=True, max_iter=10000, 

                        tol=0.0001, warm_start=False, positive=False, 

                        random_state=RNG_SEED, selection="random")

cv_opt_ll_model = cv_opt_ll_model.fit(feature_df_dummies, outcome_df)

"""

# From a previous run of what is above, for the submission

cv_opt_ll_model = Lasso(alpha=0.0001, fit_intercept=True, verbose=False,

                  precompute="auto", max_iter=10000, 

                  eps=2.2204460492503131e-16, copy_X=True, 

                  fit_path=True, positive=False)

cv_opt_ll_model = cv_opt_ll_model.fit(feature_df_dummies, outcome_df)

ll_rmse = 0.14276387188

"""



ll_preds = cv_opt_ll_model.predict(test_df_dummies)

ll_cv_preds = cross_val_predict(cv_opt_ll_model, feature_df_dummies, outcome_df, 

                                cv=stack_folds)



fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(np.log10(cv_opt_ll.alphas_), np.mean(cv_opt_ll.mse_path_, axis=1))

ax.set_title(("lars lasso regularization path (alpha vs rmse)\n"

             "best alpha = %s" % cv_opt_ll.alpha_))

plt.legend()



fig = plt.figure(figsize=(8, 50))

ax = fig.add_subplot(111)

ax.barh(np.arange(len(cv_opt_ll.coef_), 0, -1), cv_opt_ll.coef_,

       tick_label=feature_df_dummies.columns,)

ax.set_title("Lars lasso coefs")

plt.show()
# k nearest neighbors regression

# l_1, l_2, l_p, l_inf

metrics = ["euclidean", "manhattan", "minkowski", "chebyshev"]

n_neighbors_list = np.arange(4, 11, 1)



if not os.path.isfile("cv_opt_kn.pkl") or overwrite_models:

    kn = KNeighborsRegressor(n_jobs=4, p=3)

    kn_param_grid = {"n_neighbors": n_neighbors_list,

                    "weights": ["uniform", "distance"],

                    "metric": metrics}

    kn_gs = GridSearchCV(estimator=kn, param_grid=kn_param_grid, scoring="neg_mean_squared_error", 

                         fit_params=None, cv=rkf_cv)

    cv_opt_kn = kn_gs.fit(feature_df, outcome_df)

    joblib.dump(cv_opt_kn, "cv_opt_kn.pkl")

else:

    cv_opt_kn = joblib.load("cv_opt_kn.pkl")



# NB: loss is negative mean squared error

kn_rmse = np.sqrt(-cv_opt_kn.best_score_)

print(cv_opt_kn.best_score_, kn_rmse)

print(cv_opt_kn.best_estimator_)

    

cv_opt_kn_model = cv_opt_kn.best_estimator_

"""

# From a previous run of what is above, for the submission

cv_opt_kn_model = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',

            metric_params=None, n_jobs=1, n_neighbors=9, p=2,

            weights='distance')

cv_opt_kn_model = cv_opt_kn_model.fit(feature_df, outcome_df) 

kn_rmse = 0.198152549818

"""



kn_preds = cv_opt_kn_model.predict(test_df)

kn_cv_preds = cross_val_predict(cv_opt_kn_model, feature_df, outcome_df, cv=stack_folds)



uniform_run = cv_opt_kn.cv_results_["param_weights"] == "uniform"

distance_run = cv_opt_kn.cv_results_["param_weights"] == "distance"

best_metric = cv_opt_kn.best_params_["metric"]

has_best_metric = cv_opt_kn.cv_results_["param_metric"] == best_metric

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(n_neighbors_list, 

        np.sqrt(-cv_opt_kn.cv_results_["mean_test_score"][uniform_run & has_best_metric]),

       label="uniform")

ax.plot(n_neighbors_list, 

        np.sqrt(-cv_opt_kn.cv_results_["mean_test_score"][distance_run & has_best_metric]),

       label="distance")

ax.set_title("Knn CV (%s) (#nn / weights vs rmse)\nBest params: %s, %s" % \

             tuple(list(cv_opt_kn.best_params_.values())))

plt.legend()

plt.show()
# Gradient boosted trees regression



if not os.path.isfile("cv_opt_xgb.pkl") or overwrite_models:

    xgb = XGBRegressor(random_state=RNG_SEED, n_estimators=500, n_jobs=4)

    reg_ratios = [0.1, 0.5, 0.9]

    xgb_param_grid = {"max_depth": [1, 2, 3, 5],

                      "learning_rate": [0.05, 0.1, 0.2],

                      "reg_lambda": reg_ratios,

                      "reg_alpha": reg_ratios}

    xgb_gs = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, 

                          scoring="neg_mean_squared_error", 

                          fit_params=None, cv=rkf_cv)

    cv_opt_xgb = xgb_gs.fit(feature_df, outcome_df)

    joblib.dump(cv_opt_xgb, "cv_opt_xgb.pkl")

else:

    cv_opt_xgb = joblib.load("cv_opt_xgb.pkl")

    

# NB: loss is negative mean squared error

xgb_rmse = np.sqrt(-cv_opt_xgb.best_score_)

print(cv_opt_xgb.best_score_, xgb_rmse)

print(cv_opt_xgb.best_estimator_)



cv_opt_xgb_model = cv_opt_xgb.best_estimator_

"""

# From a previous run of what is above, for the submission

cv_opt_xgb_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

       max_depth=2, min_child_weight=1, missing=None, n_estimators=500,

       n_jobs=1, nthread=None, objective='reg:linear',

       random_state=RNG_SEED, reg_alpha=0.1, reg_lambda=0.5,

       scale_pos_weight=1, seed=None, silent=True, subsample=1)

cv_opt_xgb_model =  cv_opt_xgb_model.fit(feature_df, outcome_df)

xgb_rmse = 0.123629669672

"""



xgb_preds = cv_opt_xgb_model.predict(test_df)

xgb_cv_preds = cross_val_predict(cv_opt_xgb_model, feature_df, outcome_df, cv=stack_folds)



# feature importances

fig = plt.figure(figsize=(8, 30))

ax = fig.add_subplot(111)

ax.barh(np.arange(len(cv_opt_xgb_model.feature_importances_), 0, -1), 

        cv_opt_xgb_model.feature_importances_,

        tick_label=feature_df.columns)

ax.set_title(cv_opt_xgb.best_score_)

plt.show()
# Gradient boosted trees regression (again)



if not os.path.isfile("cv_opt_lgb.pkl") or overwrite_models:

    lgb = LGBMRegressor(random_state=RNG_SEED, n_estimators=500, n_jobs=4)

    reg_ratios = [0.1, 0.5, 0.9]

    lgb_param_grid = {"max_depth": [1, 3, 5, -1],

                      "learning_rate": [0.05, 0.1, 0.2],

                      "reg_lambda": reg_ratios,

                      "reg_alpha": reg_ratios}

    lgb_gs = GridSearchCV(estimator=lgb, param_grid=lgb_param_grid, 

                          scoring="neg_mean_squared_error", 

                          fit_params=None, cv=rkf_cv)

    cv_opt_lgb = lgb_gs.fit(feature_df, outcome_df)

    joblib.dump(cv_opt_lgb, "cv_opt_lgb.pkl")

else:

    cv_opt_lgb = joblib.load("cv_opt_lgb.pkl")

    

# NB: loss is negative mean squared error

lgb_rmse = np.sqrt(-cv_opt_lgb.best_score_)

print(cv_opt_lgb.best_score_, lgb_rmse)

print(cv_opt_lgb.best_estimator_)



cv_opt_lgb_model = cv_opt_lgb.best_estimator_

"""

# From a previous run of what is above, for the submission

cv_opt_lgb_model = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

       learning_rate=0.05, max_depth=3, min_child_samples=20,

       min_child_weight=0.001, min_split_gain=0.0, n_estimators=500,

       n_jobs=-1, num_leaves=31, objective=None, random_state=RNG_SEED,

       reg_alpha=0.5, reg_lambda=0.5, silent=True, subsample=1.0,

       subsample_for_bin=200000, subsample_freq=1)

cv_opt_lgb_model =  cv_opt_lgb_model.fit(feature_df, outcome_df)

lgb_rmse = 0.128340967645

"""

lgb_preds = cv_opt_lgb_model.predict(test_df)

lgb_cv_preds = cross_val_predict(cv_opt_lgb_model, feature_df, outcome_df, cv=stack_folds)



# feature importances

fig = plt.figure(figsize=(8, 30))

ax = fig.add_subplot(111)

ax.barh(np.arange(len(cv_opt_lgb_model.feature_importances_), 0, -1), 

        cv_opt_lgb_model.feature_importances_,

        tick_label=feature_df.columns)

ax.set_title(cv_opt_lgb.best_score_)

plt.show()
# random forest



if not os.path.isfile("cv_opt_rf.pkl") or overwrite_models:

    rf = RandomForestRegressor(n_estimators=500, random_state=RNG_SEED, n_jobs=4)

    rf_param_grid = {"min_samples_split": [1.0, 3, 5],

                      "max_features": ["sqrt", "log2"]}

    rf_gs = GridSearchCV(estimator=rf, param_grid=rf_param_grid, scoring="neg_mean_squared_error", 

                         fit_params=None, cv=rkf_cv)

    cv_opt_rf = rf_gs.fit(feature_df, outcome_df)

    joblib.dump(cv_opt_rf, "cv_opt_rf.pkl")

else:

    cv_opt_rf = joblib.load("cv_opt_rf.pkl")

    

# NB: loss is negative mean squared error

rf_rmse = np.sqrt(-cv_opt_rf.best_score_)

print(cv_opt_rf.best_score_, rf_rmse)

print(cv_opt_rf.best_estimator_)



cv_opt_rf_model = cv_opt_rf.best_estimator_

"""

# From a previous run of what is above, for the submission

cv_opt_rf_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

           max_features='sqrt', max_leaf_nodes=None,

           min_impurity_decrease=0.0, min_impurity_split=None,

           min_samples_leaf=1, min_samples_split=3,

           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,

           oob_score=False, random_state=RNG_SEED, verbose=0,

           warm_start=False)

cv_opt_rf_model = cv_opt_rf_model.fit(feature_df, outcome_df)

rf_rmse = 0.141737550372

"""



rf_preds = cv_opt_rf_model.predict(test_df)

rf_cv_preds = cross_val_predict(cv_opt_rf_model, feature_df, outcome_df, cv=stack_folds)



# feature importances

fig = plt.figure(figsize=(8, 30))

ax = fig.add_subplot(111)

ax.barh(np.arange(len(cv_opt_rf_model.feature_importances_), 0, -1), 

        cv_opt_rf_model.feature_importances_,

        tick_label=feature_df.columns)

ax.set_title(cv_opt_rf.best_score_)

plt.show()
# gaussian process



# Note: the kernel’s hyperparameters are optimized during fitting

if not os.path.isfile("cv_opt_gp.pkl") or overwrite_models:

    gp = GaussianProcessRegressor(normalize_y=True, random_state=RNG_SEED)

    gp_param_grid = {"kernel": [RBF() + WhiteKernel(), RationalQuadratic() + WhiteKernel()]}

    gp_gs = GridSearchCV(estimator=gp, param_grid=gp_param_grid, scoring="neg_mean_squared_error", 

                         fit_params=None, cv=rkf_cv)

    cv_opt_gp = gp_gs.fit(feature_df, outcome_df)

    joblib.dump(cv_opt_gp, "cv_opt_gp.pkl")  

else:

    cv_opt_gp = joblib.load("cv_opt_gp.pkl")



# NB: loss is negative mean squared error    

gp_rmse = np.sqrt(-cv_opt_gp.best_score_)

print(cv_opt_gp.best_score_, gp_rmse)

print(cv_opt_gp.best_estimator_)



cv_opt_gp_model = cv_opt_gp.best_estimator_

"""

# From a previous run of what is above, for the submission

cv_opt_gp_model = GaussianProcessRegressor(alpha=1e-10, copy_X_train=True,

             kernel=RationalQuadratic(alpha=1, length_scale=1) + WhiteKernel(noise_level=1),

             n_restarts_optimizer=0, normalize_y=True,

             optimizer='fmin_l_bfgs_b', random_state=RNG_SEED)

cv_opt_gp_model = cv_opt_gp_model.fit(feature_df, outcome_df)

gp_rmse = 0.132643420585

"""



gp_preds = cv_opt_gp_model.predict(test_df)

gp_cv_preds = cross_val_predict(cv_opt_gp_model, feature_df, outcome_df, cv=stack_folds)

# model stacking



# fold predictions of the base learners to be used for training the

# meta model

base_stack_cv_preds = np.vstack([en_cv_preds, ll_cv_preds, kn_cv_preds,

                                 xgb_cv_preds, lgb_cv_preds, rf_cv_preds, 

                                 gp_cv_preds]).T



# meta model

# meta = LinearRegression()

meta = KernelRidge()

# cross-validation (note: using the same stack folds) to assess 

# the accuracy of the meta model

stack_cv_preds = cross_val_predict(meta, base_stack_cv_preds, outcome_df, cv=stack_folds)

# rmse

print(np.sqrt(np.mean((stack_cv_preds - outcome_df) ** 2)))



# train the  meta model on the fold predictions from the base models

meta = meta.fit(base_stack_cv_preds, outcome_df)

# print(meta.coef_)

base_stack_test_preds = np.vstack([en_preds, ll_preds, kn_preds,

                                   xgb_preds, lgb_preds, rf_preds, 

                                   gp_preds]).T



# use the base models' full predictions as input to the meta model

test_preds = meta.predict(base_stack_test_preds)

final_preds = np.exp(test_preds)      



"""

# basic weighting scheme

errors = np.array([en_rmse, ll_rmse, kn_rmse, xgb_rmse, rf_rmse, gp_rmse])

print(errors)

model_weights = np.exp(1 / np.array(errors))

# model_weights = 1 / np.array(errors)

model_weights /= np.sum(model_weights)

print(model_weights)



don't forget to turn the log price into price, silly you!

final_preds = np.exp(np.sum(all_preds * model_weights, axis=1))

"""      



submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_preds})

submission.to_csv('house_prices_submission.csv', index=False)



all_preds = np.array([en_preds, ll_preds, kn_preds, xgb_preds, 

                      lgb_preds, rf_preds, gp_preds]).T

plt.figure()

sns.pairplot(pd.DataFrame(all_preds))

plt.figure()

sns.heatmap(np.corrcoef(all_preds.T))

plt.show()
