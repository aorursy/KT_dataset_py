import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from scipy import stats

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import Ridge, Lasso, LassoCV, ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.cross_decomposition import PLSRegression

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error

from datetime import datetime



SEED = 42



# Prevent Pandas from truncating displayed dataframes

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



sns.set(style="white", font_scale=1.2)

plt.rcParams["figure.figsize"] = [10,8]
train_ = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_ = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")



# Prepare our base training/test data

train = train_.copy().drop(columns="Id")

test = test_.copy().drop(columns="Id")
train.head()
def rmse_cv(model, X, y):

    """Return the cross validated RMSE scores for a given model, training data and target data."""

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))

    return rmse
def generate_output(preds, save=False):

    """Collate predicted values into an output file and save as .csv."""

    output = submission.copy()

    output["SalePrice"] = preds

    

    if save:

        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        output.to_csv("submissions/submission-"+date_time+".csv", index=False)

    return output
def plot_coefs(model, X):

    """Display top 10 positive and top 10 negative model coefficients in a bar chart."""

    fig, ax = plt.subplots()

    coefs = pd.Series(model.coef_, index=X.columns)

    important_coefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])

    sns.barplot(x=important_coefs, y=important_coefs.index, orient='h', ax=ax)

    ax.grid()

    plt.title("Top 20 Coefficients - Regularized Regression Model")

    plt.tight_layout()

    plt.show()
def save_params_score(model_name, best_params, best_score):

    """Save optimized parameters and scores."""

    hyperparam_df.loc[model_name, "Best Params"] = [best_params]

    hyperparam_df.loc[model_name, "Best Score"] = best_score
def tune_hyperparameters(model_name, model, params, scaler):

    """Perform grid search to determine optimal hyperparameters and best score."""

    grid = GridSearchCV(model, params, n_jobs=4, cv=5, scoring="neg_mean_squared_error", refit=True, verbose=3)

    pipeline = make_pipeline(scaler, grid)

    pipeline.fit(train, y)

    best_params = grid.best_params_

    best_score = np.round(np.sqrt(-grid.best_score_), 5)

    

    # Save params and score to hyperparameter df

    save_params_score(model_name, best_params, best_score)

    

    return best_params, best_score
# https://www.kaggle.com/fiorenza2/journey-to-the-top-10

class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):

    """Create an ensemble model."""

    def __init__(self, regressors=None):

        self.regressors = regressors



    def fit(self, X, y):

        for regressor in self.regressors:

            regressor.fit(X, y)



    def predict(self, X):

        self.predictions_ = list()

        for regressor in self.regressors:

            self.predictions_.append((regressor.predict(X).ravel()))

        return (np.mean(self.predictions_, axis=0))
def compare_predictions(file1, file2):

    """Concat two output files and find the difference between their predicted SalePrices."""

    directory = "submissions/"

    a = pd.read_csv(directory + file1)

    b = pd.read_csv(directory + file2)

    

    c = pd.concat([a["SalePrice"], b["SalePrice"]], axis=1)

    c.columns = ["a", "b"]

    c["Diff"] = c["b"] - c["a"]

    return c
temp = pd.concat([train.drop(columns="SalePrice"), test])

nulls = temp.isnull().sum()[temp.isnull().sum() > 0].sort_values(ascending=False).to_frame().rename(columns={0: "MissingVals"})

nulls["MissingValsPct"] = nulls["MissingVals"] / len(temp)

nulls
# Very helpful: https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset



for df in [train, test]:

    ## EASIER TO HANDLE:

    

    # PoolQC -> data description says NA = No Pool

    df["PoolQC"].fillna(value="None", inplace=True)

    # MiscFeature -> data description says NA = None

    df["MiscFeature"].fillna(value="None", inplace=True)

    # Alley -> data description says NA = No alley access

    df["Alley"].fillna(value="None", inplace=True)

    # Fence -> data description says NA = No fence

    df["Fence"].fillna(value="None", inplace=True)

    # FireplaceQu -> data description says NA = No fireplace

    df["FireplaceQu"].fillna(value="None", inplace=True)

    # Garage features -> data description says NA = No garage

    df["GarageType"].fillna(value="None", inplace=True)

    df["GarageFinish"].fillna(value="None", inplace=True)

    df["GarageQual"].fillna(value="None", inplace=True)

    df["GarageCond"].fillna(value="None", inplace=True)

    df["GarageArea"].fillna(value=0, inplace=True)

    df["GarageCars"].fillna(value=0, inplace=True)

    # Basement features -> data description says NA = No garage

    df["BsmtCond"].fillna(value="None", inplace=True)

    df["BsmtExposure"].fillna(value="None", inplace=True)

    df["BsmtQual"].fillna(value="None", inplace=True)

    df["BsmtFinType1"].fillna(value="None", inplace=True)

    df["BsmtFinSF1"].fillna(value=0, inplace=True)

    df["BsmtFinType2"].fillna(value="None", inplace=True)

    df["BsmtFinSF2"].fillna(value=0, inplace=True)

    df["TotalBsmtSF"].fillna(value=0, inplace=True)

    df["BsmtUnfSF"].fillna(value=0, inplace=True)

    df["BsmtFullBath"].fillna(value=0, inplace=True)

    df["BsmtHalfBath"].fillna(value=0, inplace=True)

    # Functional -> data description says assume typical

    df["Functional"].fillna(value="Typ", inplace=True)

    

    ## LESS CLEAR:

    

    # LotFrontage -> assume median

    df["LotFrontage"].fillna(value=df["LotFrontage"].median(), inplace=True)

    # GarageYrBlt -> assume equal to YearBuilt

    df["GarageYrBlt"].fillna(value=df["YearBuilt"], inplace=True)

    # MasVnrType -> NA most likely means no masonry veneer

    df["MasVnrType"].fillna(value="None", inplace=True)

    df["MasVnrArea"].fillna(value=0, inplace=True)

    # Utilities -> assume the mode

    df["Utilities"].fillna(value=df["Utilities"].mode()[0], inplace=True)

    # SaleType -> assume the mode

    df["SaleType"].fillna(value=df["SaleType"].mode()[0], inplace=True)

    # KitchenQual -> assume the mode

    df["KitchenQual"].fillna(value=df["KitchenQual"].mode()[0], inplace=True)

    # Electrical -> assume the mode

    df["Electrical"].fillna(value=df["Electrical"].mode()[0], inplace=True)    

    # MSZoning -> assume the mode

    df["MSZoning"].fillna(value=df["MSZoning"].mode()[0], inplace=True)  

    # Exterior1st -> assume the mode

    df["Exterior1st"].fillna(value=df["Exterior1st"].mode()[0], inplace=True)  

    # Exterior2nd -> assume the mode

    df["Exterior2nd"].fillna(value=df["Exterior2nd"].mode()[0], inplace=True)
fig, ax = plt.subplots(1, 2, figsize=(10,5))



sns.distplot(train["LotFrontage"].dropna(), ax=ax[0])

ax[0].axvline(x=train["LotFrontage"].mean(), ymin=0, ymax=1, color='r', label="Mean")

ax[0].axvline(x=train["LotFrontage"].median(), ymin=0, ymax=1, color='g', label="Median")

ax[0].legend()

ax[0].set_title("LotFrontage Distribution")





sns.distplot(train["GarageYrBlt"].dropna(), ax=ax[1])

ax[1].axvline(x=train["GarageYrBlt"].mean(), ymin=0, ymax=1, color='r', label="Mean")

ax[1].axvline(x=train["GarageYrBlt"].median(), ymin=0, ymax=1, color='g', label="Median")

ax[1].legend()

ax[1].set_title("GarageYrBlt Distribution")



plt.show()
mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300_000)

mask = mask.replace(to_replace=[1, 0], value=["Outlier", "Data"])

sns.scatterplot(x=train["GrLivArea"], y=train["SalePrice"], hue=mask)

plt.title("Spotting Outliers - GrLivArea > 4000 sq ft & SalePrice < $300,000")

plt.show()



train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300_000)]
# https://www.kaggle.com/fiorenza2/journey-to-the-top-10

outliers = [88, 462, 523, 588, 632, 968, 1298, 1324]

train = train.drop(outliers)
mask = test["GarageYrBlt"] == 2207

sns.scatterplot(x=test["YearBuilt"], y=test["GarageYrBlt"], hue=mask, legend=False)

plt.title("Spotting Outliers - Incorrect GarageYrBlt Value")

plt.show()
# Assume that 2207 was meant to be 2007

test.loc[test["GarageYrBlt"] == 2207, "GarageYrBlt"] = 2007
# By making these substitutions, the columns are automatically cast to datatype object   

for df in [train, test]:

    df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                               50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                               80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                               150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"}}, inplace=True)
# https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition

for df in [train, test]:

    df = df.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

    

    df['BsmtFinType1_Unf'] = 1*(df['BsmtFinType1'] == 'Unf')

    df['HasWoodDeck'] = (df['WoodDeckSF'] == 0) * 1

    df['HasOpenPorch'] = (df['OpenPorchSF'] == 0) * 1

    df['HasEnclosedPorch'] = (df['EnclosedPorch'] == 0) * 1

    df['Has3SsnPorch'] = (df['3SsnPorch'] == 0) * 1

    df['HasScreenPorch'] = (df['ScreenPorch'] == 0) * 1

    df['YearsSinceRemodel'] = df['YrSold'].astype(int) - df['YearRemodAdd'].astype(int)

    df['Total_Home_Quality'] = df['OverallQual'] + df['OverallCond']

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']

    df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])

    df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

    df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

    df['TotalBsmtSF'] = df['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

    df['2ndFlrSF'] = df['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

    df['GarageArea'] = df['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

    df['GarageCars'] = df['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)

    df['LotFrontage'] = df['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)

    df['MasVnrArea'] = df['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)

    df['BsmtFinSF1'] = df['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

    df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
numeric_features = np.array([c for c in train.select_dtypes(include=[np.number]).columns if c != "SalePrice"])

numeric_features
categorical_features = np.array(train.select_dtypes(include=[np.object]).columns)

categorical_features
ordinal_features = ["ExterQual", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", 

                "LandSlope", "ExterCond", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", 

                "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "LotShape", "Utilities"]

nominal_features = list(set(categorical_features) - set(ordinal_features))
for df in [train, test]:

    df.replace({"BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

               "BsmtExposure" : {"None" : 0, "No": 0, "Mn" : 1, "Av": 2, "Gd" : 3},

               "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

               "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

               "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

               "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

               "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

               "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

               "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},

               "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

               "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

               "GarageFinish" : {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},

               "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

               "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

               "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

               "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

               "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

               "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                , inplace=True)

    

    df[ordinal_features] = df[ordinal_features].astype('int64')
nominal_data = pd.concat([train[nominal_features], test[nominal_features]])

nominal_data.head()
train.shape, test.shape, nominal_data.shape
nominal_data = pd.get_dummies(nominal_data)

nominal_data.head()
train_dummies = nominal_data[:train.shape[0]]

test_dummies = nominal_data[train.shape[0]:]

train_dummies.shape, test_dummies.shape
y = train["SalePrice"]

train = pd.concat([train[numeric_features], train[ordinal_features], train_dummies], axis=1)

test = pd.concat([test[numeric_features], test[ordinal_features], test_dummies], axis=1)
train.shape, test.shape, set(train.columns) - set(test.columns)
y = np.log1p(y) # SalePrice

skew_threshold = 0.5 # As a general rule of thumb, |skew| > 0.5 is considered moderately skewed



for df in [train[numeric_features], test[numeric_features]]:

    skewness = df.apply(lambda x: stats.skew(x))

    skewness = skewness[np.abs(skewness) > skew_threshold]

    skewed_cols = skewness.index

    log_transformed_cols = np.log1p(df[skewed_cols])

    df[skewed_cols] = log_transformed_cols

    print("Transforming {} features...".format(len(skewed_cols)))



train[numeric_features] = train[numeric_features].astype('int64')

test[numeric_features] = test[numeric_features].astype('int64')
def compare_ridge_lasso(X, y):

    ridge_alphas = np.arange(2, 50, 1)

    cv_ridge = [rmse_cv(Ridge(alpha=alpha), X, y).mean() for alpha in ridge_alphas]

    cv_ridge = pd.Series(cv_ridge, index=ridge_alphas)

    best_alpha_ridge = ridge_alphas[np.argmin(cv_ridge.values)]



    lasso_alphas = np.arange(0.0001, 0.005, 0.0002)

    cv_lasso = [rmse_cv(Lasso(alpha=alpha, tol=0.1), X, y).mean() for alpha in lasso_alphas]

    cv_lasso = pd.Series(cv_lasso, index=lasso_alphas)

    best_alpha_lasso = lasso_alphas[np.argmin(cv_lasso.values)]

    

    print("Best RMSE for Ridge Regression = {:.4f} for alpha = {}".format(cv_ridge.min(), best_alpha_ridge))

    print("Best RMSE for Lasso Regression = {:.4f} for alpha = {}".format(cv_lasso.min(), best_alpha_lasso))

    

    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    

    sns.lineplot(x=ridge_alphas, y=cv_ridge, ax=ax[0])

    ax[0].set_title("Ridge Regression - Alpha Optimization")

    ax[0].set_xlabel("Alpha")

    ax[0].set_ylabel("RMSE")

    

    sns.lineplot(x=lasso_alphas, y=cv_lasso, ax=ax[1])

    ax[1].set_title("Lasso Regression - Alpha Optimization")

    ax[1].set_xlabel("Alpha")

    ax[1].set_ylabel("RMSE")

    

    plt.tight_layout()

    plt.show()

    

    return Ridge(alpha=best_alpha_ridge) if cv_ridge.min() < cv_lasso.min() else Lasso(alpha=best_alpha_lasso)
numeric_model = compare_ridge_lasso(train[numeric_features], y)
numeric_model.fit(train[numeric_features], y)

plot_coefs(numeric_model, train[numeric_features])
numeric_ordinal_features = np.append(numeric_features, ordinal_features)

numeric_ordinal_model = compare_ridge_lasso(train[numeric_ordinal_features], y)

numeric_ordinal_model.fit(train[numeric_ordinal_features], y)
plot_coefs(numeric_ordinal_model, train[numeric_ordinal_features])
full_model = compare_ridge_lasso(train, y)

full_model.fit(train, y)
# Baseline model predictions:

preds = full_model.predict(test)

output = generate_output(preds, False)

output.sort_values(by="SalePrice", ascending=False).head()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))



sns.distplot(train_["SalePrice"], bins=25, ax=ax[0])

ax[0].set_title("Distribution of Training House Prices")

ax[0].set_xlim([0,700000])



sns.distplot(output["SalePrice"], bins=60, ax=ax[1])

ax[1].set_title("Distribution of Predicted House Prices")

ax[1].set_xlim([0,700000])



plt.tight_layout()

plt.show()
plot_coefs(full_model, train)
# https://www.kaggle.com/humananalog/xgboost-lasso

cols_to_drop = ["MSZoning_C (all)", "MSSubClass_SC160", "Condition2_PosN", "Exterior1st_ImStucc",

                "RoofMatl_Membran", "RoofMatl_Metal", "RoofMatl_Roll", "Condition2_RRAe", 

                "Condition2_RRAn", "Condition2_RRNn", "Heating_Floor", "Heating_OthW", 

                "Electrical_Mix", "MiscFeature_TenC", "MSSubClass_SC150", "Exterior1st_Stone",

                "Exterior2nd_Other", "HouseStyle_2.5Fin"]

train = train.drop(columns=cols_to_drop)

test = test.drop(columns=cols_to_drop)
ridge = Ridge(random_state=SEED)

lasso = Lasso(random_state=SEED)

elnt = ElasticNet(random_state=SEED)

kr = KernelRidge()

svr = SVR()

knn = KNeighborsRegressor(n_jobs=-1)

pls = PLSRegression()

dt = DecisionTreeRegressor(random_state=SEED)

rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)

gb = GradientBoostingRegressor(random_state=SEED)

xgb = XGBRegressor(random_state=SEED, n_jobs=-1, objective="reg:squarederror")



models = [("Ridge", ridge), ("Lasso", lasso), ("ElasticNet", elnt), ("KernelRidge", kr), \

          ("SVR", svr), ("KNNeighbors", knn), ("PLS", pls), ("DecisionTree", dt), ("RandomForest", rf), \

          ("GradientBoosting", gb), ("XGBoost", xgb)]
hyperparam_df = pd.DataFrame(index=[m[0] for m in models],

                             columns=["Baseline Score - StandardScaler", "Baseline Score - RobustScaler", 

                                      "Best Score", "Best Params"])
rmses = []

for model in models:

    pipeline = make_pipeline(StandardScaler(), model[1])

    print("Scoring {}...".format(model[0]))

    rmse = np.round(np.sqrt(-cross_val_score(pipeline, train, y, scoring="neg_mean_squared_error", cv=5)).mean(), 3)

    rmses.append(rmse)

    hyperparam_df.loc[model[0], "Baseline Score - StandardScaler"] = rmse
rmses_robust = []

for model in models:

    pipeline = make_pipeline(RobustScaler(), model[1])

    print("Scoring {}...".format(model[0]))

    rmse = np.round(np.sqrt(-cross_val_score(pipeline, train, y, scoring="neg_mean_squared_error", cv=5)).mean(), 3)

    rmses_robust.append(rmse)

    hyperparam_df.loc[model[0], "Baseline Score - RobustScaler"] = rmse
hyperparam_df
hyperparam_df[["Baseline Score - StandardScaler", "Baseline Score - RobustScaler"]] = hyperparam_df[["Baseline Score - StandardScaler", "Baseline Score - RobustScaler"]].astype('float64')
fig, ax = plt.subplots()

sns.scatterplot(x=hyperparam_df.index, y=np.log1p(hyperparam_df["Baseline Score - StandardScaler"]), s=200, label="StandardScaler", ax=ax)

sns.scatterplot(x=hyperparam_df.index, y=np.log1p(hyperparam_df["Baseline Score - RobustScaler"]), s=200, label="RobustScaler", ax=ax)

plt.title("Impact of StandardScaler vs RobustScaler on RMSE Cross Val Scores")

plt.ylabel("log(RMSE)")

plt.xticks(rotation=90)

ax.grid()

plt.show()
fig, ax = plt.subplots()

sns.scatterplot(x=hyperparam_df.index, y=np.log1p(hyperparam_df["Baseline Score - RobustScaler"]), s=200, hue=hyperparam_df["Baseline Score - RobustScaler"], palette='coolwarm_r', legend=False, ax=ax)

plt.title("RMSE Cross Val Scores by Model (RobustScaler)")

plt.xticks(rotation=90)

ax.grid()

plt.show()
ridge_param_grid = {"alpha": np.arange(5, 30, 1), 

                    "random_state": [SEED]}



# Optimized Params:

ridge_param_grid = {'alpha': [14], 'random_state': [42]}



ridge = Ridge()

ridge_best_params, ridge_best_score = tune_hyperparameters("Ridge", ridge, ridge_param_grid, RobustScaler())

ridge_best_params, ridge_best_score
lasso_param_grid = {"alpha": np.arange(0.0001, 0.001, .00002), 

                    "random_state": [SEED]}



# Optimized Params:

lasso_param_grid = {'alpha': [0.00092], 'random_state': [42]}



lasso = Lasso()

lasso_best_params, lasso_best_score = tune_hyperparameters("Lasso", lasso, lasso_param_grid, StandardScaler())

lasso_best_params, lasso_best_score
elnt_param_grid = {"alpha": [0.0001, 0.0002, 0.0003, 0.01, 0.1, 2], 

                      "l1_ratio": [0.2, 0.85, 0.95, 0.98, 1], 

                      "random_state": [SEED]}



# Optimized Params:

elastic_param_grid = {'alpha': [0.0002], 'l1_ratio': [1], 'random_state': [42]}



elnt = ElasticNet()

elnt_best_params, elnt_best_score = tune_hyperparameters("ElasticNet", elnt, elnt_param_grid, RobustScaler())

elnt_best_params, elnt_best_score
kr_param_grid = {"alpha": [0.25, 0.4, 0.5, 0.6, 0.75, 1], 

                     "kernel": ["linear", "polynomial"], "degree": [2, 3], "coef0": [1.5, 2, 3]}



# Optimized Params:

kernel_param_grid = {'alpha': [0.25], 'coef0': [1.5], 'degree': [2], 'kernel': ['linear']}



kr = KernelRidge()

kr_best_params, kr_best_score = tune_hyperparameters("KernelRidge", kr, kr_param_grid, RobustScaler())

kr_best_params, kr_best_score
dt_param_grid = {"max_depth": [3, 4, 5],

                "min_samples_leaf": [2, 3, 4],

                "min_samples_split": [3, 4, 5],

                "random_state": [SEED]}



# Optimized Params:

dt_param_grid = {'max_depth': [5],

                 'min_samples_leaf': [4],

                 'min_samples_split': [3],

                 'random_state': [42]}



dt = RandomForestRegressor()

dt_best_params, dt_best_score = tune_hyperparameters("DecisionTree", dt, dt_param_grid, StandardScaler())

dt_best_params, dt_best_score
rf_param_grid = {"n_estimators": [50, 100, 500, 1000], 

                "max_depth": [1, 2, 3, 4, 5],

                "min_samples_leaf": [2, 3, 4],

                "min_samples_split": [3, 4, 5],

                "random_state": [SEED]}



# Optimized Params:

rf_param_grid = {}



rf = RandomForestRegressor()

rf_best_params, rf_best_score = tune_hyperparameters("RandomForest", rf, rf_param_grid, StandardScaler())

rf_best_params, rf_best_score
knn_param_grid = {"n_neighbors": [3, 4, 5, 6], 

                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],

                "leaf_size": [10, 20, 30, 40, 50],

                "p": [1, 2]}



# Optimized Params:

knn_param_grid = {'algorithm': ['auto'], 'leaf_size': [10], 'n_neighbors': [4], 'p': [1]}



knn = KNeighborsRegressor()

knn_best_params, knn_best_score = tune_hyperparameters("KNNeighbors", knn, knn_param_grid, StandardScaler())

knn_best_params, knn_best_score
xgb_param_grid = {"colsample_bytree": [0.1, 0.2],

                 "learning_rate": [0.01, 0.05],

                 "max_depth": [3, 4],

                 "n_estimators": [2000],

                 "random_state": [42]}



# Optimized Params:

xgb_param_grid = {'colsample_bytree': [0.2], 'learning_rate': [0.01], 'max_depth': [4], 'n_estimators': [2000], 'random_state': [42]}



xgb = XGBRegressor(n_jobs=-1, objective="reg:squarederror")

xgb_best_params, xgb_best_score = tune_hyperparameters("XGBoost", xgb, xgb_param_grid, StandardScaler())

xgb_best_params, xgb_best_score
svr_param_grid = {"C": [20, 30, 40, 50, 60, 70],

                 "epsilon": [0.1, 0.01, 0.001],

                 "gamma": [0.0005, 0.0001, 0.00005]}



# Optimized Params:

svr_param_grid = {'C': [50], 'epsilon': [0.01], 'gamma': [0.0001]}



svr = SVR()

svr_best_params, svr_best_score = tune_hyperparameters("SVR", svr, svr_param_grid, StandardScaler())

svr_best_params, svr_best_score
pls_param_grid = {"n_components": [2, 3, 4, 5]}



# Optimized Params:

pls_param_grid = {"n_components": [5]}



pls = PLSRegression()

pls_best_params, pls_best_score = tune_hyperparameters("PLS", pls, pls_param_grid, StandardScaler())

pls_best_params, pls_best_score
gb_param_grid = {"n_estimators": [1000, 3000],

                "learning_rate": [0.01],

                "max_depth": [4],

                "loss": ["huber"],

                "max_features": ["sqrt"],

                "min_samples_leaf": [15],

                "min_samples_split": [10]}



# Optimized Params:

gb_param_grid = {'learning_rate': [0.01],

                 'loss': ['huber'],

                 'max_depth': [4],

                 'max_features': ['sqrt'],

                 'min_samples_leaf': [15],

                 'min_samples_split': [10],

                 'n_estimators': [3000]}



gb = GradientBoostingRegressor()

gb_best_params, gb_best_score = tune_hyperparameters("GradientBoosting", gb, gb_param_grid, StandardScaler())

gb_best_params, gb_best_score
hyperparam_df[["Best Score"]] = hyperparam_df[["Best Score"]].astype('float64')

best_baseline_scores = np.min(hyperparam_df[["Baseline Score - StandardScaler", "Baseline Score - RobustScaler"]], axis=1)



fig, ax = plt.subplots()

sns.scatterplot(x=hyperparam_df.index, y=np.log1p(best_baseline_scores), s=200, label="Baseline Score", ax=ax)

sns.scatterplot(x=hyperparam_df.index, y=np.log1p(hyperparam_df["Best Score"]), s=200, label="Best Score", ax=ax)

plt.title("Optimized vs Baseline RMSE Cross Val Scores")

plt.ylabel("log(RMSE)")

plt.xticks(rotation=90)

ax.grid()

plt.show()
hyperparam_df.sort_values(by="Best Score", inplace=True)

hyperparam_df
# Lasso

scaler = RobustScaler()

lasso = Lasso(**lasso_best_params)

lasso.fit(scaler.fit_transform(train), y)

predictions_lasso = np.expm1(lasso.predict(scaler.transform(test)))



# SVR

scaler = StandardScaler()

svr = SVR(**svr_best_params)

svr.fit(scaler.fit_transform(train), y)

predictions_svr = np.expm1(svr.predict(scaler.transform(test)))



# XGB

# https://www.kaggle.com/fiorenza2/journey-to-the-top-10

scaler = StandardScaler()

_ = xgb_best_params.pop("random_state") if "random_state" in xgb_best_params else ""

xgb1 = XGBRegressor(**xgb_best_params, random_state = 42)

xgb2 = XGBRegressor(**xgb_best_params, random_state = 6666)

xgb3 = XGBRegressor(**xgb_best_params, random_state = 0)

xgb_ensemble = CustomEnsembleRegressor([xgb1, xgb2, xgb3])

xgb_ensemble.fit(scaler.fit_transform(train), y)

predictions_xgb = np.expm1(xgb_ensemble.predict(scaler.transform(test)))



# Collate Predictions

predictions = pd.concat([pd.Series(predictions_lasso), 

                         pd.Series(predictions_svr), 

                         pd.Series(predictions_xgb)], axis=1).head()

predictions.columns = ["Lasso", "SVR", "XGBoost"]
predictions.head()
# Average Predictions

predictions_avg = (predictions_lasso + predictions_svr + predictions_xgb) / 3

output = generate_output(predictions_avg, save=False)

output.head()