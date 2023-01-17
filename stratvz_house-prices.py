import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from scipy.stats import skew

from scipy.special import boxcox1p



%matplotlib inline

sns.set()
data_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col="Id")

data_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col="Id")
train = data_train.drop(["SalePrice"], axis=1).copy()

train_labels = data_train["SalePrice"].copy()

test = data_test.copy()
train.head()
sns.distplot(train_labels, fit=norm);

fig = plt.figure()

res = stats.probplot(train_labels, plot=plt)
train_labels = np.log1p(train_labels)
sns.distplot(train_labels, fit=norm);

fig = plt.figure()

res = stats.probplot(train_labels, plot=plt)
# Search for missing values

count_null_frequent = []

for col in train.columns:

    count_null_frequent.append([col, len(train[pd.isnull(train[col])]), 

                                len(train[pd.isnull(train[col])])/len(train[col])]) 



count_null = pd.DataFrame(count_null_frequent, 

                                   columns=["Name", "Null", "Perc"])

count_null = count_null.sort_values(by=['Perc'], ascending=False)

count_null.head(20)
# PoolQC, MiscFeature, Alley have many Nulls

train.drop(["PoolQC","MiscFeature", "Alley"], axis=1, inplace=True)

test.drop(["PoolQC","MiscFeature", "Alley"], axis=1, inplace=True)
def fillna_data(data):

    #FireplaceQu: Fireplace quality, NA -> No Fireplace

    data["FireplaceQu"].fillna("No", inplace=True)



    #Fence: Fence quality, NA -> No Fence

    data["Fence"].fillna("No", inplace=True)



    #GarageType: Garage location, NA -> No Garage

    data["GarageType"].fillna("No", inplace=True)



    #GarageYrBlt: Year garage was built

    data["GarageYrBlt"].fillna(0, inplace=True)



    #GarageQual: Garage quality, NA -> Garage

    data["GarageQual"].fillna("No", inplace=True)



    #GarageCond: Garage condition, NA -> No Garage

    data["GarageCond"].fillna("No", inplace=True)



    #GarageFinish: Interior finish of the garage, NA -> No Garage

    data["GarageFinish"].fillna("No", inplace=True)



    #BsmtFinType2: Rating of basement finished area (if multiple types), NA -> No Basement 

    data["BsmtFinType2"].fillna("No", inplace=True)



    #BsmtExposure: Refers to walkout or garden level walls, NA -> No Basement

    data["BsmtExposure"].fillna("NA", inplace=True)



    #BsmtCond: Evaluates the general condition of the basement, NA -> No Basement

    data["BsmtCond"].fillna("No", inplace=True)



    #BsmtFinType1: Rating of basement finished area, NA -> No Basement

    data["BsmtFinType1"].fillna("No", inplace=True)



    #BsmtQual: Evaluates the height of the basement, NA -> No Basement

    data["BsmtQual"].fillna("No", inplace=True)

    

    return data
train = fillna_data(train)

test = fillna_data(test)
# OverallCond: Rates the overall condition of the house -> categorical variable

train["OverallCond"] = train["OverallCond"].astype("str")

test["OverallCond"] = test["OverallCond"].astype("str")
# Combine YrSold and MoSold

train["DateSold"] = train["YrSold"]+(train["MoSold"]-1)/12

train.drop(["YrSold", "MoSold"], axis=1, inplace=True)



test["DateSold"] = test["YrSold"]+(test["MoSold"]-1)/12

test.drop(["YrSold", "MoSold"], axis=1, inplace=True)
null_and_frequent = []

for col in train.columns:

    null_and_frequent.append([col, len(train[pd.isnull(train[col])]), 

                                train[col].value_counts().max()/len(train[col])]) 



null_and_frequent = pd.DataFrame(null_and_frequent, 

                                   columns=["Name", "Null", 

                                            "Frequent"])

null_and_frequent = null_and_frequent.sort_values(by=['Frequent'], ascending=False)

null_and_frequent.head(20)
# Search for variables that consist of almost one value

col_frequent = list(null_and_frequent[null_and_frequent['Frequent'] > 0.985]['Name'])



train.drop(col_frequent, axis=1, inplace=True)

test.drop(col_frequent, axis=1, inplace=True)



print(col_frequent)
# All other missing values are replaced by most frequent

imputer = SimpleImputer(strategy="most_frequent")

train_transform = pd.DataFrame(imputer.fit_transform(train), 

                               columns=train.columns, index=train.index)

test = pd.DataFrame(imputer.transform(test), 

                    columns=test.columns, index=test.index)

print(imputer.statistics_)
# Search for numeric and categorical variables

name_num_col, name_obj_col = [], []



for col in train.columns:

    if train[col].dtype == object:

        name_obj_col.append(col)

    else:

        name_num_col.append(col)

        

print("Obj:\n", name_obj_col, "\n\nNum:\n", name_num_col)
# Correlation between variables

corrmat = pd.concat([train, train_labels], axis=1).corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
# Search for not important variables

not_important_col = []

for col in name_num_col:

    plt.plot(train_transform[col].values, 

             train_labels.values, 

             'bo', alpha=0.4)

    plt.ylabel('SalePrice', fontsize=13)

    plt.xlabel(col, fontsize=13)

    coef_corr = np.corrcoef(train_transform[col].values.astype(float), 

                            train_labels.values)[0][1]

    if abs(coef_corr) < 0.1:

        not_important_col.append(col)

    plt.suptitle(col + " " + 

                 str(coef_corr))

    plt.show()
print(not_important_col)
train_transform.drop(not_important_col, axis=1, inplace=True)

test.drop(not_important_col, axis=1, inplace=True)

for col in not_important_col:

    name_num_col.remove(col)
train_transform[name_obj_col].head()
onehot_encoder = OneHotEncoder(categories="auto", sparse=False)

onehot_encoder.fit(pd.concat([train_transform[name_obj_col], 

                              test[name_obj_col]], axis=0))



train_cat = pd.DataFrame(onehot_encoder.transform(train_transform[name_obj_col]), 

                         columns=onehot_encoder.get_feature_names(),

                         index=train_transform[name_obj_col].index)

test_cat = pd.DataFrame(onehot_encoder.transform(test[name_obj_col]), 

                        columns=onehot_encoder.get_feature_names(),

                        index=test[name_obj_col].index)
train_cat.head()
train_num = pd.DataFrame(train_transform[name_num_col].copy(), dtype=float)

test_num = pd.DataFrame(test[name_num_col].copy(), dtype=float)
skew_df = pd.DataFrame([[col, skew(train_num[col])] for col in name_num_col],

                       columns=["name", "skew"]).sort_values(by=['skew'],

                                                             ascending=False)

skew_df
skew_df = skew_df[abs(skew_df['skew']) > 0.75]

lam = 0.15

for col in skew_df["name"]:

    train_num[col] = boxcox1p(np.array(train_num[col], dtype="float"), lam)

    test_num[col] = boxcox1p(np.array(test[col], dtype="float"), lam)
train_num.head(5)
# Combine new numeric and categorical variables

train_prepared = pd.concat([train_num, train_cat], axis=1)

test = pd.concat([test_num, test_cat], axis=1)

train_prepared.head()
import warnings

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor



warnings.filterwarnings('ignore')
train_labels_cat = pd.cut(train_labels,

                          bins=[10., 11., 12., 13., np.inf],

                          labels=[1, 2, 3, 4])



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for train_index, test_index in split.split(train_prepared, train_labels_cat):

    X_train, X_test = train_prepared.iloc[train_index], train_prepared.iloc[test_index]

    y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
models = {"Ridge1": make_pipeline(RobustScaler(), Ridge(alpha=1, 

                                                        random_state=42)),

          "Ridge2": make_pipeline(RobustScaler(), Ridge(alpha=0.1, 

                                                        random_state=42)),

          "Ridge3": make_pipeline(RobustScaler(), Ridge(alpha=0.001, 

                                                        random_state=42)),

          "Lasso1": make_pipeline(RobustScaler(), Lasso(alpha=1, 

                                                        random_state=42)),

          "Lasso2": make_pipeline(RobustScaler(), Lasso(alpha=0.1, 

                                                        random_state=42)),

          "Lasso3": make_pipeline(RobustScaler(), Lasso(alpha=0.001, 

                                                        random_state=42)),

          "RandForest1": RandomForestRegressor(n_estimators=10, 

                                                random_state=42),

          "RandForest2": RandomForestRegressor(n_estimators=100, 

                                               random_state=42),

          "DecisTree1": DecisionTreeRegressor(random_state=42), 

          "DecisTree2": DecisionTreeRegressor(max_depth=10, 

                                              random_state=42), 

          "SVR_poly1": make_pipeline(RobustScaler(), 

                                     SVR(kernel='poly', C=1000, gamma="auto")), 

          "SVR_poly2": make_pipeline(RobustScaler(), 

                                     SVR(kernel='poly', C=1, gamma="auto")), 

          "SVR_rbf1": make_pipeline(RobustScaler(), 

                                    SVR(kernel='rbf', C=1000, gamma="auto")), 

          "SVR_rbf2": make_pipeline(RobustScaler(), 

                                    SVR(kernel='rbf', C=1, gamma="auto")), 

          "SVR_line1": make_pipeline(RobustScaler(), 

                                     SVR(kernel='linear', C=1000, gamma="auto", max_iter=10000)),

          "SVR_line2": make_pipeline(RobustScaler(), 

                                     SVR(kernel='linear', C=1, gamma="auto", max_iter=10000)),

          "XGB1": XGBRegressor(n_estimators=100, 

                               objective='reg:squarederror'), 

          "XGB2": XGBRegressor(n_estimators=1000, 

                               objective='reg:squarederror')}
models_scores = []

for model in models:

    scores = cross_val_score(models[model], X_train, y_train,

                             scoring="neg_mean_squared_error", cv=5)

    scores = np.sqrt(-scores)

    models_scores.append([model, scores.mean(), scores.std()])

    

models_scores = pd.DataFrame(models_scores, columns =["Name", "Score", "Std"]).sort_values(by=['Score'])

models_scores.head(10)
lasso_model = make_pipeline(RobustScaler(), Lasso(alpha=0.00045, random_state=42))

gs_svr = make_pipeline(RobustScaler(), SVR(C=4, kernel='rbf', gamma=0.002))

xboost = make_pipeline(RobustScaler(), XGBRegressor(n_estimators=2100, random_state=42,  max_depth=3, 

                       subsample=0.5, colsample_bytree=0.5, reg_alpha=0.3, reg_lambda=0.7,

                       gamma=0.01, learning_rate=0.04,

                       objective='reg:squarederror', min_child_weight=2))
lasso_model.fit(X_train, y_train)

gs_svr.fit(X_train, y_train)

xboost.fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, xboost.predict(X_test)))
test_predict = 0.65*lasso_model.predict(X_test) + 0.3*xboost.predict(X_test) + 0.05*gs_svr.predict(X_test) 

np.sqrt((mean_squared_error(y_test, test_predict)))
predict = 0.65*lasso_model.predict(test) + 0.3*xboost.predict(test) + 0.05*gs_svr.predict(test)

predict = pd.DataFrame({'Id': test.index, 'SalePrice': np.expm1(predict)})
predict.head(5)
predict.to_csv('predict.csv',index=False, header=True)
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")