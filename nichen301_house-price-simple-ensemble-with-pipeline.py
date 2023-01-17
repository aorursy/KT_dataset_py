import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')



import warnings

warnings.filterwarnings('ignore')



from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from scipy.stats import skew

from scipy.special import boxcox1p

from sklearn.decomposition import PCA, KernelPCA

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split



import eli5

from eli5.sklearn import PermutationImportance



from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import SVR, LinearSVR

from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge

from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor
train = pd.read_csv('../input/train.csv')
train.drop(train[(train["GrLivArea"]>4000) & (train["SalePrice"]<300000)].index, inplace=True)
n_train = len(train) # need this to restore X_train from X_full after data transformations.

X_train = train.drop(['Id', 'SalePrice'], axis=1)

y_train = train.SalePrice



test = pd.read_csv('../input/test.csv')

X_test = test.drop('Id', axis=1)



X_full = pd.concat([X_train, X_test], ignore_index=True)
# #correlation matrix

# corrmat = train.corr()

# plt.subplots(figsize=(12, 9))

# sns.heatmap(corrmat, square=True);



# plt.subplots(figsize=(12,9))

# sns.boxenplot(train.OverallQual, train.SalePrice)



# sns.jointplot(x='GrLivArea', y='SalePrice', data=train, kind='hex', 

#               gridsize=20)



# sns.jointplot(x='LotFrontage', y='SalePrice', data=train, kind='hex', 

#               gridsize=20)



# sns.jointplot(x='LotArea', y='SalePrice', data=train, kind='hex', 

#               gridsize=20)



# sns.boxplot(train.MSSubClass, np.log(train.SalePrice))



# sns.boxplot(train.MSZoning, np.log(train.SalePrice))



# sns.boxplot(train.Street, np.log(train.SalePrice))



# sns.boxplot(train.Alley, np.log(train.SalePrice))



# sns.boxplot(train.LotShape, np.log(train.SalePrice))



# corrmat.LotFrontage.sort_values(ascending=False)

# train.LotArea.plot.hist(bins=100)



# df = train.copy()



# df['LotAreaCut'] = pd.qcut(df.LotArea,10)



# df.groupby('LotAreaCut')['LotFrontage'].median().plot.bar()



# df.groupby('Neighborhood')['LotFrontage'].median().plot.bar()



# g = sns.FacetGrid(train, col="Neighborhood", col_wrap=3)

# g.map(sns.violinplot, "LotFrontage")



# g = sns.FacetGrid(df, col="LotAreaCut", col_wrap=3)

# g.map(sns.violinplot, "LotFrontage")



# sns.violinplot(train.LotArea)



# sns.jointplot(train.LotFrontage, np.log(train.SalePrice), kind='hex')



# gb = df.groupby(['LotAreaCut', 'Neighborhood'])['LotFrontage']



# gb = df.groupby(['LotAreaCut', 'Neighborhood'])['LotFrontage']
na_info_train = train.isnull().sum()

na_info_test = test.isnull().sum()



na_info = pd.concat([na_info_train, na_info_test], axis=1)

na_info.columns = ['train', 'test']



na_info['total'] = na_info.train + na_info.test

na_info['total_pct'] = na_info.total / (len(train) + len(test))

na_info['dtype'] = train.dtypes[na_info.index]



na_info.sort_values(['total', 'test', 'train'], ascending=False, inplace=True)

na_info = na_info[na_info.total > 0]



na_info
class AwesomeImputer(BaseEstimator, TransformerMixin):

    def __init__(self):

        # These columns are to be filled with 0

        self.fill_with_zero = [

            "MasVnrArea",

            "BsmtUnfSF",

            "TotalBsmtSF",

            "GarageCars",

            "BsmtFinSF2",

            "BsmtFinSF1",

            "GarageArea",

        ]

        # These are to be filled with None

        self.fill_with_none = [

            "PoolQC" ,

            "MiscFeature",

            "Alley",

            "Fence",

            "FireplaceQu",

            "GarageQual",

            "GarageCond",

            "GarageFinish",

            "GarageYrBlt",

            "GarageType",

            "BsmtExposure",

            "BsmtCond",

            "BsmtQual",

            "BsmtFinType2",

            "BsmtFinType1",

            "MasVnrType",

        ]

        # These are to be filled with the mode of the column

        self.fill_with_mode =[

            "MSZoning",

            "BsmtFullBath",

            "BsmtHalfBath",

            "Utilities",

            "Functional",

            "Electrical",

            "KitchenQual",

            "SaleType",

            "Exterior1st",

            "Exterior2nd",

        ]

        # Fill these with the column mean

        self.fill_with_mean = [

            "LotFrontage"

        ]



    def fit(self, X, y=None):

        # Fit the mode imputer

        self.imp_mode = SimpleImputer(strategy='most_frequent')

        self.imp_mode.fit(X[self.fill_with_mode])

        # Fit the mean imputer

        self.imp_mean = SimpleImputer(strategy="mean")

        self.imp_mean.fit(X[self.fill_with_mean])

        return self



    def transform(self, X):

        # Transform all the columns having missing values

        X[self.fill_with_zero] = X[self.fill_with_zero].fillna(0)

        X[self.fill_with_none] = X[self.fill_with_none].fillna("None")

        X[self.fill_with_mode] = self.imp_mode.transform(X[self.fill_with_mode])

        X[self.fill_with_mean] = self.imp_mean.transform(X[self.fill_with_mean])

        assert X.isnull().sum().sum() == 0

        return X
class NumToStr(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.to_str = [

            "MSSubClass",

            "BsmtFullBath",

            "BsmtHalfBath",

            "HalfBath",

            "BedroomAbvGr",

            "KitchenAbvGr",

            "MoSold",

            "YrSold",

            "YearBuilt",

            "YearRemodAdd",

            "LowQualFinSF",

            "GarageYrBlt",

        ]

    

    def fit(self,X,y=None):

        return self

    

    def transform(self, X):

        X[self.to_str] = X[self.to_str].astype(str)

        return X
# class YearTrans(BaseEstimator, TransformerMixin):

#     def __init__(self):

#         pass

    

#     def fit(self,X,y=None):

#         return self

    

#     def transform(self,X):

#         X.YearBuilt = X.YearBuilt - X.YearBuilt.min()

#         X.YearRemodAdd = X.YearRemodAdd - X.YearRemodAdd.min()

#         X.GarageYrBlt = X.GarageYrBlt - X.GarageYrBlt.min()

# #         lab=LabelEncoder()

# #         X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])

# #         X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])

# #         X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])

#         return X
class RemoveSkew(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=5, lmbda=0):

        self.threshold = threshold

        self.lmbda = lmbda

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        X_numeric = X.select_dtypes(exclude=["object"])

        skewness = X_numeric.apply(lambda x: skew(x))

        skew_f = skewness[abs(skewness) >= self.threshold].index

        X[skew_f] = X[skew_f].apply(lambda col: boxcox1p(col, self.lmbda))

        return X

        
class Dummies(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        self._X = pd.get_dummies(X).head(1)

        return self



    def transform(self, X):

        X = pd.get_dummies(X)

        _, X = self._X.align(X, axis=1, join='left', fill_value=0)

        return X
class AddFeature(BaseEstimator, TransformerMixin):

    def __init__(self, level=1):

        self.level = level

    

    def fit(self,X, y=None):

        return self

    

    def transform(self,X):

        if self.level >= 1:

            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]

            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            

            X["Prod_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]

        if self.level >= 2:

            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]

            X["Rooms"] = X["FullBath"]+X["TotRmsAbvGrd"]

            X["PorchArea"] = X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]



        return X
class AwesomeScaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.scaler = RobustScaler()

    

    def fit(self, X, y=None):

        self.scaler.fit(X)

        return self

    

    def transform(self, X):

        # Apply scaler while preserving pd.DataFrame format. Otherwise it casts X to np.ndarray

        X[X.columns] = self.scaler.transform(X)

        return X
class AverageModel(BaseEstimator, RegressorMixin):

    def __init__(self, models, weights='average'):

        self.models = models

        self.weights = weights

        

    def fit(self, X, y):

        self.fit_models = [

            clone(model).fit(X, y)

            for model in self.models

        ]

        return self

        

    def predict(self, X):

        n_models = len(self.fit_models)

        if self.weights == 'average':

            weights = np.ones(n_models) / n_models

        else:

            assert len(self.weights) == n_models

            weights = np.array(self.weights)

            

        predicts = pd.DataFrame([model.predict(X) for model in self.fit_models])

        avr_predict = predicts.apply(lambda col: np.inner(weights, col))

        avr_predict = np.array(avr_predict)

        return avr_predict

        
def rmse_cv(model, X, y):

    scores = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))

    return (scores.mean(), scores.std())
preprocessing = [

    AwesomeImputer(), 

    NumToStr(),

    RemoveSkew(threshold=1),

    Dummies(),

    AddFeature(),

    AwesomeScaler(), 

]
y_train = np.log1p(y_train) # unskew the SalePrice target feature.



pipe_prep = make_pipeline(*preprocessing)

X_full_prep = pipe_prep.fit_transform(X_full)



X_train_prep = X_full_prep[:n_train]

X_test_prep = X_full_prep[n_train:]
# from scipy.stats import randint, uniform



# def random_search(model, param_dist, X, y, n_iter=10):

#     assert len(param_dist) <= 6

#     rand_search = RandomizedSearchCV(model, param_dist, scoring="neg_mean_squared_error", cv=5, n_iter=n_iter)

#     rand_search.fit(X, y)

#     print(np.sqrt(-rand_search.best_score_), rand_search.best_params_)

#     f, axes = plt.subplots(len(param_dist)//2 + 1, 2)

#     for idx, param in enumerate(param_dist):

#         ax = axes.flatten()[idx]

#         sns.scatterplot(rand_search.cv_results_['param_{}'.format(param)], \

#                         np.sqrt(-rand_search.cv_results_['mean_test_score']), ax=ax)

#         ax.set_xlabel(param)

#         ax.set_ylabel('rmse')
# # best: alpha=0.0005

# random_search(Lasso(max_iter=2000), {"alpha": uniform(0, 0.001)}, pre_train_X, train_y)
# # best: alpha=40

# random_search(Ridge(alpha=40), {"max_iter": randint(1000, 100000)},\

#               pre_train_X, train_y, n_iter=20)
# random_search(SVR(C=3, gamma=0.001, epsilon=0.01), {}, pre_train_X, train_y)
# # random_search(ElasticNet(alpha=0.001), {"l1_ratio": uniform(0.4, 0.4)},\

#               pre_train_X, train_y, n_iter=20)
# random_search(GradientBoostingRegressor(),

#               {"learning_rate": uniform(0, 0.2), "n_estimators": randint(100, 1000)}, pre_train_X, train_y, n_iter=40)



# random_search(GradientBoostingRegressor(),

#               {"subsample": uniform(0, 1)}, pre_train_X, train_y, n_iter=40)

# # 0.57



# rmse_cv(GradientBoostingRegressor(n_estimators=100), pre_train_X, train_y)



# rmse_cv(GradientBoostingRegressor(n_estimators=100, learning_rate=0.05), pre_train_X, train_y)



# rmse_cv(XGBRegressor(), pre_train_X, train_y)
# GradientBoostingRegressor?
models = [Ridge(alpha=40, max_iter=10000),

          Lasso(alpha=0.0005, max_iter=10000),

          ElasticNet(alpha=0.001, l1_ratio=0.6, max_iter=10000),

          SVR(C=3, gamma=0.001, epsilon=0.01),

          BayesianRidge(),

          KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),

          GradientBoostingRegressor(),

          XGBRegressor(),

         ]
# for model in models:

#     mean, std = rmse_cv(model, X_train_prep, y_train)

#     print("{} score: {} {}".format(model.__str__()[:8], mean, std))
# model = RandomForestRegressor()

# subtrain_X, val_X, subtrain_y, val_y = train_test_split(pre_train_X, train_y)

# model.fit(subtrain_X, subtrain_y)

# perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)

# eli5.show_weights(perm, feature_names = val_X.columns.tolist())
models_subm = [

          Ridge(alpha=40, max_iter=10000),

          Lasso(alpha=0.0005, max_iter=10000),

          ElasticNet(alpha=0.001, l1_ratio=0.6, max_iter=10000),

          SVR(C=3, gamma=0.001, epsilon=0.01),

          BayesianRidge(),

          KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),

]
model_subm = AverageModel(models=models_subm)

# rmse_cv(model_subm, pre_train_X, train_y)
model_subm.fit(X_train_prep, y_train)

y_pred = np.expm1(model_subm.predict(X_test_prep))

result = pd.DataFrame({'Id':test.Id, 'SalePrice':y_pred})

result.to_csv("submission.csv",index=False)