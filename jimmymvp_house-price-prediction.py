import pandas as pd

import sklearn #rt sklearn

import torch

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import random



# take fraction of the top features regarding xgboost

TAKE_TOP_FRAC =  .85

# do SVD decomposition based on how many components, None means no SVD decomposition

SVD_COMPONENTS = 400

# standardize the data, not really needed when doing Box-Cox transform

STANDARDIZE = True

# num cross-validation folds

NUM_FOLDS = 5

# number of different weights to try for the ensemble

NUM_WEIGHT_WIGGLES = 40

# random state

RANDOM_STATE = random.randint(0,int(2**16))

# calculate cross-validation scores

CROSSVAL_SCORES = False

# calculate training set weights based on difference between training and test distributions

DENSITY_RATIO_ESTIMATION = False

# use XGB importances for polynomial and cubic features

POLY_USE_XGB = True
# read in the dataset

df_train = pd.read_csv('../input/train.csv')

df_test = df_test_orig = pd.read_csv('../input/test.csv')

# we also want to analyze the whole distribution of the data

df_all = df_all_orig = pd.concat([df_train.drop(columns=['SalePrice']), df_test])
# take a look at general columns values

df_train.describe()
#to see how the data values are distributed in the training set

df_train.hist(bins=20, figsize=(20,15))

plt.show()
# how the data is distributed in the test set

df_test.hist(bins=20, figsize=(20,15))

plt.show()
# Compute the correlation matrix

corr = df_train.corr()

# list the features with small correlation with sale price

cutoff = 0.05

print(f"Features with abs correlation lower than {cutoff}:")

print(corr.SalePrice[np.abs(corr.SalePrice) < cutoff])
from string import ascii_letters

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="white")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#skewness and kurtosis

print("Skewness: %f" % np.log(df_train['SalePrice']).skew())

print("Kurtosis: %f" % np.log(df_train['SalePrice']).kurt())



sns.distplot(np.log(df_train['SalePrice']));
#Visualize how your important variables are distributed 

from pandas.plotting import scatter_matrix

columns = ['SalePrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',

             '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']



scatter_matrix(df_train[columns], figsize=(20, 20));
rows_to_drop = []

sns.scatterplot(x='GarageArea', y='SalePrice', data=df_train)
rows_to_drop += list(df_train[df_train.SalePrice>700000 ][df_train.GarageArea > 0].index.values)
sns.scatterplot(y='SalePrice', x='TotalBsmtSF', data=df_train)
rows_to_drop+= list(df_train[df_train.TotalBsmtSF>6000 ].index.values)
sns.scatterplot(y='SalePrice', x='GrLivArea', data=df_train)
rows_to_drop+= list(df_train[df_train.GrLivArea>4000][df_train.SalePrice < 300000].index.values)
# sale price is less skewed when we take the logarithm

#histogram

sns.distplot(np.log(df_train['SalePrice']));
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

#drop outlier rows

#df_train = df_train.drop(rows_to_drop)

df_train_orig = df_train

y = np.log(df_train.SalePrice)

df_all = all_data = pd.concat([df_train.drop(columns=['SalePrice']), df_test])
# list missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")



all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")



all_data["Alley"] = all_data["Alley"].fillna("None")



all_data["Fence"] = all_data["Fence"].fillna("None")



all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")



#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

    

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

    

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)



all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



all_data["Functional"] = all_data["Functional"].fillna("Typ")



all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])



all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])



all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])



all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])



all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")



#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})



missing_data.head()
# transforming numerical that are categorical

#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



all_data = all_data.drop(['Utilities'], axis=1)



#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(all_data[c].values) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))

# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# standardize data



# if STANDARDIZE:

#     standardize_columns = [

#         "1stFlrSF",

#         "2ndFlrSF",

#         "GrLivArea",

#         "BedroomAbvGr",

#         "KitchenAbvGr",

#         "TotalSF"

#     ]

#     def standardize(df):

#         df = df.copy()

#         numerics = ['float16', 'float32', 'float64']

#         for c in standardize_columns:

#             if c == 'SalePrice' or c == 'Id' or c not in df or str(df[c].dtype) in ['category', 'object']:

#                 continue

#             # standardize

#             df[c] = (df[c]-df[c].mean())/df[c].std()

#         return df



#     all_data = standardize(all_data)

from scipy import stats

from scipy.stats import norm, skew #for some statistics





numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)

    

#all_data[skewed_features] = np.log1p(all_data[skewed_features])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
all_data
all_data = pd.get_dummies(all_data)

print(all_data.shape)
df_train = all_data.iloc[:len(df_train)]

df_test = all_data.iloc[len(df_train):]

X_train = df_train.values

X_test = df_test.values

X = all_data.values

df_all = all_data



y = y.astype(np.float64)
df_all
import xgboost

xgb = xgboost.XGBRegressor()



xgb.fit(X_train, y)

importances = xgb.feature_importances_

# Plot the feature importances of the forest

indices = np.argsort(importances)[::-1]



indices = indices[:int(len(X_train[0])*TAKE_TOP_FRAC)]



X_train = X_train[:,indices]

X_test  = X_test[:, indices]

X = X[:, indices]
df_train['SalePrice'] = y

corr = df_train.select_dtypes(np.float64).corr()

# low-correlated columns



indices = np.argsort(corr.SalePrice.values)

corr.SalePrice.iloc[indices]

# take top ten

most_correlated = corr.SalePrice.iloc[indices].index.values[-11:-1]
# most correlated columns

print("Most correlated columns:")

print(most_correlated)

print("XGB most importaint indices:")

print(indices[:20])
from sklearn import preprocessing

# create polynomial features

pfeatures = preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

cfeatures = preprocessing.PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)





# most correlated columns

df_corr = df_all[most_correlated]

X_corr = df_corr.values

X_important = X[:, indices[:20]]



X_trans = X_corr

if POLY_USE_XGB:

    X_trans = X_important



pfeatures.fit(X_trans)

cfeatures.fit(X_trans)

X_poly= pfeatures.transform(X_trans)

X_cubic = cfeatures.transform(X_trans)



# concatenate 2nd order and 3rd order polynomial features

X = np.hstack([X,X_poly, X_cubic])



X_train, X_test = X[:len(df_train)], X[len(df_train):]



X_train.shape
if DENSITY_RATIO_ESTIMATION:

    from scipy.stats import norm

    from densratio import densratio



    result = densratio(X, X_test)

    weights = result.compute_density_ratio(X)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LassoCV

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate

from sklearn.metrics import mean_squared_error

import lightgbm

import numpy as np

import warnings

warnings.filterwarnings('ignore')


from sklearn.decomposition import TruncatedSVD



if not SVD_COMPONENTS is None:



    svd = TruncatedSVD(n_components=SVD_COMPONENTS, n_iter=80, random_state=RANDOM_STATE)

    svd.fit(X_train, y)



    decomposition = svd



    X = svd.transform(X)

    X_train = svd.transform(X_train)

    X_test = svd.transform(X_test)



# standardise data

if STANDARDIZE:

    from sklearn.preprocessing import StandardScaler

    sc=StandardScaler()



    sc.fit(X)

    X_train = sc.transform(X_train)

    X_test = sc.transform(X_test)
#Validation function

def rmsle_cv(model):

    kf = KFold(NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE).get_n_splits()

    rmse= np.sqrt(-cross_val_score(model,X_train, y, scoring="neg_mean_squared_error", cv = kf))

    return rmse



def rmsle(model):

    rmse= np.sqrt(np.mean((model.predict(X_train)-y)**2))

    return rmse



def rmsle_score(y, y_):

    rmse= np.sqrt(np.mean((y-y_)**2))

    return rmse
sorted(sklearn.metrics.SCORERS.keys())
# fit SVM regression

model_svm = sklearn.svm.SVR(kernel='rbf', C=4)

model_svm = make_pipeline(RobustScaler(), model_svm)

if CROSSVAL_SCORES:

    print(f'Mean CV score:{np.mean(rmsle_cv(model_svm))}')
# fit XGBOOST

model_xgb = xgboost.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.04, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =RANDOM_STATE, nthread = -1)

model_xgb.fit(X_train,y, sample_weight=None)

if CROSSVAL_SCORES:

    print(f'Mean CV score:{np.mean(rmsle_cv(model_xgb))}')
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=RANDOM_STATE))

if CROSSVAL_SCORES:

    print(f'Mean CV score:{np.mean(rmsle_cv(lasso))}')
# fit kernel ridge regression

tunable_params = dict(alpha=[0.2, 0.3, 0.6, 0.7], degree=[1,2,3])

fixed_params = dict(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

KRR = KernelRidge(**fixed_params)

if CROSSVAL_SCORES:

    print(f'Mean CV score:{np.mean(rmsle_cv(KRR))}')
# fit LGBM

model_lgb = lightgbm.LGBMRegressor(objective='regression',num_leaves=11,

                              learning_rate=0.04, n_estimators=1000,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.4619,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11,

                              sample_weight=None)

if CROSSVAL_SCORES:

    print(f'Mean CV score:{np.mean(rmsle_cv(model_lgb))}')
from sklearn.ensemble import RandomForestRegressor

# fit random forest

forest = RandomForestRegressor(n_estimators=200)

if CROSSVAL_SCORES:

    print(f'Score:{np.mean(rmsle_cv(forest))}')
import keras

from keras.wrappers.scikit_learn import KerasRegressor, BaseWrapper

from keras.layers import Dense, Input

from keras.optimizers import Adam



class NeuralNet(BaseEstimator, KerasRegressor):

    def __init__(self, num_features, epochs, batch_size, model=None, verbose=0):

        self._model = keras.Sequential([

            Dense(256, activation='relu', input_shape=(num_features,)),

            Dense(128, activation='relu'),

            Dense(64),

            Dense(1)

        ]) if model is None else model

        optimizer = Adam(lr=4e-3)

        

        self._model.compile(optimizer, loss='mse')

        self._epochs = epochs

        self._batch_size = batch_size

        self._num_features = num_features

        self._verbose = verbose

        

    def fit(self, X, y):

        self._model.fit(X, y, epochs=self._epochs, batch_size=self._batch_size, verbose=self._verbose)



    

    def predict(self, X):

        return self._model.predict(X).flatten()

    

    def get_params(self, deep=True):

        return {'num_features' : self._num_features, 

                'batch_size': self._batch_size,

               'epochs': self._epochs,

               'verbose' : self._verbose}



    def set_params(self, **parameters):

        for parameter, value in parameters.items():

            setattr(self, parameter, value)

        return self



# fit neural net

neural_net = NeuralNet(len(X[0]), 100, 64)

if CROSSVAL_SCORES:

    print(f'Score:{np.mean(rmsle_cv(neural_net))}')
from joblib import Parallel, delayed

import numpy as np



class MetaModel(BaseEstimator, RegressorMixin):

    def __init__(self, models, fit_sub=False):

        self._models = models

        self._fitted = []

        self._fit_sub = fit_sub

    def fit(self, X, y):

        

        if self._fit_sub:

            for m in self._models:

                m.fit(X,y)

        

        predictions = np.vstack([model.predict(X) for model in self._models]).T              

        self._meta = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=1.0)

        self._meta.fit(predictions, y)

        

        

    def predict(self, X):

        predictions = np.vstack([model.predict(X) for model in self._models]).T

        #return self._meta.predict(X)+ np.mean(predictions, axis=1)

        return  self._meta.predict(predictions)



    def get_params(self, deep=True):

        # suppose this estimator has parameters "alpha" and "recursive"

        return {"models": self._models, "fit_sub" : self._fit_sub}



    def set_params(self, **parameters):

        for parameter, value in parameters.items():

            setattr(self, parameter, value)

        return self



    

mmodel =  MetaModel([KRR, model_lgb, lasso, neural_net, model_xgb], fit_sub=True)



if CROSSVAL_SCORES: 

    score = np.mean(rmsle_cv(mmodel))

    print(f'Mean CV score:{score}')

mmodel.fit(X_train,y)

model = mmodel
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)



models = [clone(model_lgb), clone(lasso), clone(model_xgb), clone(KRR) ,MetaModel([KRR, model_lgb, clone(lasso)], fit_sub=True)]

sam = StackingAveragedModels(models, clone(KRR), NUM_FOLDS)

if CROSSVAL_SCORES:

    print(f'Score: {rmsle_cv(sam)}')
# fit final stacking model

sam.fit(X_train,y)
def rmsle_score(y, y_):

    return np.sqrt(np.mean((y-y_)**2))
model_lgb = lightgbm.LGBMRegressor(objective='regression',num_leaves=11,

                              learning_rate=0.04, n_estimators=2000,

                              max_bin = 65, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2619,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf =11)



model_lgb.fit(X_train, y)

y_ = model_lgb.predict(X_test)



# generate submission for tis model

df_eval = pd.DataFrame({

    'Id' : df_test_orig['Id'],

    'SalePrice': np.exp(y_)

    })

df_eval.to_csv(f'submission_lgbm.csv',header=True, columns=['Id', 'SalePrice'], index=False)

    
import xgboost as xgb

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.04, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =RANDOM_STATE, nthread = -1)

model_xgb.fit(X_train, y)



# generate submission for tis model

y_ = model_xgb.predict(X_test)

df_eval = pd.DataFrame({

    'Id' : df_test_orig['Id'],

    'SalePrice': np.exp(y_)

    })

df_eval.to_csv(f'submission_xgb.csv',header=True, columns=['Id', 'SalePrice'], index=False)

    
class Ensemble(BaseEstimator, RegressorMixin):

    def __init__(self, base_models, coefs):

        self.base_models = base_models

        self._coefs = coefs

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        pass

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        prediction = np.vstack([m.predict(X)*c for m, c in zip(self.base_models, self._coefs)]).T.sum(axis=1)

        

        return prediction

    
predictions = model.predict(X_train)

score = rmsle_score(predictions, y)

print(f'Training set score: {score}')
from tqdm import tqdm

w = np.asarray([.6, .2, .2])

# truncated normal

def dist(w):

    return w/np.sum(w)



def truncated_nromal(w=w, std=0.1):

    return dist(np.clip(np.random.normal(loc=w, scale=0.1), std, np.inf))



weight_vectors = [truncated_nromal() for i in range(NUM_WEIGHT_WIGGLES)]

w_with_scores = []



for w in tqdm(weight_vectors):

    e = Ensemble([sam, model_xgb, model_lgb], w)

    y_ = e.predict(X_train)

    score = rmsle_score(y, y_)

    y_ = e.predict(X_test)

    print(f'Score: {score} Weights: {w}')

    w_with_scores.append((score, w, y_))

    

    

w_with_scores = list(sorted(w_with_scores, key=lambda x: x[0]))

# output predictions based on top 60% of wiggled weights

for score, w, y_ in tqdm(w_with_scores[:int(0.6*NUM_WEIGHT_WIGGLES)]):

    df_eval = pd.DataFrame({

    'Id' : df_test_orig['Id'],

    'SalePrice': np.exp(y_)

    })

    df_eval.to_csv(f'submission_{score}_{w[0]}_{w[1]}_{w[2]}.csv',header=True, columns=['Id', 'SalePrice'], index=False)

    