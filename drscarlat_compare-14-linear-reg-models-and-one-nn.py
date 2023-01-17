# IMPORT MODULES
# TURN ON the GPU !!!
# If importing dataset from outside -  Internet must be "connected"

import os
from operator import itemgetter    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from pandas.tools.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import lightgbm as lgb


import tensorflow as tf

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
#from keras.utils.np_utils import to_categorical

import tarfile
from six.moves import urllib

print(os.getcwd())
print("Modules imported \n")
#print("Files in current directory:")
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
# LOAD data

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housingRaw = load_housing_data()
housingRaw.columns
housing = housingRaw.copy()
print("housing ", housing.shape)

# Missing Data
ColsMissingValues = housing.isnull().sum()
print("There are ", len(ColsMissingValues[ColsMissingValues>0]), " features with missing values")
#print("_"*80)
all_data_na = (housing.isnull().sum() / len(housing)) * 100
all_data_na = all_data_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(len(ColsMissingValues[ColsMissingValues>0])))
# VISUALIZATION

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population",
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

housing.hist(bins=50, figsize=(20,15))
plt.show()

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self,additional=1):
        self.additional = additional
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.additional==1:
            X["rooms_per_household"] = X["total_rooms"]/X["households"]
            X["bedrooms_per_room"] = X["total_bedrooms"]/X["total_rooms"]
            X["population_per_household"]=X["population"]/X["households"]
            X["income_per_pop_household"]=X["median_income"]/X["population_per_household"]
            X["rooms_squared"]=X["rooms_per_household"]*X["rooms_per_household"]
            X["room_mult_income"]=X["rooms_squared"]*X["median_income"]    
        else:
            X["rooms_per_household"] = X["total_rooms"]/X["households"]
            X["bedrooms_per_room"] = X["total_bedrooms"]/X["total_rooms"]
            X["population_per_household"]=X["population"]/X["households"]
            X["income_per_pop_household"]=X["median_income"]/X["population_per_household"]
            X["rooms_squared"]=X["rooms_per_household"]*X["rooms_per_household"]
            X["room_mult_income"]=X["rooms_squared"]*X["median_income"]
            
        return X

# as initially, the models were UNDERFITTING - I've added some features
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = 1-np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = 1-np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self,skew=0.75):
        self.skew = skew
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_numeric=X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        #X = pd.concat([pd.get_dummies(X[['ocean_proximity']]), X[]], axis=1)
        return X
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        lab = LabelEncoder()
        X['ocean_proximity'] = lab.fit_transform(X['ocean_proximity'])
              
        return X
class imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):       
        imp = Imputer(strategy="median")
        Xcopy = X
        Ximputed = imp.fit_transform(Xcopy)
        X= Ximputed
        X = pd.DataFrame(Ximputed, columns = Xcopy.columns)
        
        return X
class scalerFI(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):       
        scl = StandardScaler()
        Xcopy = pd.DataFrame(X)
        Xscaled = scl.fit_transform(Xcopy)
        X = pd.DataFrame(Xscaled, columns = Xcopy.columns)
        
        return X
# Prepare the DATA for Feature Importance

# PIPELINE
pipeFI = Pipeline([
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1)), 
    ('imputer', imputer()),
    ('scalerFI', scalerFI()),
    ])

# RELOAD A FRESH COPY
housingFI = housingRaw.copy()

#y = np.log(housing.median_house_value)
yFI = housingFI.median_house_value

print("housingFI BEFORE pipeline", housingFI.shape)

housingWpriceFI = pd.DataFrame(housingFI)
housingNoPriceFI = housingWpriceFI.drop("median_house_value", axis=1)

print("housingFI without the label ", housingNoPriceFI.shape)

FullDataPipeFI = pipeFI.fit_transform(housingNoPriceFI)
housingFI = pd.DataFrame(FullDataPipeFI)

print("housingFI AFTER pipeline ", housingFI.shape)
print("yFI ", yFI.shape)
# AFTER Pipeline

housingFI.hist(bins=50, figsize=(20,15))
plt.show()
# FEATURE IMPORTANCE - Needs the original cols names (see classes above) in order to see the features' NAMES and not their numbers
# Useful even AFTER PCA - check the relevance of features for prediction

trainFinalFI = pd.DataFrame(housingFI)
yFinalFI = yFI

lasso=Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
lasso.fit(trainFinalFI,yFinalFI)

FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=trainFinalFI.columns)

# Focus on those with 0 importance
#print(FI_lasso.sort_values("Feature Importance",ascending=False).to_string())
#print("_"*80)
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()
# Prepare the DATA & PIPELINE for models evaluation with TRAIN and TEST sets

# PIPELINE
pipe = Pipeline([
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1)), 
    ('impute_num', Imputer(strategy="median")),
    ])
###                   ** skew_dummies is taking care of the labelencoder AND onehotencoder with get_dummies() !  **


# RELOAD A FRESH COPY
housing = housingRaw.copy()

#y = np.log(housing.median_house_value)
y = housing.median_house_value

n_train = 16500

y_train = np.array(pd.DataFrame(y[:n_train])).reshape(-1,1).ravel()
y_test = np.array(pd.DataFrame(y[n_train:])).reshape(-1,1).ravel()

print("housing BEFORE pipeline", housing.shape)

housingWprice = pd.DataFrame(housing)
housingNoPrice = housingWprice.drop("median_house_value", axis=1)

print("housing without the label ", housingNoPrice.shape)

FullDataPipe = pipe.fit_transform(housingNoPrice)
housing = pd.DataFrame(FullDataPipe)

print("housing AFTER pipeline ", housing.shape)
print("y ", y.shape)
print("_"*100)

# Any Missing Data ?
ColsMissingValues = housing.isnull().sum()
print("There are ", len(ColsMissingValues[ColsMissingValues>0]), " features with missing values")
all_data_na = (housing.isnull().sum() / len(housing)) * 100
all_data_na = all_data_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(len(ColsMissingValues[ColsMissingValues>0])))
print("_"*100)

# Split TRAIN / TEST and then SCALE them SEPARATELY

trainSet = pd.DataFrame(housing[:n_train])
testSet = pd.DataFrame(housing[n_train:])

# Scaler should be run separately on train and test to prevent information leaking from test into train and eventually overfitting
scaler = StandardScaler()
trainX = scaler.fit_transform(trainSet)
testX = scaler.fit_transform(testSet)
print("trainX and testX have been scaled separately")
print("trainX ",trainX.shape)
print("y_train ",y_train.shape)
print("testX ",testX.shape)
print("y_test ",y_test.shape)
# CROSS VALIDATION

def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
# Lin reg ALL 14 models HYPERPARAMS NOT optimized

models = [LinearRegression(),Ridge(),Lasso(),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
          ElasticNet(),SGDRegressor(),BayesianRidge(),KernelRidge(),ExtraTreesRegressor(),XGBRegressor(),lgb.LGBMRegressor()]
names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb", "LightGBM"]
# Run the models and compare

ModScores = {}

for name, model in zip(names, models):
    score = rmse_cv(model, trainX, y_train)
    ModScores[name] = score.mean()
    print("{}: {:.2f}".format(name,score.mean()))

print("_"*100)
for key, value in sorted(ModScores.items(), key = itemgetter(1), reverse = False):
    print(key, value)
# SEARCH GRID FOR HYPERPARAMS OPTIMIZATION - One model at a time

model = XGBRegressor()

param_grid = [
{},
]

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(trainX, y_train)

print(grid_search.best_estimator_)
# 7 MODELS with HYPERPARAMS optimized
models = [RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False),
          
           lgb.LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               learning_rate=0.1, max_depth=-1, min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
               n_jobs=-1, num_leaves=31, objective='regression', random_state=None,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=1),
          
          XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1),
          
          GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False),
          
          ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None,
              max_features='auto', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
              oob_score=False, random_state=None, verbose=0, warm_start=False),
          
          SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
              kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
          
          Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
               normalize=False, random_state=None, solver='auto', tol=0.001),
         
          BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
               fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
               normalize=False, tol=0.001, verbose=False)
         ]

names = ["RandomForest", "LGB", "XGB", "GBR", "ExtraTrees", "SVR", "Ridge", "BayesRidge"]
    
# LEARNING CURVE

model = GradientBoostingRegressor()

title = "Learning Curves "
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(model, title, trainX, y_train, ylim=(0.01, 0.3), cv=cv, n_jobs=4)
# Model fit and evaluation on test

model = GradientBoostingRegressor()

model.fit(trainX, y_train)

final_predictions = model.predict(testX)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) 

print("rmse on test ", final_rmse)
# Mapping data from the linear models above to NN below

del model
del models
from keras import models
from keras.models import Sequential

x_val = testX
partial_x_train = trainX
y_val = y_test
partial_y_train = y_train

print("partial_x_train ", partial_x_train.shape)
print("partial_y_train ", partial_y_train.shape)

print("x_val ", x_val.shape)
print("y_val ", y_val.shape)
# NN MODEL
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(partial_x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
# CV with NN

train_data = partial_x_train
train_targets = partial_y_train

num_epochs = 100
BatchSize = 64

k = 4
num_val_samples = len(train_data) // k
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
    
    model = build_model()
    
    history = model.fit(partial_train_data, partial_train_targets,validation_data=(x_val, y_val),
    epochs=num_epochs, batch_size=BatchSize, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
results = model.evaluate(x_val, y_val)
print("Loss and MAE on TEST")
print("results ", results)

history_dict = history.history
history_dict.keys()
# VALIDATION LOSS curves

plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

## VALIDATION MAE curves

plt.clf()
acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
plt.plot(epochs, acc, 'bo', label='Training MAE')
plt.plot(epochs, val_acc, 'b', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Model fit and evaluation on test
# Set the num of Epochs and Batch Size according to learning curves

model = build_model()
model.fit(train_data, train_targets, epochs=100, batch_size=64, verbose=0)
test_mse_score, test_mae_score = model.evaluate(x_val, y_val)

print("test_mae_score on test ", test_mae_score)