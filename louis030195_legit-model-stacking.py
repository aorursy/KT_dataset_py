# Vectors
import numpy as np

# Matrix, non-numeric data
import pandas as pd

# Algorithms
from sklearn.linear_model import Lasso, ElasticNet
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# Trees
from sklearn.tree import DecisionTreeRegressor

# Ensembles
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor

# Model stacking
from sklearn.pipeline import make_pipeline

# Preprocessing
from sklearn.preprocessing import RobustScaler

# Statistics
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

# Dimension reduction
from sklearn.decomposition import PCA

# Clustering
from sklearn.cluster import KMeans

# Neural network
import tensorflow as tf

# Metrics
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score, roc_auc_score

# Data splitting, cross-validation, hyperparameters optimization
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold

# Visualisation
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from xgboost import plot_importance

# Files
import os
# Lecture du fichier contenant le jeu d'entrainement
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
# Stockage de la colonne Id
train_ID = train['ID']
test_ID = test['ID']

# Nous retirons la colonne Id qui n'est pas utile pour l'entrainement
train.drop('ID', axis = 1, inplace = True)
test.drop('ID', axis = 1, inplace = True)
plt.hist(train.target)
plt.title('Variance')
plt.show()
pca = PCA(n_components=2)
pca.fit(train.drop('target', axis=1))

print(pca.explained_variance_ratio_)  
print(pca.singular_values_)
train_pca = pca.transform(train.drop('target', axis=1))
print(train.drop('target', axis=1).shape)
print(train_pca.shape)
train_new = pca.inverse_transform(train_pca)
plt.scatter(train_new[:, 0], train_new[:, 1], alpha=0.8)
plt.scatter(train_new[:, 0], train_new[:, 1],
            c=train.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('rainbow', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
train.info()
train_shape = train.shape[0]
test_shape = test.shape[0]
y_train = train.target.values
total_data = pd.concat((train, test), sort=True).reset_index(drop=True)
total_data.drop(['target'], axis=1, inplace=True)
total_data.isnull().values.any()
# Stockage de la colonne à prédire dans la variable y avec une transformation pour avoir une distribution logarithmique
y_train = np.log1p(train["target"])
cols_with_onlyone_val = total_data.columns[train.nunique() == 1]
total_data.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
print(train.shape, total_data.shape)
model = RandomForestRegressor(n_jobs=-1, random_state=7)
model.fit(train, y_train)

col = pd.DataFrame({'importance': model.feature_importances_, 'feature': train.columns}).sort_values(
    by=['importance'], ascending=[False])[:1000]['feature'].values
total_data = total_data[col]
total_data.shape
train = total_data[:train_shape]
test = total_data[train_shape:]
def rmsle_cv(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))
GBoost = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5))
model_xgb = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1))
model_lgb = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11))
scores = [rmsle_cv(GBoost).mean(), rmsle_cv(model_xgb).mean(), rmsle_cv(model_lgb).mean()]
plt.plot(['GBoost','xgb','lgb'], scores)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (GBoost, model_lgb, model_xgb))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_models.fit(train, y_train)
predictions = np.expm1(averaged_models.predict(test))
soumission = pd.DataFrame({'ID': test_ID, 'target': predictions})
soumission.to_csv('result.csv', index=False)