import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('ggplot')
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
train = pd.read_csv('../input/train_fe_3.csv')
test = pd.read_csv('../input/test.csv')
scaler = RobustScaler()

train.drop(train.columns[0], axis = 1, inplace = True)
deal_probability = train['deal_probability']

del train['deal_probability']
gc.collect()

drop_cols = []

for column in train:
    if(train[column].dtype != int and train[column].dtype != float):
        drop_cols.append(column)
        
train.drop(drop_cols, axis = 1, inplace = True)


for column in train:
    if(train[column].dtype == int or train[column].dtype == float):
        train[column].fillna(0.5, inplace = True)
        train[column] = train[column].abs()

X_scaled = scaler.fit(train).transform(train)
y_log = np.log1p(deal_probability)
lasso = Lasso(alpha = 0.001)
lasso.fit(X_scaled, y_log)
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index = train.columns)
FI_lasso.sort_values("Feature Importance",ascending = False)
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()
