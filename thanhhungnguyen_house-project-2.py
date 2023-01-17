import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.head()
test.head()
X_train = train.drop(columns = ['Id','SalePrice'])
y_train = train.iloc[:,-1]
X_test = test
X_train.isna().sum().sort_values(ascending = False)
X_test.isnull().sum().sort_values(ascending = False)
X_train.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
X_test.drop(columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

cat_features = list(X_train.select_dtypes(include = ['object']).columns)
num_features = list(X_train.select_dtypes(exclude = ['object']).columns)

num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)
tr_pipe = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=1, max_depth=8, min_samples_leaf=0.1))
    ]
)
np.random.seed(1)
cv_results = cross_val_score(tr_pipe, X_train, y_train, cv = 10)
print('Mean CV Accuracy:', np.mean(cv_results))
param_grid = {
    "max_depth":[6,8,10,12],
    "min_samples_leaf":[20,40,100],
}
np.random.seed(1)
grid_search = GridSearchCV(tr_pipe,param_grid, cv=10, refit = True)
grid_search.fit(X_train, y_train)

#print(grid_search.best_score_)
#print(grid_search.best_params_)
