import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import norm, skew
test = pd.read_csv("Test_data.csv")
train = pd.read_csv("Train_data.csv")

cat_cols = ['F3', 'F4', 'F5', 'F7', 'F8', 'F9', 'F11', 'F12']
num_cols = ['F13', 'F15', 'F16', 'F17', 'F19', 'F14', 'F6', 'F10']
output = ['O/P']

train['F19'] = train['F13']*train['F15']
test['F19'] = test['F13']*test['F15']

X = train[cat_cols+num_cols]

y = train[output]
sns.distplot(y , fit=norm);
sns.distplot(np.log1p(y) , fit=norm);
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), square=True)
plt.suptitle("Correlation Heatmap")
plt.show();
corr_with_sale_price = train.corr()["O/P"].sort_values(ascending=False)
plt.figure(figsize=(14,6))
corr_with_sale_price.drop("O/P").plot.bar()
plt.show();
numeric_feats = X.dtypes[X.dtypes != "object"].index

skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.5]
skewed_feats = skewed_feats.index
X[skewed_feats] = np.log1p(X[skewed_feats])
test[skewed_feats] = np.log1p(test[skewed_feats])
#X_test = test[cat_cols+num_cols]
X_train, X_val = X[:10000], X[10000:]
y_train, y_val = y[:10000], y[10000:]
y_trains = np.log1p(y_train)
X.shape
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

poly.fit(X_train)

X_poly = pd.DataFrame(poly.transform(X_train))
X_val_poly = pd.DataFrame(poly.transform(X_val))

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

select = SelectFromModel(
    RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    max_features=20)

select.fit(X_poly, y_train)
X_1_1 = pd.DataFrame(select.transform(X_poly))
X_1_2 = pd.DataFrame(select.transform(X_val_poly))
print(X_1_1.shape)
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

param_grid = {'n_estimators': [306],
'learning_rate': [0.46],
              'max_depth': [5],
              'min_child_weight': [2],
              'gamma': [1],
              'subsample': [0.9],
             }

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state=0)

grid = GridSearchCV(XGBRegressor(random_state=0), param_grid, cv = 5, n_jobs=12, verbose=8, scoring="neg_root_mean_squared_error")

grid.fit(X_train, y_trains)

param_grid2 = {'bootstrap': [True, False],
 'max_depth': [20, 40, 80, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [400, 600, 800, 1000, 1200, 1400]}

grid2 = GridSearchCV(RandomForestRegressor(random_state=0), param_grid2, cv = 4, n_jobs=11, verbose=15, scoring="neg_root_mean_squared_error")

grid2.fit(X, y)
from sklearn.metrics import mean_squared_error
score1 = grid.predict(X_val)
score1 = np.expm1(score1)
print("Val ", mean_squared_error(y_val, score1, squared=False))
score2 = grid.predict(X_train)
score2 = np.expm1(score2)
print("train ", mean_squared_error(y_train, score2, squared=False))
print("best params: {}".format(grid.best_params_))
y_test = grid.predict(X_test)
y_test = np.expm1(y_test)
y_test
output = pd.DataFrame({'Id': test['Unnamed: 0'],'PredictedValue': y_test})
output.to_csv('submission.csv', index=False)
