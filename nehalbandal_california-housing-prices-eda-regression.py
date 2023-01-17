%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/california-housing-prices/housing.csv")

print(data.shape)

data.head()
data_explore = data.copy()
data_explore.info()
data_explore.describe()
data_explore.hist(figsize=(15, 8))
columns = ['households', 'population', 'total_bedrooms', 'total_rooms']

plt.figure(figsize=(15, 8))

sns.boxplot(data=data_explore[columns])

plt.ylim((-100, 7000))
Q1 = data_explore.quantile(0.25)

Q3 = data_explore.quantile(0.75)

IQR = Q3 - Q1

((data_explore < (Q1 - 1.5 * IQR)) | (data_explore > (Q3 + 1.5 * IQR))).sum()
data_explore['total_bedrooms'].mean(), data_explore['total_bedrooms'].median()
median = data_explore['total_bedrooms'].median()

data_explore['total_bedrooms'].fillna(value=median, inplace=True)

data_explore['total_bedrooms'].isna().sum()
import matplotlib.image as mpimg

california_img=mpimg.imread('../input/images/calfornia_img.jpg')

california_state=mpimg.imread('../input/images/calfornia_state.jpg')
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)

plt.imshow(california_state)

plt.axis('off')

plt.subplot(1, 2, 2)

ax = plt.gca()

data_explore.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=data_explore["population"]/100,

             label="population", figsize=(15,6), c="median_house_value", cmap=plt.get_cmap("jet"), ax=ax)

ax.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05])
data_explore["rooms_per_household"] = data_explore["total_rooms"]/data_explore["households"]

data_explore["bedrooms_per_room"] = data_explore["total_bedrooms"]/data_explore["total_rooms"]
data_explore_dummies = pd.get_dummies(data_explore) 

plt.figure(figsize=(18, 10))

corr_matrix = data_explore_dummies.corr(method='pearson')

sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(data_explore[attributes], figsize=(15, 10))

plt.show()
data_capped = data[data['median_house_value']>=500000]

data = data[data['median_house_value']<500000]

data_capped.shape, data.shape
data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
data["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["income_cat"]):

    strat_train_set = data.iloc[train_index]

    strat_test_set = data.iloc[test_index]



strat_train_set.drop("income_cat", axis = 1, inplace=True)

strat_test_set.drop("income_cat", axis = 1, inplace=True)

strat_train_set.shape, strat_test_set.shape
X_train = strat_train_set.drop('median_house_value', axis=1)

y_train = strat_train_set['median_house_value'].copy()

X_test = strat_test_set.drop('median_house_value', axis=1)

y_test = strat_test_set['median_house_value'].copy()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PowerTransformer, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, households_ix = 3, 4, 6    # column ids



class CombineAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

        return np.c_[X, rooms_per_household, bedrooms_per_room]
num_attrs = list(X_train.columns)

num_attrs.remove('ocean_proximity')

cat_attrs = ['ocean_proximity',]
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),

                         ('attribs_adder', CombineAttributesAdder()),

                         ('scaler', PowerTransformer(method='yeo-johnson', standardize=True))])



pre_process = ColumnTransformer([("nums", num_pipeline, num_attrs),

                                   ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attrs)], remainder='passthrough')
X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)

X_train_transformed.shape, X_test_transformed.shape
feature_columns = list(X_train.columns)

feature_columns.extend(['rooms_per_household','bedrooms_per_room'])

new_cols = list(X_train['ocean_proximity'].unique())

feature_columns.extend(new_cols)

feature_columns.remove('ocean_proximity')
from sklearn.model_selection import cross_val_score



results=[]



def cv_results(model, X, y):

    scores = cross_val_score(model, X, y, cv = 7, scoring="neg_root_mean_squared_error", n_jobs=-1)

    rmse_scores = -scores

    rmse_scores = np.round(rmse_scores, 3)

    print('CV Scores: ', rmse_scores)

    print('rmse: {},  S.D.:{} '.format(np.mean(rmse_scores), np.std(rmse_scores)))

    results.append([model.__class__.__name__, np.mean(rmse_scores), np.std(rmse_scores)])
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(alpha=1, penalty='l1', random_state=42)

sgd_reg.fit(X_train_transformed, y_train)
feature_imp = [ col for col in zip(feature_columns,sgd_reg.coef_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
cv_results(sgd_reg, X_train_transformed, y_train)
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(criterion="mse", random_state=42)

tree_reg.fit(X_train_transformed, y_train)
cv_results(tree_reg, X_train_transformed, y_train)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(criterion='mse', n_estimators=100, n_jobs=-1, random_state=42)

forest_reg.fit(X_train_transformed, y_train)
feature_imp = [ col for col in zip(feature_columns,forest_reg.feature_importances_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
cv_results(forest_reg, X_train_transformed, y_train)
from xgboost import XGBRegressor
xgb_reg = XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, objective='reg:squarederror', random_state=42)

xgb_reg.fit(X_train_transformed, y_train)
feature_imp = [ col for col in zip(feature_columns,xgb_reg.feature_importances_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
cv_results(xgb_reg, X_train_transformed, y_train)
result_df = pd.DataFrame(data=results, columns=['Model', 'RMSE', 'S.D'])

result_df
from sklearn.model_selection import GridSearchCV
rf_grid_parm=[{'n_estimators':[50, 100, 300], 'max_depth':[8, 16, 24]}]

rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_grid_parm, cv=5, scoring="neg_root_mean_squared_error", return_train_score=True, n_jobs=-1)

rf_grid_search.fit(X_train_transformed, y_train)
rf_grid_search.best_params_, -rf_grid_search.best_score_
cvres = rf_grid_search.cv_results_

print("Results for each run of Random Forest Regression...")

for train_mean_score, test_mean_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):

    print(-train_mean_score, -test_mean_score, params)
best_forest_reg = rf_grid_search.best_estimator_

best_forest_reg
xgb_grid_parm=[{'n_estimators':[50, 100, 300], 'max_depth':[6, 8, 12]}]

xgb_grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror', learning_rate=0.1, n_jobs=-1, random_state=42), xgb_grid_parm, cv=5, scoring="neg_root_mean_squared_error", return_train_score=True, n_jobs=-1)

xgb_grid_search.fit(X_train_transformed, y_train)
xgb_grid_search.best_params_, -xgb_grid_search.best_score_
cvres = xgb_grid_search.cv_results_

print("Results for each run of XGBoost Regression...")

for train_mean_score, test_mean_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):

    print(-train_mean_score, -test_mean_score, params)
best_xgb_reg = xgb_grid_search.best_estimator_

best_xgb_reg
cv_results(best_forest_reg, X_test_transformed, y_test)
cv_results(best_xgb_reg, X_test_transformed, y_test)
combine_data = pd.concat([strat_train_set, strat_test_set], axis=0)

combine_data.shape
y_train_pred = best_xgb_reg.predict(X_train_transformed)

y_test_pred = best_xgb_reg.predict(X_test_transformed)
y_pred = np.concatenate([y_train_pred, y_test_pred], axis=0)

y_pred.shape
combine_data['predicted_value'] = y_pred
combine_data.head()
combine_data.describe()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

combine_data['median_house_value'].hist()

plt.title('Observed Median House Value')

plt.subplot(1, 2, 2)

combine_data['predicted_value'].hist()

plt.title('Predicted Median House Value')

plt.show()
plt.figure(figsize=(12, 8))

plt.scatter(combine_data['median_income'], combine_data['median_house_value'], c='green', alpha=0.7, label="Observed")

plt.scatter(combine_data['median_income'], combine_data['predicted_value'], c='red', alpha=0.7, label="Predicted")

plt.xlabel('Median Income')

plt.ylabel('Median House Value')

plt.legend()

plt.show()
plt.figure(figsize=(18, 10))

fig, ax = plt.subplots(nrows=1, ncols=2)

combine_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(15,8), c="median_house_value", cmap=plt.get_cmap("jet"), ax=ax[0], colorbar=False)

ax[0].imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05])

ax[0].set_title('Observed Median House Values')

combine_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,figsize=(15,8), c="predicted_value", cmap=plt.get_cmap("jet"), ax=ax[1], colorbar=False)

ax[1].imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05])

ax[1].set_title('Predicted Median House Values')

plt.show()