import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/insurance/insurance.csv")

print(data.shape)

data.head()
data_explore = data.copy()
data_explore.info()
data_explore.describe()
Q1 = data_explore.quantile(0.25)

Q3 = data_explore.quantile(0.75)

IQR = Q3 - Q1

outliers = ((data_explore < (Q1 - 1.5 * IQR)) | (data_explore > (Q3 + 1.5 * IQR))).sum()

outliers[outliers>0]
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

sns.boxplot(x='charges', data=data_explore)

plt.subplot(1, 2, 2)

sns.boxplot(x='bmi', data=data_explore)

plt.show()
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

data_explore["sex_enc"] = label_encoder.fit_transform(data_explore["sex"])

print(label_encoder.classes_)

data_explore["smoker_enc"] = label_encoder.fit_transform(data_explore["smoker"])

print(label_encoder.classes_)

data_explore["region_enc"] = label_encoder.fit_transform(data_explore["region"])

print(label_encoder.classes_)
sns.pairplot(data_explore)
data_explore['charges'].hist()

plt.xlabel('Charges')
sns.catplot(x="smoker", kind="count",hue = 'sex', data=data_explore, legend_out=False )

plt.title("Distribution of Smokers on Basis of Gender")

plt.show()
data_explore[(data_explore['smoker']=='yes') & (data_explore['sex']=='female')]['charges'].count(), data_explore[(data_explore['smoker']=='yes') & (data_explore['sex']=='male')]['charges'].count()
data_explore_male = data_explore[data_explore["sex"]=="male"]

data_explore_female = data_explore[data_explore["sex"]=="female"]

data_explore_non_smoker = data_explore[data_explore["smoker"]=="no"]

data_explore_smoker = data_explore[data_explore["smoker"]=="yes"]
data_explore_smoker.age.hist()

plt.xlabel('Age')
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

data_explore_smoker['charges'].hist()

plt.title('Distribution of Charges for Smokers')

plt.subplot(1, 2, 2)

data_explore_non_smoker['charges'].hist()

plt.title('Distribution of Charges for Non-Smokers')

plt.show()
data_explore_smoker['charges'].min()
fig, ax = plt.subplots()

data_explore.plot(kind="scatter", x="age", y="charges", alpha=0.5, c="smoker_enc", cmap=plt.get_cmap("brg"), colorbar=False, ax=ax, figsize=(8, 4))

plt.title("Distribution of treatment charges over ages\nSmokers - Green   Non-smokers: Blue")

plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2)

fig.set_figheight(6)

fig.set_figwidth(12)

data_explore_smoker.plot(kind="scatter", x="age", y="charges", ax=ax[0])

ax[0].set_title("Smokers treatment charges distribution over age")

data_explore_non_smoker.plot(kind="scatter", x="age", y="charges", ax=ax[1])

ax[1].set_title("Non Smokers treatment charges distribution over age")

plt.show()
fig, ax = plt.subplots()

data_explore.plot(kind="scatter", x="age", y="charges", alpha=0.7, c="sex_enc", cmap=plt.get_cmap("brg"), colorbar=False, ax=ax, figsize=(8, 4))

plt.title("Distribution of charges over ages\nMale - Green   Female: Blue")

plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2)

fig.set_figheight(6)

fig.set_figwidth(15)

data_explore_male.plot(kind="scatter", x="age", y="charges",alpha=0.7,c="smoker_enc", cmap=plt.get_cmap("brg"), colorbar=False, ax=ax[0])

ax[0].set_title("Males treatment charges distribution over age\nSmokers - Green   Non-smokers: Blue")

data_explore_female.plot(kind="scatter", x="age", y="charges",alpha=0.7,c="smoker_enc", cmap=plt.get_cmap("brg"), colorbar=False, ax=ax[1])

ax[1].set_title("Females treatment charges distribution over age\nSmokers - Green   Non-smokers: Blue")

plt.show()
fig, ax = plt.subplots()

data_explore.plot(kind="scatter", x="bmi", y="charges", alpha=0.7, c="smoker_enc", cmap=plt.get_cmap("PiYG"), colorbar=False, ax=ax)

plt.title("Distribution of charges over bmi\nSmokers - Green   Non-smokers: Red")

plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2)

fig.set_figheight(6)

fig.set_figwidth(15)

data_explore_smoker.plot(kind="scatter", x="bmi", y="charges", c="age", cmap=plt.get_cmap("jet"), colorbar=False, ax=ax[0])

ax[0].set_title("Smokers treatment charges distribution over BMI")

data_explore_non_smoker.plot(kind="scatter", x="bmi", y="charges", c="age", cmap=plt.get_cmap("jet"), colorbar=True, ax=ax[1])

ax[1].set_title("Non-Smokers treatment charges distribution over BMI")

plt.show()
corr_matrix = data_explore.corr()



plt.figure(figsize=(12, 6))

sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), annot=True, square=True)

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit



stratified_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in stratified_data.split(data, data["smoker"]):

    stratified_train_set = data.iloc[train_idx]

    stratified_test_set = data.iloc[test_idx]

    

stratified_train_set.shape, stratified_test_set.shape
y_train = stratified_train_set['charges'].copy()

X_train = stratified_train_set.drop(columns='charges', axis=1)



y_test = stratified_test_set['charges'].copy()

X_test = stratified_test_set.drop(columns='charges', axis=1)
cat_attrs = ['sex', 'smoker', 'region']

num_attrs = ['age', 'bmi', 'children']
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
pre_process = ColumnTransformer([('scaler', StandardScaler(), num_attrs),

                                ('encode', OneHotEncoder(), cat_attrs)], remainder='passthrough')
X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)
X_train_transformed.shape, X_train.iloc[0, :], X_train_transformed[0]
feature_columns = list(X_train.columns)

new_col_name = ['female', 'male', 'smoker_no', 'smoker_yes', 'northeast', 'northwest', 'southeast', 'southwest']

feature_columns.extend(new_col_name)

feature_columns = [ col for col in feature_columns if not col in cat_attrs]

feature_columns
from sklearn.model_selection import cross_val_score



results = []



def cv_results(model, X, y):

    scores = cross_val_score(model, X, y, cv = 7, scoring="neg_mean_squared_error", n_jobs=-1)

    rmse_scores = np.sqrt(-scores)

    rmse_scores = np.round(rmse_scores, 2)

    print('CV Scores: ', rmse_scores)

    print('rmse: {},  S.D.:{} '.format(np.mean(rmse_scores), np.std(rmse_scores)))

    results.append([model.__class__.__name__, np.mean(rmse_scores), np.std(rmse_scores)])
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

linear_reg.fit(X_train_transformed, y_train)
feature_imp = [ col for col in zip(feature_columns,linear_reg.coef_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
cv_results(linear_reg, X_train_transformed, y_train)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
from sklearn.pipeline import Pipeline



poly_reg = Pipeline([('poly_features', poly_features),

                    ('linear_reg', LinearRegression(n_jobs=-1))])
poly_reg.fit(X_train_transformed, y_train)
cv_results(poly_reg, X_train_transformed, y_train)
from sklearn.svm import SVR
svr_reg = SVR(C=1, kernel='rbf')

svr_reg.fit(X_train_transformed, y_train)
cv_results(svr_reg, X_train_transformed, y_train)
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(criterion='mse', random_state=42)

tree_reg.fit(X_train_transformed, y_train)
feature_imp = [ col for col in zip(feature_columns,tree_reg.feature_importances_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
cv_results(tree_reg, X_train_transformed, y_train)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, criterion='mse', n_jobs=-1, random_state=42)

forest_reg.fit(X_train_transformed, y_train)
feature_imp = [ col for col in zip(feature_columns,forest_reg.feature_importances_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
cv_results(forest_reg, X_train_transformed, y_train)
from sklearn.ensemble import AdaBoostRegressor
ada_reg = AdaBoostRegressor(loss='linear', n_estimators=100, learning_rate=0.01, random_state=42)

ada_reg.fit(X_train_transformed, y_train)
cv_results(ada_reg, X_train_transformed, y_train)
from xgboost import XGBRegressor
xgb_reg = XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.1, objective='reg:squarederror', random_state=42)

xgb_reg.fit(X_train_transformed, y_train)
cv_results(xgb_reg, X_train_transformed, y_train)
result_df = pd.DataFrame(data=results, columns=['Model', 'RMSE', 'S.D'])

result_df
from sklearn.model_selection import GridSearchCV
xgb_grid_parm=[{'n_estimators':[25, 50, 75, 100], 'learning_rate':[0.001, 0.01, 0.1, 0.5, 1], 'max_depth':[3, 6, 8, 12] }]

xgb_grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42), xgb_grid_parm, cv=5, scoring="neg_mean_squared_error", return_train_score=True, n_jobs=-1)

xgb_grid_search.fit(X_train_transformed, y_train)
xgb_grid_search.best_params_
cvres = xgb_grid_search.cv_results_

print("Results for each run of XGBoost Regression...")

for train_mean_score, test_mean_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-train_mean_score), np.sqrt(-test_mean_score), params)
best_xgb_reg = xgb_grid_search.best_estimator_

best_xgb_reg
cv_results(best_xgb_reg, X_test_transformed, y_test)
# R2-Score

best_xgb_reg.score(X_train_transformed, y_train), best_xgb_reg.score(X_test_transformed, y_test)
combine_data = pd.concat([stratified_train_set, stratified_test_set], axis=0)
combine_data.shape
combine_data['smoker_enc'] = label_encoder.fit_transform(combine_data['smoker'])
y_train_pred = best_xgb_reg.predict(X_train_transformed)

y_test_pred = best_xgb_reg.predict(X_test_transformed)
y_pred = np.concatenate([y_train_pred, y_test_pred], axis=0)
combine_data['predicted_charges'] = y_pred
combine_data.head()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

plt.scatter(combine_data['age'], combine_data['charges'], c=combine_data["smoker_enc"], cmap=plt.get_cmap("brg"), alpha=0.7)

plt.title("Distribution of Observed Charges\nNon-Smoker: blue, Smoker: green")

plt.subplot(1, 2, 2)

plt.scatter(combine_data['age'], combine_data['predicted_charges'], c=combine_data["smoker_enc"], cmap=plt.get_cmap("brg"), alpha=0.7)

plt.title("Distribution of Predicted Charges\nNon-Smoker: blue, Smoker: green")

plt.show()
combine_data_smoker = combine_data[combine_data['smoker']=='yes']

combine_data_non_smoker = combine_data[combine_data['smoker']=='no']
combine_data_smoker.describe()
combine_data_non_smoker.describe()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

combine_data_smoker['charges'].hist()

plt.title('Observed Charges for Smokers')

plt.subplot(1, 2, 2)

combine_data_smoker['predicted_charges'].hist()

plt.title('Predicted Charges for Non-Smokers')

plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

combine_data_non_smoker['charges'].hist()

plt.title('Observed Charges for Non-Smokers')

plt.subplot(1, 2, 2)

combine_data_non_smoker['predicted_charges'].hist()

plt.title('Predicted Charges for Non-Smokers')

plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

plt.scatter(combine_data_smoker['age'], combine_data_smoker['charges'], c='green')

plt.scatter(combine_data_smoker['age'], combine_data_smoker['predicted_charges'], c='red')

plt.title("Analysis of Predicted Charges for Smokers\nObserved Charges: green, Predicted Charges: red")

plt.subplot(1, 2, 2)

plt.scatter(combine_data_non_smoker['age'], combine_data_non_smoker['charges'], c='green')

plt.scatter(combine_data_non_smoker['age'], combine_data_non_smoker['predicted_charges'], c='red')

plt.title("Analysis of Predicted Charges for Non-Smokers\nObserved Charges: green, Predicted Charges: red")

plt.show()