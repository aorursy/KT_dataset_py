import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

print(data.shape)

data.head()
data_explore = data.copy()

data_explore = data_explore.drop(columns="Id", axis=1)
data_explore.info()
nulls = data_explore.isna().sum()

nulls[nulls>0]
na_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageType", "GarageFinish", "GarageCond", "GarageQual"]



data_explore[na_cols] = data_explore[na_cols].fillna("NA")
data_explore["Alley"].value_counts()
nulls = data_explore.isna().sum()

nan_cols = nulls[nulls>0].index

data_explore[nan_cols].info()
from sklearn.impute import SimpleImputer



num_imputer = SimpleImputer(strategy="mean")

cat_imputer = SimpleImputer(strategy="most_frequent")
num_nans = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

cat_nans = ['MasVnrType', 'Electrical', 'FireplaceQu']

data_explore[num_nans] = num_imputer.fit_transform(data[num_nans])

data_explore[cat_nans] = cat_imputer.fit_transform(data[cat_nans])
nulls = data_explore.isna().sum()

nan_cols = nulls[nulls>0].index

nan_cols
data_explore.head()
data_explore['MSSubClass'] = data_explore['MSSubClass'].astype(str)



cat_attrs = []

num_attrs = []

columns = list(data_explore.columns)

for col in columns:

    if data_explore[col].dtype=='O':

        cat_attrs.append(col)

    else:

        num_attrs.append(col)
data_explore.describe()
Q1 = data_explore.quantile(0.25)

Q3 = data_explore.quantile(0.75)

IQR = Q3 - Q1

outliers = ((data_explore < (Q1 - 1.5 * IQR)) | (data_explore > (Q3 + 1.5 * IQR))).sum()

outliers[outliers>0]
data_explore["SalePrice"].hist()
plt.hist(data_explore["SalePrice"].apply(np.log))

plt.show()
plt.figure(figsize=(85, 16))

corr_matrix = data_explore.corr()

sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True, cbar=False)

plt.tight_layout()
corr_matrix['SalePrice'].sort_values(ascending=False)
features_to_viz = ['GrLivArea', 'GarageArea', 'TotalBsmtSF']

i=1

plt.style.use("seaborn")

plt.figure(figsize=(15, 6))

for feature in features_to_viz:

    plt.subplot(1, 3, i)

    i=i+1

    plt.scatter(data_explore[feature], data_explore['SalePrice'])

    plt.title("Sale Price Vs "+feature)
plt.figure(figsize=(10, 6))

sns.boxplot(x='OverallQual', y='SalePrice', data=data_explore)
plt.figure(figsize=(18, 8))

sns.boxplot(x='YearBuilt', y='SalePrice', data=data_explore)

plt.xticks(rotation=90);
plt.scatter(data_explore['GrLivArea'], data_explore['SalePrice'], c=data_explore['TotRmsAbvGrd'], cmap="Set2_r")

plt.title('SalePrice Vs. GrLivArea')

plt.colorbar().set_label('# of Total Rooms Above Ground', fontsize=14)
data_explore['GarageCars'].value_counts()
plt.scatter(data_explore['GrLivArea'], data_explore['SalePrice'], c=data_explore['GarageCars'], cmap="Set2_r")

plt.title('SalePrice Vs. GrLivArea')

plt.colorbar().set_label('Capacity of # Cars in Garage', fontsize=14)
plt.scatter(data_explore['GrLivArea'], data_explore['SalePrice'], c=data_explore['YearBuilt'].astype('int'), cmap="rainbow")

plt.title('SalePrice Vs. GrLivArea')

plt.colorbar().set_label('YearBuilt', fontsize=14)
features_to_viz = ['ExterQual', 'GarageQual', 'KitchenQual', 'FireplaceQu', 'BsmtQual', 'BsmtExposure',]

i=1

plt.figure(figsize=(15, 10))

for col in features_to_viz:

    plt.subplot(3, 2, i)

    sns.boxplot(y=col, x='SalePrice', data=data_explore, orient='h')

    i+=1
features_to_viz = ['BldgType', 'HouseStyle', 'Foundation', 'MSZoning',]

i=1

plt.figure(figsize=(15, 10))

for col in features_to_viz:

    plt.subplot(3, 2, i)

    sns.boxplot(y=col, x='SalePrice', data=data_explore, orient='h')

    i+=1
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

sns.boxplot(y='SaleType', x='SalePrice', data=data_explore)

plt.subplot(1, 2, 2)

sns.boxplot(y='SaleCondition', x='SalePrice', data=data_explore)

plt.show()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

print(cols)

sns.pairplot(data_explore[cols])

plt.show()
X = data.drop(columns=['SalePrice'], axis=1)

y = data['SalePrice'].copy()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_log_train = np.log(y_train)

y_log_test = np.log(y_test)
na_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageType", "GarageFinish", "GarageCond", "GarageQual", "PoolQC", "Fence", "MiscFeature"]

cat_attrs = [cat for cat in cat_attrs if not cat in na_cols]

num_attrs.remove('SalePrice')
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PowerTransformer, OneHotEncoder
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="mean")),

                        ('transformer', PowerTransformer(method='yeo-johnson', standardize=True))])



cat_pipeline_1 = Pipeline([('cat_na_fill', SimpleImputer(strategy="constant", fill_value='NA')),

                          ('encoder', OneHotEncoder(handle_unknown='ignore'))])



cat_pipeline_2 = Pipeline([('cat_nan_fill', SimpleImputer(strategy="most_frequent")),

                          ('encoder', OneHotEncoder(handle_unknown='ignore'))])
pre_process = ColumnTransformer([('drop_id', 'drop', ['Id']),

                                ('cat_pipeline_1', cat_pipeline_1, na_cols),

                                ('cat_pipeline_2', cat_pipeline_2, cat_attrs),

                                ('num_pipeline', num_pipeline, num_attrs)], remainder='passthrough')
X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)
X_train_transformed.shape, X_test_transformed.shape
oh_na_cols = list(pre_process.transformers_[1][1]['encoder'].get_feature_names(na_cols))

oh_nan_cols = list(pre_process.transformers_[2][1]['encoder'].get_feature_names(cat_attrs))

feature_columns = oh_na_cols+oh_nan_cols + num_attrs
from sklearn.model_selection import GridSearchCV, KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.linear_model import ElasticNet
elastic_net_grid_param = [{'l1_ratio': list(np.linspace(0, 1, 10)), 'alpha': [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1]}]

elastic_net_grid_search = GridSearchCV(ElasticNet(random_state=42), elastic_net_grid_param, cv=kf, scoring='neg_root_mean_squared_error', return_train_score=True, n_jobs=-1)

elastic_net_grid_search.fit(X_train_transformed, y_log_train)
train_results=[]

train_results.append(['Elastic Net', elastic_net_grid_search.best_params_, -elastic_net_grid_search.best_score_])

elastic_net_grid_search.best_params_, -elastic_net_grid_search.best_score_
best_elastic_net_reg = elastic_net_grid_search.best_estimator_

best_elastic_net_reg
feature_imp = [ col for col in zip(feature_columns, best_elastic_net_reg.coef_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp[:15]
from sklearn.svm import SVR
svr_grid_param = [{'C':list(np.linspace(0.1, 1, 10)), 'epsilon':[0.01, 0.05, 0.1, 0.5, 1]}]

svr_grid_search = GridSearchCV(SVR(kernel="poly", degree=2), svr_grid_param, cv=kf, scoring="neg_root_mean_squared_error", return_train_score=True, n_jobs=-1)

svr_grid_search.fit(X_train_transformed, y_log_train)
train_results.append(['SVR', svr_grid_search.best_params_, -svr_grid_search.best_score_])

svr_grid_search.best_params_, -svr_grid_search.best_score_
best_svr_reg = svr_grid_search.best_estimator_

best_svr_reg
from sklearn.ensemble import RandomForestRegressor
rf_grid_param = [{'max_features':[0.2, 0.4, 0.6, 'auto'], 'max_depth':[8, 12, 16, 20]}]

rf_grid_search = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), rf_grid_param, cv=kf, scoring='neg_root_mean_squared_error', return_train_score=True, n_jobs=-1)

rf_grid_search.fit( X_train_transformed, y_log_train)
train_results.append(['Random Forest', rf_grid_search.best_params_, -rf_grid_search.best_score_])

rf_grid_search.best_params_, -rf_grid_search.best_score_
best_rf_reg = rf_grid_search.best_estimator_

best_rf_reg
feature_imp = [ col for col in zip(feature_columns, best_rf_reg.feature_importances_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp[:15]
from xgboost import XGBRegressor
xgb_grid_parm=[{'max_depth':[4, 6, 8, 12], 'subsample':[0.5, 0.75, 1.0]}]

xgb_grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1), xgb_grid_parm, cv=kf, scoring="neg_root_mean_squared_error", return_train_score=True, n_jobs=-1)

xgb_grid_search.fit(X_train_transformed, y_log_train)
train_results.append(['XGBoost', xgb_grid_search.best_params_, -xgb_grid_search.best_score_])

xgb_grid_search.best_params_, -xgb_grid_search.best_score_
cvres = xgb_grid_search.cv_results_

for train_mean_score, test_mean_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):

    print(-train_mean_score, -test_mean_score, params)
best_xgb_reg = xgb_grid_search.best_estimator_

best_xgb_reg
feature_imp = [ col for col in zip(feature_columns, best_xgb_reg.feature_importances_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp[:15]
from sklearn.ensemble import StackingRegressor

from sklearn.linear_model import ElasticNetCV
base_estimators = [('elastic_net', best_elastic_net_reg), ('svr', best_svr_reg), ('rf', best_rf_reg), ('xgb', best_xgb_reg)]



stack_reg = StackingRegressor(estimators=base_estimators, final_estimator=ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], random_state=42), cv=kf, passthrough=False, n_jobs=-1)

stack_reg.fit(X_train_transformed, y_log_train)
from sklearn.model_selection import cross_val_score



stack_rmse_scores = cross_val_score(stack_reg, X_train_transformed, y_log_train, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)

stack_rmse = np.round(np.mean(-stack_rmse_scores), 4)

train_results.append(['Stacking', '', stack_rmse])
pd.set_option('display.max_colwidth', -1)



train_models_df = pd.DataFrame(train_results, columns=['Model', 'Best Paramas', 'RMSLE'])

train_models_df
results = dict()

best_models = [best_elastic_net_reg, best_svr_reg, best_rf_reg, best_xgb_reg, stack_reg]

model_names = []

model_rmse = []



for model in best_models:

    test_rmse_scores = cross_val_score(model, X_test_transformed, y_log_test, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)

    test_rmse_scores = np.round(-test_rmse_scores,4)

    test_rmse = np.round(np.mean(test_rmse_scores),4)

    model_names.append(model.__class__.__name__)

    model_rmse.append(test_rmse)
def plot_results(model_names, model_rmse):

        

    plt.figure(figsize=(12, 5))

    x_indexes = np.arange(len(model_names))     

    width = 0.15                            

    

    plt.barh(x_indexes, model_rmse)

    for i in range(len(x_indexes)):

        plt.text(x=model_rmse[i], y=x_indexes[i], s=str(model_rmse[i]), fontsize=12)

    

    plt.xlabel("Mean RMSLE", fontsize=14)

    plt.yticks(ticks=x_indexes, labels=model_names, fontsize=14)

    plt.title("Results on Test Dataset")

    plt.show()
plot_results(model_names, model_rmse)
best_model = best_models[np.argmin(model_rmse)]

best_model
y_train_pred = best_model.predict(X_train_transformed)

y_test_pred = best_model.predict(X_test_transformed)

y_train_pred = np.exp(y_train_pred)

y_test_pred = np.exp(y_test_pred)

predicted = np.concatenate([y_train_pred, y_test_pred], axis=0)

obsereved = np.concatenate([y_train, y_test], axis=0)
combine_data = pd.concat([X_train, X_test], axis=0)

combine_data['SalePrice'] = obsereved

combine_data['Predicted_SalePrice'] = predicted

combine_data.shape
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

ax = combine_data['SalePrice'].hist()

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.subplot(1, 2, 2)

ax = combine_data['Predicted_SalePrice'].hist()

for p in ax.patches:

        ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.show()
plt.figure(figsize=(12, 6))

plt.scatter(combine_data['GrLivArea'], combine_data['SalePrice'], label="Observed")

plt.scatter(combine_data['GrLivArea'], combine_data['Predicted_SalePrice'] , c='green', label="Predicted")

plt.xlabel('GrLivArea')

plt.ylabel('Sale Price')

plt.legend()

plt.show()
final_model = Pipeline([('pre_process', pre_process),

                       ('best_model', best_model)])

final_model.fit(X_train, y_log_train)
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
log_predictions = final_model.predict(test_data)

predictions = np.exp(log_predictions)
test_predictions = pd.DataFrame(test_data['Id'])

test_predictions['SalePrice'] = predictions.copy()
test_predictions.head()
test_predictions.to_csv("./submission.csv", index=False)