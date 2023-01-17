#!pip install xgboost
#!pip install lightgbm
#!pip install catboost
#close warnings
import warnings
warnings.simplefilter(action='ignore')

#import libraries for linear models
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

#import libraries for non-linear models (additional to linear models)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
#read dataset
df_hitters=pd.read_csv("../input/hitters-baseball-data/Hitters.csv")
#copy dataset in case of reloading dataset immediately
df=df_hitters.copy()
df.head()
df.shape
df.info()
#get a summary of descriptive statistics
df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
#check number of missing values
df.isnull().sum().sort_values(ascending=False).head()
#get columns names according to variable types >> int64, float64, object
#it helps us to use fancy indexes, manipulate different types separately and relatively reduce memory usage instead of creating two dataframes
cat_cols=[col for col in df.columns if df[col].dtype=='object']
num_cols=[col for col in df.columns if col not in cat_cols]
#as a summary, there are 20 variables and 'Salary' is target variable. 
#rest 19 variables (16 are numerical, 3 are categorical/object) are independent variables.
#there are 59 null values only in target variable. 
#detect outliers for target variable
sns.boxplot(x=df['Salary'])
#before filling missing values in salary, trim/correct them slightly
upper_limit=df['Salary'].quantile(0.95)
outliers_upper=df[df["Salary"] > upper_limit]
df.loc[df["Salary"] > upper_limit, "Salary"] = upper_limit
#check outliers again
sns.boxplot(x=df['Salary'])
#make a decision about handling NaN values: 1. Drop them 2. Fill them.
#there are 59 missing values in 322 rows. 18% is not a small ratio and can affect results. Filling seems a better way.
#if dropping is being preferred, code: df.dropna(inplace=True)
#fill missing values according to categorical variables and mean
df['Salary']=df.groupby(['League','Division'])['Salary'].transform(lambda x: x.fillna(x.mean()))
#no missing values:)
df.isnull().sum().sort_values(ascending=False).head()
#check categorical variables and number of subcategories
print(df['League'].value_counts())
print(df['Division'].value_counts())
print(df['NewLeague'].value_counts())
#all categorical variables consist of two subcategories. Then, use Label Encoding (LE).
#LE assigns values as 0-1 (means to model: coequal variables)
le_League=LabelEncoder()
le_Division=LabelEncoder()
le_NewLeague=LabelEncoder()
df['League']=le_League.fit_transform(df['League'])
df['Division']=le_Division.fit_transform(df['Division'])
df['NewLeague']=le_NewLeague.fit_transform(df['NewLeague'])
#to get original categorical values below inverse code can be used
#le_League.inverse_transform(df['League'])
#le_Division.inverse_transform(df['League'])
#le_NewLeague.inverse_transform(df['League'])
df.head()
#Feature Scaling: Normalization of Numerical Variables (Except from target variable)
#First remove Salary variable from numerical columns - you don't want to normalize it
num_cols.remove('Salary')
norm_num_df=preprocessing.normalize(df[num_cols])
norm_num_df=pd.DataFrame(norm_num_df, columns=num_cols)
norm_num_df.head()
#change dataframe with normalized variables
df=pd.concat([norm_num_df, df[cat_cols], df['Salary']], axis=1)
df.head()
#get dependent and independent values
y=df[['Salary']] #dependent/target variable
X=df.drop(['Salary'], axis=1)  #independent variable
#divide dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
#for linear models, according to models' documentation and logic of penalty, it is advised that alpha, l1_ratio values are between 0 and 1.
#if you define alpha=0 it turns models to linear regression w/o penalties:)
#Lineer: Primitive Model
lin_reg=LinearRegression().fit(X_train, y_train)
y_pred_lin_reg=lin_reg.predict(X_test)
lin_reg_rmse=np.sqrt(mean_squared_error(y_test, y_pred_lin_reg))
lin_reg_rmse
#Ridge: Primitive Model
rid_reg=Ridge().fit(X_train, y_train)
y_pred_rid_reg=rid_reg.predict(X_test)
rid_reg_rmse=np.sqrt(mean_squared_error(y_test, y_pred_rid_reg))
print(rid_reg_rmse)

#Ridge: CV Model
alpha_rid = [0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]
rid_reg_cv=RidgeCV(alphas = alpha_rid, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
rid_reg_cv.fit(X_train, y_train)
print(rid_reg_cv.alpha_)

#Ridge: Tuned Model
rid_reg_tuned=Ridge(rid_reg_cv.alpha_).fit(X_train, y_train)
y_pred_rid_reg_tuned=rid_reg_tuned.predict(X_test)
rid_reg_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_rid_reg_tuned))
rid_reg_tuned_rmse
#Lasso: Primitive Model
las_reg=Lasso().fit(X_train, y_train)
y_pred_las_reg=las_reg.predict(X_test)
las_reg_rmse=np.sqrt(mean_squared_error(y_test, y_pred_las_reg))
print(las_reg_rmse)

#Lasso: CV Model
alpha_las = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1, 1.1]
las_reg_cv = LassoCV(alphas = alpha_las, cv = 10, normalize = True)
las_reg_cv.fit(X_train, y_train)
print(las_reg_cv.alpha_)

#Lasso: Tuned Model
las_reg_tuned = Lasso(alpha = las_reg_cv.alpha_).fit(X_train,y_train)
y_pred_las_reg_tuned = las_reg_tuned.predict(X_test)
las_reg_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_las_reg_tuned))
las_reg_tuned_rmse
#ElasticNet: Primitive Model
enet_reg=ElasticNet().fit(X_train, y_train)
y_pred_enet_reg=enet_reg.predict(X_test)
enet_reg_rmse=np.sqrt(mean_squared_error(y_test, y_pred_enet_reg))
print(enet_reg_rmse)

#ElasticNet: CV Model
enet_reg_params = {"l1_ratio": [0.001, 0.01, 0.1, 0.5, 0.9, 1, 1.1],
              "alpha":[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 1.1]}
enet_reg_cv = GridSearchCV(enet_reg, enet_reg_params, cv = 10).fit(X, y)
print(enet_reg_cv.best_params_)

#ElasticNet: Tuned Model
enet_reg_tuned = ElasticNet(**enet_reg_cv.best_params_).fit(X_train,y_train)
y_pred_enet_reg_tuned = enet_reg_tuned.predict(X_test)
enet_reg_tuned_rmse = np.sqrt(mean_squared_error(y_test,y_pred_enet_reg_tuned))
enet_reg_tuned_rmse
#KNN: Primitive Model
knn_model=KNeighborsRegressor().fit(X_train, y_train)
print(knn_model)
y_pred_knn_model=knn_model.predict(X_test)
y_pred_knn_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_knn_model))
y_pred_knn_model_rmse
#KNN: CV Model
knn_params={"n_neighbors": np.arange(2,20,1)}
knn_cv_model=GridSearchCV(knn_model, knn_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
print(knn_cv_model.best_params_)
#KNN: Tuned Model
knn_tuned=KNeighborsRegressor(**knn_cv_model.best_params_).fit(X_train, y_train)
y_pred_knn_tuned=knn_tuned.predict(X_test)
y_pred_knn_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_knn_tuned))
y_pred_knn_tuned_rmse
#SVR: Primitive Model
svr_model=SVR().fit(X_train, y_train)
print(svr_model)
y_pred_svr_model=svr_model.predict(X_test)
y_pred_svr_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_svr_model))
y_pred_svr_model_rmse
#SVR: CV Model
svr_params={"C": (0.01, 0.1, 0.5, 0.9, 1),
           "kernel": ('rbf', 'linear')}
svr_cv_model=GridSearchCV(svr_model, svr_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
print(svr_cv_model.best_params_)
#SVR: Tuned Model
svr_tuned=SVR(**svr_cv_model.best_params_).fit(X_train, y_train)
y_pred_svr_tuned=svr_tuned.predict(X_test)
y_pred_svr_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_svr_tuned))
y_pred_svr_tuned_rmse
#ANN: Primitive Model
#independent variables were already scaled/normalized, then no need to be scaled again
ann_model=MLPRegressor(random_state=42).fit(X_train, y_train)
print(ann_model)
y_pred_ann_model=ann_model.predict(X_test)
y_pred_ann_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_ann_model))
y_pred_ann_model_rmse
#ANN: CV Model
ann_params = {"alpha": [0.001, 0.01, 0.1, 0.2, 0.3, 0.5], 
             "hidden_layer_sizes": [(5,5), (10,10), (20,20), (100,100)],
             "solver": ['lbfgs', 'sgd', 'adam']}
ann_cv_model=GridSearchCV(ann_model, ann_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
print(ann_cv_model.best_params_)
#ANN: Tuned Model
ann_tuned=MLPRegressor(**ann_cv_model.best_params_, random_state=42).fit(X_train, y_train)
y_pred_ann_tuned=ann_tuned.predict(X_test)
y_pred_ann_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_ann_tuned))
y_pred_ann_tuned_rmse
#CART: Primitive Model
cart_model = DecisionTreeRegressor(random_state=42).fit(X_train,y_train)
print(cart_model)
y_pred_cart_model=cart_model.predict(X_test)
y_pred_cart_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_cart_model))
y_pred_cart_model_rmse
#CART: CV Model
cart_params = {"max_depth": [2, 3, 4, 5, 10, None],
              "min_samples_split": [2, 5, 10, 12, 20]}
cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10, n_jobs = -1, verbose=2).fit(X_train, y_train)
print(cart_cv_model.best_params_)

#CART: Tuned Model
cart_tuned=DecisionTreeRegressor(**cart_cv_model.best_params_, random_state=42).fit(X_train, y_train)
y_pred_cart_tuned=cart_tuned.predict(X_test)
y_pred_cart_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_cart_tuned))
y_pred_cart_tuned_rmse
#RF: Primitive Model
rf_model = RandomForestRegressor(random_state=42).fit(X_train,y_train)
print(rf_model)
y_pred_rf_model=rf_model.predict(X_test)
y_pred_rf_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_rf_model))
y_pred_rf_model_rmse
#RF: CV Model
rf_params = {"max_depth": [5, 8, 10, None],
             "max_features": [3, 5, 10, 15, 17],
             "min_samples_split": [2, 3, 5, 10],
             "n_estimators": [100, 200, 500]}
rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1, verbose=2).fit(X_train, y_train)
print(rf_cv_model.best_params_)

#RF: Tuned Model
rf_tuned=RandomForestRegressor(**rf_cv_model.best_params_, random_state=42).fit(X_train, y_train)
y_pred_rf_tuned=rf_tuned.predict(X_test)
y_pred_rf_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
y_pred_rf_tuned_rmse
#GBM: Primitive Model
gbm_model = GradientBoostingRegressor(random_state=42).fit(X_train,y_train)
print(gbm_model)
y_pred_gbm_model=gbm_model.predict(X_test)
y_pred_gbm_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_gbm_model))
y_pred_gbm_model_rmse
#GBM: CV Model
gbm_params = {"learning_rate": [0.01, 0.1, 0.5],
             "max_depth": [2, 3, 4],
             "n_estimators": [1000, 1500, 2000],
             "subsample": [0.2, 0.3, 0.5],
             "loss": ["ls","lad","quantile"]}
gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv = 10, n_jobs = -1, verbose=2).fit(X_train, y_train)
print(gbm_cv_model.best_params_)

#GBM: Tuned Model
gbm_tuned=GradientBoostingRegressor(**gbm_cv_model.best_params_, random_state=42).fit(X_train, y_train)
y_pred_gbm_tuned=gbm_tuned.predict(X_test)
y_pred_gbm_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_gbm_tuned))
y_pred_gbm_tuned_rmse
#XGB: Primitive Model
xgb_model = XGBRegressor(random_state=42).fit(X_train,y_train)
print(xgb_model)
y_pred_xgb_model=xgb_model.predict(X_test)
y_pred_xgb_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_xgb_model))
y_pred_xgb_model_rmse
#XGB: CV Model
xgb_params = {"learning_rate": [0.01, 0.1, 0.5],
             "max_depth": [2, 3, 5, 8],
             "n_estimators": [100, 200, 1000],
             "colsample_bytree": [0.5, 0.8, 1]}
xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv = 10, n_jobs = -1, verbose=2).fit(X_train, y_train)
print(xgb_cv_model.best_params_)

#XGB: Tuned Model
xgb_tuned=XGBRegressor(**xgb_cv_model.best_params_, random_state=42).fit(X_train, y_train)
y_pred_xgb_tuned=xgb_tuned.predict(X_test)
y_pred_xgb_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_xgb_tuned))
y_pred_xgb_tuned_rmse
#LGBM: Primitive Model
lgbm_model = LGBMRegressor(random_state=42).fit(X_train,y_train)
print(lgbm_model)
y_pred_lgbm_model=lgbm_model.predict(X_test)
y_pred_lgbm_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_lgbm_model))
y_pred_lgbm_model_rmse
#LGBM: CV Model
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
             "max_depth": [2, 3, 4, 5],
             "n_estimators": [200, 500, 700, 1000],
             "colsample_bytree": [0.6, 0.7, 0.8, 1]}
lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv = 10, n_jobs = -1, verbose=2).fit(X_train, y_train)
print(lgbm_cv_model.best_params_)

#LGBM: Tuned Model
lgbm_tuned=LGBMRegressor(**lgbm_cv_model.best_params_, random_state=42).fit(X_train, y_train)
y_pred_lgbm_tuned=lgbm_tuned.predict(X_test)
y_pred_lgbm_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_lgbm_tuned))
y_pred_lgbm_tuned_rmse
#CATB: Primitive Model
catb_model = CatBoostRegressor(verbose=False, random_state=42).fit(X_train,y_train)
print(catb_model)
y_pred_catb_model=catb_model.predict(X_test)
y_pred_catb_model_rmse=np.sqrt(mean_squared_error(y_test, y_pred_catb_model))
y_pred_catb_model_rmse
#CATB: CV Model
catb_params = {"learning_rate": [0.01, 0.1, 0.5],
               "iterations": [100, 200, 500],
              "depth": [3, 5, 8]}
catb_cv_model = GridSearchCV(catb_model, catb_params, cv = 10, n_jobs = -1).fit(X_train, y_train)
print(catb_cv_model.best_params_)

#CATB: Tuned Model
catb_tuned=CatBoostRegressor(**catb_cv_model.best_params_, verbose=False, random_state=42).fit(X_train, y_train)
y_pred_catb_tuned=lgbm_tuned.predict(X_test)
y_pred_catb_tuned_rmse=np.sqrt(mean_squared_error(y_test, y_pred_catb_tuned))
y_pred_catb_tuned_rmse
pd.set_option('display.max_colwidth', -1)
results = pd.DataFrame({"Model Name": ["Primitive Test Errors", "Tuning Params", "Tuned Test Errors"],
                        "Linear Reg": [lin_reg_rmse, np.nan, np.nan],
                        "Ridge Reg": [rid_reg_rmse, rid_reg_cv.alpha_, rid_reg_tuned_rmse],
                        "Lasso Reg": [las_reg_rmse, las_reg_cv.alpha_, las_reg_tuned_rmse],
                        "ElasticNet Reg": [enet_reg_rmse, enet_reg_cv.best_params_, las_reg_tuned_rmse],
                        "KNN": [y_pred_knn_model_rmse, knn_cv_model.best_params_, y_pred_knn_tuned_rmse],
                        "SVR": [y_pred_svr_model_rmse, svr_cv_model.best_params_, y_pred_svr_tuned_rmse],
                        "ANN": [y_pred_ann_model_rmse, ann_cv_model.best_params_, y_pred_ann_tuned_rmse],
                        "CART": [y_pred_cart_model_rmse, cart_cv_model.best_params_, y_pred_cart_tuned_rmse],
                        "RF": [y_pred_rf_model_rmse, rf_cv_model.best_params_, y_pred_rf_tuned_rmse],
                        "GBM": [y_pred_gbm_model_rmse, gbm_cv_model.best_params_, y_pred_gbm_tuned_rmse],
                        "XGB": [y_pred_xgb_model_rmse, xgb_cv_model.best_params_, y_pred_xgb_tuned_rmse],
                        "LGBM": [y_pred_lgbm_model_rmse, lgbm_cv_model.best_params_, y_pred_lgbm_tuned_rmse],
                        "CATB": [y_pred_catb_model_rmse, catb_cv_model.best_params_, y_pred_catb_tuned_rmse]
                        })

results.set_index("Model Name", inplace=True)
results.T.sort_values(by="Tuned Test Errors", ascending=True)
#GBM: Feature Importances & Visualization
importance=pd.DataFrame({'importance': gbm_tuned.feature_importances_ * 100},
                       index=X_train.columns)

importance.sort_values(by='importance', axis=0, ascending=True). plot(kind='barh', color='g')

plt.xlabel('Variable Importances')
plt.gca().legend_=None
#RF: Feature Importances & Visualization
importance=pd.DataFrame({'importance': rf_tuned.feature_importances_ * 100},
                       index=X_train.columns)

importance.sort_values(by='importance', axis=0, ascending=True). plot(kind='barh', color='g')

plt.xlabel('Variable Importances')
plt.gca().legend_=None
#XGB: Feature Importances & Visualization
importance=pd.DataFrame({'importance': xgb_tuned.feature_importances_ * 100},
                       index=X_train.columns)

importance.sort_values(by='importance', axis=0, ascending=True). plot(kind='barh', color='g')

plt.xlabel('Variable Importances')
plt.gca().legend_=None