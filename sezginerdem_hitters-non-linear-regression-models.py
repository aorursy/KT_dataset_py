import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from warnings import filterwarnings
filterwarnings('ignore')
hitters = pd.read_csv('../input/hitters/Hitters.csv')
hitters.head()
#Exploratory Data Analysis
#Structural information of the data set
hitters.info()
hitters.isnull().sum()
dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']]) 
dummies.head()
X_ = hitters.drop(['League', 'Division', 'NewLeague'], axis=1).astype('float64') 

hitters = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1) 

hitters.info()
hitters.describe().T
df_1 = hitters.dropna()
df_1.head()
df_1.info()
df = hitters.copy()
df['Salary'].fillna(df['Salary'].mean(), inplace = True) 
df_2 = df.copy()
df_2.info()
hitters.head()
null = hitters[hitters['Salary'].isnull()]
# Selection of observations with missing data
null.head()
df = hitters.dropna() #Delete observations with missing data
X_train = df.drop('Salary', axis = 1) #Train set definition
X_train.head()
y_train = df[['Salary']] #Determination of the dependent variable of the train set
y_train.head()
X_test = null.drop('Salary', axis = 1) #Defining observations with missing data in the data set as a test set
X_test.head()
gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
gbm_model_pred_test = gbm_model.predict(X_test)
gbm_model_pred_test
X_test['Salary'] = gbm_model_pred_test
df_3 = pd.concat([df, X_test], ignore_index = True)
df_3.head()
df_3.info()
df_3.describe().T
df_3.info()
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df_3)
df_scores = clf.negative_outlier_factor_
df_scores[0:20]
np.sort(df_scores)
np.sort(df_scores)[16]
threshold_value = np.sort(df_scores)[16]
threshold_value
outlier_df = df_scores > threshold_value
df_3[df_scores == threshold_value]

pressure_value = df_3[df_scores == threshold_value]
outlier = df_3[~outlier_df] 
outlier.to_records(index=False)
res = outlier.to_records(index=False)
res[:] = pressure_value.to_records(index = False)
outlier = pd.DataFrame(res, index = df_3[~outlier_df].index)
outlier.describe().T
n_outlier = df_3[outlier_df]
n_outlier.describe().T
df_4 = pd.concat([n_outlier, outlier], ignore_index = True)
df_4.describe().T
df_3.info()
df_5 = df_3[df_scores > threshold_value]
df_5.info()
df.info()
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df)
df6_scores = clf.negative_outlier_factor_
df6_scores[0:20]
np.sort(df6_scores)

np.sort(df6_scores)[8]
threshold_value6 = np.sort(df6_scores)[8]
threshold_value6
outlier_df6 = df6_scores > threshold_value6
outlier_df6
df[df6_scores == threshold_value6]
pressure_value6 = df[df6_scores == threshold_value6]
outlier6 = df[~outlier_df6] 

outlier6.to_records(index=False)
res6 = outlier6.to_records(index=False)

res6[:] = pressure_value6.to_records(index = False)

n_outlier6 = df[outlier_df6]
n_outlier6.describe().T
outlier6 = pd.DataFrame(res6, index = df[~outlier_df6].index)
outlier6.describe().T
df_6 = pd.concat([n_outlier6, outlier6], ignore_index = True)
df_6.describe().T
df_7 = n_outlier6
df_7.info()
df_8 = hitters.copy()
df_8.info()
cat_df = df_8.select_dtypes(include=["uint8"])
cat_df.head()
print(cat_df.League_N.unique())
print(cat_df["League_N"].value_counts().count())
print(cat_df["League_N"].value_counts())
print(df_8["League_N"].value_counts().plot.barh())
df_8.groupby('League_N')['Salary'].mean()
print(cat_df.Division_W.unique())
print(cat_df["Division_W"].value_counts().count())
print(cat_df["Division_W"].value_counts())
print(df_8["Division_W"].value_counts().plot.barh())
df_8.groupby('Division_W')['Salary'].mean()
print(cat_df.NewLeague_N.unique())
print(cat_df["NewLeague_N"].value_counts().count())
print(cat_df["NewLeague_N"].value_counts())
print(df_8["NewLeague_N"].value_counts().plot.barh())
df_8.groupby('NewLeague_N')['Salary'].mean()
Experience = []
for ex in df_8['Years']:
    if ex < 5:
        Experience.append(1)
    elif (ex >= 5) & (ex < 10):
        Experience.append(2)
    elif (ex >= 10) & (ex < 15):
        Experience.append(3)
    elif (ex >= 15) & (ex < 20):
        Experience.append(4)
    else:
        Experience.append(5)
df_8['Experience'] = Experience
df_8.groupby(['League_N', 'Division_W', 'NewLeague_N'])['Salary'].mean()
df_8.groupby(['League_N', 'Division_W', 'NewLeague_N', 'Experience'])['Salary'].mean()
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 1), "Salary"] = 145.961538
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 2), "Salary"] = 774.434536
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 3), "Salary"] = 918.073533
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 4), "Salary"] = 614.375000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 2), "Salary"] = 850.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 3), "Salary"] = 833.333333
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 1), "Salary"] = 203.821429
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 2), "Salary"] = 528.108696
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 3), "Salary"] = 786.916700
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 4), "Salary"] = 479.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 1), "Salary"] = 96.666667
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 0) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 3), "Salary"] = 825.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 1), "Salary"] = 70.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 2), "Salary"] = 525.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 3), "Salary"] = 500.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 4), "Salary"] = 1050.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 1), "Salary"] = 313.753320
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 2), "Salary"] = 776.095190
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 3), "Salary"] = 949.010143
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 0) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 4), "Salary"] = 486.111000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 1), "Salary"] = 565.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 2), "Salary"] = 405.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 0) & (df_8['Experience'] == 3), "Salary"] = 250.000000
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 1), "Salary"] = 188.138889
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 2), "Salary"] = 538.114053
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 3), "Salary"] = 723.452429
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 4), "Salary"] = 763.666600
df_8.loc[(df_8["Salary"].isnull()) & (df_8["League_N"] == 1) & (df_8['Division_W'] == 1) & (df_8["NewLeague_N"] == 1) & (df_8['Experience'] == 5), "Salary"] = 475.000000

df_8.info()
df_8['AtBat_rate'] = df_8["CAtBat"] / df_8["Years"]
df_8['Hits_rate'] = df_8["CHits"] / df_8["Years"]
df_8['HmRun_rate'] = df_8["CHmRun"] / df_8["Years"]
df_8['Runs_rate'] = df_8["CRuns"] / df_8["Years"]
df_8['RBI_rate'] = df_8["CRBI"] / df_8["Years"]
df_8['Walks_rate'] = df_8["CWalks"] / df_8["Years"]

df_8['1986_AtBat_rate'] = df_8["AtBat"] / df_8["CAtBat"]
df_8['1986_Hits_rate'] = df_8["Hits"] / df_8["CHits"]
df_8['1986_HmRun_rate'] = df_8["HmRun"] / df_8["CHmRun"]
df_8['1986_Runs_rate'] = df_8["Runs"] / df_8["CRuns"]
df_8['1986_RBI_rate'] = df_8["RBI"] / df_8["CRBI"]
df_8['1986_Walks_rate'] = df_8["Walks"] / df_8["CWalks"]
df_8.info()
df_8 = df_8.dropna()
df_8.info()
def compML(df, y, alg):
    #train-test distinction
    y = df[y]
    X = df.drop('Salary', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=46)
    #modeelling
    model = alg().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    model_name = alg.__name__
    print("  for data set  ", model_name, " Model Test Error: ",RMSE)
models = [LinearRegression,
          Ridge,
          Lasso,
          ElasticNet,
          LGBMRegressor, 
          XGBRegressor, 
          GradientBoostingRegressor, 
          RandomForestRegressor, 
          DecisionTreeRegressor,
          MLPRegressor,
          KNeighborsRegressor, 
          SVR]
for i in models:
    compML(df_1, "Salary", i)
for i in models:
    compML(df_2, "Salary", i)
for i in models:
    compML(df_3, "Salary", i)
for i in models:
    compML(df_4, "Salary", i)
for i in models:
    compML(df_5, "Salary", i)
for i in models:
    compML(df_6, "Salary", i)
for i in models:
    compML(df_7, "Salary", i)
for i in models:
    compML(df_8, "Salary", i)
df_4.head()
y = df_4['Salary']
X = df_4.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
knn_model = KNeighborsRegressor().fit(X_train, y_train)
knn = KNeighborsRegressor()
knn_params = {"n_neighbors": np.arange(1,30,1)}
knn_cv_model = GridSearchCV(knn, knn_params, cv = 10).fit(X_train, y_train)
knn_cv_model.best_params_
knn_tuned = KNeighborsRegressor(n_neighbors = knn_cv_model.best_params_["n_neighbors"]).fit(X_train, y_train)
knn_tuned_y_pred = knn_tuned.predict(X_test)
knn_tuned_RMSE = np.sqrt(mean_squared_error(y_test, knn_tuned_y_pred))
knn_tuned_RMSE
df_4.head()
y = df_4['Salary']
X = df_4.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
svr_model = SVR("linear") 
svr_params = {"C": [0.1,0.5,1,3]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv = 5, verbose = 2, n_jobs = -1).fit(X_train, y_train)
svr_cv_model.best_params_
svr_tuned = SVR("linear", C = 3).fit(X_train, y_train)
svr_model_y_pred = svr_tuned.predict(X_test)
svr_model_tuned_RMSE = np.sqrt(mean_squared_error(y_test, svr_model_y_pred))
svr_model_tuned_RMSE
df_4.head()
y = df_4['Salary']
X = df_4.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)
mlp_model = MLPRegressor().fit(X_train_scaled, y_train)
mlp_params = {"alpha": [0.1, 0.01, 0.02, 0.001, 0.0001], 
             "hidden_layer_sizes": [(10,20), (5,5), (100,100)]}
mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10, verbose = 2, n_jobs = -1).fit(X_train_scaled, y_train)
mlp_cv_model.best_params_
mlp_tuned = MLPRegressor(alpha = 0.001, hidden_layer_sizes = (100,100)).fit(X_train_scaled, y_train)
mlp_y_pred = mlp_tuned.predict(X_test_scaled)
mlp_tuned_RMSE = np.sqrt(mean_squared_error(y_test, mlp_y_pred))
mlp_tuned_RMSE
df_4.head()
y = df_4['Salary']
X = df_4.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
cart_model = DecisionTreeRegressor()
cart_model.fit(X_train, y_train)
cart_params = {"max_depth": [2,3,4,5,10,20],
              "min_samples_split": [2,10,5,30,50,10]}
cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10, verbose = 2, n_jobs = -1).fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeRegressor(max_depth = 4, min_samples_split = 2).fit(X_train, y_train)
cart_model_y_pred = cart_tuned.predict(X_test)
cart_tuned_RMSE = np.sqrt(mean_squared_error(y_test, cart_model_y_pred))
cart_tuned_RMSE
df_4.head()
y = df_4['Salary']
X = df_4.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
rf_model = RandomForestRegressor(random_state = 46).fit(X_train, y_train)
rf_model
rf_params = {"max_depth": [5,8,10],
            "max_features": [2,5,10],
            "n_estimators": [200, 500, 1000, 2000],
            "min_samples_split": [2,10,80,100]}
rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
rf_cv_model.best_params_
rf_model = RandomForestRegressor(random_state = 46, 
                                 max_depth = 8,
                                max_features = 5,
                                min_samples_split = 2,
                                 n_estimators = 500)
rf_tuned = rf_model.fit(X_train, y_train)
rf_y_pred = rf_tuned.predict(X_test)
rf_tuned_RMSE = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_tuned_RMSE
rf_tuned.feature_importances_*100
Importance = pd.DataFrame({'Importance':rf_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None
df_4.head()
y = df_4['Salary']
X = df_4.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
gbm_model
gbm_params = {"learning_rate": [0.001,0.1,0.01],
             "max_depth": [3,5,8],
             "n_estimators": [100,200,500],
             "subsample": [1,0.5,0.8],
             "loss": ["ls","lad","quantile"]}
gbm_cv_model = GridSearchCV(gbm_model, 
                            gbm_params, 
                            cv = 10, 
                            n_jobs=-1, 
                            verbose = 2).fit(X_train, y_train)
gbm_cv_model.best_params_
gbm_tuned = GradientBoostingRegressor(learning_rate = 0.1,
                                     loss = "lad",
                                     max_depth = 3,
                                     n_estimators = 100,
                                     subsample = 1).fit(X_train, y_train)
gbm_tuned_y_pred = gbm_tuned.predict(X_test)
gbm_tuned_RMSE = np.sqrt(mean_squared_error(y_test, gbm_tuned_y_pred))
gbm_tuned_RMSE
Importance = pd.DataFrame({'Importance':gbm_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None
y = df_4['Salary']
X = df_4.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
xgb = XGBRegressor()
xgb
xgb_params = {"learning_rate": [0.1,0.01,0.5],
             "max_depth": [2,3,4,5,8],
             "n_estimators": [100,200,500,1000],
             "colsample_bytree": [0.4,0.7,1]}
xgb_cv_model  = GridSearchCV(xgb,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
xgb_cv_model.best_params_
xgb_tuned = XGBRegressor(colsample_bytree = 0.4, 
                         learning_rate = 0.1, 
                         max_depth = 4, 
                         n_estimators = 100).fit(X_train, y_train)
xgb_tuned_y_pred = xgb_tuned.predict(X_test)
xgb_tuned_RMSE = np.sqrt(mean_squared_error(y_test, xgb_tuned_y_pred))
xgb_tuned_RMSE
df_4.head()
y = df_4['Salary']
X = df_4.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
lgb_model = LGBMRegressor()
lgb_model
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,2,3,4,5,6,7,8,9,10]}
lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.1, 
                          max_depth = 2, 
                          n_estimators = 200).fit(X_train, y_train)
lgbm_tuned_y_pred = lgbm_tuned.predict(X_test)
lgbm_tuned_RMSE = np.sqrt(mean_squared_error(y_test, lgbm_tuned_y_pred))
lgbm_tuned_RMSE
cat_df = df_4
y = cat_df['Salary']
X = cat_df.drop('Salary', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
catb_model = CatBoostRegressor()
catb_params = {"iterations": [200,500,100],
              "learning_rate": [0.01,0.1],
              "depth": [3,6,8]}
catb_cv_model = GridSearchCV(catb_model, 
                           catb_params, 
                           cv = 5, 
                           n_jobs = -1, 
                           verbose = 2).fit(X_train, y_train)
catb_cv_model.best_params_
catb_tuned = CatBoostRegressor(depth = 6, iterations = 500, learning_rate = 0.01).fit(X_train, y_train)
catb_tuned_y_pred = catb_tuned.predict(X_test)
catb_tuned_RMSE = np.sqrt(mean_squared_error(y_test, catb_tuned_y_pred))
catb_tuned_RMSE