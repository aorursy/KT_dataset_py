import numpy as np

import pandas as pd 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import neighbors

from sklearn.svm import SVR

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from xgboost import XGBRegressor



from warnings import filterwarnings

filterwarnings('ignore')
def data_prep():

    df = pd.read_csv("../input/hitters/Hitters.csv") #calling the dataset



    #we added variables that would enable our models to explain better

    #values of some variables on year basis

    df["Mean_CAtBat"] = df["CAtBat"] / df["Years"]

    df["Mean_CHits"] = df["CHits"] / df["Years"]

    df["Mean_CHmRun"] = df["CHmRun"] / df["Years"]

    df["Mean_Cruns"] = df["CRuns"] / df["Years"]

    df["Mean_CRBI"] = df["CRBI"] / df["Years"]

    df["Mean_CWalks"] = df["CWalks"] / df["Years"]



    #variables that affect the model less were removed from the dataset

    df = df.drop(['AtBat','Hits','HmRun','Runs','RBI','Walks','Assists',

                  'Errors',"PutOuts",'League','NewLeague', 'Division'], axis=1)



    #missing data filled in according to KNN

    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors = 4)

    df_filled = imputer.fit_transform(df)

    df = pd.DataFrame(df_filled,columns = df.columns)



    #Suppression for contradictory observations in Salary variable

    Q1 = df.Salary.quantile(0.25)

    Q3 = df.Salary.quantile(0.75)

    IQR = Q3-Q1

    lower = Q1 - 1.5*IQR

    upper = Q3 + 1.5*IQR

    df.loc[df["Salary"] > upper,"Salary"] = upper

    

    y = df["Salary"]

    X = df.drop("Salary",axis=1)

    

    #standardizing distributions

    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                        test_size=0.20, 

                                                        random_state=46)

    

    return X_train, X_test, y_train, y_test
def modeling(X_train, X_test, y_train, y_test):

    #KNN

    

    knn_params = {'n_neighbors': 6}



    knn_tuned = KNeighborsRegressor(**knn_params).fit(X_train, y_train)

    

    #test error

    y_pred = knn_tuned.predict(X_test)

    knn_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse KNN")

    print(knn_final)

    

    #SVR

    svr_params = {'C': 1000}



    svr_tuned = SVR(**svr_params).fit(X_train, y_train)

    

    #test error

    y_pred = svr_tuned.predict(X_test)

    svr_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse SVR")

    print(svr_final)

    

    #CART

    cart_params = {'max_depth': 5, 'min_samples_split': 50}

    

    cart_tuned = DecisionTreeRegressor(**cart_params).fit(X_train, y_train)

    

    #test error

    y_pred = cart_tuned.predict(X_test)

    cart_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse CART")

    print(cart_final)

    

    

    #RANDOM FOREST

    rf_params = {'max_depth': 5, 'max_features': 2, 'min_samples_split': 2, 'n_estimators': 900}



    rf_tuned = RandomForestRegressor(**rf_params).fit(X_train, y_train)



    #test error

    y_pred = rf_tuned.predict(X_test)

    rf_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse RF")

    print(rf_final)

    

    #GBM

    

    gbm_params = {'learning_rate': 0.01,

    'loss': 'lad', 

    'max_depth': 5,

    'n_estimators': 500,

    'subsample': 0.5}

    

    gbm_tuned = GradientBoostingRegressor(**gbm_params).fit(X_train, y_train)



    #test error

    y_pred = gbm_tuned.predict(X_test)

    gbm_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse GBM")

    print(gbm_final)

    

    #XGBoost    

    xgb_params = {'colsample_bytree': 1,

    'learning_rate': 0.01,

    'max_depth': 2,

    'n_estimators': 500}

    

    xgb_tuned = XGBRegressor(**xgb_params).fit(X_train, y_train)

    

    y_pred = xgb_tuned.predict(X_test)

    xgb_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse XGBoost")

    print(xgb_final)

    

    #Light GBM

    lgbm_params = {'colsample_bytree': 1,

    'learning_rate': 0.01,

    'max_depth': 5,

    'n_estimators': 200}

    

    lgbm_tuned = LGBMRegressor(**lgbm_params).fit(X_train, y_train)

    

    y_pred = lgbm_tuned.predict(X_test)

    lbg_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse LightGBM")

    print(lbg_final)

    

    #CatBoost

    catb_params = {'depth': 10, 'iterations': 500, 'learning_rate': 0.1}

    

    catb_tuned = CatBoostRegressor(**catb_params).fit(X_train, y_train)

    

    y_pred = catb_tuned.predict(X_test)

    cat_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse CatBoost")

    print(cat_final)

    

    #Neural Networks

    mlp_params = {'alpha': 0.1, 'hidden_layer_sizes': (1000, 100, 10)}

    

    mlp_tuned = MLPRegressor(**mlp_params).fit(X_train, y_train)

    

    y_pred = mlp_tuned.predict(X_test)

    neural_final = np.sqrt(mean_squared_error(y_test, y_pred))

    print("final rmse Neural Networks")

    print(neural_final)

    

    results = pd.DataFrame({"Test Tuned Error":[knn_final, svr_final, cart_final, 

                                                rf_final,gbm_final, xgb_final, 

                                                lbg_final, cat_final, neural_final]})

    results.index= ["KNN", "SVR","CART","Random Forests","GBM","XGBoost", "LightGBM", 

                    "CatBoost", "Neural Networks"]

    print(results)
X_train, X_test, y_train, y_test = data_prep()
modeling(X_train, X_test, y_train, y_test)