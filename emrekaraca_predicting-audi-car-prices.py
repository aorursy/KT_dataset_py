from warnings import filterwarnings

filterwarnings('ignore')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.metrics import r2_score,mean_squared_error
evaluation = pd.DataFrame({

    "Model":[],

    "R2 Score (train)":[],

    "Adjusted R2 Score (train)":[],

    "R2 Score (test)":[],

    "Adjusted R2 Score (test)":[],

    "Root Mean Squared Error(RMSE) (train)":[],

    "Root Mean Squared Error(RMSE) (test)":[],

    "R2 Score (5-Fold Cross Validation)":[],

    "Root Mean Squared Error(RMSE) (5-Fold Cross Validation)":[]

})
# Function to calculate adjusted r2 score:

# where  n  is the number of instances and  k  is the number of features.

def adjustedR2(r2_score,n,k):

    return 1-(((1-r2_score)*(n-1))/(n-k-1))
# Model Evaluation Function



def evaluateModel(model,X_train,X_test,y_train,y_test,model_name):

    

    if(model_name=="Polynomial Regression"):

        n_train=X_train.shape[1]

        n_test=X_test.shape[1]

    else:

        n_train=len(X_train.columns)

        n_test=len(X_test.columns)

        

    y_predict_test = model.predict(X_test)

    y_predict_train = model.predict(X_train)

    

    r2_score_train = float(format(r2_score(y_train,y_predict_train),'.3f'))

    

    r2_score_test = float(format(r2_score(y_test,y_predict_test),'.3f'))

    

    rmse_train = float(format(np.sqrt(mean_squared_error(y_train,y_predict_train)),'.3f'))

    

    rmse_test = np.sqrt(mean_squared_error(y_test,y_predict_test))

    

    ad_r2_score_train = float(format(adjustedR2(r2_score_train,X_train.shape[0],n_train),'.3f'))

    

    ad_r2_score_test = float(format(adjustedR2(r2_score_test,X_test.shape[0],n_test),'.3f'))

                              

    r2_score_mean = float(format(cross_val_score(model,X_train,y_train,cv=5).mean(),'.3f'))

                              

    rmse_mean = -float(format(cross_val_score(model,X_train,y_train,cv=5,scoring="neg_root_mean_squared_error").mean(),'.3f'))

    

    r = evaluation.shape[0]

    evaluation.loc[r]=[model_name,

                       r2_score_train,ad_r2_score_train,

                       r2_score_test,ad_r2_score_test,

                       rmse_train,rmse_test,

                      r2_score_mean,

                       rmse_mean]

    

    return evaluation.sort_values(by = 'Root Mean Squared Error(RMSE) (5-Fold Cross Validation)', ascending=True)
# read and load data

df = pd.read_csv("../input/used-car-dataset-ford-and-mercedes/audi.csv")
df.head()
df.info()
df.describe()
df_2=df.copy()


df_2=pd.concat((df_2,pd.get_dummies(df_2["model"]),pd.get_dummies(df_2["transmission"]),pd.get_dummies(df_2["fuelType"]))

               ,axis=1)



df_2.drop(["model","transmission","fuelType"],axis=1,inplace=True)
# correlation



plt.figure(figsize=(30,30))

sns.heatmap(df_2.corr(),annot=True)

plt.show()
df_2.drop("Petrol",axis=1,inplace=True)
#  Count plot on fuel type



fig, ax1 = plt.subplots(figsize=(5,4))

graph = sns.countplot(ax=ax1,x='fuelType', data=df)

graph.set_xticklabels(graph.get_xticklabels())

for p in graph.patches:

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height/2,height ,ha="center",fontsize=10)
df_2.drop("Hybrid",axis=1,inplace=True)
# Count plot on the transmission



fig, ax1 = plt.subplots(figsize=(5,4))

graph = sns.countplot(ax=ax1,x='transmission', data=df)

graph.set_xticklabels(graph.get_xticklabels())

for p in graph.patches:

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height/2,height ,ha="center",fontsize=8)
# Pairplotting



sns.pairplot(df)

plt.show()
df_2.drop(index=df_2[df_2["mileage"]>160000].index,inplace=True)

df_2.drop(index=df_2[df_2["year"]<2000].index,inplace=True)



df_2.drop(index=df_2[df_2["engineSize"]==0].index,inplace=True)
stdn_scaler = StandardScaler().fit(df_2[["year","mileage","tax","mpg","engineSize"]])



df_2[["year","mileage","tax","mpg","engineSize"]] = stdn_scaler.transform(df_2[["year","mileage","tax","mpg","engineSize"]])
train,test = train_test_split(df_2,test_size=0.25,random_state=42)



X_train = train.drop("price",axis=1)

y_train = train["price"]



X_test = test.drop("price",axis=1)

y_test = test["price"]
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression().fit(X_train,y_train)
evaluateModel(lin_reg,X_train,X_test,y_train,y_test,"Linear Regression")
from sklearn.linear_model import Ridge

ridge_reg=Ridge().fit(X_train,y_train)
evaluateModel(ridge_reg,X_train,X_test,y_train,y_test,"Ridge Regression")
from sklearn.linear_model import Lasso

lasso_reg = Lasso().fit(X_train,y_train)
evaluateModel(lasso_reg,X_train,X_test,y_train,y_test,"Lasso Regression")
from sklearn.linear_model import ElasticNet

elastic_reg = ElasticNet().fit(X_train,y_train)
evaluateModel(elastic_reg,X_train,X_test,y_train,y_test,"Elastic Net Regression")
params={"alpha":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],"max_iter":[1000,2500,5000,7500,10000],

       "l1_ratio":[0.3,0.4,0.5,0.6,0.7]}



elastic_gs=GridSearchCV(ElasticNet(random_state=42),param_grid=params,cv=5,scoring="neg_root_mean_squared_error")

elastic_gs.fit(X_train,y_train)



elastic_reg_best=elastic_gs.best_estimator_.fit(X_train,y_train)



evaluateModel(elastic_reg_best,X_train,X_test,y_train,y_test,"Best Elastic Net Regression")
from scipy.special import factorial

factorial(38)/(factorial(35)*factorial(3))
from sklearn.preprocessing import PolynomialFeatures



poly_features = PolynomialFeatures(degree=2, include_bias=False)



X_train_poly = poly_features.fit_transform(X_train[["year","mileage","tax","mpg","engineSize"]])

X_test_poly= poly_features.fit_transform(X_test[["year","mileage","tax","mpg","engineSize"]])



poly_reg=LinearRegression()

poly_reg.fit(X_train_poly,y_train)



evaluateModel(poly_reg,X_train_poly,X_test_poly,y_train,y_test,"Polynomial Regression")
from sklearn.svm import LinearSVR



svm_reg = LinearSVR()

svm_reg.fit(X_train, y_train)
evaluateModel(svm_reg,X_train,X_test,y_train,y_test,"Linear SVM Regression")
params={"C":[100,1000,10000],

       "dual":[True,False],"epsilon":[1500,4500,7500],

       "fit_intercept":[True,False],"max_iter":[5000,7500,10000]}



svm_gs=GridSearchCV(LinearSVR(),param_grid=params,cv=5,n_jobs=-1,scoring="neg_root_mean_squared_error")

svm_gs.fit(X_train,y_train)



best_linear_svm_reg=svm_gs.best_estimator_

best_linear_svm_reg.fit(X_train,y_train)

evaluateModel(best_linear_svm_reg,X_train,X_test,y_train,y_test,"Best Linear SVM Regression")
from sklearn.svm import SVR

nonlinear_svm_reg=SVR()

nonlinear_svm_reg.fit(X_train,y_train)

evaluateModel(nonlinear_svm_reg,X_train,X_test,y_train,y_test,"Nonlinear SVM Regression")
best_nonlinear_svm_reg = SVR(C=100000,degree=2,epsilon=1000,gamma=0.1,kernel="rbf",max_iter=10000)

best_nonlinear_svm_reg.fit(X_train,y_train)

evaluateModel(best_nonlinear_svm_reg,X_train,X_test,y_train,y_test,"Best Nonlinear SVM Regression")
from sklearn.tree import DecisionTreeRegressor

dec_reg=DecisionTreeRegressor()

dec_reg.fit(X_train,y_train)
evaluateModel(dec_reg,X_train,X_test,y_train,y_test,"Decision Tree Regressor")
from sklearn.ensemble import RandomForestRegressor

rand_reg=RandomForestRegressor().fit(X_train,y_train)

evaluateModel(rand_reg,X_train,X_test,y_train,y_test,"Random Forest Regressor")
from sklearn.ensemble import AdaBoostRegressor

adaboost_reg=AdaBoostRegressor().fit(X_train,y_train)
evaluateModel(adaboost_reg,X_train,X_test,y_train,y_test,"Adaboost Regressor")
from sklearn.ensemble import GradientBoostingRegressor

gb_reg=GradientBoostingRegressor().fit(X_train,y_train)
evaluateModel(gb_reg,X_train,X_test,y_train,y_test,"Gradient Boosting Regressor")
from sklearn.metrics import mean_squared_error

gbrt=GradientBoostingRegressor(max_depth=2,n_estimators=10000)

gbrt.fit(X_train,y_train)



errors=[mean_squared_error(y_test,y_pred) for y_pred in gbrt.staged_predict(X_test)]



best_n_est=np.argmin(errors)+1



gbrt_best=GradientBoostingRegressor(n_estimators=best_n_est,max_depth=2).fit(X_train,y_train)

evaluateModel(gbrt_best,X_train,X_test,y_train,y_test,"Gradient Boosting Regressor with optimum n_estimators")
import xgboost



xgb_reg=xgboost.XGBRegressor()

xgb_reg.fit(X_train,y_train)

evaluateModel(xgb_reg,X_train,X_test,y_train,y_test,"XGB Regressor")
xgb_reg_early_stopping=xgboost.XGBRegressor()

xgb_reg_early_stopping.fit(X_train,y_train,

                           eval_set=[(X_test,y_test)],early_stopping_rounds=2)

evaluateModel(xgb_reg_early_stopping,X_train,X_test,y_train,y_test,"XGB Regressor with early stopping")
from sklearn.ensemble import VotingRegressor



voting_reg=VotingRegressor(

estimators=[("v_lin_reg",lin_reg),

            ("v_rid_reg",ridge_reg),

            ("v_lasso_reg",lasso_reg),

            ("v_elastic_reg_best",elastic_reg_best),

            ("v_elastic_reg",elastic_reg),

            ("v_lin_svm_reg",svm_reg),

            ("v_best_lin_svm_reg",best_linear_svm_reg),

            ("v_nonlinear_svm_reg",nonlinear_svm_reg),

            ("v_best_nonlinear_svm_reg",best_nonlinear_svm_reg),

            ("v_dec_reg",dec_reg),

            ("v_rand_reg",rand_reg),

            ("v_adaboost_reg",adaboost_reg),

            ("v_gb_reg",gb_reg),

            ("v_gb_reg_best",gbrt_best),

            ("v_xgb_reg",xgb_reg),

            ("v_xgb_reg_best",xgb_reg_early_stopping)

            ]

)



voting_reg.fit(X_train,y_train)
evaluateModel(voting_reg,X_train,X_test,y_train,y_test,"Voting Regression")
evaluation.sort_values(by="Root Mean Squared Error(RMSE) (5-Fold Cross Validation)")