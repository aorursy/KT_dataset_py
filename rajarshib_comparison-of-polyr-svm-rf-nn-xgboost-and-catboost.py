import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing as p


data=pd.read_csv('../input/insurance/insurance.csv')
data.head(10)

data.describe()
#Finding correlations between variables(Only numerical ones)
data.corr()
X=data.iloc[:,0:6].values
y=data.iloc[:,-1].values

print(X[0:5,:])
print("\n")
print(y[0:5])


from sklearn.model_selection import train_test_split

#splitting the dataset in Train:Test=75:25
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) 
print(data['sex'].unique())
print(data['smoker'].unique())
print(data['region'].unique())

#Label encoding of 'sex' column
label_en=p.LabelEncoder()
X_train[:,1]=label_en.fit_transform(X_train[:,1]) 
X_test[:,1]=label_en.transform(X_test[:,1])


#Label encoding 'smoker' Column
X_train[:,4]=label_en.fit_transform(X_train[:,4]) 
X_test[:,4]=label_en.transform(X_test[:,4])

#Label encoding of 'region' Column
X_train[:,5]=label_en.fit_transform(X_train[:,5])
X_test[:,5]=label_en.transform(X_test[:,5])



print(X_train[0:5,:])
#One hot encoding of 'region' column
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 


columnTransformer = ColumnTransformer([('encoder', 
                                        OneHotEncoder(), 
                                        [5])], 
                                        remainder='passthrough') 
X_train = np.array(columnTransformer.fit_transform(X_train), dtype = np.float)
X_test = np.array(columnTransformer.transform(X_test), dtype = np.float)

print(X_train[0:5,:])
from sklearn.preprocessing import StandardScaler
scaler_X=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)

print(X_train[0:5,:])
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics

poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
POLYreg = LinearRegression()
POLYreg.fit(X_poly, y_train)

X_poly_test=poly_reg.transform(X_test)
POLY_cod=POLYreg.score(X_poly_test,y_test)
print('Coefficient of determination(R^2) = {}'.format(POLY_cod)) 
mse=(metrics.mean_squared_error(POLYreg.predict(X_poly_test),y_test))
print('Mean squared error = {}'.format(mse))
POLY_rmse=mse**0.5
print('Root Mean squared error = {}'.format(POLY_rmse))

y_pred = POLYreg.predict(poly_reg.transform(X_test))
plt.scatter(y_test,y_pred)

plt.plot(y_test,y_test,color='red')
plt.title('Polynomial Regression')
plt.xlabel('Actual')
plt.ylabel('Predicted')

scaler_y=StandardScaler()


#Reshaping to 2D array as per as StandardScaler function requirements
y_train_svr=y_train.reshape(len(y_train),1)
y_test_svr=y_test.reshape(len(y_test),1)

y_train_svr=scaler_y.fit_transform(y_train_svr)
y_test_svr=scaler_y.transform(y_test_svr)

y_train_svr=y_train_svr.reshape(len(y_train_svr))
y_test_svr=y_test_svr.reshape(len(y_test_svr))

print(y_train_svr[:5])



from sklearn import metrics
from sklearn.svm import SVR
SVRreg = SVR(kernel = 'rbf')
SVRreg.fit(X_train, y_train_svr)

y_pred_svr=(scaler_y.inverse_transform(SVRreg.predict(X_test))).flatten()

SVR_cod=SVRreg.score(X_test,y_test_svr)
print('Coefficient of determination(R^2) = {}'.format(SVR_cod)) 
mse=metrics.mean_squared_error(y_pred_svr,y_test)
print('Mean squared error = {}'.format(mse))
SVR_rmse=mse**0.5
print('Root Mean squared error = {}'.format(SVR_rmse))

plt.scatter(y_test,y_pred_svr)

plt.plot(y_test,y_test,color='red')
plt.title('SVR')
plt.xlabel('Actual')
plt.ylabel('Predicted')

from catboost import CatBoostRegressor
CATreg = CatBoostRegressor()
CATreg.fit(X_train,y_train)
CAT_cod=CATreg.score(X_test,y_test)
print('Coefficient of determination(R^2) = {}'.format(CAT_cod)) 
mse=(metrics.mean_squared_error(CATreg.predict(X_test),y_test))
print('Mean squared error = {}'.format(mse))
CAT_rmse=mse**0.5
print('Root Mean squared error = {}'.format(CAT_rmse))

plt.scatter(y_test,CATreg.predict(X_test))
plt.plot(y_test,y_test,color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('CatBoost')
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(random_state=0)

parameters_grid = {
    'hidden_layer_sizes': [(3,3),(3,4,3),(3,5,3),(10,),(5,5),(5,4,5),(4,3,4)],
    'activation': ['relu'],
    'solver': ['lbfgs'],
    'max_iter': [5000]
}
from sklearn.model_selection import GridSearchCV

gcv=GridSearchCV(estimator=mlp,param_grid=parameters_grid,cv=10,scoring='neg_root_mean_squared_error',n_jobs=-1)
search=gcv.fit(X_train,y_train)
search.best_params_

from sklearn.neural_network import MLPRegressor
NNreg = MLPRegressor(random_state=0, max_iter=5000,hidden_layer_sizes=(3,5,3),solver='lbfgs').fit(X_train, y_train)
NN_cod=NNreg.score(X_test,y_test)
print('Coefficient of determination(R^2) = {}'.format(NN_cod)) 
mse=(metrics.mean_squared_error(NNreg.predict(X_test),y_test))
print('Mean squared error = {}'.format(mse))
NN_rmse=mse**0.5
print('Root Mean squared error = {}'.format(NN_rmse))


plt.scatter(y_test,NNreg.predict(X_test))
plt.plot(y_test,y_test,color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')

from xgboost import XGBRegressor
xgb=XGBRegressor(random_state=0)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}
gcv = GridSearchCV(
    estimator=xgb,
    param_grid=parameters,
    scoring = 'neg_root_mean_squared_error',
    n_jobs = -1,
    cv = 10,
    verbose=True
)

search=gcv.fit(X_train,y_train)
search.best_params_



from xgboost import XGBRegressor
XGBreg = XGBRegressor(learning_rate=0.05,max_depth=3,n_estimators=100,random_state=0)
XGBreg.fit(X_train,y_train)
XGB_cod=XGBreg.score(X_test,y_test)
print('Coefficient of determination(R^2) = {}'.format(XGB_cod)) 
mse=(metrics.mean_squared_error(XGBreg.predict(X_test),y_test))
print('Mean squared error = {}'.format(mse))
XGB_rmse=mse**0.5
print('Root Mean squared error = {}'.format(XGB_rmse))


plt.scatter(y_test,XGBreg.predict(X_test))
plt.plot(y_test,y_test,color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('XGBoost')
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(random_state=0)

parameters={'max_depth': [10, 20, 30, 40, 50, None],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [50,100,200]}
gcv = GridSearchCV(
    estimator=rf,
    param_grid=parameters,
    scoring = 'neg_root_mean_squared_error',
    n_jobs = -1,
    cv = 10,
    verbose=True
)


search=gcv.fit(X_train,y_train)
search.best_params_

from sklearn.ensemble import RandomForestRegressor
RFreg = RandomForestRegressor(n_estimators=100,max_depth=10,min_samples_leaf=4,min_samples_split=10,random_state=0)
RFreg.fit(X_train,y_train)
RF_cod=RFreg.score(X_test,y_test)
print('Coefficient of determination(R^2) = {}'.format(RF_cod)) 
mse=(metrics.mean_squared_error(RFreg.predict(X_test),y_test))
print('Mean squared error = {}'.format(mse))
RF_rmse=mse**0.5
print('Root Mean squared error = {}'.format(RF_rmse))


plt.scatter(y_test,RFreg.predict(X_test))
plt.plot(y_test,y_test,color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest')
data = {'Model':  ['PolyR', 'SVM','RF','NN','XGBoost','CatBoost'],
        'RMSE': [POLY_rmse, SVR_rmse,RF_rmse,NN_rmse,XGB_rmse,CAT_rmse],
        'Coeff Of Det':[POLY_cod,SVR_cod,RF_cod,NN_cod,XGB_cod,CAT_cod]
        }

df = pd.DataFrame (data, columns = ['Model','RMSE','Coeff Of Det'])

df