#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

#read in data 
#training set consists house id 1- 1460
data= pd.read_csv('../input/train.csv',index_col=0)
#testing set consists house id 1461 - 2919
testdata=pd.read_csv('../input/test.csv', index_col=0)
#insert saleprice column in testdata just so the they have the same # of columns
testdata["SalePrice"]=0

#concate the dataframes together so I can clean the data together. They will be seperated later
total=pd.concat([data,testdata])
#Fill NAN with 0's
total=total.apply(lambda x: x.fillna(0))
# conver object to dummy variables
columns=total.columns[data.dtypes == 'object']
total=pd.get_dummies(total, columns =columns)
print(total.head())

#now split the data again
data = total.iloc[0:1460,:]
testdata= total.iloc[1460:, :]
# Now remove SalePrice column from test set
testdata.drop(["SalePrice"],axis=1)
print(data.head())
print(testdata.head())
#split out validation data set
Y=data[['SalePrice']]
data.drop(["SalePrice"], axis=1)
X=data
testdata.drop(["SalePrice"], axis=1)
print(testdata)
print(X)
print(Y)
#Start Testing a couple  models' performance
#Try Lasso, ElasticNet, Ridge, SVR(kernel ='rbf') etc
#import packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor 
#feature extraction with SelectKBest
from sklearn.feature_selection import SelectKBest 
#Statistical tests can be used to select those features that have the strongest relationship with the output variable. 
#The scikit-learn library provides the SelectKBest class2 that can be used with a suite of different statistical tests to select a specifit number of features.
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

#Split out validation dataset
X_train,X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size= 0.2,random_state=7)
num_folds=10
scoring = 'mean_squared_error'
seed=7

print('test')

#Using pipelines and for loops to find the best model(s)
from sklearn.pipeline import Pipeline
pipelines= []
pipelines.append(('scaledLasso', Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5)),('Lasso', Lasso())])))
pipelines.append(('scaledRidge', Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5)),('Ridge', Ridge())])))
pipelines.append(('scaledEN', Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5)),('EN', ElasticNet())])))
pipelines.append(('scaledSVR', Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5)),('SVR', SVR())])))
pipelines.append(('scaledRFR', Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5)),('RFR', RandomForestRegressor())])))
pipelines.append(('scaledGBR', Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5)),('GBR', GradientBoostingRegressor())])))
pipelines.append(('scaledABR', Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5)),('ABR', AdaBoostRegressor())])))
results=[]
names=[]
for name, model in pipelines:
    kfold=KFold(n_splits=num_folds, random_state = seed)
    cv_results= cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(),cv_results.std()))

    #scaledGBR: -1044598568.165649 (737380927.867408)
    #scaledRFR: -1054476883.359236 (620683514.287574)
#scaledEN: -1335370698.879367 (624704717.051779)
#scaledLasso: -1320054330.183205 (640006413.003961)
#scaledRidge: -1319943605.816805 (639756928.179752)
#scaledSVR: -6369795963.599862 (1838658478.481387)
#scaledABR: -1500073218.467181 (969069134.458727)

   
#Fine tuning GradientBoostingRegressor model (the best model)
scaler=Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5))]).fit(X_train)
rescaledX=scaler.transform(X_train)
learning_rate =[0.01,0.1]
n_estimators = [200, 500]
max_depth = [10,50]
min_samples_leaf = [1, 2]

param_grid = {'n_estimators': n_estimators, 
              'learning_rate': learning_rate,
             'max_depth': max_depth,
             'min_samples_leaf': min_samples_leaf}

model=GradientBoostingRegressor()
kfold=KFold(n_splits=num_folds, random_state=seed)
grid=GridSearchCV(estimator=model, param_grid=param_grid, scoring= scoring, cv=kfold)
grid_result=grid.fit(rescaledX,Y_train)

print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_params_))

#Best: -1027985518.359070 using {'learning_rate': 0.1, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 200}



#Finalize the model

scaler = Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5))]).fit(X_train)
rescaledX=scaler.transform(X_train)
##Best: -1027985518.359070 using {'learning_rate': 0.1, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 200}
# Pipeline([('Scaler',StandardScaler()),('pca',PCA(n_components=5)),('GBR', GradientBoostingRegressor())])
model=GradientBoostingRegressor(learning_rate=0.1, max_depth=10, min_samples_leaf=2, n_estimators=200)
model.fit(rescaledX,Y_train)
rescaledTestdataX=scaler.transform(testdata)

prediction=model.predict(rescaledTestdataX)
print(prediction)

testdata["SalePrice"]=prediction
output=testdata[["SalePrice"]]
print(output.head(20))

output.to_csv("prediction_output_ML_test3.csv")
