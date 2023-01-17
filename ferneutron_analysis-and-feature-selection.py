"""Handle data"""
import numpy as np
import pandas as pd

"""Metrics"""
from sklearn.metrics import mean_absolute_error

"""Feature Selection"""
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV

"""Regressors"""
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('test.csv')
#test_ID = test['ID']
y_train = train['target']
#y_train = np.log1p(y_train)
train.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
#test.drop("ID", axis = 1, inplace = True)
columns_one_value = [element for element, ele in train.items() 
                     if (pd.Series(train[element], name=element)).nunique() == 1]
train = train.drop(columns_one_value, axis=1)
#test = test.drop(columns_one_value, axis=1)
train = train.round(16)
#test = test.round(16)
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i + 1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
            
train.drop(colsToRemove, axis=1, inplace=True) 
#test.drop(colsToRemove, axis=1, inplace=True) 
print("Shape train: ", train.shape)
#print("Shape test: ", test.shape)
pca = PCA()
pca.fit(train)
# Plotting to visualize the best number of elements
plt.figure(1, figsize=(9, 8))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('Number of Feautres')
plt.ylabel('Variance Ratio')
ytrain = np.array(y_train)
ytrain = ytrain.astype('int')
# Initialize SelectKBest function
X = SelectKBest(chi2, k=100).fit_transform(train, ytrain)
X.shape
RandForest_K_best = RandomForestRegressor()      
RandForest_K_best.fit(X, ytrain)
ypred = RandForest_K_best.predict(X)
mae = mean_absolute_error(ytrain, ypred)
print("MAE with 100 features: ", mae)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', learning_rate=0.1, max_depth=3, n_estimators=300) 
# Initialize the RFECV function setting 3-fold cross validation
rfecv = RFECV(estimator=xg_reg, step=1, cv=3)
rfecv = rfecv.fit(X, y_train)
print('Best number of features :', rfecv.n_features_)
# Plotting the best features with respect to the Cross Validation Score
plt.figure()
plt.xlabel("Number of Features")
plt.ylabel("Score of Selected Features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
Xnew = X[:,rfecv.support_]
print(Xnew.shape)
xg_reg.fit(Xnew ,ytrain)
ypred = xg_reg.predict(Xnew)
mae = mean_absolute_error(ytrain, ypred)
print("MAE with 57 features: ", mae)
