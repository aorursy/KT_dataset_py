import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.columns
# Insight into Dependent variable SalePrice
df_train.SalePrice.describe()
# Histogram for SalePrice
sns.distplot(df_train.SalePrice)
# OverallQual
plt.figure(figsize=(10,6))
sns.boxplot(x='OverallQual',y='SalePrice',data=df_train)
# OverallCond
plt.figure(figsize=(10,6))
sns.boxplot(x='OverallCond',y='SalePrice',data=df_train)
corr_matrix = df_train.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix,cmap='viridis')
#Extracting Highly Related features to SalePrice
cols = corr_matrix.nlargest(10,columns='SalePrice')['SalePrice'].index
corr_matrix2 = np.corrcoef(df_train[cols].values.T)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix2, cbar=True,cmap='viridis', annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols])
total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
missing_df = pd.concat([total,percent],axis =1,keys=['total','percent'])
missing_df.head(20)
# Removing Extra features
train = df_train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
test = df_test[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
train.head()
#YearBuilt
#assuming dataset is prepared in 2013 as max value in YearBuilt is 2010
train.YearBuilt = 2013 - train.YearBuilt
test.YearBuilt = 2013 - test.YearBuilt
test.TotalBsmtSF = test.TotalBsmtSF.replace(np.nan,1046)
test.GarageArea = test.GarageArea.replace(np.nan,473)
X_train = train.iloc[:,1:].values
X_test = test.copy().values
y_train = train.SalePrice.values
X_test.shape,X_train.shape,y_train.shape
test.GarageArea.mean()
test.TotalBsmtSF.mean()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,mean_squared_error
def apply_alg(regressor):
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_train)
    
    print("--"*15)
    print("MSE : ",mean_squared_error(y_train,y_pred))
    accuracies = cross_val_score(regressor,X_train,y_train,cv = 10)
    print("Accuracy mean %f Accuracy Std %f"%(accuracies.mean(),accuracies.std()))
    print("--"*15)
#Linear Regression
from sklearn.linear_model import LinearRegression
regressorL = LinearRegression()
apply_alg(regressorL)
#Support Vector Regressor
from sklearn.svm import SVR
regressorSVR = SVR(kernel = 'rbf',C=10,epsilon=0.3,gamma=0.1)
apply_alg(regressorSVR)
#Ridge Regresor
from sklearn.linear_model import Ridge
regressorRD = Ridge(alpha=0.05,normalize=True)
apply_alg(regressorRD)
#Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressorDT = DecisionTreeRegressor(random_state = 0)
apply_alg(regressorDT)
#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 20, random_state = 0)
apply_alg(regressorRF)
y_output = regressorSVR.predict(X_test)
y_output = sc_y.inverse_transform(y_output)
output = df_test[['Id']]
output['SalePrice'] =y_output
output.to_csv('submissionSVR.csv',mode = 'w',index=False)
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': y_output})
my_submission.to_csv('submission.csv', index=False)
#Grid Search for Support Vector Regressor
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [10], 'kernel': ['rbf'], 'gamma': [0.05,0.8,0.1],'epsilon':[0.3,0.5,0.7,1]}]
grid_search = GridSearchCV(estimator = regressorSVR,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy,best_parameters)
output.head()