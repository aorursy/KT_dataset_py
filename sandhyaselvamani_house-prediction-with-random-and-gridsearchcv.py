import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
df_train= pd.read_csv('train.csv')
df_test= pd.read_csv('test.csv')
#print(df_train.columns)
#print(df_test.columns)
#df_test=df_train[df_train.columns]
print(df_test.shape)
print(df_train.shape)
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, SGDRegressor, Lasso, ElasticNet, Lars, LassoLars, HuberRegressor, BayesianRidge, PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
random_clf = ensemble.GradientBoostingRegressor()
df_train.head()
plt.figure(figsize=(26,8))
sns.heatmap(df_train.corr(),annot = True)
plt.show()
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)
df_train = df_train.drop((missing_data[missing_data['Total'] > 0.80]).index,1)

total_test = df_test.isnull().sum().sort_values(ascending=False)
percent_test = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
#missing_data.head(25)

df_test = df_test.drop((missing_data[missing_data['Total'] > 0.80]).index,1)
#Sel_cols=df_train.corr()['SalePrice']
#Sel_cols=Sel_cols.reset_index()
#Sel_cols['index']
#Sel1_col=Sel_cols['index'].where((Sel_cols['SalePrice'] > 0.4) & (Sel_cols['index']!= 'SalePrice')).dropna()
##Sel_col_list= Sel1_col.tolist()
#print(Sel_col_list)
#df_train.filter(Sel_col_list) 
#df_train.dropna()
print(df_test.shape)
print(df_train.shape)
# Categorical boolean mask
categorical_feature_mask = df_train.dtypes==object
# filter categorical columns using mask and turn it into alist
categorical_cols = df_train.columns[categorical_feature_mask]
print(categorical_cols)
categorical_feature_mask_test = df_test.dtypes==object
# filter categorical columns using mask and turn it into alist
categorical_cols_test = df_test.columns[categorical_feature_mask_test].tolist()
#df_train.drop(['MSZoning'],axis=1)
df_train.columns
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_train[categorical_cols] = df_train[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))
df_test[categorical_cols_test] = df_test[categorical_cols_test].apply(lambda col: labelencoder.fit_transform(col.astype(str)))



df_train.isnull().sum().sort_values(ascending=False).head(20)
#saleprice correlation matrix
k = 15 #number of variables for heatmap
plt.figure(figsize=(16,8))
corrmat = df_train.corr()
# picking the top 15 correlated features
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
df_train = df_train[cols]

#df_test  = df_test[cols]

#print(df_test.isnull().sum().sort_values(ascending=False).head(20))
print(df_test.columns)

#df_test=df_test[cols.drop('SalePrice')]
#print(df_test.shape)
#df_test.isnull().sum().sort_values(ascending=False).head(20)
print(df_train.columns)
print(df_train.columns.isnull())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('SalePrice', axis=1), df_train['SalePrice'], test_size=0.3, random_state=99)
plt.hist(X_train)
y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)
y_train
y_train.shape=y_train.shape[0]
from time import time
from sklearn.metrics import max_error, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
def benchmark(reg):
    print('_' * 80)
    print("Training: ")
    print(reg)
    t0 = time()
    reg.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = reg.predict(X_test)
    y_true = list(y_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = reg.score(X_test, y_test)
    if score < 0 or score > 1:
        print("score: FAIL")
        score = 0
    else:
        print("score:   %0.3f" % score)
        
    if hasattr(reg, 'coef_'):
        print("intercept_: ", reg.intercept_)
        evs = explained_variance_score(y_test, pred)
        print('EVS (Explained Variance Score): {0:.2f}'.format(evs), ' [Best value is 1.0]')
        mre = max_error(y_test, pred)
        print('MRE (Max Residual Error): {0:.2f}'.format(mre), ' [Best value is 0.0]')
        mae = mean_absolute_error(y_test, pred)
        print('MAE (Mean Absolute Error): {0:.2f}'.format(mae), ' [Best value is 0.0]')
        mse = mean_squared_error(y_test, pred)
        print('MSE (Mean Squared Error): {0:.2f}'.format(mse), ' [Best value is 0.0]')
        r_2 = r2_score(y_test, pred)
        print('R^2 (Coefficient of Determination): {0:.2f}'.format(r_2), ' [Best value is 1.0]')
        print()

    print()
    reg_descr = str(reg).split('(')[0]
    return reg_descr, score, train_time, test_time, reg
result =[]
print('=' * 80)
knn_reg=KNeighborsRegressor()
#results.append(benchmark((KNeighborsRegressor())))
params = {'leaf_size': [10,5,20,30],'n_neighbors':[4,5,7,9,11],'p':[3,5,2]}
clf = GridSearchCV(knn_reg, param_grid = params ,scoring = 'neg_mean_squared_error',cv=4)
clf.fit(X_train, y_train)
print(clf.best_params_)
plt.figure(figsize=(15,8))
knn_predictions = clf.predict(X_test)
knn_predictions= knn_predictions.reshape(-1,1)
plt.scatter(y_test,knn_predictions, c= 'brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
print('=' * 80)
print('Ridge')
knn_reg=KNeighborsRegressor()
results.append(benchmark((Ridge(alpha=1.0))))

print('=' * 80)
print('Lasso')
results.append(benchmark((Lasso(max_iter=4000))))


print('=' * 80)
print('RandomForestRegressor')
results.append(benchmark((RandomForestRegressor(1000))))
# Create a pipeline
pipe = Pipeline([("classifier", KNeighborsRegressor())])
# Create dictionary with candidate learning algorithms and their hyperparameters
search_space = [
                {"classifier": [KNeighborsRegressor()],
                 "classifier__leaf_size": [10,5,20,30],
                
                  "classifier__n_neighbors":[4,5,7,9,11]
                 },
                {"classifier": [ensemble.GradientBoostingRegressor()],
                  "classifier__n_estimators" : [1200], 
                  "classifier__max_features":[5,10]
                },

                {"classifier": [RandomForestRegressor()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_leaf":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]}]
# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(X_train, y_train)
print(best_model.best_params_)
from sklearn import metrics
#print('MSE:', metrics.mean_squared_error(y_test, predictions))

y_test.shape=y_test.shape[0]
##df_train=df_train.drop('SalePrice',axis=1)
#df_test=df_test[df_train.columns]
df_test.shape
df_train.shape

#params = {'n_estimators': [1200],'max_features':[5,10]}
#clf = GridSearchCV(grid_clf, param_grid = params ,scoring = 'neg_mean_squared_error',cv=5)
#clf.fit(X_train, y_train)
#print(clf.best_params_)
random_clf.fit(X_train, y_train)
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 800, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 20, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [3,1,4,5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
               
print(random_grid)


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = random_clf , param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=45, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return mse
    
def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return mse

def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error

train_error, test_error = calc_metrics(X_train, y_train, X_test, y_test,rf_random)
train_error, test_error = round(train_error, 3), round(test_error, 3)

print('train error: {} | test error: {}'.format(train_error, test_error))
print('train/test: {}'.format(round(test_error/train_error, 1)))
print(rf_random.best_params_)
clf_pred=rf_random.predict(X_test)
print(rf_random.best_params_)
clf_pred= clf_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))
print('MSE:', metrics.mean_squared_error(y_test, clf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))
plt.figure(figsize=(15,8))
plt.scatter(y_test,clf_pred, c= 'brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
#error_rate=np.array([metrics.mean_squared_error(y_test, predictions),metrics.mean_squared_error(y_test, clf_pred),metrics.mean_squared_error(y_test, lgb_pred)])
#plt.figure(figsize=(16,5))
#plt.plot(error_rate)
metrics.mean_squared_error(y_test, clf_pred)
a = pd.read_csv('test.csv')
test_id = a['Id']
a = pd.DataFrame(test_id, columns=['Id'])
a[["Id"]] = a[["Id"]].apply(pd.to_numeric)
#a = a.astype(int)
#a = a.apply(str)
test = sc_X.fit_transform(df_test)
df_test.shape
df_cols=df_train.drop('SalePrice',axis=1)
print(df_cols.columns)
#df_test.columns=df_cols[df_cols.columns]

df_test= df_train.filter(['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
       '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
       'Fireplaces', 'BsmtFinSF1', 'Foundation', 'WoodDeckSF'], axis=1)
df_test.shape
test_prediction_clf=rf_random.predict(df_test)
test_prediction_clf= test_prediction_clf.reshape(-1,1)




test_prediction_clf
test_prediction_clf =sc_y.inverse_transform(test_prediction_clf)
test_prediction_clf = pd.DataFrame(test_prediction_clf, columns=['SalePrice'])
test_prediction_clf.head()

result = pd.concat([a,test_prediction_clf], axis=1)
result.to_csv('submission.csv',index=False)

