import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
#sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing, svm
from sklearn.feature_selection import SelectFromModel
sns.set_style('whitegrid')
%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# display all
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv', index_col='Id')

# combine train and test data 
df_all = pd.concat([train_data.drop('SalePrice', axis=1), test_data], sort=True)  #df without the target




df_all.head()
df_all.shape
df_all.info()
numeric_features = df_all.select_dtypes(include=['int64','float64'])
categorical_features = df_all.select_dtypes(include=['object'])
print('Numeric Features:',len(list(numeric_features.columns)))
print('Categorical Features:',len(list(categorical_features.columns)))
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))
sns.heatmap(train_data.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Trian data')

sns.heatmap(test_data.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');
missing = df_all.isnull().sum().sort_values(ascending=False)
percentage=(df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing,percentage],axis=1,keys=['Missing','%']) 
missing_data[missing_data != 0].dropna()
# numeric_features = df_all.loc[:, df_all.dtypes != np.object]
# imputer = KNNImputer(n_neighbors=60)
# df_all.loc[:, df_all.dtypes != np.object] = imputer.fit_transform(numeric_features)
imp = SimpleImputer(missing_values=np.nan, strategy='median')
df_all.loc[:, df_all.dtypes != np.object] = imp.fit_transform(numeric_features)
df_all.head()
edit_values = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType',
              'GarageQual','PoolQC','Fence','MiscFeature','MasVnrType', 'GarageCond', 'GarageFinish']

for col in edit_values:
    df_all[col].fillna('NA',inplace=True)
df_all.Exterior1st.fillna(value='VinylSd', inplace=True)

df_all.Exterior2nd.fillna(value='VinylSd', inplace=True)

df_all.KitchenQual.fillna(value='TA', inplace=True)

df_all.SaleType.fillna(value='WD', inplace=True)

df_all.Utilities.fillna(value='AllPub', inplace=True)

df_all.Electrical.fillna(value='SBrkr', inplace=True)

df_all.Functional.fillna(value='Typ', inplace=True)

df_all.MSZoning.fillna(value='RL', inplace=True)
# Missung values 
print('Missing values:' ,df_all.isnull().sum().sum())
df_all['GarageYrBlt']= df_all['GarageYrBlt'].astype(int)
# adding the target to our df
df_all = pd.concat([df_all, train_data['SalePrice']], axis=1)
normal_sp = df_all['SalePrice'].dropna().map(lambda i: np.log(i) if i > 0 else 0)
print(df_all['SalePrice'].skew())
print(normal_sp.skew())

fig, ax = plt.subplots(ncols=2, figsize=(12,6))
df_all.hist('SalePrice', ax=ax[0])
normal_sp.hist(ax=ax[1])
plt.show
fig, ax = plt.subplots(figsize=(14, 8))
k = 10 #number of variables for heatmap
corrmat = df_all.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_all[cols].dropna().values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values, ax=ax, cmap="YlGnBu")
ax.set_ylim(0 ,10)
plt.show()
# correlation with the target
corr_matrix = df_all.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)
fig, axes = plt.subplots(ncols=4, nrows=4, 
                         figsize=(5 * 5, 5 * 5), sharey=True)
axes = np.ravel(axes)
cols = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual',
        'BsmtCond','GarageQual','GarageCond', 'MSSubClass','MSZoning',
        'Neighborhood','BldgType','HouseStyle','Heating','Electrical','SaleType']
for i, c in zip(np.arange(len(axes)), cols):
    ax = sns.boxplot(x=c, y='SalePrice', data=df_all, ax=axes[i], palette="Set2")
    ax.set_title(c)
    ax.set_xlabel("")
Q1 = df_all.quantile(0.25)
Q3 = df_all.quantile(0.75)
IQR = Q3 - Q1
outliars = (df_all < (Q1 - 5 * IQR)) | (df_all > (Q3 + 5 * IQR))
#removing bad columns and outliars
no_outliars_df = df_all.drop(['EnclosedPorch', 'KitchenAbvGr'], axis=True)
rm_rows = ['LotArea', 'MasVnrArea', 'PoolArea', 'OpenPorchSF', 'LotFrontage', 'TotalBsmtSF','1stFlrSF',
           'GrLivArea', 'BsmtFinSF1', 'WoodDeckSF']
df_all.drop(['EnclosedPorch', 'KitchenAbvGr'], axis=True, inplace=True)
for row in rm_rows:
    no_outliars_df.drop(no_outliars_df[row][outliars[row]].index, inplace=True)
object_features =df_all.loc[:, df_all.dtypes == np.object]   #object features
df_all= pd.get_dummies(df_all, columns=object_features.columns.values, drop_first=True)
no_outliars_df = pd.get_dummies(no_outliars_df, columns=object_features.columns.values, drop_first=True)
# Traing Data with Outliers
newtraining=df_all.loc[  : 1460]
# Testing Data with Outliers
newtesting=df_all.loc[1461 : ].drop('SalePrice', axis=1)
# newtraining['SalePrice'] = np.log(newtraining['SalePrice'])
# lab_enc = preprocessing.LabelEncoder()
# newtraining['SalePrice'] = la
# Traing Data without Outliers
no_outliars_training = no_outliars_df.loc[  : 1460]
# Testing Data without Outliers
no_outliars_test = no_outliars_df.loc[1461 : ].drop('SalePrice', axis=1)
y = newtraining['SalePrice']
X = newtraining.drop('SalePrice', axis=1)
ss = StandardScaler()
Xs =ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    Xs, y, test_size=.3, random_state=1)
yo = no_outliars_training['SalePrice']
Xo = no_outliars_training.drop('SalePrice', axis=1)
sso = StandardScaler()
Xso =sso.fit_transform(Xo)
Xo_train, Xo_test, yo_train, yo_test = train_test_split(
    Xso, yo, test_size=30, random_state=1)
lr = LinearRegression()
model= lr.fit(X_train, y_train)
print('Train Score:',model.score(X_train, y_train))
print('Test Score :',model.score(X_test, y_test))
lr_predictions = model.predict(newtesting) 
lro = LinearRegression()
model_o= lro.fit(Xo_train, yo_train)
print('Train Score:', model_o.score(Xo_train, yo_train))
print('Test Score :',model_o.score(Xo_test, yo_test))
lro_predictions = model_o.predict(Xso) 
lasso = Lasso(alpha=.0002)
lasso.fit(X_train, y_train)
print('Train Score:',lasso.score(X_train, y_train))
print('Test Score: ', lasso.score(X_test, y_test))
lasso_predictions = lasso.predict(newtesting)
# sqrt(mean_squared_error(submission['SalePrice'],lasso_predictions))
lasso_o = Lasso(alpha=.2)
lasso_o.fit(Xo_train, yo_train)
print('Train Score:',lasso_o.score(Xo_train, yo_train))
print('Test Score:',lasso_o.score(Xo_test, yo_test))
lasso_cv = LassoCV(alphas=np.logspace(-4, 4, 1), cv=5,random_state=1)
lasso_cv.fit(X_train, y_train)
print('Train Score :',lasso_cv.score(X_train, y_train))
print('Test Score:',lasso_cv.score(X_test, y_test))
lasso_cv_predictions = lasso_cv.predict(newtesting) 
lasso_cv_o = LassoCV(cv=10,random_state=1)
lasso_cv_o.fit(Xo_train, yo_train)
print('Train Score:',lasso_cv_o.score(Xo_train, yo_train))
print('Test Score:',lasso_cv_o.score(Xo_test, yo_test))
ridge = Ridge(alpha=1) 
ridge.fit(X_train, y_train)
print('Train Score:',ridge.score(X_train, y_train))
print('Test Score:',ridge.score(X_test, y_test))
ridge_predictions = ridge.predict(newtesting) 
ridge_o = Ridge(alpha=.01) 
ridge_o.fit(Xo_train, yo_train)
print('Train Score:',ridge_o.score(Xo_train, yo_train))
print('Test Score:',ridge_o.score(Xo_test, yo_test))
ridgecv = RidgeCV(alphas=np.logspace(-4, 4, 1))
ridgecv.fit(X_train, y_train)
print('Train Score:',ridgecv.score(X_train, y_train))
print('Test Score:',ridgecv.score(X_test, y_test))
ridgeCV_predictions = ridgecv.predict(newtesting) 
ridgecv_o = RidgeCV(alphas=np.logspace(-4, 4, 1)) 
ridgecv_o.fit(Xo_train, yo_train)
print('Train Score:',ridgecv_o.score(Xo_train, yo_train))
print('Test Score:',ridgecv_o.score(Xo_test, yo_test))
elastic=ElasticNet(.00001)
elastic = elastic.fit(X_train, y_train)
print('Train Score:',elastic.score(X_train, y_train))
print('Test Score:',elastic.score(X_test, y_test))
elastic_predictions = elastic.predict(newtesting) 
elastic_o=ElasticNet(.00001)
elastic_o = elastic_o.fit(Xo_train, yo_train)
print('Train Score:',elastic_o.score(Xo_train, yo_train))
print('Test Score:',elastic_o.score(Xo_test, yo_test))
elastic_cv=ElasticNetCV(alphas=np.logspace(-10, 6, 1))
elastic_cv = elastic_cv.fit(X_train, y_train)
print('Train Score:',elastic_cv.score(X_train, y_train))
print('Test score:',elastic_cv.score(X_test, y_test))
elastic_cv_o=ElasticNetCV(alphas=np.logspace(-4, 4, 1))
elastic_cv_o = elastic_cv_o.fit(Xo_train, yo_train)
print('Train Score :',elastic_cv_o.score(Xo_train, yo_train))
print('Test score  :',elastic_cv_o.score(Xo_test, yo_test))
tree = DecisionTreeClassifier(max_depth = 28)
tree.fit(X, y)
print('Score : ',tree.score(X, y))
tree_predictions = tree.predict(newtesting)
# sqrt(mean_squared_error(submission['SalePrice'], tree_predictions))
tree_o = DecisionTreeClassifier(max_depth = 29)
tree_o.fit(Xo, yo)
print('Score : ',tree_o.score(Xo, yo))
randomF = RandomForestRegressor(max_depth=50)
randomF.fit(X, y)
print('Train score :',randomF.score(X, y))
# from sklearn.model_selection import KFold
# cv=KFold(n_splits=5, shuffle=True, random_state=1)
# cross_val_score(randomF, X, y, cv=cv)
# cross_val_score(randomF, X, y, cv=cv).mean()
randomF_predictions = randomF.predict(newtesting) 
# sqrt(mean_squared_error(submission_tree['SalePrice'], randomF_predictions))
# param_grid = {
#     'n_estimators': [i for i in range(50,1000,10)],
#     'max_features': [86],
#     'max_depth' :[i for i in range(1,100,1)]
# }
# rand = RandomForestRegressor(n_jobs=-1)

# gs = GridSearchCV(rand, 
#                   param_grid, 
#                   cv=5)
# gs.fit(X, y)
# gs.best_params_
# gs.best_score_
randomF_o = RandomForestRegressor(max_depth = 50)
randomF_o.fit(Xo, yo)
print('train score : ',randomF_o.score(Xo, yo))
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(X_train, y_train)
print('Train score : ',neigh.score(X_train, y_train))
print('Test score  : ',neigh.score(X_test, y_test))
neigh_predictions = neigh.predict(newtesting)
# sqrt(mean_squared_error(submission_tree['SalePrice'], neigh_predictions))
neigh_o = KNeighborsRegressor(n_neighbors=3)
neigh_o.fit(Xo_train, yo_train)
print('Train score : ',neigh_o.score(Xo_train, yo_train))
print('Test score  : ',neigh_o.score(Xo_test, yo_test))
svm_l = svm.SVC(kernel='linear')
svm_l.fit(X, y)
svm_l.score(X, y)
cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(svm_l, X, y, cv=cv, n_jobs=-1).mean()
svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X, y)
svm_rbf.score(X, y)
cross_val_score(svm_rbf, X, y, cv=5, n_jobs=-1).mean()
svm_p = svm.SVC(kernel='poly')
svm_p.fit(X, y)
svm_p.score(X, y)
cross_val_score(svm_p, X, y, cv=5, n_jobs=-1).mean()
svm_rbf = svm.SVC(kernel='rbf', gamma=0.001)
cross_val_score(svm_rbf, X, y, cv=5).mean()
list_of_Scores = list()
# LinearRegression
results = {'Model':'LinearRegression',
           'Train Score': model.score(X_train, y_train),
           'Test Score':model.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)
# Lasso
results = {'Model':'Lasso',
           'Train Score':lasso.score(X_train, y_train),
           'Test Score': lasso.score(X_test, y_test),
           'Kaggle Score':0.59683}
list_of_Scores.append(results)
# LassoCV
results = {'Model':'LassoCV',
           'Train Score': lasso_cv.score(X_train, y_train),
           'Test Score':lasso_cv.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# Ridg
results = {'Model':'Ridg',
           'Train Score': ridge.score(X_train, y_train),
           'Test Score':ridge.score(X_test, y_test),
           'Kaggle Score':0.36706}
list_of_Scores.append(results)

# RidgCV
results = {'Model':'RidgCV',
           'Train Score': ridgecv.score(X_train, y_train),
           'Test Score':ridgecv.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# ElasticNet
results = {'Model':'ElasticNet',
           'Train Score': elastic.score(X_train, y_train),
           'Test Score':elastic.score(X_test, y_test),
           'Kaggle Score':6.63994}
list_of_Scores.append(results)

# ElasticNetCV
results = {'Model':'ElasticNetCV',
           'Train Score':elastic_cv.score(X_train, y_train),
           'Test Score':elastic_cv.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# DecisionTreeRegressor
results = {'Model':'DecisionTreeRegressor',
           'Train Score':tree.score(X, y),
           'Test Score':None,
           'Kaggle Score':0.25525}
list_of_Scores.append(results)

# RandomForest
results = {'Model':'RandomForest',
           'Train Score':randomF.score(X, y),
           'Test Score':None,
           'Kaggle Score':0.14824}
list_of_Scores.append(results) 

# KNeighborsRegressor
results = {'Model':'KNeighborsRegressor',
           'Train Score': neigh.score(X_train, y_train),
           'Test Score':neigh.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# SVM
results = {'Model':'SVM',
           'Train Score': svm_l.score(X, y),
           'Test Score':None,
           'Kaggle Score':None}
list_of_Scores.append(results)
df_results = pd.DataFrame(list_of_Scores)
df_results
list_of_Scores_o = list()
# LinearRegression
results_o = {'Model':'LinearRegression',
           'Train Score': lro.score(Xo_train, yo_train),
           'Test Score':lro.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)


# Lasso
results_o = {'Model':'Lasso',
           'Train Score': lasso_o.score(Xo_train, yo_train),
           'Test Score':lasso_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# LassoCv
results_o = {'Model':'LassoCv',
           'Train Score': lasso_cv_o.score(Xo_train, yo_train),
           'Test Score':lasso_cv_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# Ridg
results_o = {'Model':'Ridg',
           'Train Score':ridge_o.score(Xo_train, yo_train),
           'Test Score':ridge_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# RidgCV
results_o = {'Model':'RidgCV',
           'Train Score':ridgecv_o.score(Xo_train, yo_train),
           'Test Score':ridgecv_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# ElasticNet
results_o = {'Model':'ElasticNet',
           'Train Score':elastic_o.score(Xo_train, yo_train),
           'Test Score':elastic_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# ElasticNetCV
results_o = {'Model':'ElasticNetCV',
           'Train Score':elastic_cv_o.score(Xo_train, yo_train),
           'Test Score':elastic_cv_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# DecisionTreeRegressor
results_o = {'Model':'DecisionTreeRegressor',
           'Train Score':tree_o.score(Xo, yo),
           'Test Score':None}
list_of_Scores_o.append(results_o)

# RandomForest
results_o = {'Model':'RandomForest',
           'Train Score':randomF_o.score(Xo, yo),
           'Test Score':None}
list_of_Scores_o.append(results_o)

# KNeighborsRegressor
results_o = {'Model':'KNeighborsRegressor',
           'Train Score':neigh_o.score(Xo_train, yo_train),
           'Test Score':neigh_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)
df_results_o = pd.DataFrame(list_of_Scores_o)
df_results_o
submission_randomF = submission.copy()
submission_randomF['SalePrice'] = randomF_predictions
submission_randomF['SalePrice'].head()
# submission_randomF.to_csv('sample_submissionRandom.csv')