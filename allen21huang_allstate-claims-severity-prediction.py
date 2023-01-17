# import and some default settings

import warnings

import itertools

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from scipy import sparse

from sklearn import metrics

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.svm import LinearSVC

from scipy.stats import skew, boxcox

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split



warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

%matplotlib inline
# load our dataset 

dataset = pd.read_csv("/kaggle/input/allstate-claims-severity/train.csv")

model_index = len(dataset)

dataset.head()
submission = pd.read_csv("/kaggle/input/allstate-claims-severity/test.csv")

full_dataset = pd.concat([dataset,submission]).reset_index(drop=True)
full_dataset.info()
full_dataset.describe()
dataset.shape,submission.shape,full_dataset.shape
# split our dataset

Y = dataset["loss"]

X = dataset.drop(['id', 'loss'], axis= 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
# group features

cat_variables = []

con_variables = []

id_col = 'id'

target_col = 'loss'



for i in dataset.columns:

    if i[:2] == 'ca':

        cat_variables.append(i)

    if i[:2] == 'co':

        con_variables.append(i)
print("The continuous variables: ",con_variables)

print("The categorial variables: ",cat_variables)
# check the distribution of continuous column

count = 1



for i in range(len(con_variables)):

    fig = plt.figure(figsize = (15,25))

    sns.set_style('darkgrid')

    plt.subplot(len(con_variables),2,count)

    sns.violinplot(x_train[con_variables[i]],palette="hls")

    plt.title("Train")

    

    plt.subplot(len(con_variables),2,count+1)

    sns.violinplot(x_test[con_variables[i]],palette="Paired")

    plt.title("Test")

    count += 2
# plot the heatmap of correlation matrix

plt.figure(figsize=(15,12))

sns.heatmap(x_train.corr(),cmap='coolwarm',linecolor='white',linewidths=0.5,annot=True)

plt.show()
# find out high correlated features

highcorr = []

corr = x_train.corr() 

threshold = 0.9



for i in range(corr.shape[0]):

    for j in range(corr.shape[1]):

        if i == j:

            continue

        elif (corr.iloc[i,j] > threshold) | (corr.iloc[i,j] < -threshold):

            highcorr.extend([corr.iloc[i].name,corr.iloc[:,j].name])

        else:

            continue
highcorr = list(set(highcorr))

highcorr
sns.set_style('darkgrid')

sns.pairplot(dataset[highcorr],plot_kws=dict(s=4, edgecolor="w", linewidth=.01),markers='o')
sns.jointplot(x="cont1",y="cont9",data=dataset,kind='hex')
sns.jointplot(x="cont12",y="cont11",data=dataset,kind='hex')
def dropColumn(dataset,drop_col,inp=False):

    dataset.drop(drop_col,axis=1, inplace=inp)

    if inp == False:

        return dataset
drop_col = ["cont11","cont1"]

dropColumn(full_dataset,drop_col,inp=True)
for i in drop_col:

    con_variables.remove(i)
# viusalize the distribution of loss, which is our target

# thera are many outliers

sns.set_style('darkgrid')

plt.figure(figsize=(10,6))

sns.boxplot(y_train)
# it's a very skewed distribution

plt.figure(figsize=(10,6))

sns.distplot(y_train)
# apply log(1+loss) we can get a normal distribution

plt.figure(figsize=(10,6))

sns.distplot(np.log1p(y_train))
def catEncode(dataset):

    # ensure the features being converted to numeric

    le = LabelEncoder()

    dataset[cat_variables] = dataset[cat_variables].apply(lambda col: le.fit_transform(col))

    # Then I will convert it to a sparse matrix which uses way less memory as compared to dense matrix

    OneHot = OneHotEncoder(sparse=True)

    return OneHot.fit_transform(dataset[cat_variables])
full_dataset_sparse = catEncode(full_dataset)

full_dataset_sparse.shape
# calculate skewness of each numeric features

skewed_cols = full_dataset.loc[:,con_variables].apply(lambda x: skew(x.dropna()))

print(skewed_cols.sort_values())
plt.figure(figsize=(10,6))

sns.distplot(full_dataset["cont9"])
plt.figure(figsize=(10,6))

sns.distplot(full_dataset["cont8"])
# apply box-cox transformations

skewed_cols = skewed_cols[abs(skewed_cols) > 0.25].index.values

for skewed_col in skewed_cols:

    full_dataset[skewed_col],lam = boxcox(full_dataset[skewed_col] + 1)
skewed_cols = full_dataset.loc[:,con_variables].apply(lambda x: skew(x.dropna()))

print(skewed_cols.sort_values())
# apply standard scaling

SSL = StandardScaler()



for con_col in con_variables:

     full_dataset[con_col] = SSL.fit_transform(full_dataset[con_col].values.reshape(-1,1))
# we use the following two methods to evaluate our model



def logregobj(labels, preds):

    con = 2

    x =preds-labels

    grad =con*x / (np.abs(x)+con)

    hess =con**2 / (np.abs(x)+con)**2

    return grad, hess 



def log_mae(y,yhat):

    return mean_absolute_error(np.exp(y), np.exp(yhat))



log_mae_scorer = metrics.make_scorer(log_mae, greater_is_better = False)
Y = np.log(full_dataset[:model_index]["loss"]+200)

X = full_dataset[:model_index].drop(['id', 'loss'], axis= 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
linear_reg = linear_model.LinearRegression()

linear_reg.fit(x_train,y_train)
# figure out the coefficient of each feature



fig,ax = plt.subplots(figsize=(15,10))

plt.xticks(rotation=45) 

tick_spacing = 3

ax.plot(x_train.columns,linear_reg.coef_,label='LR')

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.title("Feature coefficient of Linear Regression Model")

plt.xlabel("Features")

plt.ylabel("Coefficient")

plt.legend()

plt.show()
y_pred = linear_reg.predict(x_test)

log_mae(y_test,y_pred)
alpha = [1, 5, 10, 20, 30, 40, 50]



ridge = Ridge()

parameters = {'alpha': alpha}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(x_train, y_train)
ridge_regressor.best_params_
y_pred = ridge_regressor.predict(x_test)

log_mae(y_test,y_pred)
rrg = linear_model.Ridge(alpha=40)

rrg.fit(x_train, y_train)
larg = linear_model.Lasso(alpha=1e-7)

larg.fit(x_train, y_train)
y_pred = larg.predict(x_test)

log_mae(y_test,y_pred)
fig,ax = plt.subplots(figsize=(15,10))

plt.xticks(rotation=45) 

tick_spacing = 3

ax.plot(x_train.columns,linear_reg.coef_,c='r',label='LR')

ax.plot(x_train.columns,larg.coef_,c='g',label="Lasso")

ax.plot(x_train.columns,rrg.coef_,c='b',label="Ridge")

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.title("Feature coefficient of Three Regression Model")

plt.xlabel("Features")

plt.ylabel("Coefficient")

plt.legend()

plt.show()
sub_x = full_dataset[model_index:]
sub_x.drop(["loss","id"],axis=1,inplace=True)
final_predict = np.exp(ridge_regressor.predict(sub_x)) - 200
results1 = pd.DataFrame()

results1['id'] = full_dataset[model_index:].id

results1['loss'] = final_predict

results1.to_csv("sub.csv", index=False)

print("Submission created.")
full_data_sparse = sparse.hstack((full_dataset_sparse,full_dataset[con_variables]), format='csr')

print(full_data_sparse.shape)



model_x = full_dataset_sparse[:model_index]

submission_x = full_dataset_sparse[model_index:]

model_y = np.log(full_dataset[:model_index].loss.values + 200)

ID = full_dataset.id[:model_index].values
def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):

## grid search for the best model

    model = GridSearchCV(estimator=est,

                         param_grid=param_grid,

                         scoring=log_mae_scorer,

                         verbose=10,

                         n_jobs=n_jobs,

                         iid=True,

                         refit=refit,

                         cv=cv)

    # fit grid search model

    model.fit(train_x, train_y)

    print("Best score: %0.3f" % model.best_score_)

    print("Best parameters set:", model.best_params_)

    print("Scores:", model.grid_scores_)

    return model
param_grid = {'objective':[logregobj],

              'learning_rate':[0.02, 0.04, 0.06, 0.08],

              'n_estimators':[1500],

              'max_depth': [9],

              'min_child_weight':[50],

              'subsample': [0.78],

              'colsample_bytree':[0.67],

              'gamma':[0.9],

              'nthread': [-1],

              'seed' : [1234]}



while False:

    model = search_model(model_x,

                         model_y,

                         xgb.XGBRegressor(),

                         param_grid,

                         n_jobs=1,

                         cv=4,

                         refit=True)
rgr = xgb.XGBRegressor(seed = 1234, 

                       learning_rate = 0.01, # smaller, better results, more time

                       n_estimators = 1500, # Number of boosted trees to fit.

                       max_depth=9, # the maximum depth of a tree

                       min_child_weight=50,

                       colsample_bytree=0.67, # the fraction of columns to be randomly samples for each tree

                       subsample=0.78, # the fraction of observations to be randomly samples for each tree

                       gamma=0.9, # Minimum loss reduction required to make a further partition on a leaf node of the tree, 

                       # the larger, the more conservative 

                       nthread = -1, # Number of parallel threads used to run xgboost.

                       silent = False # Whether to print messages while running boosting.

                      )

rgr.fit(model_x, model_y)
pred_y = np.exp(rgr.predict(submission_x)) - 200
plt.figure(figsize=(12,8))

plt.bar(range(len(rgr.feature_importances_)), rgr.feature_importances_,c='royalblue')

plt.ylim(0,0.1)

plt.show()
xgb.plot_importance(rgr,max_num_features=5,importance_type='weight')
np.argsort(rgr.feature_importances_)
results2 = pd.DataFrame()

results2['id'] = full_dataset[model_index:].id

results2['loss'] = pred_y

results2.to_csv("sub2.csv", index=False)

print("Submission created.")