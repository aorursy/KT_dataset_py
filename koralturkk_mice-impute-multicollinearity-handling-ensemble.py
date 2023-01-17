from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder

from sklearn.compose import make_column_transformer

import matplotlib.pyplot as plt

import seaborn as sns

from fancyimpute import IterativeImputer

import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

from tensorflow.keras.optimizers import Adam

from tensorflow import keras

from sklearn.model_selection import KFold
#Lets start-off by loading data

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
n_train = train.shape[0]

n_test = test.shape[0]



test_id = test["Id"]



train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)
print("AllData")

(all_data_rows, all_data_columns) = all_data.shape

print(" Number of rows: {} \n Number of columns: {}".format(all_data_rows, all_data_columns))

print(train.sample(3))



def display_missing(df):

    for col in df.columns.tolist():

        if df[col].isnull().sum()>0:

            print("{} column missing values: {} / {}".format(col, df[col].isnull().sum(),df.shape[0]))



display_missing(all_data)
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})



f, ax = plt.subplots(figsize=(15, 20))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

plt.show()
def filterProblematicColumns(df,threshold):

    listOfColumnNames = []

    for col in df.columns.tolist():

        if df[col].isnull().sum()> threshold:

            listOfColumnNames.append(col)

            print(col)

    

    return listOfColumnNames



portion = 0.2

threshold = all_data.shape[0] * portion





columnsToDrop = filterProblematicColumns(all_data, threshold)



all_data = all_data.drop(columns=columnsToDrop)
columns_with_missing_values = all_data.loc[:, all_data.isnull().any()]

missing_columns = columns_with_missing_values.columns.tolist()



print("Columns with Missing Values: ","\n", "\n", missing_columns, "\n")

print(columns_with_missing_values.describe())

print(all_data.shape)

print("\n", "--------------", "\n")

numcols = all_data.select_dtypes(include = np.number).columns



#Lets start by plotting a heatmap to determine if any variables are correlated

plt.figure(figsize = (12,8))

sns.heatmap(data= all_data[numcols].corr())

plt.show()

plt.gcf().clear()

def corr_missing_values(df, columns): 

    for column in columns:

        df_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

        df_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

        print(df_corr[df_corr['Feature 1'] == column])

        print("")



#corr_missing_values(all_data, [x for x in missing_columns if x in numcols])

numeric_columns = all_data.select_dtypes(include = np.number).columns.tolist()

nominal_columns = ["MSZoning","Street","LandContour","LotConfig","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","Foundation","Heating","CentralAir","Electrical","GarageType","SaleCondition"]

ordinal_columns = ["LotShape","Utilities","LandSlope","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","HeatingQC","KitchenQual","Functional","GarageFinish","GarageQual","GarageCond","PavedDrive","SaleType"]



#Check if numbers match, to make sure no columns are left out

print(all_data.shape[1])

print(len(numeric_columns), len(nominal_columns), len(ordinal_columns))

## Ordinal Encoding (by skipping null values)



ordinal_enc_dict = {}

for col_name in ordinal_columns:

    ordinal_enc_dict[col_name] = OrdinalEncoder()



    col = all_data[col_name]

    col_not_null = col[col.notnull()]

    reshaped_vals = col_not_null.values.reshape(-1,1)



    encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)

    all_data.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)



#Check if the values are encoded and no column has been skipped.   

print(all_data[ordinal_columns].head())
print(display_missing(all_data[nominal_columns]))
## Imputation with mode

nom_cols_withnull = all_data[nominal_columns].columns[all_data[nominal_columns].isnull().any()].tolist()



most_common_imputed = all_data[nom_cols_withnull].apply(lambda x: x.fillna(x.value_counts().index[0]))



for col_name in most_common_imputed.columns:

    all_data[col_name] = most_common_imputed[col_name]
nom_df = pd.get_dummies(all_data[nominal_columns], prefix=nominal_columns)



for col_name in nom_df.columns:

    all_data[col_name] = nom_df[col_name]



all_data = all_data.drop(columns= nominal_columns)



print(all_data)
MICE_imputer = IterativeImputer()

ordinal_mice = all_data.copy(deep = True)



ordinal_mice.iloc[:,:] = np.round(MICE_imputer.fit_transform(ordinal_mice))



for col_name in ordinal_columns:

    all_data[col_name] = ordinal_mice[col_name]



for col_name in numeric_columns:

    all_data[col_name] = ordinal_mice[col_name]



### Skewness in SalePrice



fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey= False, figsize= (8,4))



sns.distplot(train['SalePrice'], ax = ax0)

sns.distplot(np.log1p(train['SalePrice']), ax = ax1)



ax0.set(title= "Sale Price Distribution")

ax1.set(title= "Sale Price Distribution in Logarithmic Scale")
# Transforming y to avoid skewness



y = np.log1p(train["SalePrice"] )
x_axis_features = ["LotArea", "TotRmsAbvGrd", "GrLivArea", "OverallQual","OverallCond","YearBuilt","YrSold","MoSold"]







def subplots_vs_saleprice(df,x_features, no_cols, fig_size=(20,15)):

    

    no_features = len(x_features)

    number_of_rows = no_features // no_cols +1

    

    

    fig, axs = plt.subplots(number_of_rows, no_cols, figsize= fig_size)

    

    feature_index = 0

    

    for nrow in range(number_of_rows):

        for ncol in range(no_cols):  

            

            axs[nrow, ncol].scatter(df[x_features[feature_index]].values, np.log1p(train["SalePrice"].values))

            axs[nrow, ncol].set_title('{} vs SalePrice'.format(x_features[feature_index]))

            feature_index += 1

            

            if feature_index == no_features:

                break

          

            



    for ax in axs.flat:

        ax.set(xlabel='', ylabel='SalePrice (logScale)')



    # Hide x labels and tick labels for top plots and y ticks for right plots.

    for ax in axs.flat:

        ax.label_outer()



        

subplots_vs_saleprice(train, x_axis_features, 3)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()





X = all_data.loc[:n_train-1,:]

test = all_data.loc[n_train:,:]



X_copy = X.copy(deep = False)

test_copy = test.copy(deep = False)



X = scaler.fit_transform(X)

test = scaler.fit_transform(test)
from sklearn.linear_model import Ridge, Lasso, LinearRegression  

from sklearn.kernel_ridge import KernelRidge  

from sklearn.model_selection import GridSearchCV

import numpy as np  

from sklearn import metrics  

from sklearn.metrics import mean_squared_error  

import xgboost as xgb  

from sklearn.linear_model import ElasticNet  

from sklearn.svm import SVR









print("-----------Stats for SVR-----------------", "\n")



svr = SVR(epsilon = 0.01)

parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]

grid_svr = GridSearchCV(svr, parameters, cv = 10, scoring="neg_mean_squared_error", verbose=0,n_jobs = -1)

grid_svr.fit(X, y)

print(pd.DataFrame(grid_svr.cv_results_))





print("-----------Stats for ElasticNet-----------------", "\n")



elastic_net = ElasticNet(selection="random")

elastic_params = {"max_iter": [1, 5, 10],

                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],

                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}



grid_elastic= GridSearchCV(estimator = elastic_net, param_grid= elastic_params, scoring="neg_mean_squared_error", cv=5, verbose=0,n_jobs = -1)

grid_elastic.fit(X,y)

print(pd.DataFrame(grid_elastic.cv_results_))







print("-----------Stats for XGB-----------------", "\n")



xg_boost = xgb.XGBRegressor(objective='reg:squarederror')

house_price_dmatrix = xgb.DMatrix(data = X, label=y)

xgb_params = {"learning_rate":[0.01,0.1,0.5,0.9],"n_estimators":[200],"subsample": [0.3,0.5,0.9]}

grid_xgb= GridSearchCV(estimator = xg_boost, param_grid= xgb_params, scoring="neg_mean_squared_error", cv=5, verbose=0,n_jobs = -1)

grid_xgb.fit(X,y)

print(pd.DataFrame(grid_xgb.cv_results_))







print("-----------Stats for KernelRidge-----------------", "\n")



kernel_ridge = KernelRidge()



param_grid_kr = {'alpha': [0.001,0.003,0.01,0.3,0.1,0.3,1,3,10,30,100,300,1000,3000,100000],

              'kernel':['polynomial'], 

              'degree':[2,3,4,5,6,7,8],

              'coef0':[0,1,1.5,2,2.5,3,3.5,10]}

kernel_ridge = GridSearchCV(kernel_ridge, 

                 param_grid = param_grid_kr, 

                 scoring = "neg_mean_squared_error", 

                 cv = 5,

                 n_jobs = -1,

                 verbose = 0)



kernel_ridge.fit(X,y)

print(pd.DataFrame(kernel_ridge.cv_results_))

k_best = kernel_ridge.best_estimator_

kernel_ridge.best_score_



param_grid = {"alpha": [0.001,0.003,0.01,0.3,0.1,0.3,1,3,10,30,100,300,1000,3000,100000]}



print("-----------Stats for Ridge-----------------", "\n")

ridge = Ridge()  

grid_ridge = GridSearchCV(ridge, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1, verbose = 0) 

grid_ridge.fit(X, y)

print(pd.DataFrame(grid_ridge.cv_results_))



print("-----------Stats for Lasso-----------------", "\n")

lasso = Lasso()  

grid_lasso = GridSearchCV(lasso, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1, verbose=0) 

grid_lasso.fit(X, y)

print(pd.DataFrame(grid_lasso.cv_results_))



print("-----------Scoreboard for Kernel Ridge-----------------", "\n")

print(kernel_ridge.best_score_)

print(kernel_ridge.best_estimator_.alpha)



print("-----------Scoreboard for Ridge-----------------", "\n")

print(grid_ridge.best_score_)

print(grid_ridge.best_estimator_.alpha)



print("-----------Scoreboard for Lasso-----------------", "\n")



print(grid_lasso.best_score_)

print(grid_lasso.best_estimator_.alpha)



print("-----------Scoreboard for XGB-----------------", "\n")

print(grid_xgb.best_score_)

print(grid_xgb.best_params_)



print("-----------Scoreboard for Elastic-----------------", "\n")

print(grid_elastic.best_score_)

print(grid_elastic.best_params_)



print("-----------Scoreboard for SVR-----------------", "\n")

print(grid_svr.best_score_)

print(grid_svr.best_params_)
from sklearn.ensemble import StackingRegressor

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score



estimators = [('ridge', grid_ridge.best_estimator_) ,

              ('lasso', grid_lasso.best_estimator_),

              ("elastic", grid_elastic.best_estimator_),

              ('kernel_ridge', kernel_ridge.best_estimator_),

              ("svr", grid_svr.best_estimator_)

             ]



stack = StackingRegressor(estimators=estimators, final_estimator=grid_xgb.best_estimator_)



print("\n ---Score for Stack--- ")

print(cross_val_score(stack, X, y, cv=10, scoring="neg_mean_squared_error", n_jobs=-1).mean())
lasso=grid_lasso.best_estimator_

lasso.fit(X,y)

feature_importance_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=X_copy.columns)

feature_importance_lasso.sort_values("Feature Importance",ascending=False)

feature_importance_lasso[feature_importance_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))



plt.xticks(rotation=90)

plt.show()



abs_feature_importance_lasso = pd.DataFrame({"Feature Importance":abs(lasso.coef_)}, index=X_copy.columns)



print(abs_feature_importance_lasso[abs_feature_importance_lasso["Feature Importance"]!=0].sort_values("Feature Importance", ascending =False).head(10))
corr_matrix = X_copy.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

corr_threshold = 0.85

to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]



X_new = X_copy.copy(deep=False)

X_new =X_new.drop(X_new[to_drop], axis=1)

X_new = scaler.fit_transform(X_new)



test_new = test_copy.copy(deep=False)

test_new =test_new.drop(test_new[to_drop], axis=1)

test_new = scaler.fit_transform(test_new)



print("-----------Stats for ElasticNet-----------------", "\n")



elastic_net = ElasticNet()

elastic_params = {"max_iter": [1, 5, 10],

                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],

                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}



grid_elastic= GridSearchCV(estimator = elastic_net, param_grid= elastic_params, scoring="neg_mean_squared_error", cv=10, verbose=0,n_jobs=-1)

grid_elastic.fit(X_new,y)

print(pd.DataFrame(grid_elastic.cv_results_))





print("-----------Stats for SVR-----------------", "\n")

svr = SVR(epsilon = 0.01)

parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]

grid_svr = GridSearchCV(svr, parameters, cv = 10,n_jobs=-1)

grid_svr.fit(X, y)

print(pd.DataFrame(grid_svr.cv_results_))







print("-----------Stats for XGB-----------------", "\n")

xg_boost = xgb.XGBRegressor(objective='reg:squarederror')

house_price_dmatrix = xgb.DMatrix(data = X_new, label=y)

xgb_params = {"learning_rate":[0.01,0.1,0.5,0.9],"n_estimators":[200],"subsample": [0.3,0.5,0.9]}



grid_xgb= GridSearchCV(estimator = xg_boost, param_grid= xgb_params, scoring="neg_mean_squared_error", cv=10, verbose=0,n_jobs=-1)

grid_xgb.fit(X_new,y)

print(pd.DataFrame(grid_xgb.cv_results_))



print("-----------Stats for KernelRidge-----------------", "\n")

kernel_ridge = KernelRidge()

param_grid_kr = {'alpha': [0.001,0.003,0.01,0.3,0.1,0.3,1,3,10,30,100,300,1000,3000,100000],

              'kernel':['polynomial'], 

              'degree':[2,3,4,5,6,7,8],

              'coef0':[0,1,1.5,2,2.5,3,3.5,10]}

kernel_ridge = GridSearchCV(kernel_ridge, 

                 param_grid = param_grid_kr, 

                 scoring = "neg_mean_squared_error", 

                 cv = 10,

                 n_jobs = -1,

                 verbose = 0)

kernel_ridge.fit(X_new,y)

print(pd.DataFrame(kernel_ridge.cv_results_))

k_best = kernel_ridge.best_estimator_

kernel_ridge.best_score_



param_grid = {"alpha": [0.001,0.003,0.01,0.3,0.1,0.3,1,3,10,30,100,300,1000,3000,100000]}



print("-----------Stats for Ridge-----------------", "\n")

ridge = Ridge()  

grid_ridge = GridSearchCV(ridge, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1, verbose = 0) 

grid_ridge.fit(X_new, y)

print(pd.DataFrame(grid_ridge.cv_results_))



print("-----------Stats for SVR-----------------", "\n")

grid_ridge = GridSearchCV(ridge, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1, verbose = 0) 

grid_ridge.fit(X_new, y)

print(pd.DataFrame(grid_ridge.cv_results_))



print("-----------Stats for Lasso-----------------", "\n")

lasso = Lasso()  

grid_lasso = GridSearchCV(lasso, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1, verbose=0) 

grid_lasso.fit(X_new, y)

print(pd.DataFrame(grid_lasso.cv_results_))



print("-----------Scoreboard for Kernel Ridge-----------------", "\n")

print(kernel_ridge.best_score_)

print(kernel_ridge.best_estimator_.alpha)



print("-----------Scoreboard for Ridge-----------------", "\n")

print(grid_ridge.best_score_)

print(grid_ridge.best_estimator_.alpha)



print("-----------Scoreboard for Lasso-----------------", "\n")



print(grid_lasso.best_score_)

print(grid_lasso.best_estimator_.alpha)



print("-----------Scoreboard for XGB-----------------", "\n")

print(grid_xgb.best_score_)

print(grid_xgb.best_params_)



print("-----------Scoreboard for Elastic-----------------", "\n")

print(grid_elastic.best_score_)

print(grid_elastic.best_params_)  



print("-----------Scoreboard for SVR-----------------", "\n")

print(grid_svr.best_score_)

print(grid_svr.best_params_)
estimators = [('ridge', grid_ridge.best_estimator_) ,

              ('lasso', grid_lasso.best_estimator_),

              ("elastic", grid_elastic.best_estimator_),

              ('kernel_ridge', kernel_ridge.best_estimator_),

              ]



stack = StackingRegressor(estimators=estimators, final_estimator=grid_xgb.best_estimator_ )



print("\n ---Score for Stack--- ")

print(cross_val_score(stack, X_new, y, cv=10, scoring="neg_mean_squared_error",n_jobs=-1).mean())
from sklearn.decomposition import PCA, KernelPCA



pca = PCA(n_components=190)

X_pca = pca.fit_transform(X)

test_pca = pca.transform(test)
from sklearn.linear_model import Ridge, Lasso, LinearRegression  

from sklearn.kernel_ridge import KernelRidge  

from sklearn.model_selection import GridSearchCV

import numpy as np  

from sklearn import metrics  

from sklearn.metrics import mean_squared_error  

import xgboost as xgb  

from sklearn.linear_model import ElasticNet  

from sklearn.svm import SVR



print("-----------Stats for SVR-----------------", "\n")



svr = SVR(epsilon = 0.01)

parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]

grid_svr = GridSearchCV(svr, parameters, cv = 10, scoring="neg_mean_squared_error", verbose=0,n_jobs=-1)

grid_svr.fit(X_pca, y)

print(pd.DataFrame(grid_svr.cv_results_))





print("-----------Stats for ElasticNet-----------------", "\n")



elastic_net = ElasticNet()

elastic_params = {"max_iter": [1, 5, 10],

                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],

                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}



grid_elastic= GridSearchCV(estimator = elastic_net, param_grid= elastic_params, scoring="neg_mean_squared_error", cv=5, verbose=0,n_jobs=-1)

grid_elastic.fit(X_pca,y)

print(pd.DataFrame(grid_elastic.cv_results_))







print("-----------Stats for XGB-----------------", "\n")



xg_boost = xgb.XGBRegressor(objective='reg:squarederror')

house_price_dmatrix = xgb.DMatrix(data = X, label=y)

xgb_params = {"learning_rate":[0.01,0.1,0.5,0.9],"n_estimators":[200],"subsample": [0.3,0.5,0.9]}

grid_xgb= GridSearchCV(estimator = xg_boost, param_grid= xgb_params, scoring="neg_mean_squared_error", cv=5, verbose=0,n_jobs=-1)

grid_xgb.fit(X_pca,y)

print(pd.DataFrame(grid_xgb.cv_results_))







print("-----------Stats for KernelRidge-----------------", "\n")



kernel_ridge = KernelRidge()



param_grid_kr = {'alpha': [0.001,0.003,0.01,0.3,0.1,0.3,1,3,10,30,100,300,1000,3000,100000],

              'kernel':['polynomial'], 

              'degree':[2,3,4,5,6,7,8],

              'coef0':[0,1,1.5,2,2.5,3,3.5,10]}

kernel_ridge = GridSearchCV(kernel_ridge, 

                 param_grid = param_grid_kr, 

                 scoring = "neg_mean_squared_error", 

                 cv = 5,

                 n_jobs = -1,

                 verbose = 0)



kernel_ridge.fit(X_pca,y)

print(pd.DataFrame(kernel_ridge.cv_results_))

k_best = kernel_ridge.best_estimator_

kernel_ridge.best_score_



param_grid = {"alpha": [0.001,0.003,0.01,0.3,0.1,0.3,1,3,10,30,100,300,1000,3000,100000]}



print("-----------Stats for Ridge-----------------", "\n")

ridge = Ridge()  

grid_ridge = GridSearchCV(ridge, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1, verbose = 0) 

grid_ridge.fit(X_pca, y)

print(pd.DataFrame(grid_ridge.cv_results_))



print("-----------Stats for Lasso-----------------", "\n")

lasso = Lasso()  

grid_lasso = GridSearchCV(lasso, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1, verbose=0) 

grid_lasso.fit(X_pca, y)

print(pd.DataFrame(grid_lasso.cv_results_))



print("-----------Scoreboard for Kernel Ridge-----------------", "\n")

print(kernel_ridge.best_score_)

print(kernel_ridge.best_estimator_.alpha)



print("-----------Scoreboard for Ridge-----------------", "\n")

print(grid_ridge.best_score_)

print(grid_ridge.best_estimator_.alpha)



print("-----------Scoreboard for Lasso-----------------", "\n")



print(grid_lasso.best_score_)

print(grid_lasso.best_estimator_.alpha)



print("-----------Scoreboard for XGB-----------------", "\n")

print(grid_xgb.best_score_)

print(grid_xgb.best_params_)



print("-----------Scoreboard for Elastic-----------------", "\n")

print(grid_elastic.best_score_)

print(grid_elastic.best_params_)



print("-----------Scoreboard for SVR-----------------", "\n")

print(grid_svr.best_score_)

print(grid_svr.best_params_)
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score



estimators = [('ridge', grid_ridge.best_estimator_) ,

              ('lasso', grid_lasso.best_estimator_),

              ("elastic", grid_elastic.best_estimator_),

              ('kernel_ridge', kernel_ridge.best_estimator_),

              ("svr", grid_svr.best_estimator_)

             ]



stack = StackingRegressor(estimators=estimators, final_estimator=grid_xgb.best_estimator_)



print("\n ---Score for Stack--- ")

print(cross_val_score(stack, X_pca, y, cv=10, scoring="neg_mean_squared_error",n_jobs=-1).mean())
fitted_stack = stack.fit(X,y)

predictions = np.expm1(fitted_stack.predict(test))



output = pd.DataFrame({'Id': test_id, 'SalePrice': predictions})

output.to_csv('stack.csv', index=False)
### Things to do



### Add new features with feature engineering

### Try removing unimportant features for performance