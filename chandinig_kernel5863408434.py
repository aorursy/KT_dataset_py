# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


pd.set_option('max_columns', 105)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

%matplotlib inline

sns.set()



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)

#warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# setting the number of cross validations used in the Model part 

nr_cv = 5



# switch for using log values for SalePrice and features     

use_logvals = 1    

# target used for correlation 

target = 'SalePrice_Log'

    

# only columns with correlation above this threshold value  

# are used for the ML Regressors in Part 3

min_val_corr = 0.4    

    

# switch for dropping columns that are similar to others already used and show a high correlation to these     

drop_similar = 1
def get_best_score(grid):

    

    best_score = np.sqrt(-grid.best_score_)

    print(best_score)    

    print(grid.best_params_)

    print(grid.best_estimator_)

    

    return best_score
def print_cols_large_corr(df, nr_c, targ) :

    corr = df.corr()

    corr_abs = corr.abs()

    print (corr_abs.nlargest(nr_c, targ)[targ])
def plot_corr_matrix(df, nr_c, targ) :

    

    corr = df.corr()

    corr_abs = corr.abs()

    cols = corr_abs.nlargest(nr_c, targ)[targ].index

    cm = np.corrcoef(df[cols].values.T)



    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))

    sns.set(font_scale=1.25)

    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 

                fmt='.2f', annot_kws={'size': 10}, 

                yticklabels=cols.values, xticklabels=cols.values

               )

    plt.show()
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print(train.shape)

print("*"*50)

print(test.shape)
print(train.info())

print("*"*50)

print(test.info())
train.head()

train.describe()

test.head()

test.describe()

sns.distplot(train['SalePrice']);

#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
train['SalePrice_Log'] = np.log(train['SalePrice'])



sns.distplot(train['SalePrice_Log']);

# skewness and kurtosis

print("Skewness: %f" % train['SalePrice_Log'].skew())

print("Kurtosis: %f" % train['SalePrice_Log'].kurt())

# dropping old column

train.drop('SalePrice', axis= 1, inplace=True)
numerical_feats = train.dtypes[train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = train.dtypes[train.dtypes == "object"].index

print("Number of Categorical features: ", len(categorical_feats))
print(train[numerical_feats].columns)

print("*"*100)

print(train[categorical_feats].columns)
train[numerical_feats].head()

train[categorical_feats].head()

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# columns where NaN values have meaning e.g. no pool etc.

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MSZoning', 'Utilities']



# replace 'NaN' with 'None' in these columns

for col in cols_fillna:

    train[col].fillna('None',inplace=True)

    test[col].fillna('None',inplace=True)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)
# fillna with mean for the remaining columns: LotFrontage, GarageYrBlt, MasVnrArea

train.fillna(train.mean(), inplace=True)

test.fillna(test.mean(), inplace=True)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)
train.isnull().sum().sum()

test.isnull().sum().sum()

for col in numerical_feats:

    print('{:15}'.format(col), 

          'Skewness: {:05.2f}'.format(train[col].skew()) , 

          '   ' ,

          'Kurtosis: {:06.2f}'.format(train[col].kurt())  

         )
sns.distplot(train['GrLivArea']);

#skewness and kurtosis

print("Skewness: %f" % train['GrLivArea'].skew())

print("Kurtosis: %f" % train['GrLivArea'].kurt())
sns.distplot(train['LotArea']);

#skewness and kurtosis

print("Skewness: %f" % train['LotArea'].skew())

print("Kurtosis: %f" % train['LotArea'].kurt())
for df in [train,test]:

    df['GrLivArea_Log'] = np.log(df['GrLivArea'])

    df.drop('GrLivArea', inplace= True, axis = 1)

    df['LotArea_Log'] = np.log(df['LotArea'])

    df.drop('LotArea', inplace= True, axis = 1)

    

    

    

numerical_feats = train.dtypes[train.dtypes != "object"].index
sns.distplot(train['GrLivArea_Log']);

#skewness and kurtosis

print("Skewness: %f" % train['GrLivArea_Log'].skew())

print("Kurtosis: %f" % train['GrLivArea_Log'].kurt())
sns.distplot(train['LotArea_Log']);

#skewness and kurtosis

print("Skewness: %f" % train['LotArea_Log'].skew())

print("Kurtosis: %f" % train['LotArea_Log'].kurt())
nr_rows = 12

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



li_num_feats = list(numerical_feats)

li_not_plot = ['Id', 'SalePrice', 'SalePrice_Log']

li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]





for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_plot_num_feats):

            sns.regplot(train[li_plot_num_feats[i]], train[target], ax = axs[r][c])

            stp = stats.pearsonr(train[li_plot_num_feats[i]], train[target])

            #axs[r][c].text(0.4,0.9,"title",fontsize=7)

            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    

plt.show()
train = train.drop(

    train[(train['OverallQual']==10) & (train['SalePrice_Log']<12.3)].index)
train = train.drop(

    train[(train['GrLivArea_Log']>8.3) & (train['SalePrice_Log']<12.5)].index)
corr = train.corr()

corr_abs = corr.abs()



nr_num_cols = len(numerical_feats)

ser_corr = corr_abs.nlargest(nr_num_cols, target)[target]



cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)

cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)
print(ser_corr)

print("*"*30)

print("List of numerical features with r above min_val_corr :")

print(cols_abv_corr_limit)

print("*"*30)

print("List of numerical features with r below min_val_corr :")

print(cols_bel_corr_limit)
for catg in list(categorical_feats) :

    print(train[catg].value_counts())

    print('#'*50)
li_cat_feats = list(categorical_feats)

nr_rows = 15

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y=target, data=train, ax = axs[r][c])

    

plt.tight_layout()    

plt.show()
catg_strong_corr = [ 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 

                     'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']



catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 

                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 

                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 

                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 

                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 

                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 

                  'SaleCondition' ]
nr_feats = len(cols_abv_corr_limit)

plot_corr_matrix(train, nr_feats, target)

id_test = test['Id']



to_drop_num  = cols_bel_corr_limit

to_drop_catg = catg_weak_corr



cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 



for df in [train, test]:

    df.drop(cols_to_drop, inplace= True, axis = 1)
test.columns
catg_list = catg_strong_corr.copy()

catg_list.remove('Neighborhood')



for catg in catg_list :

    #sns.catplot(x=catg, y=target, data=train, kind='boxen')

    sns.violinplot(x=catg, y=target, data=train)

    plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(16, 5)

sns.violinplot(x='Neighborhood', y=target, data=train, ax=ax)

plt.xticks(rotation=45)

plt.show()
for catg in catg_list :

    g = train.groupby(catg)[target].mean()

    print(g)
# 'MSZoning'

msz_catg2 = ['RM', 'RH']

msz_catg3 = ['RL', 'FV'] 





# Neighborhood

nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']

nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']



# Condition2

cond2_catg2 = ['Norm', 'RRAe']

cond2_catg3 = ['PosA', 'PosN'] 



# SaleType

SlTy_catg1 = ['Oth']

SlTy_catg3 = ['CWD']

SlTy_catg4 = ['New', 'Con']
for df in [train, test]:

    

    df['MSZ_num'] = 1  

    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    

    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        

    

    df['NbHd_num'] = 1       

    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    

    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    



    df['Cond2_num'] = 1       

    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    

    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    

    

    df['Mas_num'] = 1       

    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 

    

    df['ExtQ_num'] = 1       

    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     

    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     

    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     

   

    df['BsQ_num'] = 1          

    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     

    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     

 

    df['CA_num'] = 0          

    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    



    df['Elc_num'] = 1       

    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 





    df['KiQ_num'] = 1       

    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     

    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     

    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      

    

    df['SlTy_num'] = 2       

    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  

    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  

    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4  
new_col_num = ['MSZ_num', 'NbHd_num', 'Cond2_num', 'Mas_num', 'ExtQ_num', 'BsQ_num', 'CA_num', 'Elc_num', 'KiQ_num', 'SlTy_num']



nr_rows = 4

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(new_col_num):

            sns.regplot(train[new_col_num[i]], train[target], ax = axs[r][c])

            stp = stats.pearsonr(train[new_col_num[i]], train[target])

            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    

plt.show()
catg_cols_to_drop = ['Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']



corr1 = train.corr()

corr_abs_1 = corr1.abs()



nr_all_cols = len(train)

ser_corr_1 = corr_abs_1.nlargest(nr_all_cols, target)[target]



print(ser_corr_1)

cols_bel_corr_limit_1 = list(ser_corr_1[ser_corr_1.values <= min_val_corr].index)





for df in [train, test] :

    df.drop(catg_cols_to_drop, inplace= True, axis = 1)

    df.drop(cols_bel_corr_limit_1, inplace= True, axis = 1) 
corr2 = train.corr()

corr_abs_2 = corr2.abs()



nr_all_cols = len(train)

ser_corr_2 = corr_abs_2.nlargest(nr_all_cols, target)[target]



print(ser_corr_2)
train.head()

test.head()

corr = train.corr()

corr_abs = corr.abs()



nr_all_cols = len(train)

print (corr_abs.nlargest(nr_all_cols, target)[target])
nr_feats=len(train.columns)

plot_corr_matrix(train, nr_feats, target)
cols = corr_abs.nlargest(nr_all_cols, target)[target].index

cols = list(cols)



if drop_similar == 1 :

    for col in ['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt'] :

        if col in cols: 

            cols.remove(col)
cols = list(cols)

print(cols)
feats = cols.copy()

feats.remove('SalePrice_Log')



print(feats)
train_ml = train[feats].copy()

test_ml  = test[feats].copy()



y = train[target]
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

train_ml_sc = sc.fit_transform(train_ml)

test_ml_sc = sc.transform(test_ml)

train_ml_sc = pd.DataFrame(train_ml_sc)

train_ml_sc.head()
X = train_ml.copy()

y = train[target]

X_test = test_ml.copy()



X_sc = train_ml_sc.copy()

y_sc = train[target]

X_test_sc = test_ml_sc.copy()



X.info()

X_test.info()
X.head()

X_sc.head()

X_test.head()

from sklearn.model_selection import GridSearchCV

score_calc = 'neg_mean_squared_error'
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

grid_linear = GridSearchCV(linreg, parameters, cv=nr_cv, verbose=1 , scoring = score_calc)

grid_linear.fit(X, y)



sc_linear = get_best_score(grid_linear)
linreg_sc = LinearRegression()

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

grid_linear_sc = GridSearchCV(linreg_sc, parameters, cv=nr_cv, verbose=1 , scoring = score_calc)

grid_linear_sc.fit(X_sc, y)



sc_linear_sc = get_best_score(grid_linear_sc)
linregr_all = LinearRegression()

#linregr_all.fit(X_train_all, y_train_all)

linregr_all.fit(X, y)

pred_linreg_all = linregr_all.predict(X_test)

pred_linreg_all[pred_linreg_all < 0] = pred_linreg_all.mean()
sub_linreg = pd.DataFrame()

sub_linreg['Id'] = id_test

sub_linreg['SalePrice'] = pred_linreg_all
sub_linreg.head()
from sklearn.tree import DecisionTreeRegressor



param_grid = { 'max_depth' : [7,8,9,10] , 'max_features' : [11,12,13,14] ,

               'max_leaf_nodes' : [None, 12,15,18,20] ,'min_samples_split' : [20,25,30],

                'presort': [False,True] , 'random_state': [5] }

            

grid_dtree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)

grid_dtree.fit(X, y)



sc_dtree = get_best_score(grid_dtree)



pred_dtree = grid_dtree.predict(X_test)
dtree_pred = grid_dtree.predict(X_test)

sub_dtree = pd.DataFrame()

sub_dtree['Id'] = id_test

sub_dtree['SalePrice'] = dtree_pred
sub_dtree.head()
from sklearn.ensemble import RandomForestRegressor



param_grid = {'min_samples_split' : [3,4,6,10], 'n_estimators' : [70,100], 'random_state': [5] }

grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)

grid_rf.fit(X, y)



sc_rf = get_best_score(grid_rf)
pred_rf = grid_rf.predict(X_test)



sub_rf = pd.DataFrame()

sub_rf['Id'] = id_test

sub_rf['SalePrice'] = pred_rf 



if use_logvals == 1:

    sub_rf['SalePrice'] = np.exp(sub_rf['SalePrice']) 



sub_rf.to_csv('rf.csv',index=False)
sub_rf.head(10)

os.mkdir('/kaggle/output')

sub_rf.to_csv('/kaggle/output/RF.csv',index=False)
from sklearn.neighbors import KNeighborsRegressor



param_grid = {'n_neighbors' : [3,4,5,6,7,10,15] ,    

              'weights' : ['uniform','distance'] ,

              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}



grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)

grid_knn.fit(X_sc, y_sc)



sc_knn = get_best_score(grid_knn)
pred_knn = grid_knn.predict(X_test_sc)



sub_knn = pd.DataFrame()

sub_knn['Id'] = id_test

sub_knn['SalePrice'] = pred_knn



if use_logvals == 1:

    sub_knn['SalePrice'] = np.exp(sub_knn['SalePrice']) 



sub_knn.to_csv('knn.csv',index=False)
sub_knn.head()
sub_rf.to_csv('/kaggle/output/KNN.csv',index=False)