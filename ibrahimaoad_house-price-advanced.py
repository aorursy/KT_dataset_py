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
import pandas as pd



# Read the data

X_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_test_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')



# Remove rows with missing target

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)



X_full.shape
#Fast check for any duplicated values

X_full.drop_duplicates()
#First We'll drop cells with very high NA values

#We've created the function below for resuable access

#Now we're slecting columns that has more than 33%of it's conent as null values and returning them as a list



def get_cols_with_many_nas(data, metric):

    

    col_with_nas_all = dict(data.isna().sum())

    col_with_nas_filtered = dict()

    cols_list = list()



    for (key, value) in col_with_nas_all.items():

        if value > (data.shape[0]/metric) :

            col_with_nas_filtered[key] = value

            cols_list.append(key)

    print(col_with_nas_filtered)

    return cols_list

cols_to_drop = get_cols_with_many_nas(X_full,3)
# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)



def get_cols_with_many_unique(df, cardinality_limit):

    

    cat_all_unique = dict(df.select_dtypes(include=['object']).nunique())

    cat_filtered = dict()

    cols_list_cat = list()

    for (key, value) in cat_all_unique.items():

        if value > cardinality_limit :

            cat_filtered[key] = value

            cols_list_cat.append(key)

    print(cat_filtered)

    return cols_list_cat

cat_cols_to_drop = get_cols_with_many_unique(X_full,12)
#Dropping the above from data

X_fullV1 = X_full.drop(cols_to_drop+cat_cols_to_drop, axis=1)

X_testV1 = X_test_full.drop(cols_to_drop+cat_cols_to_drop, axis=1)

print(X_fullV1.shape)

print(X_testV1.shape)
import matplotlib.pyplot as plt



#Plotting Numerical data against SalePrice to spot outliers

X_full_nums = X_fullV1.select_dtypes(exclude=['object'])



def plots(df,x,y,kind):

    df[[x,y]].plot(kind=kind,x=x,y=y)

    plt.show()



for col in X_full_nums:

    plots(X_full_nums,'SalePrice',col,'scatter')
#After digging into graphs,below is a function to eliminate outliers:



def drop_outliers(df,column,limit):

    df.drop(df[column][df[column]>limit].index, inplace=True)





outliers_conditions = [('LotFrontage',200),('LotArea',80000),('MasVnrArea',1100),('BsmtFinSF1',2200)

              ,('BsmtFinSF2',1100),('TotalBsmtSF',2800),('1stFlrSF',2800),('GrLivArea',4000),

             ('WoodDeckSF',800),('OpenPorchSF',500),('EnclosedPorch',400),('3SsnPorch',400)

             ,('ScreenPorch',400),('PoolArea',400),('MiscVal',400)]



for col, cond in outliers_conditions:

    drop_outliers(X_fullV1,col,cond)

print(X_fullV1.shape)

y = X_fullV1.SalePrice

X_fullV2= X_fullV1.drop(['SalePrice'], axis=1)

print(y.shape)

print(X_fullV2.shape)

print(X_testV1.shape)
from sklearn.impute import SimpleImputer



X_full_numsV1 = X_fullV2.select_dtypes(exclude=['object'])

X_test_numsV1 = X_testV1.select_dtypes(exclude=['object'])



num_imputer = SimpleImputer(strategy='median')



imputed_nums = pd.DataFrame(num_imputer.fit_transform(X_full_numsV1))



imputed_nums.columns = X_full_numsV1.columns

imputed_nums.index = X_full_numsV1.index



imputed_test_nums = pd.DataFrame(num_imputer.transform(X_test_numsV1))

imputed_test_nums.columns = X_test_numsV1.columns

imputed_test_nums.index = X_test_numsV1.index



print(imputed_nums.head(10))

print(imputed_test_nums.head(10))
from sklearn.preprocessing import OneHotEncoder



X_full_catV1 = X_fullV2.select_dtypes(include=['object'])

X_test_catV1 = X_testV1.select_dtypes(include=['object'])



cat_imputer = SimpleImputer(strategy='most_frequent')

imputed_cats = pd.DataFrame(cat_imputer.fit_transform(X_full_catV1))



imputed_cats.columns = X_full_catV1.columns

imputed_cats.index = X_full_catV1.index



imputed_test_cats = pd.DataFrame(cat_imputer.transform(X_test_catV1))



imputed_test_cats.columns = X_test_catV1.columns

imputed_test_cats.index = X_test_catV1.index



cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

encoded_cats = pd.DataFrame(cat_encoder.fit_transform(imputed_cats))

encoded_test_cats = pd.DataFrame(cat_encoder.transform(imputed_test_cats))



encoded_cats.index = imputed_cats.index

encoded_test_cats.index = imputed_test_cats.index

print(encoded_cats.head(10))

print(encoded_test_cats.head(10))
X_fullV3 = pd.concat([imputed_nums, encoded_cats], axis=1)

X_testV2 = pd.concat([imputed_test_nums, encoded_test_cats], axis=1)



print(X_fullV3.shape)

print(y.shape)

print(X_testV2.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_fullV4 = pd.DataFrame(scaler.fit_transform(X_fullV3))

X_fullV4.columns = X_fullV3.columns

X_fullV4.index = X_fullV3.index



X_testV3 = pd.DataFrame(scaler.transform(X_testV2))

X_testV3.columns = X_testV2.columns

X_testV3.index = X_testV2.index



X_fullV4
#But first, Let's devise a cross-validation methodology once and for all



from sklearn.model_selection import cross_val_score



def rmse_cv(model,X_train_try,y_try):

    rmse= np.sqrt(-cross_val_score(model, X_train_try, y_try, scoring="neg_mean_squared_error", cv=5))

    return(rmse)
# first import library

from sklearn.linear_model import LassoCV

from numpy import log

#now create our object

model_lasso = LassoCV(n_jobs = -1).fit(X_fullV4, log(y))

res = rmse_cv(model_lasso,X_fullV4,log(y))

print("Mean:",res.mean())

print("Min: ",res.min())
coef = pd.Series(model_lasso.coef_, index = X_fullV3.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
import matplotlib



# plotting feature importances!

imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
all_laso = dict(coef != 0)

laso_filtered = dict()

laso_list = list()



for (key, value) in all_laso.items():

    if value != 0 :

        laso_filtered[key] = value

        laso_list.append(key)

X_fullV5= X_fullV3[laso_list]

X_testV4= X_testV2[laso_list]
print(X_fullV5.shape)

print(X_testV4.shape)

print(y.shape)
X_fullV5
from sklearn.model_selection import train_test_split

# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X_fullV5, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)



print(X_testV4.shape)

print(X_train.shape)

print(X_valid.shape)

print(y_train.shape)

print(y_valid.shape)

print(y.shape)
from sklearn.linear_model import Lasso, ElasticNet, Ridge

from numpy import log

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import GridSearchCV

alphas = [.0009,.0008,.0007,.0006,.0005,.0004,.0003,.0002,.0001,

         .009,.008,.007,.006,.005,.004,.003,.002,.001,

         .09,.08,.07,.06,.05,.04,.03,.02,.01,

         1,.9,.8,.7,.6,.5,.4,.3,.2,.1

         ,50,40,30,20,10,9,8,7,6,5,4,3,2]

param_grid = [{'alpha':alphas}]

model_final = Lasso()

model_final2 = ElasticNet()

model_final3 = Ridge()



housing_grid = GridSearchCV(model_final, param_grid, cv=10,n_jobs= -1,verbose=1)

housing_grid_el = GridSearchCV(model_final2, param_grid, cv=10,n_jobs= -1,verbose=1)

housing_grid_ri = GridSearchCV(model_final3, param_grid, cv=10,n_jobs= -1,verbose=1)
housing_grid.fit(X_train, log(y_train))

housing_grid_el.fit(X_train, log(y_train))

housing_grid_ri.fit(X_train, log(y_train))
print(housing_grid.best_params_)

print(housing_grid_el.best_params_)

print(housing_grid_ri.best_params_)
model_final1 = Lasso(alpha=.0003)

model_final1.fit(X_train, log(y_train))

preds1 = model_final1.predict(X_valid)

score1 = np.sqrt(mean_squared_log_error(y_valid, np.e**preds1))

print('RMSLE:', score1)
model_final2 = ElasticNet(alpha=.0006)

model_final2.fit(X_train, log(y_train))

preds2 = model_final2.predict(X_valid)

score2 = np.sqrt(mean_squared_log_error(y_valid, np.e**preds2))

print('RMSLE:', score2)
model_final3 = Ridge(alpha=8)

model_final3.fit(X_train, log(y_train))

preds3 = model_final3.predict(X_valid)

score3 = np.sqrt(mean_squared_log_error(y_valid, np.e**preds3))

print('RMSLE:', score3)
preds_final1 = model_final1.predict(X_testV4)

preds_final2 = model_final2.predict(X_testV4)

preds_final3 = model_final3.predict(X_testV4)



output = pd.DataFrame({'Id': X_testV4.index,

                       'SalePrice1': np.e**preds_final1,

                       'SalePrice2': np.e**preds_final2,

                       'SalePrice3': np.e**preds_final3})

output
output.set_index('Id', inplace =True)
output_final = pd.DataFrame({'Id': X_testV4.index,'SalePrice': output.mean(axis =1)})

output_final
# output_final.to_csv('submission.csv', index=0)
output_final_ri = pd.DataFrame({'Id': X_testV4.index,'SalePrice': np.e**preds_final3})

output_final_ri.to_csv('submission.csv', index=0)