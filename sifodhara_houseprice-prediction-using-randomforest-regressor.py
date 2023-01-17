# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")

import missingno as msno

import sklearn

import category_encoders as ce

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

house_price = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

house_price_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
house_price.head()
house_price.shape
house_price.info()
house_price.describe()
null_check = pd.Series(round(100*(house_price.isnull().sum()/house_price.shape[0]),2))

null_check.sort_values(ascending=False)
msno.matrix(house_price)

plt.show()
null_check[null_check>30.00]
house_price.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
msno.matrix(house_price)

plt.show()
null_check[(null_check>0.00) & (null_check<30.00)]
house_price = house_price[~house_price['GarageType'].isnull()]
null_check_new = pd.Series(round(100*(house_price.isnull().sum()/house_price.shape[0]),2))

null_check_new.sort_values(ascending=False)
null_check_new[(null_check_new>0.00)&(null_check<30.00)]
impute_list = ['LotFrontage','MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical']

for i in impute_list:

    if (house_price[i].dtype=='float64'):

        house_price[i].fillna(house_price[i].mean() , inplace = True)

    else:

        house_price[i].fillna(house_price[i].mode().values[0] , inplace = True)

cols = list(house_price.columns)

null_sum = 0

for i in cols:

    null_sum = null_sum+house_price[i].isnull().sum()

print("Null Valuse in DataFrame : ",null_sum)

    
num_cols = []

cat_cols = []

for i in cols[1:len(cols)]:

    if (house_price[i].dtype=='float64')|(house_price[i].dtype=='int64'):

        num_cols.append(i)

    else:

        cat_cols.append(i)

plt.figure(figsize=(30,20))



violin_plots = []

countplot_cols = []



for i in num_cols:

    if (house_price[i].nunique()>25):

        violin_plots.append(i)

    else:

        countplot_cols.append(i)

#print(len(boxplot_cols))

        

for i in enumerate(violin_plots):

        #print(i[1])

    plt.subplot(5,4,i[0]+1)

    ax =sns.violinplot(house_price[i[1]]) ## KDE with narrow bandwidth to show individual probability lumps

    #print(i[0]+1)

    ax.set_xlabel(i[1],fontsize=15)

plt.tight_layout()

plt.show()
plt.figure(figsize=(30,30))

for i in enumerate(countplot_cols):

    plt.subplot(6,3,i[0]+1)

    ax = sns.countplot(x=i[1],data=house_price)

    ax.set_xlabel(i[1],fontsize=15)

plt.tight_layout()

plt.show()
plt.figure(figsize=(30,30))



for i in enumerate(violin_plots):

        #print(i[1])

    plt.subplot(5,4,i[0]+1)

    ax =sns.scatterplot(x=i[1],y='SalePrice',data=house_price) ## KDE with narrow bandwidth to show individual probability lumps

    #print(i[0]+1)

    ax.set_xlabel(i[1],fontsize=15)

    ax.set_ylabel("Sale Price",fontsize=15)



plt.tight_layout()

plt.show()
plt.figure(figsize=(30,40))

for i in enumerate(countplot_cols):

    plt.subplot(6,3,i[0]+1)

    ax = sns.boxplot(x=i[1],y="SalePrice",data=house_price)

    ax.set_xlabel(i[1],fontsize=15)

plt.tight_layout()

plt.show()
plt.figure(figsize = (30, 30))



# ----------------------------------------------------------------------------------------------------

# plot the data

# the idea is to iterate over each class

# extract their data ad plot a sepate density plot

large_cat_cols = []

small_cat_cols = []

for j in cat_cols:

    if (house_price[j].nunique()>5):

        large_cat_cols.append(j)

    else:

        small_cat_cols.append(j)



for i in enumerate(small_cat_cols):

    for cyl_ in house_price[i[1]].unique():

    # extract the data

        x = house_price[house_price[i[1]] == cyl_]["SalePrice"]

    # plot the data using seaborn

        plt.subplot(6,4,i[0]+1)

        ax = sns.kdeplot(x, shade=True, label = "{}".format(cyl_))

        ax.set_xlabel(i[1],fontsize=15)

        plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text

        #plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title

# set the title of the plot

plt.tight_layout()

plt.show()

plt.figure(figsize=(40,30))

for i in enumerate(large_cat_cols):

    plt.subplot(4,4,i[0]+1)

    ax = sns.scatterplot(x=i[1],y='SalePrice',hue=i[1],data=house_price)

    ax.set_xlabel(i[1],fontsize=15)

    ax.tick_params(axis="x", labelsize=15 , rotation=45)

    #plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text





plt.tight_layout()

plt.show()
y_train = house_price.pop("SalePrice")

X_train = house_price
encoder = ce.OneHotEncoder(cols=cat_cols)



X_train = encoder.fit_transform(X_train) ## one hot encoding on all variables
X_test = house_price_test.copy() ## also done one hot encoding on test set as well



X_test = encoder.fit_transform(X_test)
X_test.shape ## check shape of test
X_train.shape ## check shape of train
sel = VarianceThreshold(threshold=0.1)

sel.fit(X_train)  # fit finds the features with zero variance
# if we sum over get_support, we get the number of features that are not constant

sum(sel.get_support())
X_train = X_train[X_train.columns[sel.get_support()]] ## select variables with proper distribution of values
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

lm = LinearRegression().fit(X_train,y_train)

## created linear regression model and fit our train data

rfe = RFE(lm,42).fit(X_train,y_train)

## select 42 features after running rfe
rfe_cols = X_train.columns[rfe.support_]

## choose features provided by rfe
## check column names provided by rfe 

rfe_cols
X_train = X_train[rfe_cols] ## X_train contains only features selected by rfe.
plt.figure(figsize=(30,15))

sns.heatmap(X_train.corr(),annot=True)

plt.show()
X_train.drop(['LotShape_2','LotConfig_3','ExterQual_2','BsmtQual_2','KitchenQual_2','Exterior2nd_1','Exterior2nd_2'],axis=1,inplace=True)
X_train.shape ## checking shape
# Calculate the VIFs for the new model

import statsmodels

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('TotRmsAbvGrd',axis=1,inplace=True) ## drop feature 
vif = pd.DataFrame() ## again compute VIF

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('OverallQual',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('OverallCond',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('BedroomAbvGr',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('FullBath',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('SaleType_1',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('GarageCars',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('Foundation_1',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('Condition1_1',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('BldgType_1',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.drop('SaleCondition_1',axis=1,inplace=True)
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train.shape 
X_test = X_test[X_train.columns]
X_test.shape ## check shape
X_test.isnull().sum() ## check for null value
X_test['BsmtFullBath'].fillna(X_test['BsmtFullBath'].mode().values[0] , inplace = True) ## impute null value of BsmtFullBath with most frequent value
null_count = 0

for i in X_test.columns:

    null_count = null_count+X_test[i].isnull().sum()

print("Null Values in X_test : ",null_count)

    
from sklearn.ensemble import RandomForestRegressor ## import libraries



rfr = RandomForestRegressor(random_state=1).fit(X_train,y_train)
y_train_pred = rfr.predict(X_train) ## predict sale price
from sklearn.metrics import r2_score

r2_score_default = r2_score(y_train,y_train_pred) ## check r2 score of the model
r2_score_default
from sklearn.metrics import mean_squared_error



mse = mean_squared_error(y_train,y_train_pred)
mse
# Create the parameter grid based on the results of random search 

params = {

    'max_depth': [1, 2, 5, 10, 20 ],

    'min_samples_leaf': [10, 20, 50, 100 , 200 , 400],

    'max_features': [4 , 8 , 15 , 20],

    'n_estimators': [10, 30, 50, 100, 200]

}
from sklearn.model_selection import GridSearchCV

# Instantiate the grid search model

grid_search = GridSearchCV(estimator=rfr, param_grid=params, 

                          cv=4, n_jobs=-1, verbose=1, scoring = "r2")
%%time

grid_search.fit(X_train,y_train)
rf_best = grid_search.best_estimator_
rf_best
rf_best = rf_best.fit(X_train,y_train)
y_train_pred_tune = rf_best.predict(X_train)
r2_score_best = r2_score(y_train,y_train_pred_tune)
r2_score_best
mse_best = mean_squared_error(y_train,y_train_pred_tune)
mse_best
test_pred = rf_best.predict(X_test)
house_price_test['SalePrice'] = test_pred
house_price_test = house_price_test[['Id','SalePrice']]
house_price_test.to_csv("Submission_house_price.csv",index=False)
