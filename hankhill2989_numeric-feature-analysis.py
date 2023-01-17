# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

house_train = pd.read_csv("../input/train.csv")

house_test = pd.read_csv("../input/test.csv")
#This step puts the saleprice and ID at front of data

full = pd.concat([house_train,house_test])

cols = list(full)

cols.insert(0, cols.pop(cols.index('Id')))

cols.insert(1, cols.pop(cols.index('SalePrice')))

full = full.ix[:,cols]

#creating a SalePrice column that will be attached after all feature engineering 

saleprice = full.ix[:,1]



#separating numeric and categorical features into two datasets

numeric_feats = full.select_dtypes(include=['int','int64','float','float64'])

cat_feats = full.select_dtypes(include=['object'])



#Drop SalePrice from numeric Feats so we do not impute by accident

numeric_feats = numeric_feats.drop('SalePrice',axis=1)
miss_check = numeric_feats.isnull().sum()

print(miss_check)
#Lets take a look at GarageYrBlt, LotFrontage, and MasVnrArea

numeric_feats[['GarageYrBlt','LotFrontage', 'MasVnrArea']].describe()



#Seems like GarageYrBlt has a weird extreme value so we will cap that variable

numeric_feats['GarageYrBlt'] = np.where(numeric_feats['GarageYrBlt'] > 2010, 2010, numeric_feats['GarageYrBlt'])
#Looks like we should be okay imputing the median for all missing numeric features

#We will make an indicator for missing values of GarageYrBlt, LotFrontage and MasVnrArea

miss_list = ['GarageYrBlt','LotFrontage','MasVnrArea']

name_list = ['miss_garageyrblt','miss_LF','miss_MasVnrArea']

for i, j in zip(miss_list[0:3],name_list[0:3]):

    numeric_feats[j] = np.where(numeric_feats[i].isnull(), 1, 0)

    

numeric_feats=numeric_feats.fillna(numeric_feats.median())

miss_check = numeric_feats.isnull().sum() #Just to make sure there are no more missing features
#Doing a final check to make sure there are no more numeric variables that should be categorical

cols_list = numeric_feats.columns.tolist()

check = {}

for i in cols_list:

    check[i] = numeric_feats[i].value_counts()
cat_feats2=numeric_feats.ix[:,[4,7,8,11,12,14,17,18,22,25,36]]

numeric_feats=numeric_feats.drop(['BedroomAbvGr','BsmtFullBath','BsmtHalfBath','Fireplaces',\

'FullBath','GarageCars','HalfBath','KitchenAbvGr','MoSold','MSSubClass','YrSold'], axis=1)



#bringing SalePrice back to numeric feats

numeric_feats = pd.concat([saleprice,numeric_feats], axis=1)

cols = list(numeric_feats)

cols.insert(0, cols.pop(cols.index('Id')))

cols.insert(1, cols.pop(cols.index('SalePrice')))

numeric_feats = numeric_feats.ix[:,cols]



#Numeric Feature EDA will be the next step.

#The numeric dataset will be split back into its training and validation sets

numeric_train = numeric_feats[~numeric_feats['SalePrice'].isnull()]

numeric_test = numeric_feats[numeric_feats['SalePrice'].isnull()]
#Checking the distribution of the target variable

plt.hist(numeric_train['SalePrice'])
#The target variable has a left skew so we will do a log transform+1 to get it to normal distribution

numeric_train['SalePrice'] = np.log1p(numeric_train['SalePrice'])

#Checking histogram again

plt.hist(numeric_train['SalePrice'])
#Lets check scatterplots of the target and numeric features

x_axis=range(5)

y_axis=range(5)

column_list=numeric_train.columns.tolist()

f, axes=plt.subplots(5,5,figsize=(14,16))

for i in x_axis:

    for j in y_axis:

        title_index=2+j+len(y_axis)*i

        a=column_list[title_index]

        axes[i,j].scatter(numeric_train[a],numeric_train['SalePrice'])

        axes[i,j].set_title(a)

plt.show()
#Lets check the correlation between variables and target

correlations = numeric_train.corr()

corr_dict = correlations['SalePrice'].to_dict()

del corr_dict['Id']

print('Correlations between Numeric Features and the target:\n')

for ele in sorted(corr_dict.items(), key = lambda x: - abs(x[1])):

    print("{0}: \t{1}".format(*ele))
numeric_train = numeric_train.drop('PoolArea',axis=1)

numeric_feats = numeric_feats.drop('PoolArea',axis=1)

numeric_test = numeric_test.drop('PoolArea',axis=1)
#There are a couple of really nice variables in relation to the target

fig, axs=plt.subplots(2,2, figsize=(12,12))

sns.regplot(x='OverallQual', y='SalePrice', data=numeric_train, color='Red',ax=axs[0,0],)

sns.regplot(x='GrLivArea',y='SalePrice',data=numeric_train,color='Orange',ax=axs[0,1])

sns.regplot(x='GarageArea',y='SalePrice',data=numeric_train,color='Blue',ax=axs[1,0])

sns.regplot(x='TotalBsmtSF',y='SalePrice',data=numeric_train,color='Green',ax=axs[1,1])
#Making xtrain and y values

id_sal_values = numeric_feats.ix[:,[0,1]]

numeric_feats = numeric_feats.drop(['Id','SalePrice'], axis=1)

x_train = numeric_feats[:numeric_train.shape[0]]

x_test = numeric_feats[numeric_train.shape[0]:]

y_train = numeric_train.SalePrice



#Lets do some feature selection to see if some variables drop out

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import f_regression

from sklearn.linear_model import LassoCV

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestRegressor
#We will first start off with some forward selection and see how that does

linear_model = LinearRegression(normalize=True)

f_reg = f_regression(x_train,y_train)

x_list = x_train.columns.tolist()
#Lets try the lasso selection

model_lasso = LassoCV(alphas = [1,0.1,0.001,0.0005]).fit(x_train,y_train)

#Finding out which coefficients were taken from lasso

coef = pd.Series(model_lasso.coef_, index=x_train.columns)

print('Lasso picked' + str(sum(coef != 0)) + 'variables and eliminated the other' + str(sum(coef == 0)) + 'variables')

#Picking out the important coefficents through lasso

imp_coef = coef.sort_values()

print(imp_coef)

matplotlib.rcParams['figure.figsize'] = (10.0,10.0)

imp_coef.plot(kind='barh')

plt.title('Coefficients in the Lasso Model')
#Lets check the non-parametric random forest feature importance

rf = RandomForestRegressor(max_features='sqrt')

rf_model = rf.fit(x_train,y_train)

feat_labels = numeric_train.columns[2:]

importances=rf.feature_importances_

indices = np.argsort(importances)[::-1]

#print feature ranking

print("Feature Ranking:")

for f in range(x_train.shape[1]):

    print("%2d) %-*s %f" % (f+1, 30, feat_labels[f], importances[indices[f]]))
def rmse_cv(model):

    rmse=np.sqrt(-cross_val_score(model, x_train, y_train, scoring='mean_squared_error', cv=5))

    return(rmse)
rmse_cv(model_lasso).mean()

rmse_cv(rf_model).mean()
rmse_cv(linear_model).mean()