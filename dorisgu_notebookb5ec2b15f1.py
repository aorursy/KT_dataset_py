# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for beauty data visualization 

#invite people for the Kaggle party

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Get input

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

# Merge data together

all_data = pd.concat([df_train,df_test], keys=['train','test'])
# In case we need to log transfer the data, fill the missing value with 1

all_data.loc['test'].SalePrice.fillna(1, inplace=True)
# 1.1 Check the object

all_data.columns

all_data.dtypes

#a, b = np.unique(all_data.dtypes,return_counts=True)

#a
#descriptive statistics summary 

#Check if there exists abnormal value

for cols in all_data.columns:

    #print(type(cols))

    print(all_data[cols].describe())    

# seems like there contains "nan" or missing value in some of the variables. 
#histogram not normal needs to be transformed

sns.distplot(all_data.loc['train'].SalePrice);
sns.distplot(np.log(all_data.loc['train'].SalePrice))

all_data.loc['train'].SalePrice = np.log(all_data.loc['train'].SalePrice)
#skewness and kurtosis

print("Skewness: %f" % all_data.loc['train'].SalePrice.skew())

print("Kurtosis: %f" % all_data.loc['train'].SalePrice.kurt())
#correlation matrix

corrmat = all_data.loc['train'].corr() #excluding NA/null values

# Visulaization 

#f, ax = plt.subplots(figsize=(12, 9))

#sns.heatmap(corrmat, vmax=.8, square=True);
# drop those column are not correlated to the saleprice

corrmat_k = corrmat.nlargest(len(corrmat), 'SalePrice')['SalePrice']

drop_list_uncorrelated = []

for i in corrmat_k.index:

    if np.absolute( corrmat_k[i])< 0.1:

        drop_list_uncorrelated.append(i)



drop_list_uncorrelated
all_data = all_data.drop(drop_list_uncorrelated,1)
# 1.2 missing value 

total_tr = all_data.loc['train'].isnull().sum()

percent_tr = (all_data.loc['train'].isnull().sum()/all_data.loc['train'].isnull().count())

total_t = all_data.loc['test'].isnull().sum()

percent_t = (all_data.loc['test'].isnull().sum()/all_data.loc['test'].isnull().count())

missing_data = pd.concat([total_tr, percent_tr,total_t, percent_t], axis=1, keys=['Total_train', 'Percent_train','Total_test', 'Percent_test'])

missing_data.sort_values(by=['Percent_train'],ascending=False)[:20]

## >15%
#dealing with missing data # we should look at both for training and test set, 

#in case there exists lots of missing value only in test set



all_data = all_data.drop((missing_data[missing_data['Total_train'] > 1]).index,1) 

# drop all the columns with missing value more than 1, the Garage* and Bsmt* do not look like informative 

# And both train and test set show the same missing data distribution 
all_data = all_data.drop(all_data.loc[all_data['Electrical'].isnull()].index) # drop the row in train set with missing value in electrical 
all_data.loc['train'].isnull().sum().max() #just checking that there's no missing data in train set...
# Deal with missing value in test dataset 

total_t = all_data.loc['test'].isnull().sum()

percent_t = (all_data.loc['test'].isnull().sum()/all_data.loc['test'].isnull().count())

missing_data = pd.concat([total_t, percent_t], axis=1, keys=['Total_test', 'Percent_test'])

missing_data.sort_values(by=['Percent_test'],ascending=False)[:15]
# fill every column with its own mean value in numerical data

all_data = all_data.fillna(all_data.mean())
# fill every column with its own most frequent value in categorical data

all_data = all_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
all_data.isnull().sum().max() 
# 1.3 Outliers 

# numerical data 

corrmat = all_data.loc['train'].corr() 

u = 0.2 # minimum correlation coefficient

for k in range(1,len(corrmat)):

    corrmat_k = corrmat.nlargest(k, 'SalePrice')['SalePrice']

    if corrmat_k.values[-1]< u:

        corrmat_k = corrmat_k[:-1]

        break

cols = corrmat_k.index

cm = all_data.loc['train'][cols].corr()

#f, ax = plt.subplots(figsize=(12, 12))

#sns.heatmap(cm, vmax=.8, annot= True, square=True);

# look at the scatter plot for those cols
# visualize the scatter plot for outliers using most correlated data 

for col in cols:

    title = 'Variable: '+ col

    plt.title(title)

    plt.scatter(all_data.loc['train'][col], all_data.loc['train']['SalePrice'])

    plt.show()
# outliers in GrLivArea

out1 = all_data.loc[np.logical_and(all_data['GrLivArea']>4000,all_data['SalePrice']<12.5)]

out1
# Drop the rows in train 

tmp = all_data.copy()

tmp  = tmp.drop(out1.index[0:2])

# outliers in test ? should we refill it or ignore 
plt.scatter(tmp.loc['train']['GrLivArea'], tmp.loc['train']['SalePrice'])
# boxplot for checking outliers

for col in cols:

    title = 'Variable: '+ col

    plt.title(title)

    plt.boxplot(tmp.loc['train'][col])

    plt.show()
# Refill the data with the mean 

tmp.set_value(tmp.loc[tmp['GarageCars']>3.5]['GarageCars'].index,'GarageCars',np.around(tmp['GarageCars'].mean()))
tmp.loc[tmp['OverallQual']<2]['OverallQual']
tmp.set_value(tmp.loc[tmp['OverallQual']<2]['OverallQual'].index,'OverallQual',np.around(tmp['OverallQual'].mean()))
tmp.loc[tmp['Fireplaces']>2.5]['Fireplaces']
tmp.set_value(tmp.loc[tmp['Fireplaces']>2.5]['Fireplaces'].index,'Fireplaces',np.around(tmp['Fireplaces'].mean()))
all_data = tmp
# outliers for categorical data 

cates = all_data.select_dtypes(include=['object']).copy().columns

for i in range(len(cates)):

    plt.figure()

    sns.boxplot(x=cates[i], y='SalePrice', data=all_data.loc['train'])
np.unique(all_data.loc['train']['Utilities'],return_counts = True)

## "NoSeWa" only have one sample consider to drop this column

all_data = all_data.drop('Utilities',1)
# numerical data 

corrmat = all_data.loc['train'].corr() 

u = 0.2 # minimum correlation coefficient

for k in range(1,len(corrmat)):

    corrmat_k = corrmat.nlargest(k, 'SalePrice')['SalePrice']

    if corrmat_k.values[-1]< u:

        corrmat_k = corrmat_k[:-1]

        break

cols = corrmat_k.index

cm = all_data.loc['train'][cols].corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(cm, vmax=.8, annot= True, square=True);
# GarageCars vs GarageArea

# TotalBsmtSF vs 1stFlrSF 

# TotRmsAbvGrd vs GrLivArea

# those there pairs of data are highly correlated, so just keep one of each pair

all_data = all_data.drop('GarageArea',1)

all_data = all_data.drop('1stFlrSF',1)

all_data = all_data.drop('TotRmsAbvGrd',1)
corrmat = all_data.loc['train'].corr() 

u = 0.4 # minimum correlation coefficient

for k in range(1,len(corrmat)):

    corrmat_k = corrmat.nlargest(k, 'SalePrice')['SalePrice']

    if abs(corrmat_k.values[-1])< u:

        corrmat_k = corrmat_k[:-1]

        break

cols = corrmat_k.index

cm = all_data.loc['train'][cols].corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(cm, vmax=.8, annot= True, square=True);
#FullBath vs GrLivArea

all_data = all_data.drop(['FullBath','YearRemodAdd'],1)
cols = cols.drop('FullBath')

cols = cols.drop('YearRemodAdd')

cols
feature_nums = cols[1:]

print('Numerical features for modeling: '+feature_nums)
# categoricla data

# Looking at the correlation between categorical data with SalePrice

# by Encoding the categorical data using "get_dummies"

obj_df = all_data.loc['train'].select_dtypes(include=['object']).copy()

obj_df.head()



## 27 catogricals data 
cato = {'Min' : [],'Max':[]} # dictionary for saving the minimum and maximum correlation coeficients 

for col in obj_df.columns:

    for_dummy = obj_df[col]

    test = pd.concat([all_data.loc['train']['SalePrice'], pd.get_dummies(for_dummy, prefix=col)], axis=1)

    cato['Max'].append(np.max(test.corr().nlargest(len(test), 'SalePrice')['SalePrice'][1:]))

    cato['Min'].append(np.min(test.corr().nlargest(len(test), 'SalePrice')['SalePrice'][1:]))

   # print(col+': '+str(np.max(test.corr().nlargest(len(test), 'SalePrice')['SalePrice'][1:]))+';'+str(np.min(test.corr().nlargest(len(test), 'SalePrice')['SalePrice'][1:])))

category = pd.DataFrame(cato, index = obj_df.columns)

category

test_corr = category[np.logical_or(category['Max']<0.1,category['Min']>-0.1)] # features with smaller coefficients

# checking those features with smaller correlation coefficient:

# 'Street', 'LotConfig', 'LandSlope', 'Condition2', 'RoofMatl'

for item in test_corr.index:

    print(item)

    print('Train set')

    print(np.unique(all_data.loc['train'][item],return_counts=True))

    print('Test set')

    print(np.unique(all_data.loc['test'][item],return_counts=True))

    print('------------------------------------------------')
## there are four categorical data do not show the diverse and the correlation coefficient are low, 

## Consider dropping them

## "Heating", "Street",LandSlope,Condition2

all_data = all_data.drop('Heating',1)

all_data = all_data.drop('Street',1)

all_data = all_data.drop('LandSlope',1)

all_data = all_data.drop('Condition2',1)
len(all_data.dtypes[all_data.dtypes == 'object'].index) # 23 categorical seems like too many features 
# strict on the coefficient in order to reduce the feature 

test_high = category[np.logical_or(category['Max']>0.3,category['Min']<-0.3)]

#for i in test_high.index:

#    plt.figure()

#    sns.boxplot(x=i, y='SalePrice', data=all_data.loc['train'])



feature_cat = test_high.index

print('Categorical features for modeling: '+feature_cat)    
# 2.1 Get numerical data 

feature = all_data[feature_nums]

feature.head()

for i in feature.columns:

    plt.figure()

    sns.distplot(feature.loc['train'][i])
#  Normalization two features

## the reason why, we didn't normalize other features, is those are more like categorical data 



sns.distplot(np.log(feature['GrLivArea']))

feature['GrLivArea']=np.log(feature['GrLivArea'])

plt.figure()

sns.distplot(np.log(feature['TotalBsmtSF']+1))

feature['TotalBsmtSF']=np.log(feature['TotalBsmtSF']+1)

# 2.2 Get categorical data 

feature[feature_cat] = all_data[feature_cat]

feature.head()
# Encoding the categorical data using "get_dummies"

for col in feature.dtypes[feature.dtypes == 'object'].index:

    for_dummy = feature.pop(col)

    feature = pd.concat([feature, pd.get_dummies(for_dummy, prefix=col)], axis=1)
feature.head()

# standard the feature

scaler = StandardScaler()

scaler.fit(feature)

feature = pd.DataFrame(scaler.transform(feature),columns=feature.columns, index = feature.index)
y  = all_data.loc['train']['SalePrice']

sns.distplot(y)
# check the outliers and features again: 

TEST = pd.concat([feature.loc['train'],pd.DataFrame(y)], axis=1)

corrmat = TEST.corr() 

u = 0.34 # minimum correlation coefficient

for k in range(1,len(corrmat)):

    corrmat_k = corrmat.nlargest(k, 'SalePrice')['SalePrice']

    if abs(corrmat_k.values[-1])< u:

        corrmat_k = corrmat_k[:-1]

        break

cols = corrmat_k.index

cm = TEST[cols].corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(cm, vmax=.8, annot= True, square=True);
# take data for analysis (reducing feature)

feature = feature[cols[1:]]
#Seperate data

X_train = feature.loc['train']

X_test = feature.loc['test']
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 25, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
cv_lasso
from sklearn.linear_model import Lasso

alphas = [1e-3, 5e-3,8e-3,1e-2]

cv_lasso = [rmse_cv(Lasso(alpha = alpha, max_iter=50000)).mean()  for alpha in alphas]

pd.Series(cv_lasso, index = alphas).plot()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
lasso_preds = np.exp(model_lasso.predict(X_test))
solution = pd.DataFrame({"id":df_test.Id, "SalePrice":lasso_preds})

solution.to_csv("ridge_sol.csv", index = False)