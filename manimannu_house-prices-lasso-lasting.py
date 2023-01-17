# import data from kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np # Linear Algebra, Matrix operations .....etc

import pandas as pd # Data manipulation.

import seaborn as sns # Data Visualization.

import matplotlib.pyplot as plt # plots

%matplotlib inline 

#plot in jupyter notebook don't open new window for plotting.



# Future warnings ignore

import warnings

warnings.filterwarnings("ignore")



# Machine Learning API's

from sklearn.model_selection import cross_val_score # cross validation.

from sklearn.linear_model import LinearRegression,Lasso 

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
train_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv") # train data

test_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv") # test data.

train_df.head() # view top rows default 5 rows get displayed.
test_df.head() # view test data.
target=train_df["SalePrice"].copy()

train_df.drop("SalePrice",axis=1,inplace=True) # Drop target from train dataset.
df=pd.concat([train_df,test_df])
# check shape of combined data.

df.shape
# Droping Id column

del df["Id"]
from IPython.display import Image

Image("/kaggle/input/missing-values-mechanism/Missingtheory.png")
null_cols=df.columns[df.isna().any()] # Get cols with missing values
null_per=round(df[null_cols].isna().agg("mean").sort_values(ascending=False),5) # Get percentage of missing values of each column.
null=pd.DataFrame({"Features":null_per.index,"percentage":null_per.values}) # create data frame  for barplot.
plt.figure(figsize=(15,5)) # set width and height for plot

sns.barplot(x=null.columns[0],y=null.columns[1],data=null)

plt.xticks(rotation = 90) # Prevent labels from overlapping.

plt.show()
# Drop features with 50% values missing in it.

df.dropna(thresh=int(0.5*len(df)),inplace=True,axis=1)
df.shape # shape after dropping cols.
rem_null=null[4:]# Get remaining missing values which are less than 50%.

plt.figure(figsize=(15,5)) # set width and height for plot

sns.barplot(x=rem_null.columns[0],y=rem_null.columns[1],data=rem_null)

plt.xticks(rotation = 90) # Prevent labels from overlapping.

plt.show()
# Get numeric features and store into new variable

numeric_features=df.select_dtypes(include=np.number)
numeric_features.dtypes.value_counts() # check dtypes.
# Imputation using KNN it finds similar values to impute missing ones.

from fancyimpute import KNN

impute=KNN(k=3).fit_transform(numeric_features) # imputing
# Creating dataframe because knn imputation returns series of numpy arrays.

imputed=pd.DataFrame(impute,columns=numeric_features.columns)

imputed.head()
cast_int=imputed.dtypes[numeric_features.dtypes != imputed.dtypes] # Get dtypes to change into int format.

imputed[cast_int.index]=imputed[cast_int.index].applymap(int) # casting into original dtypes.
imputed.head()
imputed.dtypes.value_counts() # count of dtypes
# check you imputated to null values

imputed.isna().any().sum()
# Statistical information of numeric data.

imputed.describe()
# Check for missingness count

cate_features=df.select_dtypes(exclude=np.number)

cate_cols=cate_features.columns[cate_features.isna().any()]

cate_per=round(df[cate_cols].isna().agg("mean").sort_values(ascending=False),5) # Get percentage of missing values of each column.

cate_per
cate_features[cate_cols].isna().sum()
cate_features["FireplaceQu"]=cate_features["FireplaceQu"].fillna("Missing")

cols=cate_features[cate_cols].isna().sum() <100

impute_cols=cols.index

for i in impute_cols:

    cate_features[i].fillna(cate_features[i].mode()[0],inplace=True)
cate_features[cate_cols].isna().sum()
cate_features.columns.value_counts().sum()
# check distribution of target variable.

plt.figure(figsize=(8,5))

sns.distplot(target)
target.skew() # skewness
# converting right skewed to normal distribution (or) close to normal.

plt.figure(figsize=(8,5))

sns.distplot(np.log(target))

np.log(target).skew()
recombined=pd.concat([numeric_features,cate_features],axis=1)
recombined.head()
train=recombined[:train_df.shape[0]]

test=recombined[train_df.shape[0]:]

train.shape,test.shape
trainn=pd.concat([train,target],axis=1)
plt.figure(figsize=(25,15))

corr=trainn.corr(method="spearman")

sns.heatmap(corr,annot=True)
print (corr['SalePrice'].sort_values(ascending=False)[:20]) #top 15 values

print ('----------------------')

print (corr['SalePrice'].sort_values(ascending=False)[-5:]) #last 5 values`
# 1st correlated variable.

trainn['OverallQual'].unique()
#let's check the mean price per quality and plot it.

pivot = trainn.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

sns.barplot(x="OverallQual",y="SalePrice",data=trainn)
pivot = trainn.pivot_table(index='GrLivArea', values='SalePrice', aggfunc=np.median)

sns.jointplot(x="GrLivArea",y="SalePrice",data=trainn)
cate_data=recombined.select_dtypes(include="object")

cate_data.describe()
# we need to check the relation between categorical values using ANOVA .

from scipy import stats

cat = [f for f in trainn.columns if trainn.dtypes[f] == 'object']

def anova(frame):

    anv = pd.DataFrame()

    anv['features'] = cat

    pvals = []

    for c in cat:

           samples = []

           for cls in frame[c].unique():

                  s = frame[frame[c] == cls]['SalePrice'].values

                  samples.append(s)

           pval = stats.f_oneway(*samples)[1]

           pvals.append(pval)

    anv['pval'] = pvals

    return anv.sort_values('pval')



cate_data['SalePrice'] = trainn.SalePrice

k = anova(cate_data) 

k['disparity'] = np.log(1./k['pval'].values) 

plt.figure(figsize=(15,8))

sns.barplot(data=k, x = 'features', y='disparity') 

plt.xticks(rotation=90) 

plt.show()
# View all numeric features distribution.

trainn.hist(figsize=(20,18))

plt.show()
# Also visualize the categorical variables using boxplot

def boxplot(x,y,**kwargs):

            sns.boxplot(x=x,y=y)

            x = plt.xticks(rotation=90)



cat = [f for f in train.columns if trainn.dtypes[f] == 'object']



p = pd.melt(trainn, id_vars='SalePrice', value_vars=cat)

g = sns.FacetGrid (p, col='variable', col_wrap=4, sharex=False, sharey=False, height=4)

g = g.map(boxplot, 'value','SalePrice')

g
# we need to convert categorical variables in to numeric. unless it gives you any error all models works with numerical mathematical formuales.

# combine train and test data ,split target into new variable

target=trainn["SalePrice"].copy()

del trainn["SalePrice"]

fina_df=pd.concat([trainn,test])
# Look out features behavior and perform feature engineering.

# Select features which contribute to target variable and also create new .



fina_df.select_dtypes(include="object").nunique() # unique values in each categorical variable.
# perfrom label encoding on ordinal data or mapping function.

# features which are have quality are ordinal.

features_ord=["GarageQual","KitchenQual","FireplaceQu","BsmtQual","ExterQual"]

fina_df[features_ord].nunique()
# multiple column label encoder class

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline



class MultiColumnLabelEncoder:

    def __init__(self,columns = None):

        self.columns = columns # array of column names to encode



    def fit(self,X,y=None):

        return self # not relevant here



    def transform(self,X):

        '''

        Transforms columns of X specified in self.columns using

        LabelEncoder(). If no columns specified, transforms all

        columns in X.

        '''

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)
fina_df=MultiColumnLabelEncoder(columns = features_ord).fit_transform(fina_df)
fina_df.head()
# converting other categorical values to numeric using OHE , pandas get_dummies.

fina_df=pd.get_dummies(fina_df)
# splitting train and test data.

train_split=fina_df[:train_df.shape[0]]

test_split=fina_df[train_df.shape[0]:]
# log transformation target before training

y=np.log(target)
train_split.head()
# some cols still have null values that are hidden by imputation methods.

remaining=train_split.columns[train_split.isna().any()]

remaining
for i in remaining:

    train_split[i].fillna(0,inplace=True) # Filling with zeros
remaining=test_split.columns[test_split.isna().any()]

for i in remaining:

    test_split[i].fillna(0,inplace=True) # filling with zeros
# final check for null values in both train and test data.

print(train_split.columns[train_split.isna().any()])

print(test_split.columns[test_split.isna().any()])
from sklearn.preprocessing import StandardScaler

stand=StandardScaler()

values=stand.fit_transform(train_split)

X_train=pd.DataFrame(values,columns=train_split.columns)
from sklearn.preprocessing import StandardScaler

stand=StandardScaler()

values=stand.fit_transform(test_split)

X_test=pd.DataFrame(values,columns=test_split.columns)
import xgboost as xgb

# below parameters are valued by cross-validation.

regr = xgb.XGBRegressor(colsample_bytree=0.2,

                       gamma=0.0,

                       learning_rate=0.05,

                       max_depth=6,

                       min_child_weight=1.5,

                       n_estimators=7200,

                       reg_alpha=0.9,

                       reg_lambda=0.6,

                       subsample=0.2,

                       seed=42,

                       silent=1)



regr.fit(X_train, y) # train data, log transformed target.
from sklearn.metrics import mean_squared_error

# root mean square error function

def rmse(y_test,y_pred):

      return np.sqrt(mean_squared_error(y_test,y_pred)) 



# Run prediction on training set to get an idea of how well it does

y_pred = regr.predict(X_train)

y_test = y

print("XGBoost score on training set: ", rmse(y_test, y_pred))
# make prediction on the test set

y_pred_xg = regr.predict(X_test)

xg_ex = np.exp(y_pred_xg)

pred1 = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': xg_ex})

pred1.to_csv('xgboost.csv', header=True, index=False)
# Lasso

from sklearn.linear_model import Lasso



# found this best alpha through cross-validation

best_alpha = 0.00099



lasso = Lasso(alpha=best_alpha, max_iter=50000)

lasso.fit(X_train, y)



# Metrics score on train data

y_pred = lasso.predict(X_train)

y_test = y

print("Lasso score on training set: ", rmse(y_test, y_pred))
#make prediction on the test set

lasso_pred = lasso.predict(X_test)

lasso_ex = np.exp(lasso_pred)

pred1 = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': lasso_ex})

pred1.to_csv('lasso.csv', header=True, index=False)
# Gradient boosting regressor model training

from sklearn.ensemble import GradientBoostingRegressor

est = GradientBoostingRegressor(n_estimators= 1000, max_depth= 2, learning_rate= .01)

est.fit(X_train, y)
# metrics score on train data.

y_pred = est.predict(X_train)

y_test = y

print("Gradient score on training set: ", rmse(y_test, y_pred))
# submission to kaggle

GBC_pred = est.predict(X_test) # prediction

GBC_ex = np.exp(GBC_pred) # converting back to original values

pred1 = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': GBC_ex})

pred1.to_csv('GBC.csv', header=True, index=False)