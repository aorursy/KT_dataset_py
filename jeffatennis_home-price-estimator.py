import os

import numpy as np 

import pandas as pd

import seaborn as sn

import matplotlib.pyplot as plt

import geopandas as geo



import seaborn as sns; sns.set(style="ticks", color_codes=True)



from scipy.stats import zscore



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge



from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures



from sklearn import metrics

from sklearn.utils import shuffle

from sklearn.metrics import r2_score



import scipy.stats as stats



from sklearn.preprocessing import LabelEncoder



import statsmodels.api as sm



pd.options.display.float_format = '{:20,.2f}'.format



from scipy.stats import zscore



filename = "../input/house-prices-advanced-regression-techniques/train.csv"



pd.set_option('display.max_columns', 100) 



df = pd.read_csv(filename)

df.head(5)
df.describe()
obj = df.isnull().sum()

for key,value in obj.iteritems():

    print(key,",",value)

    

#Candidates to remove: Alley, PoolQC, Fence, MiscFeature

#Candidates to impute a value: LotFrontage, FireplaceQual (set to avg), BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, GarageType,GarageYrBlt (built same as home),GarageFinish
#Drop sparsely populated numeric columns

df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'], axis=1, inplace = True)

#Impute per column mean where the column has a null value - note, this function doesnt seem to work.  FireplaceQu is not working, returning as object.  will i need to go column-by-column

#on categorical features (refer to data description.txt)

dfClean = df.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'float' else x.fillna('unknown'))



obj = dfClean.isnull().sum()

for key,value in obj.iteritems():

    print(key,",",value)
df.describe()


dfClean['FireplaceQu'].value_counts()
histograms = dfClean.hist(column=["MSSubClass","LotFrontage","LotArea","OverallQual","YearBuilt","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF",

                           "GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr",

                           "Fireplaces","GarageYrBlt","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch",

                           "3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold","SalePrice"], figsize = (24,24))

dfClean.head()
g1 = sns.pairplot(dfClean, height=1.5, vars= ["MSSubClass", "MSZoning","Street","LotShape", "LandContour","Utilities","LotConfig","LandSlope","Condition1","Condition2"])
g2 = sns.pairplot(dfClean, height=2, vars= ["BldgType","HouseStyle","RoofStyle","Exterior1st","Exterior2nd",

                                             "MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual"])
g3 = sns.pairplot(dfClean, height=1.5, vars= ["BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating",

                                           "HeatingQC","CentralAir","Electrical","KitchenQual","Functional"])
g4 = sns.pairplot(dfClean, height=1.5, vars= ["FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",

                                             "PavedDrive","SaleType","SaleCondition"])
dfClean.drop(['LotArea','2ndFlrSF','KitchenAbvGr','EnclosedPorch','3SsnPorch','ScreenPorch','Street','LandContour','Utilities',

              'PoolArea','LowQualFinSF','LandSlope','Condition2','BldgType','BsmtCond','BsmtFinType2','Heating','Functional'], axis=1, inplace = True)
dfClean.describe(include='all')
corrMatrix = dfClean.corr()



plt.figure(figsize=(20,20))

sns.heatmap(corrMatrix)

plt.show()
f, ax = plt.subplots(figsize=(24,8))



sns.set_color_codes("pastel")

sns.barplot(x="Neighborhood", y="SalePrice", data=dfClean,

            label="Total", color="b")
#This compares a highl

F, p = stats.f_oneway(dfClean[dfClean.Neighborhood=='NoRidge'].SalePrice,

                      dfClean[dfClean.Neighborhood=='OldTown'].SalePrice,

                      dfClean[dfClean.Neighborhood=='Crawfor'].SalePrice,

                     dfClean[dfClean.Neighborhood=='BrkSide'].SalePrice,

                     dfClean[dfClean.Neighborhood=='StoneBr'].SalePrice)

print(p)
f, ax = plt.subplots(figsize=(24,8))



sns.set_color_codes("pastel")

sns.barplot(x="SaleCondition", y="SalePrice", data=dfClean,

            label="Total", color="b")
F, p = stats.f_oneway(dfClean[dfClean.SaleCondition=='Normal'].SalePrice,

                      dfClean[dfClean.SaleCondition=='Abnorml'].SalePrice,

                      dfClean[dfClean.SaleCondition=='Partial'].SalePrice,

                     dfClean[dfClean.SaleCondition=='AdjLand'].SalePrice,

                     dfClean[dfClean.SaleCondition=='Alloca'].SalePrice,

                      dfClean[dfClean.SaleCondition=='Family'].SalePrice

                     )

print(p)
f, ax = plt.subplots(figsize=(24,8))



sns.set_color_codes("pastel")

sns.barplot(x="Condition1", y="SalePrice", data=dfClean,

            label="Total", color="b")
F, p = stats.f_oneway(dfClean[dfClean.Condition1=='Norm'].SalePrice,

                      dfClean[dfClean.Condition1=='Feedr'].SalePrice,

                      dfClean[dfClean.Condition1=='PosN'].SalePrice,

                     dfClean[dfClean.Condition1=='Artery'].SalePrice,

                     dfClean[dfClean.Condition1=='RRAe'].SalePrice,

                      dfClean[dfClean.Condition1=='RRNn'].SalePrice,

                      dfClean[dfClean.Condition1=='RRAn'].SalePrice,

                      dfClean[dfClean.Condition1=='PosA'].SalePrice,

                      dfClean[dfClean.Condition1=='RRNe'].SalePrice

                     )

print(p)
pd.options.display.float_format = '{:20,.2f}'.format



dfClean['SalePrice_zscore'] =  zscore(dfClean['SalePrice'])



dfClean['Sqft_zscore'] =  zscore(dfClean['GrLivArea'])

dfClean['BsmSqft_zscore'] =  zscore(dfClean['TotalBsmtSF'])

dfClean['TotalSqft_zscore'] =  zscore(dfClean['TotalBsmtSF']+dfClean['GrLivArea'])



dfClean[['SalePrice_zscore','Sqft_zscore','BsmSqft_zscore','TotalSqft_zscore']].describe()
##Categorize neighborhood based on z-scare, which is the # of standard deviations AWAY from the mean.  Z-score of 0 = mean

dfClean['MegaRichArea']=0

dfClean.loc[(dfClean['SalePrice_zscore'] > 5), 'MegaRichArea'] = 1



dfClean['RichArea']=0

dfClean.loc[(dfClean['SalePrice_zscore'] > 2) & (dfClean['SalePrice_zscore'] <= 5), 'RichArea'] = 1



dfClean['AboveAvgArea']=0

dfClean.loc[(dfClean['SalePrice_zscore'] > 0.25) & (dfClean['SalePrice_zscore'] <= 2), 'AboveAvgArea'] = 1



dfClean['AvgArea']=0

dfClean.loc[(dfClean['SalePrice_zscore'] > -0.25) & (dfClean['SalePrice_zscore'] <= 0.25), 'AvgArea'] = 1



dfClean['BelowAvgArea']=0

dfClean.loc[(dfClean['SalePrice_zscore'] < -0.25) & (dfClean['SalePrice_zscore'] >= 0), 'BelowAvgArea'] = 1



dfClean['PoorArea']=0

dfClean.loc[(dfClean['SalePrice_zscore'] < 0), 'PoorArea'] = 1

dfClean[['YearBuilt','YearRemodAdd']].describe()
dfClean['EffectiveYrBlt']=(dfClean['YearRemodAdd']-dfClean['YearBuilt'])
plt.scatter(dfClean['YearRemodAdd'],dfClean['SalePrice'])


dfClean['RegularSale']=0

dfClean.loc[(dfClean.SaleCondition=='Normal'), 'RegularSale'] = 1



dfClean['AbnormalSale']=0

dfClean.loc[(dfClean.SaleCondition=='Abnorml') | (dfClean.SaleCondition=='Partial'), 'AbnormalSale'] = 1



dfClean['OtherSale']=0

dfClean.loc[(dfClean.SaleCondition=='AdjLand') | (dfClean.SaleCondition=='Alloca') | (dfClean.SaleCondition=='Family'), 'OtherSale'] = 1



dfClean[["RegularSale","AbnormalSale","OtherSale","SaleCondition"]].tail()
f, ax = plt.subplots(figsize=(24,8))



sns.set_color_codes("pastel")

sns.barplot(x="OverallCond", y="OverallQual", data=dfClean, color="b")
dfClean["TotalCondition"] = dfClean["OverallQual"]*dfClean["OverallCond"]

dfClean[["TotalCondition","OverallQual","OverallCond"]].head()
dfClean['PoorBad']=0

dfClean.loc[(dfClean.OverallCond<4) & (dfClean['PoorArea']==0), 'PoorBad'] = 1



dfClean[dfClean.PoorBad==1]
#https://datatofish.com/statsmodels-linear-regression/

#dfClean = shuffle(dfClean)



x = dfClean[['MegaRichArea','RichArea','AboveAvgArea','AvgArea','BelowAvgArea','PoorArea','TotalSqft_zscore','YearBuilt']]  #independent variables

y = dfClean['SalePrice_zscore'] #dependent or response variable



#Apply train/test splits:  this will hold out 20% of the records for testing, so training on 80% of records

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)



#x_train = sm.add_constant(x_train) # adding a constant



model = sm.OLS(y_train,x_train).fit()

predictions = model.predict(x_train) 



print_model = model.summary()

print(print_model)
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

lm = linear_model.LinearRegression(normalize=True)

model = lm.fit(x_train, y_train)



predictions = lm.predict(x_test)



plt.scatter(y_test, predictions)

plt.xlabel("True Values")

plt.ylabel("Predictions")
#Score the SciKit learn using on the TRAINING DATA - this should match the statmodel Rsquared value

model.score(x_train,y_train)
#Score the SciKit learn using on the TEST DATA

model.score(x_test,y_test)
# create an array of alpha values

alpha_range = 10.**np.arange(-10, 10)

alpha_range



# select the best alpha with RidgeCV

from sklearn.linear_model import RidgeCV

ridgeregcv = RidgeCV(alphas=alpha_range, normalize=True, scoring='neg_mean_squared_error')

ridgeregcv.fit(x_train, y_train)

ridgeregcv.alpha_



# predict method uses the best alpha value

y_pred = ridgeregcv.predict(x_test)



print("R-Square Value",r2_score(y_test,y_pred))

print("\n")

print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))

print("\n")

print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))

print("\n")

print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.linear_model import LassoCV

lassoregcv = LassoCV(n_alphas=100, normalize=True, random_state=1)

lassoregcv.fit(x_train, y_train)

print('alpha : ',lassoregcv.alpha_)



y_pred = lassoregcv.predict(x_test)



print("R-Square Value",r2_score(y_test,y_pred))

print("\n")

print ("mean_absolute_error :",metrics.mean_absolute_error(y_test, y_pred))

print("\n")

print ("mean_squared_error : ",metrics.mean_squared_error(y_test, y_pred))

print("\n")

print ("root_mean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
dfClean.head()
x = dfClean[['MegaRichArea','RichArea','AboveAvgArea','AvgArea','BelowAvgArea','PoorArea','Sqft_zscore']]  #independent variables

y = dfClean['SalePrice_zscore'] #dependent or response variable



#Apply train/test splits:  this will hold out 20% of the records for testing, so training on 80% of records

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)



x_train = sm.add_constant(x_train) # adding a constant



model = sm.OLS(y_train,x_train).fit()

predictions = model.predict(x_train) 



print_model = model.summary()

print(print_model)



#OpenPorchSF, TotRmsAbvGrd, FullBaths were NOT statistically significant and failed to boost model performance.  

#MoSold, YrSold, WoodDeckSF, GarageArea, LotFrontage and Year Built IS statistically significant and improved performance



#https://becominghuman.ai/stats-models-vs-sklearn-for-linear-regression-f19df95ad99b
x = dfClean[['MegaRichArea','RichArea','AboveAvgArea','AvgArea','BelowAvgArea','PoorArea','GrLivArea']]  #independent variables

y = dfClean['SalePrice'] #dependent or response variable



#Apply train/test splits:  this will hold out 20% of the records for testing, so training on 80% of records

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)



lm = linear_model.LinearRegression(normalize=True)

model = lm.fit(x_train, y_train)



predictions = lm.predict(x_test)



plt.scatter(y_test, predictions)

plt.xlabel("True Values")

plt.ylabel("Predictions")

#Score the SciKit learn using on the TRAINING DATA - this should match the statmodel Rsquared value

model.score(x_train,y_train)
model.score(x_test,y_test)
#https://stackoverflow.com/questions/40729162/merging-results-from-model-predict-with-original-pandas-dataframe

#This applies to alogrithm to the ENTIRE dataframe



y_hats = lm.predict(x)



dfClean.rename(columns = {'SalePrice':'SalePrice_Original'}, inplace = True) 



dfClean['SalePrice'] = y_hats

dfClean[['SalePrice_Original','SalePrice']].describe()
polynomial_features= PolynomialFeatures(degree=2)

xp = polynomial_features.fit_transform(x)



model = sm.OLS(y, xp).fit()

ypred = model.predict(xp) 



print_model = model.summary()

print(print_model)

plt.scatter(y,ypred)

dfClean[['Id','SalePrice']].to_csv('submission.csv',index=False)