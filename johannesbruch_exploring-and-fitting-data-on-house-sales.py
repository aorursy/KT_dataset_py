import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualisation with scatter plots

import matplotlib # visualising regression parameters

import seaborn as sns # visualisation with correlation matrix

from scipy.stats import norm # for standardisation

from scipy import stats # used here for visualisation of skewness
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

corrmat=df_train.corr()

f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corrmat, vmax=1, square=True)
#saleprice correlation matrix

k = 11 # number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index # selecting the k columns of the matrix with largest correlation with SalesPrice

cm = np.corrcoef(df_train[cols].values.T) # accessing the correlation coefficients to depict them as numbers

sns.set(font_scale=1) # set the size of the font of the numbers

f1, ax1 = plt.subplots(figsize=(9,9))

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
total=df_train.isnull().sum().sort_values(ascending=False)# sorted list of total missing number of elements

percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)# unintuitively, this list has the same sorting order as the above

missing_data=pd.concat([total,percent], axis=1, keys=['Total','Percent'])# concatenate into new frame

missing_data.loc[missing_data['Total']>0].head(len(df_train.columns))# print the whole frame except those variables without missing data
df_test= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')#load test data

total=df_test.isnull().sum().sort_values(ascending=False)# sorted list of total missing number of elements

percent=(df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)# unintuitively, this list has the same sorting order as the above

missing_data2=pd.concat([total,percent], axis=1, keys=['Total','Percent'])# concatenate into new frame

missing_data2.loc[missing_data2['Total']>0].head(len(df_test.columns))# print the whole frame except those variables without missing data
index_list= list(corrmat.nlargest(k, 'SalePrice')['SalePrice'].index)

index_list.pop(0)# remove SalePrice

df_test[index_list].describe()

   
for var in index_list:

    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

    data.plot.scatter(x=var, y='SalePrice');
print('Number of houses that were apparently remodelled in 1950:')

l=list(df_train['YearBuilt'].loc[df_train['YearRemodAdd']==1950])

print(len(l))

print('Number of houses that were apparently build in 1900:')

l=list(df_train['YearRemodAdd'].loc[df_train['YearBuilt']==1900])

print(len(l))

print('Number of houses that were apparently built in 1900 and remodelled in 1950:')

print(len([year for year in l if year==1950]))

print('Fraction of houses that were  remodelled:')

print(len(list(df_train['YearRemodAdd'].loc[df_train['YearBuilt']!=df_train['YearRemodAdd']]))/len(df_train['YearRemodAdd']))
df_train['FloorsSF']=pd.Series(len(df_train['YearBuilt']), index=df_train.index)

df_train['FloorsSF']=df_train['1stFlrSF']+df_train['TotalBsmtSF']+df_train['2ndFlrSF']+df_train['GrLivArea']

medianG=df_train.loc[df_train['GarageCars']==1,'GarageArea'].median()

df_train['Garage']=pd.Series(len(df_train['YearBuilt']), index=df_train.index)

df_train['Garage']=df_train['GarageCars']+1*df_train['GarageArea']/medianG

df_test['FloorsSF']=pd.Series(len(df_test['YearBuilt']), index=df_test.index)

df_test['FloorsSF']=df_test['1stFlrSF']+df_test['TotalBsmtSF']+df_test['2ndFlrSF']+df_test['GrLivArea']

medianG=df_test.loc[df_test['GarageCars']==1,'GarageArea'].median()

df_test['Garage']=pd.Series(len(df_train['YearBuilt']), index=df_test.index)

df_test['Garage']=df_test['GarageCars']+1*df_test['GarageArea']/medianG

#saleprice correlation matrix

corrmat=df_train.corr()

k = 11 # number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index # selecting the k columns of the matrix with largest correlation with SalesPrice

cm = np.corrcoef(df_train[cols].values.T) # accessing the correlation coefficients to depict them as numbers

sns.set(font_scale=1) # set the size of the font of the numbers

f1, ax1 = plt.subplots(figsize=(9,9))

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
df_train = df_train.drop((missing_data[missing_data['Total'] > 0]).index,1)

df_test = df_test.drop((missing_data[missing_data['Total'] > 0]).index,1)

# a list of other non-categorical variables that will not be included

index_list_noncat_exclude=['GrLivArea','1stFlrSF','GarageCars','GarageArea','TotalBsmtSF','2ndFlrSF','HouseStyle', 'Utilities', 'Exterior2nd', 'Condition2', 'Heating', 'Exterior1st', 'RoofMatl']

df_train = df_train.drop(index_list_noncat_exclude,1)

df_test = df_test.drop(index_list_noncat_exclude,1)

print('Print zero, if no data is missing in the new training set: ')

df_train.isnull().sum().max() #just checking that there's no missing data missing...

figa = plt.figure(figsize=(15,30))

ax1 = plt.subplot(621)

results = stats.probplot(df_train['FloorsSF'], plot=ax1)

ax1.set_title("FloorsSFTrain")

ax2 = plt.subplot(622)

results = stats.probplot(df_test['FloorsSF'], plot=ax2)

ax2.set_title("FloorsSFTest")

ax3 = plt.subplot(623)

results = stats.probplot(df_train['Garage'], plot=ax3)

ax3.set_title("GarageTrain")

ax4 = plt.subplot(624)

results = stats.probplot(df_test['Garage'], plot=ax4)

ax4.set_title("GarageTest")

ax5 = plt.subplot(625)

results = stats.probplot(df_train['YearBuilt'], plot=ax5)

ax5.set_title("YearBuiltTrain")

ax6 = plt.subplot(626)

results = stats.probplot(df_test['YearBuilt'], plot=ax6)

ax6.set_title("YearBuiltTest")

ax7 = plt.subplot(627)

results = stats.probplot(df_train['YearRemodAdd'], plot=ax7)

ax7.set_title("YearRemodAddTrain")

ax8= plt.subplot(628)

results = stats.probplot(df_test['YearRemodAdd'], plot=ax8)

ax8.set_title("YearRemodAddTest")

figb = plt.figure(figsize=(15,15))

ax1 = plt.subplot(221)

results = stats.probplot(df_train['TotRmsAbvGrd'], plot=ax1)

ax1.set_title("TotRmsAbvGrdTrain")

ax2 = plt.subplot(222)

results = stats.probplot(df_test['TotRmsAbvGrd'], plot=ax2)

ax2.set_title("TotRmsAbvGrdTest")

ax3 = plt.subplot(223)

results = stats.probplot(df_train['SalePrice'], plot=ax3)

ax3.set_title("SalePriceTrain")
# data transformation by taking the logarithm

df_train['FloorsSF'] = np.log1p(df_train['FloorsSF'])

df_test['FloorsSF'] = np.log1p(df_test['FloorsSF'])

df_train['TotRmsAbvGrd'] = np.log1p(df_train['TotRmsAbvGrd'])

df_test['TotRmsAbvGrd'] = np.log1p(df_test['TotRmsAbvGrd'])

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

figa = plt.figure(figsize=(15,15))

ax1 = plt.subplot(221)

results = stats.probplot(df_train['FloorsSF'], plot=ax1)

ax1.set_title("FloorsSFTrain")

ax2 = plt.subplot(222)

results = stats.probplot(df_test['FloorsSF'], plot=ax2)

ax2.set_title("FloorsSFTest")

ax3 = plt.subplot(223)

results = stats.probplot(df_train['TotRmsAbvGrd'], plot=ax3)

ax3.set_title("TotRmsAbvGrdTrain")

ax4 = plt.subplot(224)

results = stats.probplot(df_test['TotRmsAbvGrd'], plot=ax4)

ax4.set_title("TotRmsAbvGrdTest")

figc=plt.figure(figsize=(10,10))

ax4 = plt.subplot(111)

results = stats.probplot(df_train['SalePrice'], plot=ax4)

ax4.set_title("SalePriceTrain")
df_train['Centenarian']=pd.Series(len(df_train['YearBuilt']), index=df_train.index)

df_train['Centenarian']=0

df_train.loc[df_train['YearBuilt']<1920,'Centenarian']=1

df_test['Centenarian']=pd.Series(len(df_test['YearBuilt']), index=df_test.index)

df_test['Centenarian']=0

df_test.loc[df_test['YearBuilt']<1920,'Centenarian']=1

df_train['Remodelled']=pd.Series(len(df_train['YearBuilt']), index=df_train.index)

df_train['Remodelled']=0

df_train.loc[df_train['YearBuilt']!=df_train['YearRemodAdd'],'Remodelled']=1

df_test['Remodelled']=pd.Series(len(df_test['YearBuilt']), index=df_test.index)

df_test['Remodelled']=0

df_test.loc[df_test['YearBuilt']!=df_test['YearRemodAdd'],'Remodelled']=1

f1, ax1 = plt.subplots(figsize=(10, 10))

fig = sns.boxplot(x='Centenarian', y="SalePrice", data=df_train)

f3, ax3 = plt.subplots(figsize=(10, 10))

fig = sns.boxplot(x='Remodelled', y="SalePrice", data=df_train)
data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)

for i in range(2):

    if i <1:

        print('Among Centenarians')

        dat = data.loc[df_train['YearBuilt']<1920]

    else:

        print('Among younger buildings')

        dat = data.loc[df_train['YearBuilt']>=1920]

    corrmat=dat.corr()

    cols = corrmat.index

    cm = np.corrcoef(dat[cols].values.T) # accessing the correlation coefficients to depict them as numbers

    print('The correlation coefficient between SalePrice and YearBuilt is:')

    print(cm[0,1])
from sklearn.linear_model import LassoCV, Ridge, RidgeCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, train, y, scoring="neg_mean_squared_error", cv = 10))

    return(rmse)



#filling NA's with the mean of the column:

df_test = df_test.fillna(df_test.mean())



df_train = pd.get_dummies(df_train)

test = pd.get_dummies(df_test)

cols=list(df_train.columns)

cols.remove('SalePrice')

train=df_train[cols]

y=df_train.SalePrice



model_lasso = LassoCV(alphas = [0.5, 0.005, 0.0005]).fit(train, y)



print('RMSE on the train set:')

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = train.columns)





print("Lasso uses " + str(sum(coef != 0)) + " variables and disregards the other " +  str(sum(coef == 0)) + " variables")







imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])







matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the model")



import xgboost as xgb



dtrain = xgb.DMatrix(train, label = y)

dtest = xgb.DMatrix(test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=200)

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(train, y)

xgb_preds = np.expm1(model_xgb.predict(test)) 

lasso_preds = np.expm1(model_lasso.predict(test))

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.5*lasso_preds + 0.5*xgb_preds

solution = pd.DataFrame({"id":df_test.Id, "SalePrice":preds})
solution.to_csv('result-with-best.csv', index=False)