import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



import warnings

warnings.simplefilter(action='ignore')



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import math

import sklearn.model_selection as ms

import sklearn.metrics as sklm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
a = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

b = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Use this code to show all the 163 columns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
a.head()
print('The shape of our training set: ',a.shape[0], 'houses', 'and', a.shape[1], 'features')

print('The shape of our testing set: ',b.shape[0], 'houses', 'and', b.shape[1], 'features')

print('The testing set has 1 feature less than the training set, which is SalePrice, the target to predict  ')
num=a.select_dtypes(include='number')

numcorr=num.corr()

f,ax=plt.subplots(figsize=(9, 9))

sns.heatmap(numcorr.sort_values(by=['SalePrice'], ascending=False)[['SalePrice']], cmap='Blues')

plt.title(" Numerical features correlation with the sale price", weight='bold', fontsize=18)

plt.xticks(weight='bold')

plt.yticks(weight='bold', color='dodgerblue', rotation=0)





plt.show()
Num=numcorr['SalePrice'].sort_values(ascending=False).head(10).to_frame()



cm = sns.light_palette("cyan", as_cmap=True)



s = Num.style.background_gradient(cmap=cm)

s
plt.figure(figsize=(15,6))

plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color='crimson', alpha=0.5)

plt.title('Ground living area/ Sale price', weight='bold', fontsize=16)

plt.xlabel('Ground living area', weight='bold', fontsize=12)

plt.ylabel('Sale price', weight='bold', fontsize=12)

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
# Figure Size

fig, ax = plt.subplots(figsize=(9,6))



# Horizontal Bar Plot

title_cnt=a.Neighborhood.value_counts().sort_values(ascending=False).reset_index()

mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color=sns.color_palette('Reds',len(title_cnt)))









# Remove axes splines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)



# Remove x,y Ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.2)



# Show top values 

ax.invert_yaxis()



# Add Plot Title

ax.set_title('Most frequent neighborhoods',weight='bold',

             loc='center', pad=10, fontsize=16)

ax.set_xlabel('Count', weight='bold')





# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),

            fontsize=10, fontweight='bold', color='grey')

plt.yticks(weight='bold')





plt.show()

# Show Plot

plt.show()
# Figure Size

fig, ax = plt.subplots(figsize=(9,6))



# Horizontal Bar Plot

title_cnt=a.BldgType.value_counts().sort_values(ascending=False).reset_index()

mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color=sns.color_palette('Greens',len(title_cnt)))









# Remove axes splines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)



# Remove x,y Ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.2)



# Show top values 

ax.invert_yaxis()



# Add Plot Title

ax.set_title('Building type: Type of dwelling',weight='bold',

             loc='center', pad=10, fontsize=16)

ax.set_xlabel('Count', weight='bold')





# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),

            fontsize=10, fontweight='bold', color='grey')

plt.yticks(weight='bold')





plt.show()
plt.style.use('seaborn')

sns.set_style('whitegrid')



plt.subplots(0,0,figsize=(15,3))



a.isnull().mean().sort_values(ascending=False).plot.bar(color='black')

plt.axhline(y=0.1, color='r', linestyle='-')

plt.title('Missing values average per column: Train set', fontsize=20, weight='bold' )

plt.show()



plt.subplots(1,0,figsize=(15,3))

b.isnull().mean().sort_values(ascending=False).plot.bar(color='black')

plt.axhline(y=0.1, color='r', linestyle='-')

plt.title('Missing values average per column: Test set ', fontsize=20, weight='bold' )

plt.show()
na = a.shape[0]

nb = b.shape[0]

y_train = a['SalePrice'].to_frame()

#Combine train and test sets

c1 = pd.concat((a, b), sort=False).reset_index(drop=True)

#Drop the target "SalePrice" and Id columns

c1.drop(['SalePrice'], axis=1, inplace=True)

c1.drop(['Id'], axis=1, inplace=True)

print("Total size is :",c1.shape)
c=c1.dropna(thresh=len(c1)*0.9, axis=1)

print('We dropped ',c1.shape[1]-c.shape[1], ' features in the combined set')
allna = (c.isnull().sum() / len(c))

allna = allna.drop(allna[allna == 0].index).sort_values(ascending=False)

plt.figure(figsize=(12, 8))

allna.plot.barh(color='purple')

plt.title('Missing values average per column', fontsize=25, weight='bold' )

plt.show()
print('The shape of the combined dataset after dropping features with more than 90% M.V.', c.shape)
NA=c[allna.index.tolist()]
NAcat=NA.select_dtypes(include='object')

NAnum=NA.select_dtypes(include='number')

print('We have :',NAcat.shape[1],'categorical features with missing values')

print('We have :',NAnum.shape[1],'numerical features with missing values')
NAnum.head()
#MasVnrArea: Masonry veneer area in square feet, the missing data means no veneer so we fill with 0

c['MasVnrArea']=c.MasVnrArea.fillna(0)

#GarageYrBlt:  Year garage was built, we fill the gaps with the median: 1980

c['GarageYrBlt']=c["GarageYrBlt"].fillna(1980)

#For the rest of the columns: Bathroom, half bathroom, basement related columns and garage related columns:

#We will fill with 0s because they just mean that the hosue doesn't have a basement, bathrooms or a garage
NAcat.head()
NAcat1= NAcat.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



NAcat1 = NAcat1.style.background_gradient(cmap=cm)

NAcat1
#We start with features having just few missing value:  We fill the gap with forward fill method:

c['Electrical']=c['Electrical'].fillna(method='ffill')

c['SaleType']=c['SaleType'].fillna(method='ffill')

c['KitchenQual']=c['KitchenQual'].fillna(method='ffill')

c['Exterior1st']=c['Exterior1st'].fillna(method='ffill')

c['Exterior2nd']=c['Exterior2nd'].fillna(method='ffill')

c['Functional']=c['Functional'].fillna(method='ffill')

c['Utilities']=c['Utilities'].fillna(method='ffill')

c['MSZoning']=c['MSZoning'].fillna(method='ffill')
#Categorical missing values

NAcols=c.columns

for col in NAcols:

    if c[col].dtype == "object":

        c[col] = c[col].fillna("None")
#Numerical missing values

for col in NAcols:

    if c[col].dtype != "object":

        c[col]= c[col].fillna(0)
c.isnull().sum().sort_values(ascending=False).head()
c['TotalArea'] = c['TotalBsmtSF'] + c['1stFlrSF'] + c['2ndFlrSF'] + c['GrLivArea'] +c['GarageArea']



c['Bathrooms'] = c['FullBath'] + c['HalfBath']*0.5 



c['Year average']= (c['YearRemodAdd']+c['YearBuilt'])/2
#c['MoSold'] = c['MoSold'].astype(str)

c['MSSubClass'] = c['MSSubClass'].apply(str)

c['YrSold'] = c['YrSold'].astype(str)
cb=pd.get_dummies(c)

print("the shape of the original dataset",c.shape)

print("the shape of the encoded dataset",cb.shape)

print("We have ",cb.shape[1]- c.shape[1], 'new encoded features')
Train = cb[:na]  #na is the number of rows of the original training set

Test = cb[na:] 
fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot2grid((3,2),(0,0))

plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('yellowgreen'), alpha=0.5)

plt.axvline(x=4600, color='r', linestyle='-')

plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((3,2),(0,1))

plt.scatter(x=a['TotalBsmtSF'], y=a['SalePrice'], color=('red'),alpha=0.5)

plt.axvline(x=5900, color='r', linestyle='-')

plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((3,2),(1,0))

plt.scatter(x=a['1stFlrSF'], y=a['SalePrice'], color=('deepskyblue'),alpha=0.5)

plt.axvline(x=4000, color='r', linestyle='-')

plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((3,2),(1,1))

plt.scatter(x=a['MasVnrArea'], y=a['SalePrice'], color=('gold'),alpha=0.9)

plt.axvline(x=1500, color='r', linestyle='-')

plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((3,2),(2,0))

plt.scatter(x=a['GarageArea'], y=a['SalePrice'], color=('orchid'),alpha=0.5)

plt.axvline(x=1230, color='r', linestyle='-')

plt.title('Garage Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((3,2),(2,1))

plt.scatter(x=a['TotRmsAbvGrd'], y=a['SalePrice'], color=('tan'),alpha=0.9)

plt.axvline(x=13, color='r', linestyle='-')

plt.title('TotRmsAbvGrd - Price scatter plot', fontsize=15, weight='bold' )



a['GrLivArea'].sort_values(ascending=False).head(2)
a['TotalBsmtSF'].sort_values(ascending=False).head(1)
a['MasVnrArea'].sort_values(ascending=False).head(1)
a['1stFlrSF'].sort_values(ascending=False).head(1)
a['GarageArea'].sort_values(ascending=False).head(4)
a['TotRmsAbvGrd'].sort_values(ascending=False).head(1)
#train=Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500) & (Train['GarageArea'] < 1240)

#           & (Train['TotRmsAbvGrd'] < 13)]



#print('We removed ',Train.shape[0]- train.shape[0],'outliers')
train=Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500)]



print('We removed ',Train.shape[0]- train.shape[0],'outliers')
target=a[['SalePrice']]

target.loc[1298]
target.loc[523]
#train=Train.copy()

#pos=[30,   88,  142,  277,  308,  328,  365,  410,  438,  462,  495,

#        523,  533,  581,  588,  628,  632,  681,  688,  710,  714,  728,

#        774,  812,  874,  898,  916,  935,  968,  970, 1062, 1168, 1170,

#        1181, 1182, 1298, 1324, 1383, 1423, 1432, 14]

#target.drop(target.index[pos], inplace=True)

#train.drop(target.index[pos], inplace=True)
#pos = [1298,523, 297, 581, 1190, 1061, 635, 197,1328, 495, 583, 313, 335, 249, 706]

pos = [1298,523, 297]

target.drop(target.index[pos], inplace=True)
print('We make sure that both train and target sets have the same row number after removing the outliers:')

print( 'Train: ',train.shape[0], 'rows')

print('Target:', target.shape[0],'rows')
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('orchid'), alpha=0.5)

plt.title('Area-Price plot with outliers',weight='bold', fontsize=18)

plt.axvline(x=4600, color='r', linestyle='-')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

plt.scatter(x=train['GrLivArea'], y=target['SalePrice'], color='navy', alpha=0.5)

plt.axvline(x=4600, color='r', linestyle='-')

plt.title('Area-Price plot without outliers',weight='bold', fontsize=18)

plt.show()
print("Skewness before log transform: ", a['GrLivArea'].skew())

print("Kurtosis before log transform: ", a['GrLivArea'].kurt())
from scipy.stats import skew



#numeric_feats = c.dtypes[c.dtypes != "object"].index



#skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

#skewed_feats = skewed_feats[skewed_feats > 0.75]

#skewed_feats = skewed_feats.index



#train[skewed_feats] = np.log1p(train[skewed_feats])



print("Skewness after log transform: ", train['GrLivArea'].skew())

print("Kurtosis after log transform: ", train['GrLivArea'].kurt())
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,10))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((2,2),(0,0))

sns.distplot(a.GrLivArea, color='plum')

plt.title('Before: Distribution of GrLivArea',weight='bold', fontsize=18)

#first row sec col

ax1 = plt.subplot2grid((2,2),(0,1))

sns.distplot(a['1stFlrSF'], color='tan')

plt.title('Before: Distribution of 1stFlrSF',weight='bold', fontsize=18)





ax1 = plt.subplot2grid((2,2),(1,0))

sns.distplot(train.GrLivArea, color='plum')

plt.title('After: Distribution of GrLivArea',weight='bold', fontsize=18)

#first row sec col

ax1 = plt.subplot2grid((2,2),(1,1))

sns.distplot(train['1stFlrSF'], color='tan')

plt.title('After: Distribution of 1stFlrSF',weight='bold', fontsize=18)

plt.show()
print("Skewness before log transform: ", target['SalePrice'].skew())

print("Kurtosis before log transform: ",target['SalePrice'].kurt())
#log transform the target:

target["SalePrice"] = np.log1p(target["SalePrice"])
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

plt.hist(a.SalePrice, bins=10, color='mediumpurple',alpha=0.5)

plt.title('Sale price distribution before normalization',weight='bold', fontsize=18)

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

plt.hist(target.SalePrice, bins=10, color='darkcyan',alpha=0.5)

plt.title('Sale price distribution after normalization',weight='bold', fontsize=18)

plt.show()
print("Skewness after log transform: ", target['SalePrice'].skew())

print("Kurtosis after log transform: ",target['SalePrice'].kurt())
x=train

y=np.array(target)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=0)
from sklearn.preprocessing import RobustScaler

scaler= RobustScaler()

# transform "x_train"

x_train = scaler.fit_transform(x_train)

# transform "x_test"

x_test = scaler.transform(x_test)

#Transform the test set

X_test= scaler.transform(Test)
#from sklearn.linear_model import LinearRegression



#lreg=LinearRegression()

#MSEs=ms.cross_val_score(lreg, x, y, scoring='neg_mean_squared_error', cv=5)

#meanMSE=np.mean(MSEs)

#print(meanMSE)

#print('RMSE = '+str(math.sqrt(-meanMSE)))
import sklearn.model_selection as GridSearchCV

from sklearn.linear_model import Ridge



ridge=Ridge()

parameters= {'alpha':[x for x in range(1,101)]}



ridge_reg=ms.GridSearchCV(ridge, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

ridge_reg.fit(x_train,y_train)

print("The best value of Alpha is: ",ridge_reg.best_params_)

print("The best score achieved with Alpha=11 is: ",math.sqrt(-ridge_reg.best_score_))

ridge_pred=math.sqrt(-ridge_reg.best_score_)
ridge_mod=Ridge(alpha=15)

ridge_mod.fit(x_train,y_train)

y_pred_train=ridge_mod.predict(x_train)

y_pred_test=ridge_mod.predict(x_test)



print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_pred_train))))

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_pred_test))))   
from sklearn.linear_model import Lasso



parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}



lasso=Lasso()

lasso_reg=ms.GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

lasso_reg.fit(x_train,y_train)



print('The best value of Alpha is: ',lasso_reg.best_params_)
lasso_mod=Lasso(alpha=0.0009)

lasso_mod.fit(x_train,y_train)

y_lasso_train=lasso_mod.predict(x_train)

y_lasso_test=lasso_mod.predict(x_test)



print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_lasso_train))))

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_lasso_test))))
coefs = pd.Series(lasso_mod.coef_, index = x.columns)



imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh", color='yellowgreen')

plt.xlabel("Lasso coefficient", weight='bold')

plt.title("Feature importance in the Lasso Model", weight='bold')

plt.show()


print("Lasso kept ",sum(coefs != 0), "important features and dropped the other ", sum(coefs == 0)," features")
from sklearn.linear_model import ElasticNetCV



#alphas = [10,1,0.1,0.01,0.001,0.002,0.003,0.004,0.005,0.00054255]

#l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]



#elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)



#elasticmod = elastic_cv.fit(x_train, y_train.ravel())

#ela_pred=elasticmod.predict(x_test)

#print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, ela_pred))))

#print(elastic_cv.alpha_)
from sklearn.linear_model import ElasticNetCV



alphas = [0.000542555]

l1ratio = [0.1, 0.3,0.5, 0.9, 0.95, 0.99, 1]



elastic_cv = ElasticNetCV(cv=5, max_iter=1e7, alphas=alphas,  l1_ratio=l1ratio)



elasticmod = elastic_cv.fit(x_train, y_train.ravel())

ela_pred=elasticmod.predict(x_test)

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, ela_pred))))

print(elastic_cv.alpha_)
from xgboost.sklearn import XGBRegressor



#xg_reg = XGBRegressor()

#xgparam_grid= {'learning_rate' : [0.01],'n_estimators':[2000, 3460, 4000],

#                                     'max_depth':[3], 'min_child_weight':[3,5],

#                                     'colsample_bytree':[0.5,0.7],

#                                     'reg_alpha':[0.0001,0.001,0.01,0.1,10,100],

#                                    'reg_lambda':[1,0.01,0.8,0.001,0.0001]}



#xg_grid=GridSearchCV(xg_reg, param_grid=xgparam_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

#xg_grid.fit(x_train,y_train)

#print(xg_grid.best_estimator_)

#print(xg_grid.best_score_)
xgb= XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.5, gamma=0,

             importance_type='gain', learning_rate=0.01, max_delta_step=0,

             max_depth=3, min_child_weight=0, missing=None, n_estimators=4000,

             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,

             reg_alpha=0.0001, reg_lambda=0.01, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

xgmod=xgb.fit(x_train,y_train)

xg_pred=xgmod.predict(x_test)

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, xg_pred))))
from sklearn.ensemble import VotingRegressor



vote_mod = VotingRegressor([('Ridge', ridge_mod), ('Lasso', lasso_mod), ('Elastic', elastic_cv), 

                            ('XGBRegressor', xgb)])

vote= vote_mod.fit(x_train, y_train.ravel())

vote_pred=vote.predict(x_test)



print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, vote_pred))))
from mlxtend.regressor import StackingRegressor





stregr = StackingRegressor(regressors=[elastic_cv,ridge_mod, lasso_mod, vote_mod], 

                           meta_regressor=xgb, use_features_in_secondary=True

                          )



stack_mod=stregr.fit(x_train, y_train.ravel())

stacking_pred=stack_mod.predict(x_test)



print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, stacking_pred))))
final_test=(0.3*vote_pred+0.5*stacking_pred+ 0.2*y_lasso_test)

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, final_test))))

#VotingRegressor to predict the final Test

vote_test = vote_mod.predict(X_test)

final1=np.expm1(vote_test)



#StackingRegressor to predict the final Test

stack_test = stregr.predict(X_test)

final2=np.expm1(stack_test)



#LassoRegressor to predict the final Test

lasso_test = lasso_mod.predict(X_test)

final3=np.expm1(lasso_test)

#Submission of the results predicted by the average of Voting/Stacking/Lasso

final=(0.2*final1+0.6*final2+0.2*final3)



final_submission = pd.DataFrame({

        "Id": b["Id"],

        "SalePrice": final

    })

final_submission.to_csv("final_submission.csv", index=False)

final_submission.head()