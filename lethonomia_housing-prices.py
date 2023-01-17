import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

import random as rnd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import make_scorer

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LassoCV



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

random_state = 3

import os

#print(os.listdir("../input"))
y_values = [39.669, 39.669, 37.679, 39.813, 39.773, 34.770, 25.369, 25.590, 26.174, 26.029, 26.684, 24.694]

fig = plt.figure(figsize=(14,5), dpi=80)

_ =  sns.lineplot(x=range(len(y_values)),y=y_values,linestyle='--', marker='o', color='b')

plt.xlabel('Submission # (Not including errors)')

plt.ylabel('Percent Error')

arrowprops=dict(color='black',headwidth=2,headlength=2,width=.5)

plt.annotate(xy=(0,39.669),xytext=(0,38),arrowprops=arrowprops,

             s="Halving Sample")

plt.annotate(xy=(2, 37.679),xytext=(2, 36),arrowprops=arrowprops,

             s="5 KFolds")

plt.annotate(xy=(4,39.773),xytext=(4.2,39.773),arrowprops=arrowprops,

             s="Fixing correlation")

plt.annotate(xy=(5,34.770),xytext=(5.2,34.770),arrowprops=arrowprops,

             s="Combining columns")

plt.annotate(xy=(6,25.369),xytext=(6,26),arrowprops=arrowprops,

             s="Voting Predictor")
#'../input/train.csv'

train_df = pd.read_csv('../input/train.csv')

train_df.name = 'Train'

train_df.head()

test_df = pd.read_csv('../input/test.csv')

test_df.name = 'Test'

test_df.head()
both_df = [train_df,test_df]
for dataset in both_df:

    for column in range(len(dataset.columns)):

        if(dataset[dataset.columns[column]].dtypes == 'object'):

            dataset[dataset.columns[column]] = dataset[dataset.columns[column]].fillna('Null')

            #Replace missing string values with Null

    for i in range(len(dataset.isna().sum())):

        if dataset[dataset.columns[i]].isna().sum() > 0:

            dataset[dataset.columns[i]] = dataset[dataset.columns[i]].fillna(0)

            #Replace missing integer values with 0
#Makes a swarm plot of a selected column based on the SalePrice

def makeplot(X,Y=train_df['SalePrice']):

    _ = plt.figure(figsize=(10,5))

    _ = sns.swarmplot(x=X,y=Y,palette=sns.cubehelix_palette())

    _.set_title(X.name + " : SalePrice")
cols = train_df.columns

types = train_df.dtypes

for column in range(len(cols)):

    if column < len(cols)-1:

        print("%s(%s), "%(cols[column],types[column]), end="")

    else:

        print("%s(%s) "%(cols[column],types[column]), end="")
#List of categorical values that need to be converted to dummy values

cat_conds = ('OverallQual','Functional')
#List of failed categorical values (low correlation value)

fail_conds = ('Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'RoofMatl', 'ExterCond', 'Electrical', 

              'PoolQC', 'MiscFeature','Heating','HeatingQC','BsmtFinType1','BsmtFinType2','BsmtCond',

              'Exterior1st','Exterior2nd','RoofStyle','OverallCond','HouseStyle','BldgType','SaleType',

              'Street', 'Alley', 'LotShape','LandSlope','MasVnrType','Foundation','BsmtExposure',

              'PavedDrive','ExterQual','GarageFinish','GarageCond','Fence','LotConfig','BsmtQual','MSSubClass',

              'MSZoning','LandContour','CentralAir','KitchenQual','GarageType','SaleCondition','GarageQual',

              'FireplaceQu')
for column in fail_conds[0:2]:

    makeplot(train_df[column])
for column in cat_conds:

    makeplot(train_df[column])
for dataset in both_df:

    dataset.drop(np.array(fail_conds),axis=1,inplace=True)

train_df.head()
cols = train_df.columns

types = train_df.dtypes

for column in range(len(cols)):

    #columns and their type

    if column < len(cols)-1:

        print("%s(%s), "%(cols[column],types[column]), end="")

    else:

        print("%s(%s) "%(cols[column],types[column]), end="")
#continous values

num_conds = ('YearBuilt','YearRemodAdd','SalePrice',

             'GrLivArea','TotRmsAbvGrd','Fireplaces','FireplaceQu','GarageCars','GarageArea',

             'OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF',

             'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',

             'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath')

fail_num_conds = ('PoolArea','MiscVal','MoSold','BedroomAbvGr','KitchenAbvGr','GarageYrBlt','YrSold',

                  'LowQualFinSF','LotFrontage','LotArea','MasVnrArea')
for column in num_conds[0:5]:

    makeplot(train_df[column])
for dataset in both_df:

    dataset.drop(np.array(fail_num_conds),axis=1,inplace=True)

train_df.head()
#Combine all bath columns

for dataset in both_df:

    dataset['NumBaths'] = dataset['BsmtFullBath'] + 0.5 * dataset['BsmtHalfBath'] + dataset['FullBath'] + 0.5 * dataset['HalfBath'] 

    dataset.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath','HalfBath'],axis=1,inplace=True)

train_df['NumBaths'].head()
#Find total SF and drop other SFs

for dataset in both_df:

    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']

    dataset.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)

train_df['TotalSF'].head()
#Outside SF

for dataset in both_df:

    dataset['TotalPorchSF'] = dataset['OpenPorchSF'] + dataset['3SsnPorch'] + dataset['EnclosedPorch'] + dataset['ScreenPorch'] + dataset['WoodDeckSF']

    dataset.drop(['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'],axis=1,inplace=True)

train_df['TotalPorchSF'].head()
#YearDiff

for dataset in both_df:

    dataset['YearDiff'] = dataset['YearRemodAdd']-dataset['YearBuilt']

    dataset.drop(['YearRemodAdd','YearBuilt'],axis=1,inplace=True)

train_df['YearDiff'].head()
makeplot(train_df['YearDiff'])
_ = plt.hist(x=train_df['YearDiff'],bins=10)
year_bins = pd.qcut(train_df['YearDiff'],10,duplicates='drop').unique()

print(year_bins)
#Categorize the year difference by bin

for dataset in both_df:

    for i in range(0,len(year_bins)):

        dataset.loc[(dataset['YearDiff'] > pd.IntervalIndex(year_bins).left[i]) & (dataset['YearDiff'] <= pd.IntervalIndex(year_bins).right[i]), 'YearDiff'] = i

train_df.head()
dummies = ('OverallQual','Functional')
for column in dummies:

    dummy = pd.get_dummies(train_df[column],prefix=(train_df[column].name))

    train_df = train_df.join(dummy)

    train_df.drop([column],axis=1,inplace=True)

    #Next dataset

    dummy = pd.get_dummies(test_df[column],prefix=(test_df[column].name))

    test_df = test_df.join(dummy)

    test_df.drop([column],axis=1,inplace=True)
train_df.head()
test_df.drop(['Functional_Null'],axis=1,inplace=True)
y = train_df.copy()['SalePrice']

ids = test_df.copy()['Id']

train = train_df.copy().drop(['SalePrice','Id'],axis=1)

test = test_df.copy().drop(['Id'],axis=1)
train.head()
'''First use of the model_fitter function. kfolds is inside the function so that

I could change the number of columns as I worked.'''

def model_fitter(model,folds, X=train,y=y,test=test):

    kfolds = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    cv = cross_validate(model, X, y,cv=kfolds,scoring=make_scorer(mean_squared_log_error),

                        return_train_score=False, return_estimator=True)

    print("Best Score is: %s, located at %s"%(min(cv['test_score']),list(cv['test_score']).index(min(cv['test_score']))))

    best_rfc = cv['estimator'][list(cv['test_score']).index(min(cv['test_score']))]

    print(best_rfc) #returns the best estimator parameters from the many generated

    return(best_rfc)
rfc = RandomForestClassifier(random_state=random_state)

f_rfc = model_fitter(rfc,3,train,y,test)

#f_rfc_pred = f_rfc.predict(test)
knn = KNeighborsClassifier()

f_knn = model_fitter(knn,3,train,y,test)

#f_knn_pred = f_knn.predict(test)
dtc =DecisionTreeClassifier(random_state=random_state)

f_dtc = model_fitter(dtc,3,train,y,test)

#f_dtc_pred = f_dtc.predict(test)
gnb = GaussianNB()

f_gnb = model_fitter(gnb,3,train,y,test)

#f_gnb_pred = f_gnb.predict(test)
train.shape
estimators=[('RFC', f_rfc),('KNN', f_knn),('DTC', f_dtc),('GNB', f_gnb)]

VotingPredictor = VotingClassifier(estimators=estimators,voting='soft', n_jobs=5)

f_vp = model_fitter(VotingPredictor,3,train,y,test)

f_vp_pred = f_vp.predict(test)

print(f_vp_pred)
ans = pd.DataFrame()

ans['SalePrice'] = f_vp_pred

ans.head()
ans.index.name='Id'

ans = ans.reset_index()

ans['Id'] = test_df['Id']

ans.to_csv('ans.csv',index=False)

ans.head()

#0.39669 (Halving sample)

#0.37679 (5 KFolds)

#0.39773 (Fixing correlation)

#0.34770 (Combining columns)

#0.25369 (Voting Predictor) 107 places!
train.columns