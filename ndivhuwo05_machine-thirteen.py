# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.drop('Id', inplace = True, axis =1)

test.drop('Id', inplace = True, axis =1)

train.head()
test.head()
#variable change for later use

test_A = test
corr_overall = train.corr()

k=12



col = corr_overall.nlargest(k, 'SalePrice')['SalePrice'].index

corr_coeff = np.corrcoef(train[col].values.T)

ax = plt.subplots(figsize=(20,15))

heatmap = sns.heatmap(corr_coeff, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':12}, yticklabels=col.values, xticklabels=col.values)
train.shape
fig, plot = plt.subplots()

plot.scatter(x= train['OverallQual'], y= train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)

plt.show()
fig, plot = plt.subplots()

plot.scatter(x= train['GrLivArea'], y= train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index) #drop rows that meet this criteria
fig, plot = plt.subplots()

plot.scatter(x= train['GrLivArea'], y= train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train.shape
train_A = train.drop(['SalePrice'], axis=1)

y = pd.DataFrame(train['SalePrice'])

features = pd.concat([train_A, test_A]).reset_index(drop=True)
features.head()
#function to plot the distribution of a variable. Returns the distribution, skeweness and kurtosis

def distribution(df,column_name):

    

    sns.distplot(df[column_name], color = 'b',kde = True)

    plt.title('Distribution of ' + column_name)

    plt.xlabel(column_name)

    plt.ylabel('Number of occurences')

    

    #skewness

    skewness = df[column_name].skew()

    if (skewness > -0.5) & (skewness < 0.5):

        print('The data is fairly symmetrical with skewness of ' + str(skewness))

    elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

        print('The data is moderately skewed with skewness of ' + str(skewness))

    elif (skewness < -1) | (skewness > 1):

        print('The data is highly skewed with skewness of ' + str(skewness))

    #kurtosis    

    print('The kurtosis is ' + str(df[column_name].kurt()))

distribution(y,'SalePrice')
y_log = pd.DataFrame(np.log1p(y['SalePrice']))

distribution(y_log,'SalePrice')
numerical_features = features.dtypes[features.dtypes != "object"].index

categorical_features = features.dtypes[features.dtypes == "object"].index



numerical_df = features[numerical_features]

categorical_df = features[categorical_features]
features_na = (features.isnull().sum() / len(features)) * 100

#drop features without missing values

features_na = features_na.drop(features_na[features_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :features_na})



#plot

f, ax = plt.subplots(figsize=(10, 8))

sns.barplot(x=features_na.index, y=features_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent', fontsize=15)

plt.xticks(rotation='90')

plt.title('Percent of missing data by feature', fontsize=15)

#mean

def fill_na_num_mean(df,column_name):



    df[column_name] = df[column_name].transform(lambda x: x.fillna(round(x.mean(),1)))

    

    return df

#median

def fill_na_num_median(df,column_name):



    df[column_name] = df[column_name].transform(lambda x: x.fillna(round(x.median(),1)))

    

    return df



#Nan to None

def fill_na_cat_none(df,column_name):

    df[column_name].fillna('None',inplace=True)

    

#Nan to 0 

def fill_na_num_0(df,column_name):

    df[column_name].fillna(0,inplace=True)

    

#mode

def fill_na_mode(df,column_name):

    df[column_name].fillna(df[column_name].mode()[0], inplace = True)

    
quantitative = [f for f in features.columns if features.dtypes[f] != 'object'] #numerical

qualitative = [f for f in features.columns if features.dtypes[f] == 'object'] #categorical



f = pd.melt(features, value_vars=quantitative)

g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)

g = g.map(sns.distplot, "value")
features.drop(['MiscFeature'], inplace = True, axis = 1)

features.drop(['MiscVal'], inplace = True, axis = 1)
features.drop(['Fence'], inplace = True, axis =1)

features.drop(['Alley'], inplace = True, axis = 1)

features.drop(['PoolQC'], inplace = True, axis = 1)

features.drop(['PoolArea'], inplace = True, axis =1)
plt.hist(features['LotFrontage'])

plt.show()



#skewness

skewness = features['LotFrontage'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['LotFrontage'].kurt()))
fill_na_mode(features,'LotFrontage')
features[['MasVnrArea','MasVnrType']].head()

fill_na_cat_none(features,'MasVnrType') 

fill_na_num_0(features,'MasVnrArea')

features[['BsmtFinSF2','BsmtFinType2']].head()


fill_na_num_0(features,'BsmtFinSF2')



features[['BsmtFinSF1','BsmtFinType1']].head()



features[['BsmtUnfSF','BsmtFinType1','BsmtFinType2']].head()
fill_na_cat_none(features,'BsmtFinType1')

fill_na_num_0(features,'BsmtFinSF1')

fill_na_cat_none(features,'BsmtFinType2')

fill_na_num_0(features,'BsmtFinSF2')



features[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']].head()
fill_na_num_0(features,'TotalBsmtSF')

fill_na_num_0(features,'BsmtUnfSF')
features[['BsmtFullBath','BsmtHalfBath','TotalBsmtSF']].head()
fill_na_num_0(features,'BsmtFullBath')

fill_na_num_0(features,'BsmtHalfBath')
features[['GarageType','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']].head()
features.drop('GarageYrBlt', inplace = True, axis = 1)

features.drop(['Utilities'], axis=1, inplace = True)



fill_na_cat_none(features,'GarageType')

fill_na_cat_none(features,'GarageFinish')

fill_na_cat_none(features,'GarageQual')

fill_na_cat_none(features,'GarageCond')





fill_na_num_0(features,'GarageCars')

fill_na_num_0(features,'GarageArea')
new_numerical_features = features.dtypes[features.dtypes != "object"].index



new_categorical_features = features.dtypes[features.dtypes == "object"].index



new_num_df = features[new_numerical_features]

new_cat_df = features[new_categorical_features]



features[['BsmtQual','BsmtCond','BsmtExposure','TotalBsmtSF']].head()
fill_na_cat_none(features,'BsmtExposure')

fill_na_cat_none(features,'BsmtCond')

fill_na_cat_none(features,'BsmtQual')

fill_na_mode(features,'MSZoning')

fill_na_mode(features,'SaleType')

fill_na_mode(features,'KitchenQual')

fill_na_mode(features,'Electrical')

fill_na_mode(features,'Exterior1st')

fill_na_mode(features,'Exterior2nd')

fill_na_mode(features,'Functional')



features[['FireplaceQu','Fireplaces']].head()
fill_na_cat_none(features,'FireplaceQu')
features.head()
#encounding categorical data (ordinal)

features['LandContour'] = features['LandContour'].replace(dict(Lvl=4, Bnk=3, HLS=2, Low=1))

features['LandSlope'] = features['LandSlope'].replace(dict(Gtl=3, Mod=2, Sev=1))

features['ExterQual'] =features['ExterQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['ExterCond'] =features['ExterCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['BsmtQual'] = features['BsmtQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['BsmtCond'] =features['BsmtCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['BsmtCond'] =features['BsmtCond'].replace('None',0)

features['BsmtExposure'] =features['BsmtExposure'].replace(dict(Gd=4, Av=3, Mn=2, No=1))

features['BsmtExposure'] =features['BsmtExposure'].replace('None',0)

features['BsmtFinType1'] = features['BsmtFinType1'].replace(dict(GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1))

features['BsmtFinType1'] =features['BsmtFinType1'].replace('None',0)

features['BsmtFinType2'] = features['BsmtFinType2'].replace(dict(GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1))

features['BsmtFinType2'] = features['BsmtFinType2'].replace('None',0)

features['HeatingQC'] = features['HeatingQC'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['CentralAir'] = features['CentralAir'].replace(dict(Y=1, N=0))

features['KitchenQual'] =features['KitchenQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['Functional'] = features['Functional'].replace(dict(Typ=8, Min1=7, Min2=6, Mod=5, Maj1=4, Maj2=3, Sev=2, Sal=1))

features['FireplaceQu'] = features['FireplaceQu'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['FireplaceQu'] = features['FireplaceQu'].replace('None', 0)

features['GarageQual'] = features['GarageQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['GarageQual'] =features['GarageQual'].replace('None', 0)

features['GarageCond'] = features['GarageCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

features['GarageCond'] = features['GarageCond'].replace('None', 0)

features['LotShape'] = features['LotShape'].replace(dict(Reg=4, IR1=3, IR2=2, IR3=1))

features['GarageFinish'] = features['GarageFinish'].replace(dict(Fin=3, RFn=2, Unf=1))

features['GarageFinish'] = features['GarageFinish'].replace('None', 0)

features['PavedDrive'] =features['PavedDrive'].replace(dict(Y=3, P=2, N=1))

features =features.astype({"Functional": int})
features['BsmtQual'] =features['BsmtQual'].replace('None',0)#assigning 0 to none to turn the object into an int

features =features.astype({"KitchenQual": int})

features =features.astype({"Functional": int})



#dummy variables of none ranked columns

features['MSSubClass'] =features['MSSubClass'].astype('category')

features['MoSold'] = features['MoSold'].astype('category')

features['YrSold'] =features['YrSold'].astype('category')

cat_cols = ['MSZoning','Street','MSSubClass', 'MoSold', 'YrSold', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition']

dumies=pd.get_dummies(features[cat_cols],drop_first=True)

dumies.head()
for column in cat_cols:

    features.drop([column],axis=1,inplace=True)
features=pd.concat([ features,dumies],axis=1)

features.head()
features.dtypes
new_train = features.iloc[:1458, :]

new_test = features.iloc[1458:,:]



X_test = new_test

X_train = new_train



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_train)

X_standardize = pd.DataFrame(X_scaled, columns = X_train.columns)

X_standardize.head()
X_standardize.describe().loc['std']
from sklearn.linear_model import Ridge

from sklearn import metrics



ridge = Ridge()

ridge.fit(X_train, y_log)



y_sale = ridge.predict(X_test)

y_predict_ridge = np.expm1(y_sale)

submission = pd.DataFrame(y_predict_ridge)



#creating submission file

sample = pd.read_csv('../input/sample_submission.csv')

sample['SalePrice']=submission[0]

sample.to_csv('Ridge.csv',index = False)
#train predict

y_train = ridge.predict(X_train)
from sklearn.linear_model import Lasso,ElasticNet

from sklearn import metrics
lasso = Lasso(alpha=0.01)

lasso.fit(X_train, y_log)



y_lasso_pred = lasso.predict(X_test)

lasso_predict = np.expm1(y_lasso_pred)

submission = pd.DataFrame(lasso_predict)



#creating submission file

sample = pd.read_csv('../input/sample_submission.csv')

sample['SalePrice']=submission[0]

sample.to_csv('Lasso.csv',index = False)
y_lasso_train = lasso.predict(X_train)
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



ENet.fit(X_train, y_log)

ENet_train_pred = ENet.predict(X_train)



ENet_pred = np.expm1(ENet.predict(X_test))

sub = pd.DataFrame(ENet_pred)



sub.head()



#creating submission file

sample = pd.read_csv('../input/sample_submission.csv')

sample['SalePrice']=sub[0]

sample.to_csv('Enet.csv',index = False)

ridge.score(X_train,y_log)
ENet.score(X_train,y_log)
lasso.score(X_train,y_log)
print('Ridge MSE: ',metrics.mean_squared_error(y_log,y_train))
print('Enet MSE: ',metrics.mean_squared_error(y_log,ENet_train_pred))
print('Lasso MSE: ',metrics.mean_squared_error(y_log,y_lasso_train))
plt.scatter(np.arange(len(y_log)), ENet_train_pred, label='Predicted')

plt.scatter(np.arange(len(y_log)), y_log, label='Training')



plt.legend()

plt.show()