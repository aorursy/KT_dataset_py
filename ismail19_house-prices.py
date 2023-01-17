# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# print( os.walk('/kaggle/input'))

for dirname, _, filenames in os.walk('/kaggle/input'):
#     print(dirname, _, filenames)
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv");
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv");
train_test_data=[train,test]
train.shape,test.shape
train.head(80)
train['SalePrice'].describe()
#histogram
sns.distplot(train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice  train['GrLivArea']
var = 'GrLivArea'
data = pd.concat([ train['SalePrice'],  train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([ train['SalePrice'],  train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'],  train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'
data = pd.concat([ train['SalePrice'],  train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
#correlation matrix
corrmat =  train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef( train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot( train[cols], size = 2.5)
plt.show();
train.isnull().sum().loc[lambda x : x>0].sort_values(ascending=False)
 
# train.isnull().sum()
def analysis(feature):
    print("*********** group by value ,count ***************")
    print(feature.value_counts())
    print("*********** total null value count ***************")
    print(feature.isnull().value_counts())
    
    

col_obj = train.select_dtypes([np.object]).columns.values.tolist()
# for feature in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageType","GarageFinish","GarageQual"]:
for feature in col_obj:
    for dataset in train_test_data:
        dataset[feature].fillna("NA", inplace=True)
# LotFrontage    259
# GarageYrBlt     81
# MasVnrArea  
test.info()
# col_int_float = train.select_dtypes([np.float64,np.int64,np.uint8]).columns.values.tolist()
col_int_float = train.select_dtypes([np.float64,np.int64,np.uint8]).columns.values.tolist()
col_int_float.remove("SalePrice")
for feature in col_int_float:
    for dataset in train_test_data:
        dataset[feature].fillna(0, inplace=True)
# # analysis(train["SalePrice"])
# # analysis(test["PoolQC"])
# for dataset in train_test_data:
#     dataset["PoolQC"].fillna("NA", inplace=True)
 
# analysis(train["GarageFinish"])
# analysis(test["PoolQC"])
 


# # use pd.concat to join the new columns with your original dataframe
# train = pd.concat([train,pd.get_dummies(train ,columns=['PoolQC' ], prefix=['PoolQC' ])],axis=1)
# test = pd.concat([test,pd.get_dummies(train ,columns=['PoolQC' ], prefix=['PoolQC' ])],axis=1)
# # now drop the original 'country' column (you don't need it anymore)
# train.drop(['PoolQC' ],axis=1, inplace=True)
# test.drop(['PoolQC' ],axis=1, inplace=True)

# fill missing LotFrontage with median LotFrontage for each MSZoning (RL,RM ,FV  ,RH  ,C (all) )
# train["LotFrontage"].fillna(train.groupby("MSZoning")["LotFrontage"].transform("median"), inplace=True)
import pandas as pd
# train = pd.DataFrame(data = [['a', 123, 'ab'], ['b', 234, 'bc']],
#                      columns=['col1', 'col2', 'col3'])
# test = pd.DataFrame(data = [['c', 345, 'ab'], ['b', 456, 'ab']],
#                      columns=['col1', 'col2', 'col3'])
# train_objs_num = len(train)
# dataset = pd.concat(objs=[train, test], axis=0,sort=False)
# dataset_preprocessed = pd.get_dummies(dataset)
# train_preprocessed = dataset_preprocessed[:train_objs_num]
# test_preprocessed = dataset_preprocessed[train_objs_num:]
# # Categorical boolean mask
# categorical_feature_mask = train.dtypes==object
# # filter categorical columns using mask and turn it into a list
# categorical_cols = train.columns[categorical_feature_mask].tolist()

# # import labelencoder
# from sklearn.preprocessing import LabelEncoder
# # instantiate labelencoder object
# le = LabelEncoder()

# # apply le on categorical feature columns
# train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col))
# train[categorical_cols].head(10)

# # import OneHotEncoder
# from sklearn.preprocessing import OneHotEncoder
# # instantiate OneHotEncoder
# ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) 

# # categorical_features = boolean mask for categorical columns
# # sparse = False output an array not sparse matrix

# # apply OneHotEncoder on categorical feature columns
# X_ohe = ohe.fit_transform(train) # It returns an numpy array


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# train['MSZoning']=labelencoder_X.fit_transform(train['MSZoning' ])
# list(labelencoder_X.inverse_transform(train['MSZoning']))

# onehotencoder = OneHotEncoder(categorical_features = [0])
# onehotencoder.fit_transform(train)
# pd.get_dummies(df, prefix=['A', 'D'], columns=['A', 'D'])



# # use pd.concat to join the new columns with your original dataframe
# train = pd.concat([train,pd.get_dummies(train ,columns=['MSZoning', 'Street'], prefix=['MSZoning', 'Street'])],axis=1)
# test = pd.concat([test,pd.get_dummies(train ,columns=['MSZoning', 'Street'], prefix=['MSZoning', 'Street'])],axis=1)
# # now drop the original 'country' column (you don't need it anymore)
# train.drop(['MSZoning', 'Street'],axis=1, inplace=True)
# test.drop(['MSZoning', 'Street'],axis=1, inplace=True)


col_obj = train.select_dtypes([np.object]).columns.values.tolist()


# col_obj.values
train_objs_num = len(train)
test['SalePrice']=-1;
dataset = pd.concat(objs=[train, test], axis=0,sort=False)
dataset_preprocessed = pd.get_dummies(dataset ,columns=col_obj, prefix=col_obj)
# dataset_preprocessed = pd.concat([dataset,pd.get_dummies(dataset ,columns=col_obj, prefix=col_obj)],axis=1)
dataset_preprocessed = pd.concat([dataset[col_obj],pd.get_dummies(dataset ,columns=col_obj, prefix=col_obj)],axis=1)
dataset_preprocessed.drop(col_obj,axis=1, inplace=True)
train = dataset_preprocessed[:train_objs_num]
test = dataset_preprocessed[train_objs_num:]

test.drop(['SalePrice'],axis=1, inplace=True)
# dataset_preprocessed['SaleType_NA']
dataset_preprocessed.to_csv('1.csv', index=False)

# # use pd.concat to join the new columns with your original dataframe
# train = pd.concat([train,pd.get_dummies(train ,columns=col_obj, prefix=col_obj)],axis=1)
# test = pd.concat([test,pd.get_dummies(test ,columns=col_obj, prefix=col_obj)],axis=1)


# # now drop the original 'country' column (you don't need it anymore)
# train.drop(col_obj,axis=1, inplace=True)
# test.drop(col_obj,axis=1, inplace=True)





# for dataset in train_test_data:
#     dataset = pd.concat([dataset,pd.get_dummies(dataset['MSZoning'], prefix='MSZoning')],axis=1)
#     dataset.drop(['MSZoning'],axis=1, inplace=True)
   
   
print(train.shape,test.shape)
train.isnull().sum().loc[lambda x : x>0].sort_values(ascending=False)
test.isnull().sum().loc[lambda x : x>0].sort_values(ascending=False)
 
train.isnull().sum()

# train['MSZoning'].isnull().sum()
# train['MSZoning'].value_counts()

# MSZoning=train.groupby('MSZoning')
# MSZoning.first()
# train['Street'].value_counts()

# street_mapping={
#     'Pave':1,
#     'Grvl':2
# }

# for dataset in train_test_data:
#     dataset['Street']=dataset['Street'].map(street_mapping)
   

# train.head()
# test.head( )
train.info()
%%javascript
IPython.OutputArea.auto_scroll_threshold = 9999;
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots
def bar_chart(feature):
    survived = train[train['SalePrice']==1][feature].value_counts()
    dead = train[train['SalePrice']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['SalePrice','SalePrice']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
# bar_chart(train['SaleType_NA'])
facet = sns.FacetGrid(train, hue="SaleType_NA",aspect=4)
facet.map(sns.kdeplot,'SalePrice',shade= True)
facet.set(xlim=(0, train['SalePrice'].max()))
facet.add_legend()
 
plt.show() 
train = train.drop(['Id'], axis=1)
# test_data = test.drop("Id", axis=1).copy()
test_data = test.drop(['Id'], axis=1)


train_data = train.drop('SalePrice', axis=1)
target = train['SalePrice']
# train_data.columns-train.columns
train.shape,train_data.shape, target.shape,  test.shape
# train_data['SalePrice']



train_data.head(10)
# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
train.info()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
#Fitting multiple layer regression to the training set
# type(target)
# target.reshape((-1, 1))
from sklearn.linear_model  import LinearRegression
regressor=LinearRegression()
regressor.fit(train_data,target.T)

train_data.shape,target.shape,target.T.shape

train_data.to_csv('train_data.csv', index=False)
target.to_csv('target.csv', index=False)
test_data.to_csv('test1.csv', index=False)
#predict
test.info()
test_pred=regressor.predict(test_data)
scores = cross_val_score(regressor,train_data, target, cv = 3)
print(scores)
print(round(np.mean(scores)*100,2))
submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": test_pred
    })

submission.to_csv('submission.csv', index=False)
# from sklearn import cross_validation, linear_model

# loo = cross_val_score.LeaveOneOut(len(target))

# regr = linear_model.LinearRegression()

# scores = cross_validation.cross_val_score(regressor, train_data, target, scoring='mean_squared_error', cv=loo)

# # This will print the mean of the list of errors that were output and 
# # provide your metric for evaluation
# print scores.mean()

# scoring = 'accuracy'
# score = cross_val_score(regressor, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
# print(score)
# # Visualising the Linear Regression results
# plt.scatter(train_data, target.T, color = 'red')
# plt.plot(train_data, regressor.predict(train_data), color = 'blue')
# plt.title('Truth or Bluff (Linear Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()
clf = SVC(gamma='auto')
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Naive Bayes Score
round(np.mean(score)*100, 2)
