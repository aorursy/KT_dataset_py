# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Display all columns
pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.shape
null_values=[features for features in df.columns if df[features].isnull().sum()>1]
print(null_values)
for feature in null_values:
    print(feature,np.round(df[feature].isnull().mean(),4),'% missing values')
sns.heatmap(df.isnull())
for feature in null_values:
    data=df.copy()
    data[feature]=np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
#     sns.countplot(data[feature].isnull())
    plt.title(feature)
    plt.show()
#Drop
numerical_features=[feature for feature in df.columns if df[feature].dtypes!='O']
print("No of numerical values: ",len(numerical_features))
df[numerical_features].head()
year_f=[feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
print(year_f)
for i in year_f:
    print(i,df[i].unique())
df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Medan House Price')
plt.title('House Price vs YearSold')
year_f
for feature in year_f:
    if feature!='YrSold':
        data=df.copy()
        data[feature]=data['YrSold']-data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
        
## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<25 and feature not in year_f+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
discrete_feature
df[discrete_feature].head()
## Lets Find the realtionship between them and Sale PRice

for feature in discrete_feature:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_f+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))
## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_feature:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
data=df.copy()
for feature in continuous_feature:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['Saleprice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('Saleprice')
        plt.title(feature)
        plt.show()
for feature in continuous_feature:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        sns.boxplot(data[feature])
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
categorical_feature=[feature for feature in df.columns if data[feature].dtypes=='O']
categorical_feature


df[categorical_feature].head()
for feature in categorical_feature:
    print("The feature of {} and number of categories are {}".format(feature,len(df[feature].unique())))
for feature in categorical_feature:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
dft=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
dft.head()
dft.shape
feature_null=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtypes=='O']
for feature in feature_null:
    print("{}: {}% missing values".format(feature,np.round(df[feature].isnull().mean(),4)))
feature_nul=[feature for feature in dft.columns if df[feature].isnull().sum()>0 and dft[feature].dtypes=='O']
for feature in feature_null:
    print("{}: {}% missing values".format(feature,np.round(dft[feature].isnull().mean(),4)))
def replace_cat_feature(df,feature_null):
    data=df.copy()
    data[feature_null]=data[feature_null].fillna('Missing')
    return data
df=replace_cat_feature(df,feature_null)
df[feature_null].isnull().sum()
def replace_cat_feature(dft,feature_null):
    data=dft.copy()
    data[feature_null]=data[feature_null].fillna('Missing')
    return data
dft=replace_cat_feature(dft,feature_nul)
dft[feature_nul].isnull().sum()
df.head()
dft.head()
numerical_null=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtypes=='int' or df[feature].dtypes=='float']
for feature in numerical_null:
    print("{}: {}% missing values".format(feature,np.round(df[feature].isnull().mean(),4)))
numerical_nul=[feature for feature in dft.columns if dft[feature].isnull().sum()>0 and dft[feature].dtypes=='int' or dft[feature].dtypes=='float']
for feature in numerical_nul:
    print("{}: {}% missing values".format(feature,np.round(dft[feature].isnull().mean(),4)))
for feature in numerical_null:
    median_value=df[feature].median()
    #we are median since there are outliers
    df[feature+'nan']=np.where(df[feature].isnull(),1,0)
    #Where there is null value replace it with 1 else 0
    df[feature].fillna(median_value,inplace=True)
df[numerical_null].isnull().sum()
for feature in numerical_nul:
    median_value=dft[feature].median()
    #we are median since there are outliers
    dft[feature+'nan']=np.where(dft[feature].isnull(),1,0)
    #Where there is null value replace it with 1 else 0
    dft[feature].fillna(median_value,inplace=True)
dft[numerical_null].isnull().sum()
df.head(10)
dft.head()
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    df[feature]=df['YrSold']-df[feature]
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    dft[feature]=dft['YrSold']-dft[feature]
df.head()
dft.head()
df[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()
dft[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()
import numpy as np
num_features=['LotFrontage','LotArea','1stFlrSF','GrLivArea']
for feature in num_features:
    df[feature]=np.log(df[feature])
import numpy as np
num_features=['LotFrontage','LotArea','1stFlrSF','GrLivArea']
for feature in num_features:
    dft[feature]=np.log(dft[feature])
df.head()
categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']
categorical_features
for feature in categorical_features:
    temp=df.groupby(feature)['SalePrice'].count()/len(df)
    temp_df=temp[temp>0.01].index
    df[feature]=np.where(df[feature],df[feature],'Rare_var')
df.head(50)
feature_scale=[feature for feature in df.columns if feature not in ['Id','SalePrice'] and df[feature].dtypes!='O']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()#Scale down the data between 0 to 1
scaler.fit(df[feature_scale])
feature_scale=[feature for feature in df.columns if feature not in ['Id','SalePrice'] and df[feature].dtypes!='O']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()#Scale down the data between 0 to 1
scaler.fit(dft[feature_scale])
df.head()
dft.head()

data=pd.concat([df[['Id','SalePrice']].reset_index(drop=True),
                 pd.DataFrame(scaler.transform(df[feature_scale]),columns=feature_scale)],
                             axis=1)
data.to_csv('X_train.csv',index=False)
dtrain=pd.read_csv('X_train.csv')
dtrain.head()
data=pd.concat([dft[['Id']].reset_index(drop=True),
                 pd.DataFrame(scaler.transform(dft[feature_scale]),columns=feature_scale)],
                             axis=1)
data.to_csv('X_test.csv',index=False)
dtest=pd.read_csv('X_test.csv')
dtest.head()
from sklearn.linear_model import Lasso
from sklearn .feature_selection import SelectFromModel

y_train=dtrain['SalePrice']
x_train=dtrain.drop(['Id','SalePrice'],axis=1)
x_test=dtest.drop('Id',axis=1)
feature_sel_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
feature_sel_model.fit(x_train,y_train)
feature_sel_model.get_support()
selected_feat=x_train.columns[(feature_sel_model.get_support())]
print("Total features: ",x_train.shape[1])
print("Selected features: ",len(selected_feat))
selected_feat
x_train=x_train[selected_feat]
x_test=x_test[selected_feat]
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from xgboost import XGBRegressor
models=[]
models.append(('LR',LinearRegression()))
models.append(('RR',Ridge(alpha=0.9)))
models.append(('LasR',Lasso(alpha=0.9)))
models.append(('RF',RandomForestRegressor(min_samples_leaf=5)))
models.append(('DT',DecisionTreeRegressor()))
models.append(('KNN',KNeighborsRegressor(n_neighbors=6)))
models.append(('XG',XGBRegressor()))
models.append(('SVR',SVR()))
results=[]
names=[]
scoring='accuracy'
for name,model in models:
    l=model
    l.fit(x_train,y_train)
    print(name," ",l.score(x_train,y_train))
p=PolynomialFeatures(degree=3)
p.fit(x_train)
xt=p.transform(x_train)
l=LinearRegression()
l.fit(xt,y_train)
print("Poly ",l.score(xt,y_train))
X_train,X_test,Y_train,y_test=train_test_split(x_train,y_train,test_size=0.2,random_state=1)
X_train.shape
X_train.head()
params={
    'learning_rate':[0.05,0.1,0.15,0.2,0.25,0.3],
    'max_depth':[3,4,5,6,8,10,12,15,20,25,30,35,40,45],
    'min_child_weight':[1,3,5,7],
    'gamma':[0.0,0.1,0.2,0.3,0.4],
    'colsample_bytree':[0.1,0.4,0.5,0.7]
}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
clf=XGBRegressor()
random_search=RandomizedSearchCV(clf,param_distributions=params,n_iter=5,n_jobs=-1,cv=5,verbose=1)
random_search.fit(X_train,Y_train)
random_search.best_estimator_
random_search.best_params_

my_model=XGBRegressor(min_child_weight=5,
                      max_depth=12,
                      learning_rate=0.15,
                      gamma=0.1,
                      colsample_bytree=0.5)
my_model.fit(X_train,Y_train)
print(my_model.score(X_train,Y_train))
print(my_model.score(X_test,y_test))
# l=my_model.predict(x_test)

# my_model=DecisionTreeRegressor(criterion='mse',min_samples_leaf=15,)
# my_model.fit(x_train,y_train)
# print(my_model.score(X_train,Y_train))
# print(my_model.score(X_test,y_test))
predictions = my_model.predict(x_test)
output = pd.DataFrame({'Id': dtest['Id'],'SalePrice': predictions})
output.to_csv('Saleprice1.csv', index=False)
output