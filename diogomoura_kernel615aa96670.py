# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
x_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head(5)
pd.set_option('display.max_columns', None)
train.loc[:, train.isna().any()].isna().sum()/train.shape[0]
columns_to_drop=['Id','Alley','Fence','PoolQC','MiscFeature','FireplaceQu']
train.drop(columns_to_drop,axis=1,inplace=True)
'''
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN', strategy='median',axis=0)
imp.fit(x_train)
x_train=pd.dataframe(data=imp.transform(x_train),columns=x_train.columns)
print(x_train.isnull().sum().sort_values(ascending=False).head())
'''
train.fillna(train.mean(),inplace=True)
y_train=train.loc[:,'SalePrice']
x_train=train.drop('SalePrice',axis=1)
def dummy_df(df,todummy_list):
    for x in todummy_list:
        dummies=pd.get_dummies(df[x],prefix=x,dummy_na=False)
        df=df.drop(x,axis=1)
        df=pd.concat([df,dummies],axis=1)
    return df
x_train.info()
todummy_list=list(x_train.select_dtypes(include=['object']).columns)
print(todummy_list)
#x_train=dummy_df(x_train,todummy_list)
x_train=pd.get_dummies(x_train)
'''Distribution of features'''
def plot_histogram(x):
    plt.hist(x,color='gray',alpha=0.5)
    plt.title("Histogram of {var_name}".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
plot_histogram(x_train['LotFrontage'])
#plot by category
def plot_histogram_dv(x,y):
    plt.hist(x[y==0],alpha=0.5,label='DV-0')
    plt.hist(x[y==1],alpha=0.5,label='DV-1')
    plt.hist(x,color='gray',alpha=0.5)
    plt.title("Histogram of {var_name} by DV Category".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()
''' Tukey method-outliers detection'''
def find_outliers_tukey (x):
    q1=np.percentile(x,25)
    q3=np.percentile(x,75)
    iqr=q3-q1
    floor=q1-1.5*iqr
    ceiling=q3+1.5*iqr
    outlier_indices=list(x.index[(x<floor)|(x>ceiling)])
    outlier_values=list(x[outlier_indices])
    return outlier_indices, outlier_values
tukey_indices, tukey_values=find_outliers_tukey(x_train['LotFrontage'])
print(np.sort(tukey_values))

''' KDEUnivariate method -outliers detection'''
from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate

def find_outliers_kde(x):
    x_scaled=scale(list(map(float,x)))
    kde=KDEUnivariate(x_scaled)
    kde.fit(bw='scott',fft=True)
    pred=kde.evaluate(x_scaled)
    n=sum(pred<0.05)
    outlier_ind=np.asarray(pred).argsort()[:n]
    outlier_value=np.asarray(x)[outlier_ind]
    return outlier_ind, outlier_value
kde_indices, kde_values=find_outliers_kde(x_train['LotFrontage'])
print(np.sort(kde_values))
'''
Feature engineering
'''
'''
Interactions among features
Note: interactions amongst dummy variables belonging to the same categorical feature are always zero
Although it is very easy to calculate two-way interactions amongst all features, it is very computationally expensive
.10 features = 45 two-way interaction terms
.50 features = 1,225 two-way interaction terms
.100 features = 4,950 two-way interaction terms
.500 features = 124,750 two-way interation terms
.Recommend understanding your data and domain if possible and selectively choosing interaction terms
'''
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    #Get feature names
    combos=list(combinations(list(df.columns),2))
    colnames=list(df.columns)+['_'.join(x) for x in combos]
    #Find interactions
    poly=PolynomialFeatures(interaction_only=True,include_bias=False)
    df=poly.fit_transform(df)
    df=pd.DataFrame(df)
    df.columns=colnames
    #Remove interaction terms with all zero values
    noint_indices=[i for i,x in enumerate(list((df==0).all())) if x]
    df=df.drop(df.columns[noint_indices],axis=1)
    return df
#x_train=add_interactions(x_train)
#print(x_train.head(5))
'''Dimensionality reduction using PCA'''
from sklearn.decomposition import PCA

pca=PCA(n_components=10)
x_pca=pd.DataFrame(pca.fit_transform(x_train))
print(x_pca.head(5))
'''
Test set manipulation
'''
def preprocess_test_set(df,columns_to_drop,to_dummy_list,train_df):
    dataframe=df.drop(columns_to_drop,axis=1)
    for x in to_dummy_list:
        dummies=pd.get_dummies(dataframe[x],prefix=x,dummy_na=False)
        dataframe=dataframe.drop(x,axis=1)
        dataframe=pd.concat([dataframe,dummies],axis=1)
    dataframe=dataframe.fillna(dataframe.mean())
    return dataframe
ids=x_test.loc[:,'Id']
x_test=preprocess_test_set(x_test,columns_to_drop,todummy_list,x_train)
x_train.shape
x_test.shape
x_train,x_test=x_train.align(x_test,join='outer',axis=1)
#Fill na values created from the align in categorical values
x_train.fillna(0,inplace=True)
x_test.fillna(0,inplace=True)
#Scale the features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_test)
x_test=scaler.transform(x_test)
x_train=scaler.transform(x_train)
#Select the best features
from sklearn.feature_selection import SelectKBest, chi2, f_classif
selector = SelectKBest(f_classif, k=150)
selector.fit(x_train,y_train)
# Get columns to keep and create new dataframe with those only
cols = selector.get_support(indices=True)
print(cols)
#x_train = x_train.iloc[:,cols]
#x_test= x_test.iloc[:,cols]
x_train=x_train[:,cols]
x_test=x_test[:,cols]
from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(random_state=0)
clf.fit(x_train,y_train)
from sklearn.ensemble import AdaBoostRegressor
#clf=AdaBoostRegressor(random_state=0)
#clf.fit(x_train,y_train)
from sklearn.ensemble import BaggingRegressor
#clf=BaggingRegressor(random_state=0)
#clf.fit(x_train,y_train)
y_test=pd.concat([ids,pd.Series(clf.predict(x_test),name='SalePrice')],axis=1)
y_test.to_csv('submission.csv',index=False)
print("Done!")