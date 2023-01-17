# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import re

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df=pd.concat([df1,df_test],axis=0)

#print(df.shape)

#print(df.columns)

pd.pandas.set_option('display.max_columns',None)

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
print(df.shape)
#lets find out the missing value, its percentage and data type 

total = df.isnull().sum().sort_values(ascending=False)

percent = ((df.isnull().sum()/len(df))*100).sort_values(ascending=True)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

nan=missing_data[missing_data["Total"]>0].index

missing_data=missing_data[missing_data["Total"]>0]

nan=pd.DataFrame(nan, columns=['nan col'])

nan

j=[]

for i in nan['nan col']:

    j.append(df[i].dtype)

j=pd.DataFrame(j,columns=['type'])

nan=pd.concat([nan,j],axis=1).set_index('nan col')

nan=pd.concat([missing_data,nan],axis=1)

nan
# lets the divide the columns into numerical and categorical data  and further divide numerical data continous, discrete,and temporal data.

#df=df.drop(['Id'],axis=1)

col=df.columns

num=[]

cat=[]

for i in col:

    

    #numerical=df12[df12[i].dtype !='object']

    if (df[i].dtype =='object')==True:

        cat.append(i)

    else:

        num.append(i)

num=pd.DataFrame(num,columns=['numerical'])

cat=pd.DataFrame(cat,columns=['categorical'])

num

    #    

#print(num)

#seperating numerical data into temporal data

year=[]

for i in num['numerical']:    

    if 'yr'in i or 'YR' in i or 'YEAR' in i or 'year'in i or 'Year'in i or'Yr' in i:

        year.append(i)

year=pd.DataFrame(year,columns=['Year'])

print(year)

#seperating numerical data into cat and descrete data

num_cont=[]

num_dis=[]

for i in num['numerical']:

    if len(df[i].unique())<25:

        num_dis.append(i)

    else:

        num_cont.append(i)

    

    

num_cont=pd.DataFrame(num_cont,columns=['cont']) 

num_dis=pd.DataFrame(num_dis,columns=['dis']) 

print(num_dis)

print(num_cont)
#Relationship with categorical features

for i in num['numerical']:

    plt.scatter(df[i],df['SalePrice'])

    plt.xlabel(i)

    plt.ylabel('saleprice')

    plt.title(i)

    plt.show()

    


area=df['1stFlrSF']+df['2ndFlrSF']#+df['ScreenPorch']

print(area,df['LowQualFinSF'])




df['Functional'] = df['Functional'].fillna('Typ')

df['Electrical'] = df['Electrical'].fillna("SBrkr")

df['KitchenQual'] = df['KitchenQual'].fillna("TA")



df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])



df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df=df.drop(['PoolQC','MiscFeature','Alley','Id'],axis=1)
#categorical Feature

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))# cat ordinal 

df['Functional'] = df['Functional'].fillna('Typ')

df['Electrical'] = df['Electrical'].fillna("SBrkr")

df['KitchenQual'] = df['KitchenQual'].fillna("TA")

type1=nan[nan['type']=='object'].index

#print(type1)

for i in type1:

    if i !='PoolQC' and i !='MiscFeature' and i !='Alley' and i !='MSZoning':

        

        #print(i)

        df[i] = df[i].fillna(df[i].mode()[0])

        plt.scatter(df[i],df['SalePrice'])

        plt.xlabel(i)

        plt.ylabel('saleprice')

        plt.title(i)

        plt.show()
type1=nan[nan['type']=='object'].index

#print(type1)

for i in type1:

    if i !='PoolQC' and i !='MiscFeature' and i !='Alley':

        total = df[i].isnull().sum()

        

        print(total)

#numerical Feature

type1=nan[nan['type']!='object'].index

print(type1)

for i in type1:

    

    total = df[i].isnull().sum()

        

    #print(total)

    df[i] = df[i].fillna(df[i].mode()[0])

    total = df[i].isnull().sum()

    df[i] = df[i].fillna(df[i].mode()[0])

    plt.scatter(df[i],df['SalePrice'])

    plt.xlabel(i)

    plt.ylabel('saleprice')

    plt.title(i)

    plt.show()

    
total = df.isnull().sum().sort_values(ascending=False)

print(total)
from scipy.stats import skew





numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in df.columns:

    if df[i].dtype in numeric_dtypes: 

        numerics2.append(i)



skew_features = df[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews = pd.DataFrame({'skew':skew_features})

skews

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



high_skew = skew_features[skew_features > 0.5]

high_skew = high_skew

skew_index = high_skew.index



for i in skew_index:

    df[i]= boxcox1p(df[i], boxcox_normmax(df[i]+1))



        

skew_features2 = df[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews2 = pd.DataFrame({'skew':skew_features2})

skews2
y=df.iloc[:,-1]

y

x=df.copy()

x.drop(['SalePrice'],axis=1)

print(x)
from pandas.api.types import CategoricalDtype

d=df.columns

df1=df.copy()

dt=[]

for i in d:

    dt.append(df[i].dtype)

d=pd.DataFrame(d,columns=['column'])

dty=pd.DataFrame(dt,columns=['type'])

dty1=pd.concat([d,dty],axis=1)

for i in dty1['column']:

    if df[i].dtype=='object':

        r=df[i].unique()

        print(r)

        df[i] = df[i].astype(CategoricalDtype(categories=r, ordered = True)).cat.codes

        

print(df['MSZoning'])
df[df == 'continuous']
y_train1=df['SalePrice']

y_train1=pd.DataFrame(y_train1,columns=['SalePrice'])

y_train=y_train1.iloc[:1459,:]

df=df.drop(['SalePrice'],axis=1)

X_train=df.iloc[:1459,:]

X_test=df.iloc[1460:,:]

print(X_train.shape,y_train.shape,X_test.shape)

print(type(y_train1))


#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(df1.shape,df_test.shape)
print(y_train)
#y_train=pd.DataFrame(y_train,columns=['SalePrice'])

#y_test=pd.DataFrame(y_test,columns=['SalePrice'])

#print(y_train.shape,X_train.shape)
#z=y_train['SalePrice'].iloc(1)

#y_train['SalePrice']=y_train['SalePrice'].astype(str)

#y_test['SalePrice']=y_train['SalePrice'].astype(str)

#print(y_train.iloc[618])
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
model = ExtraTreesClassifier()

model.fit(X_train,y_train)

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

feat_importances.nlargest(30).plot(kind='barh')

plt.show()
bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X_train,y_train)



dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_train.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

x1=featureScores.sort_values(by='Score',ascending=False).head(30)

x2=x1['Specs'].tolist()

x2
print(X_train.shape,y_train.shape)
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_features=76,max_depth=10, random_state=0)

regr.fit(X_train,y_train)

score=regr.predict(X_test)

#print(score)
y_pred1=pd.DataFrame(score,columns=['SalePrice'])

type(y_pred1)
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],y_pred1],axis=1)

datasets.to_csv('sample_submission.csv',index=False)