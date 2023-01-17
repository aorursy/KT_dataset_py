# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/home-data-for-ml-course/train.csv')
df1=pd.read_csv('../input/home-data-for-ml-course/test.csv')
df.info()
k=df.copy()

df.drop(['Fence','PoolQC','MiscFeature','FireplaceQu','Alley'],axis=1,inplace=True)
df.info()
df1.info()
k1=df1.copy()

df1.drop(['Fence','PoolQC','MiscFeature','FireplaceQu','Alley'],axis=1,inplace=True)
df1.info()
df.describe()
df.head()
from fancyimpute import KNN
df.info()
miss_val=['LotFrontage','MasVnrArea']
k=df[['LotFrontage','MasVnrArea']]
# Function for KNN imputation

def knn(t,j):

    z=t

    z.loc[0,j]=np.NaN

    z=pd.DataFrame(KNN(k=3).fit_transform(z),columns=z.columns)

    return(z.loc[0,j])



# Function for imputation using mean value.

def mean(t,j):

    z=t

    z.loc[0,j]=np.NaN

    z=z.loc[:,j].fillna(z.loc[:,j].mean())

    return(z[0])



# Function for imputation using median value.

def median(t,j):

    z=t

    z.loc[0,j]=np.NaN

    z=z.loc[:,j].fillna(z.loc[:,j].median())

    return(z[0])



# Function for imputing the missing values.

# Here we have first stored a non null value of a particular column stored it in a separate variable and replaced it in the dataframe with nan.

# Then we have imputed the missing value using the mean and median and depending upon which method imputes the value closes to the actual value is used for imputing the missing values in the dataset.

def impute(t):

    for j in miss_val:

            if(sum(t.loc[:,j].isnull())!=0):

                p=mean(t,j)

                q=median(t,j)

                r=knn(t,j)

                if(abs(p-t.loc[0,j]) < abs(q-t.loc[0,j]) and abs(p-t.loc[0,j]) < abs(r-t.loc[0,j])):

                    t.loc[:,j]=t.loc[:,j].fillna(t.loc[:,j].mean())

                elif(abs(q-t.loc[0,j]) < abs(p-t.loc[0,j]) and abs(q-t.loc[0,j]) < abs(r-t.loc[0,j])):

                    t.loc[:,j]=t.loc[:,j].fillna(t.loc[:,j].median())

                else:

                    t=pd.DataFrame(KNN(k=3).fit_transform(t),columns=t.columns)

            else:

                continue

    return(t)
k=impute(k)
df[['LotFrontage','MasVnrArea']]=k[['LotFrontage','MasVnrArea']]
df.info()
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
df['GarageYrBlt']=df['GarageYrBlt'].astype(int)
df.info()
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
con_var = [ c for c in list(df.columns) if df.dtypes[c] != 'object' ]

cat_var=[c for c in list(df.columns) if df.dtypes[c] == 'object' ]
con_var
cat_var
import matplotlib.pyplot as plt

import seaborn as sb
con_var=con_var[1:]
for i in con_var:

    plt.hist(i,data=df)

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Distribution")

    plt.show()
df['YrSold']=df['YrSold'].astype(str)

df['GarageYrBlt']=df['GarageYrBlt'].astype(str)

df['YearRemodAdd']=df['YearRemodAdd'].astype(str)

df['YearBuilt']=df['YearBuilt'].astype(str)
df1.info()
miss_val=['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']
k1=df1[miss_val]
k1=impute(k1)
df1[miss_val]=k1[miss_val]
df1['GarageYrBlt']=df1['GarageYrBlt'].fillna(df1['GarageYrBlt'].median())
df1.info()
df1 = df1.apply(lambda x:x.fillna(x.value_counts().index[0]))
df1.info()
df.info()
con_var = [ c for c in list(df.columns) if df.dtypes[c] != 'object' ]

cat_var=[c for c in list(df.columns) if df.dtypes[c] == 'object' ]
d1=df[con_var]
d1.drop(['Id'],axis=1,inplace=True)
import seaborn as sb

f, ax = plt.subplots(figsize=(35, 15))

corr = d1.corr()

sb.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap=sb.diverging_palette(220, 10, as_cmap=True),annot=True,ax=ax,);
for i in con_var:

    plt.boxplot(i,data=df)

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Distribution")

    plt.show()
for i in con_var:

    plt.boxplot(i,data=df1)

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Distribution")

    plt.show()
def outlier_analysis(t,c):

    q75,q25=np.percentile(t[c],[75,25])

    iqr=q75-q25

    min=q25-(iqr*1.5)

    max=q75+(iqr*1.5)

    for i in range(t.shape[0]):

        if(t.loc[i,c] > max):

            t.loc[i,c]=max

        elif(t.loc[i,c] < min):

            t.loc[i,c]=min  

    return(t)
out_var=['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',

        'GrLivArea','BsmtFullBath','BsmtHalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',

         '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold']
# Removing outliers form the training set.

for i in out_var:

    df=outlier_analysis(df,i)
for i in con_var:

    plt.boxplot(i,data=df)

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Distribution")

    plt.show()
d1=df[con_var]
d1.drop(['Id'],axis=1,inplace=True)
import seaborn as sb

f, ax = plt.subplots(figsize=(35, 15))

corr = d1.corr()

sb.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap=sb.diverging_palette(220, 10, as_cmap=True),annot=True,ax=ax,);
cat_var
for i in cat_var:

    k=df[i].value_counts()

    k.plot(kind='pie',figsize=(20,10),legend=True)

    plt.legend(loc=0,bbox_to_anchor=(1.5,0.5))

    plt.show()
k= pd.crosstab(index=df['Neighborhood'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'Neighborhood']!='CollgCr' and df.loc[i,'Neighborhood']!='Edwards' and df.loc[i,'Neighborhood']!='Gilbert' and df.loc[i,'Neighborhood']!='NAmes'

      and df.loc[i,'Neighborhood']!='NWAmes' and df.loc[i,'Neighborhood']!='NridgHt' and df.loc[i,'Neighborhood']!='OldTown' and df.loc[i,'Neighborhood']!='Sawyer' and

      df.loc[i,'Neighborhood']!='Somerst'):

        df.loc[i,'Neighborhood']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['Condition1'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'Condition1']!='Feedr' and df.loc[i,'Condition1']!='Norm'):

        df.loc[i,'Condition1']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['Condition2'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'Condition2']!='Norm'):

        df.loc[i,'Condition2']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['HouseStyle'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'HouseStyle']!='1.5Fin' and df.loc[i,'HouseStyle']!='1Story' and df.loc[i,'HouseStyle']!='2Story'):

        df.loc[i,'HouseStyle']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['YearBuilt'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.04')
for i in range(df.shape[0]):

    if(df.loc[i,'YearBuilt']!='2005' and df.loc[i,'YearBuilt']!='2006'):

        df.loc[i,'YearBuilt']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['YearRemodAdd'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'YearRemodAdd']!='2005' and df.loc[i,'YearRemodAdd']!='2006' and df.loc[i,'YearRemodAdd']!='2007' and df.loc[i,'YearRemodAdd']!='1950'):

        df.loc[i,'YearRemodAdd']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['RoofStyle'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'RoofStyle']!='Gable' and df.loc[i,'RoofStyle']!='Hip'):

        df.loc[i,'RoofStyle']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['RoofMatl'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'RoofMatl']!='CompShg'):

        df.loc[i,'RoofMatl']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['Exterior1st'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'Exterior1st']!='HdBoard' and df.loc[i,'Exterior1st']!='MetalSd' and df.loc[i,'Exterior1st']!='Plywood' and df.loc[i,'Exterior1st']!='VinylSd' and 

      df.loc[i,'Exterior1st']!='Wd Sdng'):

        df.loc[i,'Exterior1st']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['Exterior2nd'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'Exterior2nd']!='HdBoard' and df.loc[i,'Exterior2nd']!='MetalSd' and df.loc[i,'Exterior2nd']!='Plywood' and df.loc[i,'Exterior2nd']!='VinylSd' and 

      df.loc[i,'Exterior2nd']!='Wd Sdng'):

        df.loc[i,'Exterior2nd']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['Foundation'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'Foundation']!='BrkTil' and df.loc[i,'Foundation']!='CBlock' and df.loc[i,'Foundation']!='PConc'):

        df.loc[i,'Foundation']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['Heating'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'Heating']!='GasA'):

        df.loc[i,'Heating']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['Functional'],columns="count")

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'Functional']!='Typ'):

        df.loc[i,'Functional']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['GarageYrBlt'],columns="count")

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'GarageYrBlt']!='1980'):

        df.loc[i,'GarageYrBlt']= 'Others'

    else:

        continue  
k= pd.crosstab(index=df['SaleType'],columns="count")

t=k/k.sum()

t.query('count >= 0.05')
for i in range(df.shape[0]):

    if(df.loc[i,'SaleType']!='New' and df.loc[i,'SaleType']!='WD'):

        df.loc[i,'SaleType']= 'Others'

    else:

        continue  
df.describe()
df.drop(['BsmtFinSF2','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal'],axis=1,inplace=True)
df.describe()
con_var = [ c for c in list(df.columns) if df.dtypes[c] != 'object' ]

cat_var=[c for c in list(df.columns) if df.dtypes[c] == 'object' ]
d1=df[con_var]
d1.drop(['Id'],axis=1,inplace=True)
import seaborn as sb

f, ax = plt.subplots(figsize=(35, 15))

corr = d1.corr()

sb.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap=sb.diverging_palette(220, 10, as_cmap=True),annot=True,ax=ax,);
df.drop(['LowQualFinSF','BsmtHalfBath','KitchenAbvGr'],axis=1,inplace=True)
con_var = [ c for c in list(df.columns) if df.dtypes[c] != 'object' ]

cat_var=[c for c in list(df.columns) if df.dtypes[c] == 'object' ]



d1=df[con_var]



d1.drop(['Id'],axis=1,inplace=True)



import seaborn as sb

f, ax = plt.subplots(figsize=(35, 15))

corr = d1.corr()

sb.heatmap(corr, mask=np.zeros_like(corr,dtype=np.bool),cmap=sb.diverging_palette(220, 10, as_cmap=True),annot=True,ax=ax,);
from sklearn import preprocessing
z=df.copy()
le = preprocessing.LabelEncoder()

for i in cat_var:

    df[i] = le.fit_transform(df[i])
cat_var
df.info()
from scipy import stats

for i in cat_var:

    f, p = stats.f_oneway(df[i], df['SalePrice'])

    print("P value for variable "+str(i)+" is "+str(p))
df.columns
con_var1 = [ c for c in list(df1.columns) if df1.dtypes[c] != 'object' ]

cat_var1=[c for c in list(df1.columns) if df1.dtypes[c] == 'object' ]
df1.describe()
df1['YrSold']=df1['YrSold'].astype(str)

df1['GarageYrBlt']=df1['GarageYrBlt'].astype(str)

df1['YearRemodAdd']=df1['YearRemodAdd'].astype(str)

df1['YearBuilt']=df1['YearBuilt'].astype(str)
con_var1 = [ c for c in list(df1.columns) if df1.dtypes[c] != 'object' ]

cat_var1=[c for c in list(df1.columns) if df1.dtypes[c] == 'object' ]
for i in con_var1:

    plt.boxplot(i,data=df1)

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Distribution")

    plt.show()
out_var=['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',

        'GrLivArea','BsmtFullBath','FullBath','BsmtHalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',

         '3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold']
# Removing outliers form the training set.

for i in out_var:

    df1=outlier_analysis(df1,i)
df1.describe()
df1.drop(['BsmtFinSF2','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal'],axis=1,inplace=True)
df1.describe()
con_var1 = [ c for c in list(df1.columns) if df1.dtypes[c] != 'object' ]

cat_var1=[c for c in list(df1.columns) if df1.dtypes[c] == 'object' ]
for i in con_var1:

    plt.boxplot(i,data=df1)

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Distribution")

    plt.show()
df1.drop(['LowQualFinSF','BsmtHalfBath','KitchenAbvGr'],axis=1,inplace=True)
for i in cat_var1:

    k=df1[i].value_counts()

    k.plot(kind='pie',figsize=(20,10),legend=True)

    plt.legend(loc=0,bbox_to_anchor=(1.5,0.5))

    plt.show()
for i in range(df1.shape[0]):

    if(df1.loc[i,'Neighborhood']!='CollgCr' and df1.loc[i,'Neighborhood']!='Edwards' and df1.loc[i,'Neighborhood']!='Gilbert' and df1.loc[i,'Neighborhood']!='NAmes'

      and df1.loc[i,'Neighborhood']!='NWAmes' and df1.loc[i,'Neighborhood']!='NridgHt' and df1.loc[i,'Neighborhood']!='OldTown' and df1.loc[i,'Neighborhood']!='Sawyer' and

      df1.loc[i,'Neighborhood']!='Somerst'):

        df1.loc[i,'Neighborhood']= 'Others'

    else:

        continue
 

k= pd.crosstab(index=df1['Condition1'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'Condition1']!='Feedr' and df1.loc[i,'Condition1']!='Norm'):

        df1.loc[i,'Condition1']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['Condition2'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'Condition2']!='Norm'):

        df1.loc[i,'Condition2']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['HouseStyle'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'HouseStyle']!='1.5Fin' and df1.loc[i,'HouseStyle']!='1Story' and df1.loc[i,'HouseStyle']!='2Story'):

        df1.loc[i,'HouseStyle']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['YearBuilt'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.04')



for i in range(df1.shape[0]):

    if(df1.loc[i,'YearBuilt']!='2005' and df1.loc[i,'YearBuilt']!='2006'):

        df1.loc[i,'YearBuilt']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['YearRemodAdd'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'YearRemodAdd']!='2005' and df1.loc[i,'YearRemodAdd']!='2006' and df1.loc[i,'YearRemodAdd']!='2007' and df1.loc[i,'YearRemodAdd']!='1950'):

        df1.loc[i,'YearRemodAdd']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['RoofStyle'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'RoofStyle']!='Gable' and df1.loc[i,'RoofStyle']!='Hip'):

        df1.loc[i,'RoofStyle']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['RoofMatl'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'RoofMatl']!='CompShg'):

        df1.loc[i,'RoofMatl']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['Exterior1st'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'Exterior1st']!='HdBoard' and df1.loc[i,'Exterior1st']!='MetalSd' and df1.loc[i,'Exterior1st']!='Plywood' and df1.loc[i,'Exterior1st']!='VinylSd' and 

      df1.loc[i,'Exterior1st']!='Wd Sdng'):

        df1.loc[i,'Exterior1st']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['Exterior2nd'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'Exterior2nd']!='HdBoard' and df1.loc[i,'Exterior2nd']!='MetalSd' and df1.loc[i,'Exterior2nd']!='Plywood' and df1.loc[i,'Exterior2nd']!='VinylSd' and 

      df1.loc[i,'Exterior2nd']!='Wd Sdng'):

        df1.loc[i,'Exterior2nd']= 'Others'

    else:

        continue  



#Foundation,Heating,Functional,GarageYrBlt,SaleType



k= pd.crosstab(index=df1['Foundation'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'Foundation']!='BrkTil' and df1.loc[i,'Foundation']!='CBlock' and df1.loc[i,'Foundation']!='PConc'):

        df1.loc[i,'Foundation']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['Heating'],columns="count")

k

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'Heating']!='GasA'):

        df1.loc[i,'Heating']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['Functional'],columns="count")

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'Functional']!='Typ'):

        df1.loc[i,'Functional']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['GarageYrBlt'],columns="count")

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'GarageYrBlt']!='1980'):

        df1.loc[i,'GarageYrBlt']= 'Others'

    else:

        continue  



k= pd.crosstab(index=df1['SaleType'],columns="count")

t=k/k.sum()

t.query('count >= 0.05')



for i in range(df1.shape[0]):

    if(df1.loc[i,'SaleType']!='New' and df1.loc[i,'SaleType']!='WD'):

        df1.loc[i,'SaleType']= 'Others'

    else:

        continue  
df.columns
df1.columns
len(list(df.columns))-len(list(df1.columns))
con_var1 = [ c for c in list(df1.columns) if df1.dtypes[c] != 'object' ]

cat_var1=[c for c in list(df1.columns) if df1.dtypes[c] == 'object' ]
le = preprocessing.LabelEncoder()

for i in cat_var1:

    df1[i] = le.fit_transform(df1[i])
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

X = add_constant(X)

pd.Series([variance_inflation_factor(X.values, i) 

               for i in range(X.shape[1])], 

              index=X.columns)
df['Bathrooms']=df['FullBath']+df['HalfBath']+df['BsmtFullBath']

df1['Bathrooms']=df1['FullBath']+df1['HalfBath']+df1['BsmtFullBath']
X=df[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

       'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',

       'Electrical', '2ndFlrSF', 'Bathrooms', 'BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',

       'WoodDeckSF', 'OpenPorchSF', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']]

y=df['SalePrice']      
test_data=df1[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

       'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',

       'Electrical', '2ndFlrSF', 'Bathrooms', 'BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',

       'WoodDeckSF', 'OpenPorchSF', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size = 0.2, random_state = 1)
# Random Forest 

from sklearn.ensemble import RandomForestRegressor

reg_RF=RandomForestRegressor(random_state=1,n_estimators=1000)

reg_RF.fit(X_train,y_train)
pred_RF_train=reg_RF.predict((X_train))

pred_RF_test=reg_RF.predict((X_test))
from sklearn.metrics import mean_squared_error
# RMSE on training set

np.sqrt(mean_squared_error(y_train,pred_RF_train))
# RMSE on testing set

np.sqrt(mean_squared_error(y_test,pred_RF_test))
from sklearn.metrics import r2_score
# R-squared for Random forest

r2_score(y_test, pred_RF_test)
pred_test_RF=reg_RF.predict(test_data)
# XGBoost Regressor

from sklearn.ensemble import GradientBoostingRegressor
# Training the algorithm

fit_GB = GradientBoostingRegressor(random_state=1,n_estimators=1000,tol=1e-20).fit(X_train, y_train)
# Predicting and calculating RMSE on training set

pred_train = fit_GB.predict(X_train)

rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
# RMSE on training set

rmse_for_train
# Predicting on test set

pred_test = fit_GB.predict(X_test)

rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))
# RMSE on testing set

rmse_for_test
# R squared for XGBoost Regressor

r2_score(y_test,pred_test)
pred_test_XGB=fit_GB.predict(test_data)
y_pred_train=reg_SVR.predict(X_train)

y_pred_test=reg_SVR.predict(X_test)
rmse_for_train =np.sqrt(mean_squared_error(y_train,y_pred_train))
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred_test))
# R squared for SVR

r2_score(y_test,y_pred_test)
df2=pd.DataFrame()

df2['Id']=df1['Id']

df2['SalePrice']=pred_test_RF
df2.to_csv('submission.csv',index=False)