import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import math
train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
test_df.head()
train_df['train']=1
test_df['train']=0
df=pd.concat([train_df,test_df],axis=0,sort=False)
df.head()
drop_null=[]

for i in df.columns:

    count=0

    p=0

    for  j in np.array(df[i].isnull()):

        p+=1

        if j==True:

            count+=1

    pr=(((count)/p)*100)

    print(i,'\t-\t',pr)

    if pr>=50:

        drop_null.append(i)
drop_null
df=df.drop(drop_null,axis=1)
cat_df=df.select_dtypes(include=['object'])

num_df=df.select_dtypes(exclude=['object'])
cat_df.isnull().sum()
cat_low_null=[]

cat_high_null=[]

for i in cat_df.columns:

    if cat_df[i].isnull().sum()!=0 and cat_df[i].isnull().sum()<=50:

        cat_low_null.append(i)

    elif cat_df[i].isnull().sum()>50:

        cat_high_null.append(i)

cat_low_null
cat_high_null
cat_df[cat_high_null]=cat_df[cat_high_null].fillna('none')
cat_df[cat_low_null]=cat_df[cat_low_null].fillna(cat_df.mode().iloc[0])
num_df.isnull().sum()

    
num_df['LotFrontage'].median()

(num_df['YrSold']-num_df['YearBuilt']).median()
num_df['LotFrontage']=num_df['LotFrontage'].fillna(68)
num_df['GarageYrBlt']=num_df['GarageYrBlt'].fillna(num_df['YrSold']-35)
num_df=num_df.fillna(0)
for i in cat_df.columns:

    print(cat_df[i].value_counts())

    cat_df[i].value_counts().plot(kind='bar',figsize=[10,3])

    plt.show()
cat_df=cat_df.drop(['Street','Utilities','Condition2','RoofMatl','Heating'],axis=1)
num_df['Age']=num_df['YrSold']-num_df['YearBuilt']

num_df['Age'].head()
num_df['Age'].describe()
num_df[num_df['Age']<0]
num_df.loc[num_df['YrSold']<num_df['YearBuilt'],'YrSold']=2009

num_df['Age']=num_df['YrSold']-num_df['YearBuilt']

num_df['Age'].describe()
num_df['TotalBsmtbath']=num_df['BsmtFullBath']+num_df['BsmtHalfBath']*0.5
num_df[['BsmtFullBath','BsmtHalfBath','TotalBsmtbath']].head()
num_df['TotalBath']=num_df['FullBath']+num_df['HalfBath']*0.5
num_df[['FullBath','HalfBath','TotalBath']].head()
num_df['TotalSF']=num_df['TotalBsmtSF']+num_df['1stFlrSF']+num_df['2ndFlrSF']
num_df[['TotalBsmtSF','1stFlrSF','2ndFlrSF','TotalSF']].head()
num_df.head()
for i in cat_df.columns:

    cat_df[i]=cat_df[i].astype('category')
for i in cat_df.columns:

    cat_df[i]=cat_df[i].cat.codes
cat_df.head()
df_final=pd.concat([num_df,cat_df],axis=1,sort=False)
df_final.head()
df_final=df_final.drop('Id',axis=1)
df_final.head()
df_train=df_final[df_final['train']==1]

df_train=df_train.drop(['train'],axis=1)
df_test=df_final[df_final['train']==0]

df_test=df_test.drop(['train','SalePrice'],axis=1)
sns.distplot(df_train['SalePrice'])

print('skewness of SalePrice is %f'%df_train['SalePrice'].skew())
var=['LotArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','TotalSF','GarageArea','GrLivArea']

     

for u in var:

    sns.scatterplot(x=u,y=df_train['SalePrice'],data=df_train)

    plt.title(u)

    plt.show()
yr=['YearBuilt','YearRemodAdd','YrSold']

for i in yr:

    df_train.groupby(i)['SalePrice'].median().plot()

    plt.title(i)

    plt.show()
df_train.groupby('Age')['SalePrice'].median().plot()
bath=['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','TotalBsmtbath', 'TotalBath']

for i in bath:

    sns.lineplot(i,'SalePrice',data=df_train)

    plt.show()
X=df_train.drop(['SalePrice'],axis=1)

Y=df_train['SalePrice']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=45)
y_train=y_train.values.reshape(-1,1)

y_test=y_test.values.reshape(-1,1)
from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

sc_y=StandardScaler()

x_train=sc_x.fit_transform(x_train)

x_test=sc_x.fit_transform(x_test)

y_train=sc_y.fit_transform(y_train)

y_test=sc_y.fit_transform(y_test)
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=100,random_state=0)

rfr.fit(x_train,y_train)
pred=rfr.predict(x_test)

pred=pred.reshape(-1,1)
from sklearn.metrics import r2_score

r2=r2_score(y_test,pred)
from sklearn.metrics import mean_squared_error

mse=mean_squared_error(y_test,pred)

rmse=math.sqrt(mse)
print('r2 score is - ',r2)

print('Root mean squarred error - ',rmse)
plt.figure(figsize=(18,6))

plt.plot(y_test,label='Actual')

plt.plot(pred,label='Predicted')

plt.show()



rfr.fit(X,Y)
f_pred=rfr.predict(df_test)
final_sub=pd.DataFrame({

    'Id': test_df['Id'],

    'SalePrice':f_pred

})
final_sub.to_csv('sample_submission.csv',index=False)