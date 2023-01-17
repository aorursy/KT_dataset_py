#import all libraries

import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style='whitegrid')

import matplotlib.pyplot as plt

from matplotlib import style

#sta matplotlib to inline and displays graphs below the corresponding cell.

%matplotlib inline

import os

from sklearn.datasets import *

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/Automobile.csv")

df.head()
df.shape
#data cleaning

df.isnull().sum()
df.describe().T
df.info()
df.columns
cols=['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',

       'number_of_doors', 'body_style', 'drive_wheels', 'engine_location',

       'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type',

       'number_of_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke',

       'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg',

       'highway_mpg', 'price']

for i in cols:

    def check(data):

        t=data[i].loc[data[i]=='?']

        return t

    



    g=check(df)

    print(g)
#thus the data shows there is no null values and also no special characters in the place of values
#what are the columns which are object

obj=list(df.select_dtypes(include=['object']))

obj
#what are the columns which are float and int

flint=list(df.select_dtypes(include=['int64','float64']))

flint
#checking for outliers

plt.figure(figsize=(15,8))

sns.boxplot(data=df)
df[['engine_size','peak_rpm','curb_weight','horsepower','price']].hist(figsize=(10,8),bins=6,color='Y')

plt.tight_layout()

plt.show()
print('the minimum price of car: %0.2d, the maximum price of the car: %0.2d'%(df['price'].min(),df['price'].max()))
df['make'][df['price']>=30000].count()
#there are 14 cars like that which are highly priced due to its features
d=df['make'][df['price']>=30000].value_counts().count()

print(d)

df['make'][df['price']>=30000].value_counts()
df.head()
df.aspiration.value_counts()
fig,a=plt.subplots(1,2,figsize=(10,5))

df.groupby('aspiration')['price'].agg(['mean','median','max']).plot.bar(rot=0,ax=a[0])

df.aspiration.value_counts().plot.bar(rot=0,ax=a[1])
plt.figure(1)

plt.subplot(221)

df['engine_type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red')

plt.title("Number of Engine Type frequency diagram")

plt.ylabel('Number of Engine Type')

plt.xlabel('engine-type');



plt.subplot(222)

df['body_style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='orange')

plt.title("Number of Body Style frequency diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('body-style');



plt.subplot(223)

df['number_of_doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='green')

plt.title("Number of Door frequency diagram")

plt.ylabel('Number of Doors')

plt.xlabel('num-of-doors');



plt.subplot(224)

df['fuel_type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='purple')

plt.title("Number of Fuel Type frequency diagram")

plt.ylabel('Number of vehicles')

plt.xlabel('fuel-type');





plt.tight_layout()

plt.subplots_adjust(wspace=0.3,hspace=0.5)

plt.show()
fig,a=plt.subplots(1,2,figsize=(10,2))

df.body_style.value_counts().plot.pie(explode=(0.03,0,0,0,0),autopct='%0.2f%%',figsize=(10,5),ax=a[0])

a[0].set_title('No. of cars sold')





df.groupby('body_style')['price'].agg(['mean','median','max']).sort_values(by='median',ascending=False).plot.bar(ax=a[1])

a[1].set_title('Price of each body_style')

plt.tight_layout()

plt.show()
sns.catplot(data=df, y="normalized_losses", x="symboling"  ,kind="point")
df.head()
sns.lmplot('engine_size','highway_mpg',hue='make',data=df,fit_reg=False)

plt.title('Engine_size Vs highway_mpg')

plt.show()

sns.lmplot('engine_size','city_mpg',hue='make',data=df,fit_reg=False)

plt.title('Engine_size Vs city_mpg')
df[['make','fuel_type','aspiration','number_of_doors','body_style','drive_wheels','engine_location']][df['engine_size']>=300]
# df.drive_wheels.value_counts().plot.bar()

fig,ax=plt.subplots(2,1,figsize=(15,5))

sns.countplot(x='drive_wheels',data=df,ax=ax[0])

df.groupby(['drive_wheels','make'])['price'].mean().plot.bar(ax=ax[1])

plt.grid()

plt.show()

fig,ax=plt.subplots(2,1,figsize=(15,5))

sns.countplot(x='drive_wheels',data=df,ax=ax[0])

df.groupby(['drive_wheels','body_style'])['price'].mean().plot.bar(ax=ax[1])

ax[1].set_ylabel('Price')

plt.grid()

plt.show()
dff=pd.pivot_table(df,index=['body_style'],columns=['drive_wheels'],values=['engine_size'],

                   aggfunc=['mean'],fill_value=0)



dff.plot.bar(figsize=(15,5),rot=45)

plt.show()

dff
sns.kdeplot(df['price'],shade=True)
df.price.max()
df.groupby('drive_wheels')[['city_mpg','highway_mpg']].agg('sum').plot.bar()

df.groupby('drive_wheels')[['city_mpg','highway_mpg']].agg('sum')
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),annot=True)
sns.pairplot(df,aspect=1.5)
sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg','horsepower','engine_size','curb_weight'],y_vars=['price'],kind='reg',size=2.8)
sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg'],y_vars=['horsepower'],hue='price',size=4)

plt.savefig('pairplot.png')
sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg'],y_vars=['horsepower'],kind='reg',size=4)
df=pd.read_csv("../input/Automobile.csv")

df.head()
df.shape
cols=list(df.select_dtypes(include=['object']))

cols

for i in cols:

    print(i)

#     print()

    print(df[i].value_counts())

    print()
#  We dont have any further use of ["make","body_style"] to build our model, so drop ['make','body_style'] columns



df.drop(['make','body_style'],1,inplace=True)
df.columns
df.describe(include='all').T
#For those columns which are having 2 samples lets do label encoding and those columns which are having more than 

#2 samples lets do get_dummies for that

labcols=['engine_location','number_of_doors','fuel_type','aspiration']

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in labcols:

#     print(i)

    df[i]=le.fit_transform(df[i])

# df.head()
#get dummies

dumcols=['drive_wheels','engine_type','fuel_system','number_of_cylinders']

df=pd.get_dummies(df,columns=dumcols,drop_first=True)
df.head()
y=df['price']

X=df.drop('price',1)
X.shape[1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression

lr=LinearRegression().fit(X_train,y_train)
y_pred=lr.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error

print('r2_score:',r2_score(y_test,y_pred))

# print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))
y_train_pred=lr.predict(X_train)

print('train_score:',r2_score(y_train,y_train_pred))

print('test_score:',r2_score(y_test,y_pred))
import statsmodels.api as sm

Xc=sm.add_constant(X_train)

model=sm.OLS(y_train,Xc).fit()

result=model.summary()

result
r=df.columns

r.shape
# Backward Elimination

cols=list(X_train.columns)

while(len(cols)>0):

    p=[]

    X=X_train[cols]

    Xc=sm.add_constant(X)

    model=sm.OLS(y_train,Xc).fit()

    p=pd.Series(model.pvalues.values[1:],index=cols)

    pmax=max(p)

    feature_with_p_max=p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE=cols

print(selected_features_BE) 

print()

print('count_selected_colums: ',len(cols))

    

    

    
### if the columns are more then there will be chance of increment in VARIACNE ERROR,so thats why we sorted columns based on PValues
#from 40 colums it reduced to 17 columns,continue with the model building

X_final=df[selected_features_BE]

y=df['price']
print(X_final.shape,y.shape)
from sklearn.linear_model import LinearRegression

lr=LinearRegression().fit(X_final,y)
lr.coef_
pd.DataFrame(lr.coef_,index=X_final.columns,columns=['coefficents'])
print('beta0:',lr.intercept_)
from sklearn.model_selection import KFold,cross_val_score

kfold=KFold(n_splits=3, shuffle=True, random_state=1)

cv_score=cross_val_score(lr,X_final,y,cv=kfold)

score=np.mean(cv_score)

# cv=cross_val_score(lr,X_final,y,scoring='neg_mean_squared_error',cv=kfold)

# rm=(-1)*(cv)

print('the r2_score of the model:',score)

# rmse=(np.sqrt(rm))

# print("rmse:%0.02f,variance:%0.02f" % (np.mean(rmse),np.var(rmse,ddof=1)))
print('rsquared value:',model.rsquared)
import pandas as pd

Automobile = pd.read_csv("../input/Automobile.csv")