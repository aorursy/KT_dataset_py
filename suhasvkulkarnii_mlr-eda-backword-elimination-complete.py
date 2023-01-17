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
df=pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv")
df.head(5)
df.isnull().any()
df.duplicated().sum()
df.shape
df.dtypes
df['symboling'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sb

plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

plt.title("Count of risky and safe cars in dataset")

df['symboling'].value_counts().plot(kind='bar')

plt.xlabel("Symboling: -2 : Not risky , 2: Highly Risky")

plt.ylabel("Count")

plt.legend()

plt.tight_layout()

plt.grid(True)

plt.subplot(1,2,2)

plt.title("Distribution plot for Symboling")

sb.distplot(df['symboling'])

plt.xlabel("Symboling")

plt.ylabel("Count")

plt.legend()

plt.tight_layout()

plt.show()
df['CarName']=df['CarName'].str.split(" ",expand=True)

df['CarName'].unique()

#replacing misspelled car names

df['CarName'].replace({'maxda':'mazda','Nissan':'nissan','porcshce':'porsche','toyouta':'toyota','vokswagen':'volkswagen','vw':'volkswagen'},inplace=True)
#potting total number of cars of each brand

df['CarName'].value_counts()

#Thr are 32 cars from toyota brand folowed by nissan and mazda

plt.figure(figsize=(15,9))

plt.title("Cars for each brand")

df['CarName'].value_counts().plot(kind='bar')

plt.xlabel("Car brand name")

plt.ylabel("Counts")

plt.legend()

plt.tight_layout()

plt.grid(True)

plt.show()
#Plotting to understand which brand cars are more safe or risky withprice value

plt.figure(figsize=(20,12))



plt.subplot(2,3,1)

plt.title("Counts of cars for symboling =-2")

sb.countplot(data=df[df['symboling']== -2],x='CarName')

plt.xlabel("Car Brands")

plt.xticks(rotation=90)

plt.ylabel("Count value")

plt.tight_layout()

plt.grid(True)





plt.subplot(2,3,2)

plt.title("Counts of cars for symboling =-1")

sb.countplot(data=df[df['symboling']== -1],x='CarName')

plt.xlabel("Car Brands")

plt.xticks(rotation=90)

plt.ylabel("Count value")

plt.tight_layout()

plt.grid(True)



plt.subplot(2,3,3)

plt.title("Counts of cars for symboling =0")

sb.countplot(data=df[df['symboling']== -0],x='CarName')

plt.xlabel("Car Brands")

plt.xticks(rotation=90)

plt.ylabel("Count value")

plt.tight_layout()

plt.grid(True)



plt.subplot(2,3,4)

plt.title("Counts of cars for symboling =1")

sb.countplot(data=df[df['symboling']== 1],x='CarName')

plt.xlabel("Car Brands")

plt.xticks(rotation=90)

plt.ylabel("Count value")

plt.tight_layout()

plt.grid(True)



plt.subplot(2,3,5)

plt.title("Counts of cars for symboling =2")

sb.countplot(data=df[df['symboling']== 2],x='CarName')

plt.xlabel("Car Brands")

plt.xticks(rotation=90)

plt.ylabel("Count value")

plt.tight_layout()

plt.grid(True)





plt.subplot(2,3,6)

plt.title("Counts of cars for symboling =3")

sb.countplot(data=df[df['symboling']== 3],x='CarName')

plt.xlabel("Car Brands")

plt.xticks(rotation=90)

plt.ylabel("Count value")

plt.tight_layout()

plt.grid(True)

plt.show()


#For variable fueltype and aspiration

plt.figure(figsize=(12,10))

plt.subplot(1,2,1)

plt.title("Box plot for fuel type ",fontsize=40)

sb.boxplot(x='fueltype',y='price',data=df)

plt.xlabel('fueltype',fontsize=30)

plt.ylabel('price',fontsize=30)

plt.tight_layout()

plt.subplot(1,2,2)

plt.title("Boxplot for aspration",fontsize=40)

sb.boxplot(x='aspiration',y='price',data=df)

plt.xlabel('aspration',fontsize=30)

plt.ylabel('price',fontsize=30)

plt.tight_layout()

plt.show()
df['doornumber'].value_counts()



#Plotting against doors

plt.figure(figsize=(15,8))

plt.title("Total doors and price value")

sb.boxplot(x='doornumber',y='price',data=df)

plt.xlabel("Total number of doors")

plt.ylabel("Price")

plt.tight_layout()

plt.show()
df['carbody'].value_counts()



plt.figure(figsize=(14,10))

plt.subplot(1,2,1)

plt.title("Counts of each car")

sb.countplot(data=df,x='carbody')

plt.xlabel("Type of body")

plt.ylabel("Counts")

plt.tight_layout()

plt.grid(True)

plt.subplot(1,2,2)

plt.title("Box plot for each body type")

sb.boxplot(x='carbody',y='price',data=df)

plt.xlabel("Car body build")

plt.ylabel("Price")

plt.tight_layout()

plt.show()
df['drivewheel'].value_counts()



#looking how each wheel drive are related to price

plt.figure(figsize=(12,10))

plt.subplot(1,2,1)

plt.title("Count of each brand wheel drive")

sb.countplot(x='CarName',hue='drivewheel',data=df)

plt.xlabel("Car Brands")

plt.ylabel("Counts")

plt.xticks(rotation=90)

plt.tight_layout()

plt.grid(True)

plt.subplot(1,2,2)

plt.title("Drive wheel on prices")

sb.boxplot(x='drivewheel',y='price',data=df)

plt.xlabel("Drivewheel type")

plt.ylabel("Price")

plt.tight_layout()

plt.grid(True)

plt.show()
plt.title("Engine location on prices")

sb.boxplot(x='enginelocation',y='price',data=df)

plt.xlabel("Engine location")

plt.ylabel("Price")

plt.tight_layout()

plt.grid(True)

plt.show()
df['wheelbase'].value_counts()

ma=df['price'].idxmax()

df['price'].max()

df['wheelbase'].max()
plt.figure(figsize=(12,10))

plt.title("Wheel base distribution")

sb.scatterplot(x='wheelbase',y='price',data=df)

plt.xlabel("Wheel base")

plt.xticks(rotation=90)

plt.ylabel("Price")

plt.legend()

plt.tight_layout

plt.grid(True)

plt.show()
plt.figure(figsize=(12,10))

plt.subplot(1,3,1)

plt.title("Engine Type distribution")

sb.boxplot(x='enginetype',y='price',data=df)

plt.xlabel("engnetype")

plt.ylabel("Price")

plt.tight_layout()



plt.subplot(1,3,2)

plt.title("cylindernumber  distribution")

sb.boxplot(x='cylindernumber',y='price',data=df)

plt.xlabel("cylindernumber")

plt.ylabel("Price")

plt.tight_layout()



plt.subplot(1,3,3)

plt.title("fuelsystem distribution")

sb.boxplot(x='fuelsystem',y='price',data=df)

plt.xlabel("fuelsystem")

plt.ylabel("Price")

plt.tight_layout()



plt.show()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



df[['fueltype','aspiration','doornumber','enginelocation']]=df[['fueltype','aspiration','doornumber','enginelocation']].apply(le.fit_transform)
dummy=pd.get_dummies(data=df,columns=['carbody','drivewheel','enginetype','cylindernumber','fuelsystem'])

dummy1=dummy.iloc[:,21:]

dummy1.columns

dummy1=dummy1.drop(['carbody_wagon','drivewheel_4wd','enginetype_ohcv','cylindernumber_three','fuelsystem_mfi'],axis=1)



df1=pd.concat([df,dummy1],axis=1)

df1.columns

df2=df1.drop(['car_ID','CarName','carbody','drivewheel','enginetype','cylindernumber','fuelsystem'],axis=1)

cor=df2.corr
df2.columns

df2=df2.drop(['fuelsystem_idi','compressionratio','carlength','wheelbase','curbweight','enginesize','citympg','drivewheel_rwd','enginetype_rotor'],axis=1)

core=df2.corr
# splitting test and train data set

y=df2['price']

x=df2.drop(['price'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import  LinearRegression

le=LinearRegression()

#fitting the model

le.fit(x_train,y_train)
y_pred=le.predict(x_test)
residuals=y_pred-y_test
le.score(x_test,y_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

r2_score(y_test,y_pred)

print(("MSE : {}").format(mean_squared_error(y_test,y_pred)))

print(("RMSE : {}").format(np.sqrt(mse)))
#Normality of the residuals

sb.distplot(residuals)
sb.residplot(y_test,y_pred)
#Checking linearity of the model

sb.scatterplot(x=y_test,y=y_pred)
sb.scatterplot(x=residuals,y=y_pred)
import scipy

fig,ax=plt.subplots(figsize=(8,6))

scipy.stats.probplot(residuals,plot=ax,fit=True)
#for autocorelation

from statsmodels.tsa.api import graphics as gp

gp.plot_acf(residuals,lags=40,alpha=0.05)
#Multicolinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()

vif["VIF"]=[variance_inflation_factor(x.values,i)for i in range(x.shape[1])]

vif["features"]=x.columns
#bacjkward ellimination method

import statsmodels.api as smf

x1=np.append(arr=np.ones((205,1)).astype(int),values=x,axis=1)

###0:constant : 1 age, 2 :

x1_opt=x1[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]]

model=smf.OLS(endog=y,exog=x1_opt).fit()

model.summary()
x1_opt=x1[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]]

model=smf.OLS(endog=y,exog=x1_opt).fit()

model.summary()
x1_opt=x1[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]]

model=smf.OLS(endog=y,exog=x1_opt).fit()

model.summary()
x1_opt=x1[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34]]

model=smf.OLS(endog=y,exog=x1_opt).fit()

model.summary()


x1_opt=x1[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,34]]

model=smf.OLS(endog=y,exog=x1_opt).fit()

model.summary()
x1_opt=x1[:,[0,2,6,8,9,10,11,13,18,19,20,21,22,23,24,26,27,28]]

model=smf.OLS(endog=y,exog=x1_opt).fit()

model.summary()
df3=df2.drop(['symboling','aspiration','doornumber','enginelocation','carheight',

              'highwaympg','carbody_hardtop','carbody_hatchback','carbody_sedan',

              'drivewheel_fwd','cylindernumber_four','cylindernumber_two','fuelsystem_1bbl',

              'fuelsystem_2bbl','fuelsystem_4bbl', 'fuelsystem_mpfi', 'fuelsystem_spdi',

               'fuelsystem_spfi'],axis=1)
x_final=df3.drop(['price'],axis=1)

y_final=df3['price']
le_final=LinearRegression()

le_final.fit(x_final,y_final)
le_final.score(x_final,y_final)
x_new=np.array([1,70.0,4.40,3.30,150,7300,0,1,0,0,0,0,1,0,0,0])

x_new=x_new.reshape((1,16))

y_new=le_final.predict(x_new)
print(y_new)