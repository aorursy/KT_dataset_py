# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

filename_list=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filename_list.append(os.path.join(dirname, filename))
#Joining all csv files into one dataset with one line of code :)

data=pd.concat(map(pd.read_csv,filename_list))
data.info()
data.drop(["engine size","mileage2","fuel type2","engine size2","reference"],axis=1,inplace=True)

data.head(2)
import math as m

tax=list(data["tax"])

taxe=list(data["tax(£)"])

tax_final=[]

for i in range(0, len(tax)):

    if m.isnan(tax[i]):

        if m.isnan(taxe[i]):

            tax_final.append(np.nan)

        else:

            tax_final.append(taxe[i])

    else:

        tax_final.append(tax[i])

data.drop(["tax","tax(£)"],axis=1,inplace=True)

data["tax"]=tax_final
data.info()
data.dropna(inplace=True)

data.info()
data.head()
price_list=list(data.price)

new_price_list=[]

for i in price_list:

    new_price_list.append(float(i))

data["price"]=new_price_list
data.info()
mileage_list=list(data.mileage)

new_mileage_list=[]

for i in mileage_list:

    new_mileage_list.append(float(i))

data["mileage"]=new_mileage_list
data.info()
# importing visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns
data.describe()
sns.pairplot(data)
sns.regplot(data=data, x="tax", y="price")
sns.regplot(data=data, x="engineSize", y="price")
data[data["year"]>2020]
data.model.nunique()
data.transmission.unique()
data.fuelType.unique()
data=data[data["year"]<=2020]
data.info()
i_mileage=[1/x for x in list(data.mileage)]

i_mpg=[1/x for x in list(data.mpg)]
data.drop("model", axis=1, inplace=True)
plt.scatter(i_mileage, data["price"])
plt.scatter(i_mpg, data["price"])
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score
data.info()
data=data.reset_index().drop("index",axis=1)

data
#fuel type dummies

dummies_ft=pd.get_dummies(data["fuelType"],drop_first=True)

dummies_ft["Other FT"]=dummies_ft["Other"]

dummies_ft.drop("Other",axis=1,inplace=True)

data=data.join(dummies_ft)
data.drop("fuelType", axis=1, inplace=True)
data000=data.copy()

data100=data.copy()

data010=data.copy()

data001=data.copy()

data101=data.copy()

data110=data.copy()

data011=data.copy()

data111=data.copy()
data100["mileage"]=i_mileage

data101["mileage"]=i_mileage

data110["mileage"]=i_mileage

data111["mileage"]=i_mileage
data010["mpg"]=i_mpg

data011["mpg"]=i_mpg

data110["mpg"]=i_mpg

data111["mpg"]=i_mpg
#dummies and cardinals for transmission

dummies_transmission=pd.get_dummies(data["transmission"],drop_first=True)

dummies_transmission["Other T"]=dummies_transmission["Other"]

dummies_transmission.drop("Other", axis=1, inplace=True)

cardinal_transmission={'Automatic':2, 'Manual':0, 'Semi-Auto':1, 'Other':3} 

# Other gets value 3 because it is very likely that any other type of transmission is superior to the ones listed



data000.drop("transmission",axis=1,inplace=True)

data100.drop("transmission",axis=1,inplace=True)

data010.drop("transmission",axis=1,inplace=True)

data110.drop("transmission",axis=1,inplace=True)

data000=data000.join(dummies_transmission)

data100=data100.join(dummies_transmission)

data010=data010.join(dummies_transmission)

data110=data110.join(dummies_transmission)



data001["transmission"]=data001["transmission"].map(cardinal_transmission)

data101["transmission"]=data101["transmission"].map(cardinal_transmission)

data011["transmission"]=data011["transmission"].map(cardinal_transmission)

data111["transmission"]=data111["transmission"].map(cardinal_transmission)

y000=data000["price"]

X000=data000.drop("price",axis=1)



y100=data100["price"]

X100=data100.drop("price",axis=1)



y010=data010["price"]

X010=data010.drop("price",axis=1)



y001=data001["price"]

X001=data001.drop("price",axis=1)



y101=data101["price"]

X101=data101.drop("price",axis=1)



y110=data000["price"]

X110=data000.drop("price",axis=1)



y011=data011["price"]

X011=data011.drop("price",axis=1)



y111=data111["price"]

X111=data111.drop("price",axis=1)



X_train000, X_test000, y_train000, y_test000 = train_test_split(X000, y000, test_size=0.3)



X_train001, X_test001, y_train001, y_test001 = train_test_split(X001, y001, test_size=0.3)



X_train010, X_test010, y_train010, y_test010 = train_test_split(X010, y010, test_size=0.3)



X_train100, X_test100, y_train100, y_test100 = train_test_split(X100, y100, test_size=0.3)



X_train110, X_test110, y_train110, y_test110 = train_test_split(X110, y110, test_size=0.3)



X_train101, X_test101, y_train101, y_test101 = train_test_split(X101, y101, test_size=0.3)



X_train011, X_test011, y_train011, y_test011 = train_test_split(X011, y011, test_size=0.3)



X_train111, X_test111, y_train111, y_test111 = train_test_split(X111, y111, test_size=0.3)
lr=LinearRegression()

lr.fit(X_train000, y_train000)

lr_result=lr.predict(X_test000)

print(explained_variance_score(y_test000,lr_result))
lr=LinearRegression()

lr.fit(X_train001, y_train001)

lr_result=lr.predict(X_test001)

print(explained_variance_score(y_test001,lr_result))
lr=LinearRegression()

lr.fit(X_train010, y_train010)

lr_result=lr.predict(X_test010)

print(explained_variance_score(y_test010,lr_result))
lr=LinearRegression()

lr.fit(X_train100, y_train100)

lr_result=lr.predict(X_test100)

print(explained_variance_score(y_test100,lr_result))
lr=LinearRegression()

lr.fit(X_train011, y_train011)

lr_result=lr.predict(X_test011)

print(explained_variance_score(y_test011,lr_result))
lr=LinearRegression()

lr.fit(X_train110, y_train110)

lr_result=lr.predict(X_test110)

print(explained_variance_score(y_test110,lr_result))
lr=LinearRegression()

lr.fit(X_train101, y_train101)

lr_result=lr.predict(X_test101)

print(explained_variance_score(y_test101,lr_result))
lr=LinearRegression()

lr.fit(X_train111, y_train111)

lr_result=lr.predict(X_test111)

print(explained_variance_score(y_test111,lr_result))
values=[]

for i in [0.1,1,10,100,1000,10000]:

    lasso=Lasso(alpha=i)

    lasso.fit(X_train010,y_train010)

    values.append(list(lasso.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X010.columns)

plt.xlabel("\alpha value")

plt.ylabel("Coefficient value")
lasso=Lasso(alpha=1)

lasso.fit(X_train010,y_train010)

lasso_result=lasso.predict(X_test010)

print(explained_variance_score(y_test010,lasso_result))

print(lasso.coef_)
lasso=Lasso(alpha=10)

lasso.fit(X_train010,y_train010)

lasso_result=lasso.predict(X_test010)

print(explained_variance_score(y_test010,lasso_result))

print(lasso.coef_)
values=[]

for i in [0.1,1,10,100,1000,10000]:

    lasso=Lasso(alpha=i)

    lasso.fit(X_train000,y_train000)

    values.append(list(lasso.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X000.columns)
lasso=Lasso(alpha=10)

lasso.fit(X_train000,y_train000)

lasso_result=lasso.predict(X_test000)

print(explained_variance_score(y_test000,lasso_result))

print(lasso.coef_)
lasso=Lasso(alpha=100)

lasso.fit(X_train000,y_train000)

lasso_result=lasso.predict(X_test000)

print(explained_variance_score(y_test000,lasso_result))

print(lasso.coef_)
values=[]

for i in [0.1,1,10,100,1000,10000]:

    lasso=Lasso(alpha=i)

    lasso.fit(X_train100,y_train100)

    values.append(list(lasso.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X100.columns)
lasso=Lasso(alpha=10)

lasso.fit(X_train100,y_train100)

lasso_result=lasso.predict(X_test100)

print(explained_variance_score(y_test100,lasso_result))

print(lasso.coef_)
lasso=Lasso(alpha=100)

lasso.fit(X_train100,y_train100)

lasso_result=lasso.predict(X_test100)

print(explained_variance_score(y_test100,lasso_result))

print(lasso.coef_)
values=[]

for i in [0.1,1,10,100,1000,10000]:

    lasso=Lasso(alpha=i)

    lasso.fit(X_train001,y_train001)

    values.append(list(lasso.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X001.columns)
lasso=Lasso(alpha=10)

lasso.fit(X_train001,y_train001)

lasso_result=lasso.predict(X_test001)

print(explained_variance_score(y_test001,lasso_result))

print(lasso.coef_)
lasso=Lasso(alpha=100)

lasso.fit(X_train001,y_train001)

lasso_result=lasso.predict(X_test001)

print(explained_variance_score(y_test001,lasso_result))

print(lasso.coef_)
values=[]

for i in [0.1,1,10,100,1000,10000]:

    lasso=Lasso(alpha=i)

    lasso.fit(X_train110,y_train110)

    values.append(list(lasso.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X110.columns)
lasso=Lasso(alpha=10)

lasso.fit(X_train110,y_train110)

lasso_result=lasso.predict(X_test110)

print(explained_variance_score(y_test110,lasso_result))

print(lasso.coef_)
lasso=Lasso(alpha=100)

lasso.fit(X_train110,y_train110)

lasso_result=lasso.predict(X_test110)

print(explained_variance_score(y_test110,lasso_result))

print(lasso.coef_)
values=[]

for i in [0.1,1,10,100,1000,10000]:

    lasso=Lasso(alpha=i)

    lasso.fit(X_train101,y_train101)

    values.append(list(lasso.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X101.columns)
lasso=Lasso(alpha=10)

lasso.fit(X_train101,y_train101)

lasso_result=lasso.predict(X_test101)

print(explained_variance_score(y_test101,lasso_result))

print(lasso.coef_)
lasso=Lasso(alpha=100)

lasso.fit(X_train101,y_train101)

lasso_result=lasso.predict(X_test101)

print(explained_variance_score(y_test101,lasso_result))

print(lasso.coef_)
values=[]

for i in [0.1,1,10,100,1000,10000]:

    lasso=Lasso(alpha=i)

    lasso.fit(X_train011,y_train011)

    values.append(list(lasso.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X011.columns)
lasso=Lasso(alpha=10)

lasso.fit(X_train011,y_train011)

lasso_result=lasso.predict(X_test011)

print(explained_variance_score(y_test011,lasso_result))

print(lasso.coef_)
lasso=Lasso(alpha=100)

lasso.fit(X_train011,y_train011)

lasso_result=lasso.predict(X_test011)

print(explained_variance_score(y_test011,lasso_result))

print(lasso.coef_)
values=[]

for i in [0.1,1,10,100,1000,10000]:

    lasso=Lasso(alpha=i)

    lasso.fit(X_train111,y_train111)

    values.append(list(lasso.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X111.columns)
lasso=Lasso(alpha=10)

lasso.fit(X_train111,y_train111)

lasso_result=lasso.predict(X_test111)

print(explained_variance_score(y_test111,lasso_result))

print(lasso.coef_)
lasso=Lasso(alpha=1000)

lasso.fit(X_train111,y_train111)

lasso_result=lasso.predict(X_test111)

print(explained_variance_score(y_test111,lasso_result))

print(lasso.coef_)
values=[]

for i in [0.1,1,10,100,1000,10000]:

    ridge=Ridge(alpha=i)

    ridge.fit(X_train010,y_train010)

    values.append(list(ridge.coef_))

val_arr=np.array(values)    

val_arr=np.transpose(val_arr)

values=val_arr.tolist()



plt.figure(figsize=(20,12))

for i in range(0,len(values)):

    plt.plot([0.1,1,10,100,1000,10000],values[i])

plt.xscale("log")

plt.legend(X010.columns)
ridge=Ridge(alpha=1)

ridge.fit(X_train010,y_train010)

ridge_result=ridge.predict(X_test010)

print(explained_variance_score(y_test010,ridge_result))

print(ridge.coef_)
ridge=Ridge(alpha=10)

ridge.fit(X_train010,y_train010)

ridge_result=ridge.predict(X_test010)

print(explained_variance_score(y_test010,ridge_result))

print(ridge.coef_)
ridge=Ridge(alpha=100)

ridge.fit(X_train010,y_train010)

ridge_result=ridge.predict(X_test010)

print(explained_variance_score(y_test010,ridge_result))

print(ridge.coef_)
filename_list

# entries 3,4 and 5 will be removed
filename_list1=filename_list[0:3]+filename_list[6:]

filename_list1
data2=pd.read_csv(filename_list1[0])

brand_name=filename_list1[0].split("/")[-1].split(".")[0]

data2["brand"]=[brand_name]*len(data2)

for i in filename_list1[1:]:

    aux_df=pd.read_csv(i)

    brand_name=i.split("/")[-1].split(".")[0]

    aux_df["brand"]=[brand_name]*len(aux_df)

    data2=pd.concat([data2,aux_df])

# coincidently, a lot of the features that weren't present in most datasets belong to the data we just removed 

data2=data2[data2["year"]<=2020]

data2.head()
import math as m

tax=list(data2["tax"])

taxe=list(data2["tax(£)"])

tax_final=[]

for i in range(0, len(tax)):

    if m.isnan(tax[i]):

        if m.isnan(taxe[i]):

            tax_final.append(np.nan)

        else:

            tax_final.append(taxe[i])

    else:

        tax_final.append(tax[i])

data2.drop(["tax","tax(£)"],axis=1,inplace=True)

data2["tax"]=tax_final



data2.dropna(inplace=True)

data2.drop("model", axis=1, inplace=True)
data2.info()
g = sns.FacetGrid(data2, col="brand",col_wrap=3)

g.map(sns.distplot,"price")
#dummies and cardinals for transmission

data2=data2.reset_index().drop("index",axis=1)

dummies_transmission=pd.get_dummies(data2["transmission"],drop_first=True)

dummies_transmission["Other T"]=dummies_transmission["Other"]

dummies_transmission.drop("Other",axis=1, inplace=True)

dummies_brand=pd.get_dummies(data2["brand"], drop_first=True)

dummies_ft=pd.get_dummies(data2["fuelType"],drop_first=True)

dummies_ft["Other FT"]=dummies_ft["Other"]

dummies_ft.drop("Other",axis=1, inplace=True)

data2["mpg"]=i_mpg

data2.drop(["transmission","brand","fuelType"], axis=1, inplace=True)

data2=data2.join(dummies_transmission)

data2=data2.join(dummies_brand)

data2=data2.join(dummies_ft)

data2.head()
data2.info()
y=data2["price"]

X=data2.drop("price",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr=LinearRegression()

lr.fit(X_train, y_train)

lr_res=lr.predict(X_test)

print(explained_variance_score(y_test,lr_res))