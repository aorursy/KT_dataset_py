import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")

df
df.head()
df.tail()
df.info()
df.describe()
df.columns.tolist()
df.shape
df.isna().sum()
#for rating of resturant url,adress,phone,listed_in(type)not needed so to be dropped

df=df.drop(['url', 'address','phone','listed_in(city)'], axis=1)

df
df.name.value_counts().head()
df.name.value_counts().tail()
plt.figure(figsize = (13,8))

r = df.name.value_counts()[:50].plot(kind = 'bar',color='green',fontsize=15)

r.legend(['Restaurants'])

plt.xlabel("Name of Restaurant")

plt.ylabel("Count of Restaurants")

plt.title("Name vs count of Restaurant",fontsize =15, weight = 'bold',color='red')
#no of  accepting online ordesrs

df.online_order.value_counts()
plt.figure(figsize = (13,8))

s= df.online_order.value_counts().plot(kind = 'bar',color='yellow',fontsize=15)

r.legend(['orders'])

plt.xlabel("online orders")

plt.ylabel("Count ")

plt.title("No of online orders",fontsize =15, weight = 'bold',color='red')
df['book_table'].value_counts()
plt.figure(figsize = (13,8))

s= df.book_table.value_counts().plot(kind = 'bar',color='red',fontsize=15)

r.legend(['book table'])

plt.xlabel("book_table")

plt.ylabel("no of resturants ")

plt.title("book table facility",fontsize =15, weight = 'bold',color='blue')
#location

df['location'].value_counts()[:15]
plt.figure(figsize = (13,8))

s= df.location.value_counts()[:20].plot(kind = 'bar',color='pink',fontsize=25)

r.legend(['location'])

plt.xlabel("location")

plt.ylabel("count ")

plt.title("location vs count",fontsize =15, weight = 'bold',color='blue')
df['rest_type'].value_counts()
plt.figure(figsize = (13,8))

s= df.rest_type.value_counts()[:20].plot(kind = 'bar',color='lightskyblue',fontsize=25)

r.legend(['rest type'])

plt.xlabel("rest_type")

plt.ylabel("count ")

plt.title("rest type vs count",fontsize =15, weight = 'bold',color='blue')
#rename approx cost column

df.rename(columns={'approx_cost(for two people)': 'approx_cost'}, inplace=True)
df['approx_cost'].value_counts()
plt.figure(figsize = (13,8))

s= df.approx_cost.value_counts()[:20].plot(kind = 'bar',color='orange',fontsize=25)

r.legend(['approx_cost'])

plt.xlabel("approx_cost")

plt.ylabel("count ")

plt.title("approx cost vs count",fontsize =15, weight = 'bold',color='blue')
df=df[df.dish_liked.isna()==False]

df.isna().sum()
df['dish_liked'].value_counts()

plt.figure(figsize = (13,8))

s= df.dish_liked.value_counts()[:20].plot(kind = 'bar',color='lightgreen',fontsize=25)

r.legend(['dish liked'])

plt.xlabel("dish_liked")

plt.ylabel("count ")

plt.title("approx cost vs count",fontsize =15, weight = 'bold',color='blue')
df['rates'].value_counts()
df=df[df.rates.isna()==False]
df['rates'].value_counts()
plt.figure(figsize = (13,8))

s= df.rates.value_counts()[:20].plot(kind = 'bar',color='lightgreen',fontsize=25)

r.legend(['rates'])

plt.xlabel("rates")

plt.ylabel("count ")

plt.title("rates vs count",fontsize =15, weight = 'bold',color='blue')
df['cuisines'].value_counts()
plt.figure(figsize = (12,6))

sns.countplot(x=df['rates'], hue = df['online_order'])

plt.ylabel("Restaurants that Accept/Not Accepting online orders")

plt.title("rate vs online order",weight = 'bold')
df['location'].nunique()
#creating dummies for online order,table booked as it contains categorical yes and no

df['online_order']= pd.get_dummies(df.online_order, drop_first=True)

df['book_table']= pd.get_dummies(df.book_table, drop_first=True)

df

df.drop(columns=['dish_liked','reviews_list','menu_item','listed_in(type)'], inplace  =True)
df['rest_type'] = df['rest_type'].str.replace(',' , '') 

df['rest_type'] = df['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))

df['rest_type'].value_counts().head()
df['cuisines'] = df['cuisines'].str.replace(',' , '') 

df['cuisines'] = df['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))

df['cuisines'].value_counts().head()
from sklearn.preprocessing import LabelEncoder

T = LabelEncoder()                 

df['location'] = T.fit_transform(df['location'])

df['rest_type'] = T.fit_transform(df['rest_type'])

df['cuisines'] = T.fit_transform(df['cuisines'])

#df['dish_liked'] = T.fit_transform(df['dish_liked'].
df["approx_cost"] = df["approx_cost"].astype(str).str.replace(',' , '') 
df["approx_cost"] =df["approx_cost"].astype('float')
df.head()
x = df.drop(['rates','name','approx_cost'],axis = 1)

x
y=df['rates']

y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 33)

x_train
x_test
x_test.fillna(x_train.mean(), inplace=True)
col_mask=df.isnull().any(axis=0) 
row_mask=df.isnull().any(axis=1)
df.loc[row_mask,col_mask]
np.isnan(x.values.any())
df=df[df.approx_cost.isna()==False]
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

y_pred_lr = lr.predict(x_test)
lr.score(x_test, y_test)*100
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(x_train,y_train)

y_pred_rfr = rfr.predict(x_test)
rfr.score(x_test,y_test)*100
##SVM

from sklearn import metrics

from sklearn.svm import SVC

s= SVC()

s.fit(x_train,y_train)

y_pred_s = s.predict(x_test)  
s.score(x_test,y_test)*100