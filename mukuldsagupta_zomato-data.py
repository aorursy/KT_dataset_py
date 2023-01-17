#### 1. Adjust column names and dropped irrelevant columns
#### 2. Removing duplicates
#### 3. Removing Null values
#### 4. to creat resturat type in different columns
#### 5. work all columns to split different columns
#### 6. join all columns
#### 8. drop extra columns 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("../input/zomato.csv")
data
data=data.drop(["address","url","location","phone","dish_liked","menu_item","reviews_list","listed_in(type)"],axis=1)
data.head()
data.isnull().any()
data.columns
data["rate"]=data["rate"].astype(str)
import regex
data["rate"]=data["rate"].apply(lambda x: np.squeeze(regex.findall("\d.\d",x)))
#data["rate"]=data["rate"].apply(lambda X: str(X).split("/5")[0])
data.head()
data["rate"]=data["rate"].astype(str)
data['rate'] = data["rate"].replace("[]", 0)
data["rate"]=data["rate"].astype(float)
data['rest_type'] = data["rest_type"].replace("<", "not_app")
data.head()
data['rest_type'] = data["rest_type"].replace(np.nan, "not_app")
data['cuisines'] = data["cuisines"].replace(np.nan, "not_app")
data.isnull().any()
data["approx_cost(for two people)"]=data["approx_cost(for two people)"].astype(str)
data["approx_cost(for two people)"]=data["approx_cost(for two people)"].apply(lambda x:x.replace(',',''))
data["approx_cost(for two people)"]=data["approx_cost(for two people)"].replace(np.nan,0)
data["approx_cost(for two people)"]=data["approx_cost(for two people)"].replace("nan",0)
data["approx_cost(for two people)"]=data["approx_cost(for two people)"].astype(int)
data
data.isnull().any()
d=data["rest_type"].astype(str)
d=d.to_list()
d
z1=[]
for i in d:
    k=i.split(", ")
    z1.append(k)
print(z1)
k=pd.DataFrame(z1,columns = ['Cochice', 'Pima'])
k.head()
z=k.Cochice.unique()
w=k.Pima.unique()
z_new=np.concatenate((w,z), axis=0)
z_new
for i in range(len(z_new)):
    if z_new[i] == None:
        z_new[i] = "not_app"
j=np.unique(z_new)
j
list_type=j.tolist()
j=j.tolist()
d=np.array(z1)
d
len(d)
len(j)
x=np.zeros((51717,26))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
j=le.fit_transform(j)
j
q=[]
for i in range(len(d)):
    u=d[i]
    u=le.transform(u)
    q.append(u)
    
q
for i in range(len(q)):
    oi=q[i]
    if len(oi)==1:
        oi2=oi[0]
        x[i][oi2]=1
    elif len(oi)==2:
        oi2=oi[0]
        oi3=oi[1]
        x[i][oi2]=1
        x[i][oi3]=1
    
x=x.astype(int)
rest_type=pd.DataFrame(x,columns=list_type)
rest_type
d=data["cuisines"].astype(str)
d=d.to_list()
d
z1=[]
for i in d:
    k=i.split(", ")
    z1.append(k)
print(z1)
k=pd.DataFrame(z1,columns = ['Cochice', 'Pima',"mu","ku","ll","gu","pt","ta"])
k.head()
z=k.Cochice.unique()
w=k.Pima.unique()
a=k.mu.unique()
b=k.ku.unique()
c=k.ll.unique()
d=k.gu.unique()
e=k.pt.unique()
f=k.ta.unique()
z_new=np.concatenate((w,z,a,b,c,d,e,f), axis=0)
z_new
for i in range(len(z_new)):
    if z_new[i] == None:
        z_new[i] = "not_app"
j=np.unique(z_new)
j
list_type=j.tolist()
j=j.tolist()
d=np.array(z1)
d
len(d)
len(j)
x=np.zeros((51717,108))
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
j=le1.fit_transform(j)
j
q=[]
for i in range(len(d)):
    u=d[i]
    u=le1.transform(u)
    q.append(u)
    
q
for i in range(len(q)):
    oi=q[i]
    if len(oi)==1:
        oi2=oi[0]
        x[i][oi2]=1
    elif len(oi)==2:
        oi2=oi[0]
        oi3=oi[1]
        x[i][oi2]=1
        x[i][oi3]=1
    
x=x.astype(int)
cuisines=pd.DataFrame(x,columns=list_type)
cuisines
import ast 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 10,6
plt.xkcd() # let's have some funny plot
%matplotlib inline
sns.countplot(x=data['online_order'])
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.title('Restaurants delivering online or Not')
sns.countplot(x=data['book_table'])
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.title('Restaurants providing Table booking facility:')
plt.figure(figsize=(20,10))
ax = sns.countplot(x='rate',hue='book_table',data=data)
plt.title('Rating of Restaurants vs Table Booking')
plt.show()
plt.figure(figsize=(20,10))
ax = sns.countplot(x='rate',hue='online_order',data=data)
plt.title('Rating of Restaurants vs Online Delivery')
plt.show()
from sklearn.preprocessing import LabelEncoder
le2=LabelEncoder()
le3=LabelEncoder()
le4=LabelEncoder()
le5=LabelEncoder()
data["listed_in(city)"]=le2.fit_transform(data["listed_in(city)"])
data["name"]=le2.fit_transform(data["name"])
data.head()
data["online_order"]=le2.fit_transform(data["online_order"])
data["book_table"]=le2.fit_transform(data["book_table"])
data.head()
data=data.join(rest_type)
data=data.join(cuisines,lsuffix="_c")
data.head()
data=data.drop(["rest_type","cuisines"],axis=1)
data
data.corr()
data.dtypes
X=data.drop("rate",axis=1).values
y=data.iloc[:,3:4].values
from sklearn.model_selection import train_test_split
X_train,Xtest,y_train,ytest=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.ensemble import BaggingRegressor
br=BaggingRegressor(n_estimators=30)
br.fit(X_train,y_train)
ypred1=br.predict(Xtest)
from sklearn.metrics import r2_score
r2_score(ytest,ypred1)