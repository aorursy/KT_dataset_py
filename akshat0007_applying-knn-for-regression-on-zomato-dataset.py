# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/zomato.csv",encoding = "ISO-8859-1")

country = pd.read_excel('../input/Country-Code.xlsx')

df = pd.merge(df, country, on='Country Code')

df.head()
df.shape
import matplotlib.pyplot as plt

import seaborn as sns

from pandas import DataFrame

df.dtypes
df.head()
df.describe()
df1=df.groupby(["Cuisines"])
df1.mean()

df2=df.groupby(["City"])

df2.mean()
df3=df["City"].value_counts()

df3







data_country = df.groupby(['Country'], as_index=False).count()[['Country', 'Restaurant ID']]

data_country.head()

data_country.columns = ['Country', 'No of Restaurant']

plt.figure(figsize=(20,30))

plt.bar(data_country['Country'], data_country['No of Restaurant'],color="brown")

plt.xlabel('Country')

plt.ylabel('No of Restaurant')

plt.title('No of Restaurant')

plt.xticks(rotation = 60)
data_City = df[df['Country'] =='India']

Total_city =data_City['City'].value_counts()

Total_city.plot.bar(figsize=(20,10))

plt.title('Restaurants by City')                                             

plt.xlabel('City')

plt.ylabel('No of Restaurants')

plt.show()
Cuisine_data =df.groupby(['Cuisines'], as_index=False)['Restaurant ID'].count()

Cuisine_data.columns = ['Cuisines', 'Number of Resturants']

Top10= (Cuisine_data.sort_values(['Number of Resturants'],ascending=False)).head(10)

plt.figure(figsize=(20,30))

sns.barplot(Top10['Cuisines'], Top10['Number of Resturants'])

plt.xlabel('Cuisines', fontsize=20)

plt.ylabel('Number of Resturants', fontsize=20)

plt.title('Top 10 Cuisines on Zomato', fontsize=30)

plt.show()
dummy_cuisines=pd.get_dummies(df["Has Online delivery"])

df4=dummy_cuisines.sum()

DataFrame(df4)

x=["Yes","No"]

plt.bar(x,df4,color="red")

plt.xlabel("Wether the restaurant has an Online delivery")

plt.ylabel("Count of restaurants")
import matplotlib.pyplot as plt

import seaborn as sns
pd.crosstab(df['Rating text'], df['City'])
%matplotlib inline
plt.figure(figsize=(20,10))

plt.ylabel("Number of restaurants")

plt.xlabel("Aggregate rating")

sns.barplot(df["Aggregate rating"],range(1,50))



plt.show()
from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=1, stopwords=stopwords).generate_from_frequencies(df['Restaurant Name'].value_counts()))





fig = plt.figure(1,figsize=(30,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
from sklearn import neighbors

from sklearn.metrics import mean_squared_error 

from math import sqrt

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import r2_score
plt.figure(figsize=(10,8))

plt.scatter(df["Votes"],df["Average Cost for two"],marker="*",color="green")

plt.xlabel("Number of Votes")

plt.ylabel("Average Cost for two")



df.corr()
 

corrmat = df.corr() 

  

f,ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
from sklearn.model_selection import train_test_split
x=df[['Currency']]

y=df['Average Cost for two']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

dummies=pd.get_dummies(x_train)

dummies

dummies2=pd.get_dummies(x_test)

dummies2.head()

k=[]

accu=[]

for i in range(1,50):

    model = neighbors.KNeighborsRegressor(n_neighbors = i)

    model.fit(dummies, y_train)  #fit the model

    pred=model.predict(dummies2) #make prediction on test set

    a=dummies2.shape

    accuracy = r2_score(y_test, pred)

    print("For k=",i)

    print("Accuracy is -",accuracy*100,'%') 

    k.append(i)

    accu.append(accuracy)

    

plt.plot(k,accu)

plt.xlabel("Value of K")

plt.ylabel("R2_score")
model = neighbors.KNeighborsRegressor(n_neighbors = 13)

model.fit(dummies, y_train)  #fit the model

pred=model.predict(dummies2) #make prediction on test set

a=dummies2.shape

accuracy = r2_score(y_test, pred)

for i in range(a[0]):

    print("For ",x_test.iloc[i,:])

    print("average cost for two=")

    print(pred[i])


   
accuracy = r2_score(y_test, pred)

print("Accuracy is -",accuracy*100,'%') 
x=df[['Currency','Rating text']]

y=df['Average Cost for two']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

dummies=pd.get_dummies(x_train)

dummies

dummies2=pd.get_dummies(x_test)

dummies2.head()

accur=[]

K1=[]

rmse=[]

y_test2=y_test.values##Converting y_test to numpy array



for i in range(1,50):

    model = neighbors.KNeighborsRegressor(n_neighbors = i)

    model.fit(dummies, y_train)  #fit the model

    pred=model.predict(dummies2) #make prediction on test set

    accuracy = r2_score(y_test, pred)

    error=sqrt(mean_squared_error(y_test2,pred))

    print("For K=",i)

    print("Root Mean Squared Error is-",error)

    print("Accuracy is -",accuracy*100,'%') 

    K1.append(i)

    rmse.append(error)

    accur.append(accuracy)

    

 



plt.plot(K1,rmse)



plt.xlabel("Value of K")

plt.ylabel("RMSE")
plt.plot(rmse,accur)

plt.xlabel("RMSE")

plt.ylabel("R2_score")
plt.plot(K1,accur)



plt.xlabel("Value of K")

plt.ylabel("R2_score")




a=dummies2.shape

model = neighbors.KNeighborsRegressor(n_neighbors = 2)

model.fit(dummies, y_train)  #fit the model

pred=model.predict(dummies2) #make prediction on test set

for i in range(a[0]):

    print("For ",x_test.iloc[i,:])

    print("average cost for two=")

    print(pred[i])

accuracy = r2_score(y_test, pred)

print("For K=",2)

print("Accuracy is -",accuracy*100,'%')
from sklearn.linear_model import LinearRegression
x=df[['Currency','Rating text']]

y=df['Average Cost for two']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

dummies=pd.get_dummies(x_train)

dummies

dummies2=pd.get_dummies(x_test)

dummies2.head()

linear_model=LinearRegression()

linear_model.fit(dummies,y_train)
linear_model.coef_
linear_model.intercept_
prediction=linear_model.predict(dummies2)

r2_score(prediction,y_test)
error=sqrt(mean_squared_error(y_test,prediction))

error 
x=df[['Aggregate rating','Price range']]

y=df['Average Cost for two']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.6,random_state=42)

linear_model=LinearRegression()

linear_model.fit(x_train,y_train)
linear_model.coef_
linear_model.intercept_
prediction=linear_model.predict(x_test)

r2_score(y_test,prediction)
error=sqrt(mean_squared_error(y_test,prediction))

error