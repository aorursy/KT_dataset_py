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
filename='/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

df=pd.read_csv(filename)

df=pd.DataFrame(df)

df.head()
df.isnull().sum()
df.describe()
df.info()
print("Filling the NaN values with zeroes:-\n")

df=df.fillna(0)

df.fillna(0,inplace=True)

df.head()
print("The total number of rows are",df.shape[0])

print("The total number of columns are",df.shape[1])

print("The columns are:-\n",df.columns.tolist())

print(df.dtypes)
df.drop(['last_review'],axis=1,inplace=True)

df.head()
print("Mean:\n",df.groupby('neighbourhood').mean().head())
df.neighbourhood.unique()
print("Median:-\n",df.groupby('neighbourhood').median().head())
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
ax = plt.axes()

ax.scatter(df.neighbourhood_group,df.neighbourhood)

ax.set(xlabel='Neighbourhood group',

       ylabel='Neighbourhood',

       title='Concentration of hotels in different neighbourhoods')

df['price'].hist(bins=100,grid=False, xlabelsize=12, ylabelsize=12, color='yellow')

plt.xlabel('Prices',fontsize=14.0)

plt.ylabel('Frequency',fontsize=14.0)

plt.title('Histogram of prices',fontsize=20.0)

plt.xlim([20,500])

plt.ylim([500,25000])
f,ax = plt.subplots(figsize=(10,5))

df1 = df[df.neighbourhood_group=="Manhattan"]['price']

sns.distplot(df1,color='black')

plt.show()
df_Brooklyn=df[df['neighbourhood_group']=='Brooklyn']

df_Brooklyn.head()

df_Brooklyn.neighbourhood.unique()
bxplt=sns.boxplot(y='availability_365',x='room_type',data=df_Brooklyn,width=0.5,palette="colorblind")

 

bxplt=sns.stripplot(y='availability_365',x='room_type',data=df_Brooklyn,jitter=True,marker='o',alpha=0.5,color='black')
df_locality=df_Brooklyn[df_Brooklyn['neighbourhood']=='Kensington']

df_locality2=df_Brooklyn[df_Brooklyn['neighbourhood']=='Clinton Hill']

df_locality.head()
sns.distplot(df_locality['price'], hist = False, kde = True, label='Kensington',color='yellow')

sns.distplot(df_locality2['price'],hist=False,kde=True,label='Clinton Hill',color='red')

plt.title('Comparison of prices in Kensington and Clinton Hill')

plt.xlabel('Prices')

plt.ylabel('Density')

plt.legend(prop={'size':10})
plt.stackplot(df_locality['room_type'],df_locality['availability_365'],color=['green','yellow'])

plt.xlabel('Availability')

plt.ylabel('Types of room')

plt.title('Stack plot')

plt.legend()

plt.show()
x=df.iloc[:,9:11].values

y=df.iloc[:,4].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train=scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=92)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

conf_mat=confusion_matrix(y_test, y_pred)

print(conf_mat)

print(classification_report(y_test, y_pred))
error=[]

for i in range(1,100):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i=knn.predict(X_test)

    error.append(np.mean(pred_i!=y_test))
plt.figure(figsize=(10,10))

plt.plot(range(1,100),error,color='black',linestyle='dashed',marker='*',markerfacecolor='red',markersize=10)

plt.title('Error Rate K value')

plt.xlabel('K Value')

plt.ylabel('Mean error')
type(conf_mat) 
fig,ax = plt.subplots(figsize=(12,7))

title="Confusion matrix heat map"

plt.title(title,fontsize=18)

ttl=ax.title

ttl.set_position([0.5,1.02])

ax.set_xticks([])

ax.set_yticks([])

ax.axis('off')

sns.heatmap(conf_mat, fmt="",cmap="Blues", linewidths=0.50, ax=ax)