# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns 

import plotly as py

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Mall_Customers.csv")

df.head(50)
#Checking the Null or missing values in data

df.isnull().sum()
df.duplicated().sum() # check for duplicate data
#Lets analyse the data using scatter plot to see the data trend.

df.head()

#lets see the annaul income and spending score

df.columns

plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])

#plt.scatter(df["Age"],df['Annual Income (k$)'])

plt.legend()

plt.xlabel("Income")

plt.ylabel("Spending score")

plt.figure(1 , figsize = (15 , 7))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = df)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
#lets see the same for age and income

plt.scatter(df["Age"],df['Annual Income (k$)'])

plt.legend()

plt.ylabel("Income")

plt.xlabel("Age")
#lets see spending of male and female customers

plt.hist(df['Spending Score (1-100)'])

plt.title("Spending scores of Customers")

plt.show()
df.head(10)
#high spedning customers

df_spend=df.sort_values(by="Spending Score (1-100)",ascending =False)

df_spend
#Since Gender is a male/female value so we have to encode it as 0,1

#so we can use label encoder for using it in Machine Learning model.

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

dfe=df.copy()

dfe.Gender=le.fit_transform(df.Gender) # the Gender is encoded as 0/1 where 0 is female and 1 is male.

dfe.head()
dfe.drop("CustomerID",inplace=True,axis="columns")
#Lets check for corrlation of data amongsteach other.# same as heat map

#dfe.corr(method='pearson')
dfe.corrwith(dfe["Annual Income (k$)"],axis=0)
dfe.columns
#Lets train the model using train test and split

X=dfe[['Gender', 'Spending Score (1-100)']]

#from sklearn.model_selection import train_test_split



#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size =0.2) # 20 % is test data,80% will be training data
#lets scale the data 

from sklearn.preprocessing import normalize

X= normalize(X)
from sklearn.cluster import KMeans

#creating elbow plot to find the value of K for best results

k_range= range(1,11)

SSE =[]  # Sum of square errors

for k in k_range:

    km=KMeans(n_clusters=k)

    km.fit(X)

    SSE.append(km.inertia_)
SSE  # there are coordinates of sum square errors
#plotting the elbow

plt.xlabel("k values")

plt.ylabel("sum square errors")

plt.grid(True)

plt.plot(k_range,SSE,marker="+")
#so K value can be taken as 5

km= KMeans(n_clusters=4)



y_predicted=km.fit_predict(X)

y_predicted



km.score(X) # 
#getting the absolute value of score 

from sklearn.metrics import silhouette_score

silhouette_score(X,y_predicted)