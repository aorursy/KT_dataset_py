# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans

import warnings

%matplotlib inline

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#set style of plots

sns.set_style('white')

np.random.seed(501)

#define a custom palette

customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']

sns.set_palette(customPalette)

sns.palplot(customPalette)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Mall_Customers.csv")
df.head()
df.describe()
df.isnull().sum()
def agelim(num):

    if(num<30):

        return "Youth"

    if((num>30)&(num<45)):

        return "middle"

    else:

        return "old"

    
df["age_lev"]=df["Age"].apply(agelim)

df.head()
plt.figure(figsize=(12,8))

cnt=df['age_lev'].value_counts()

sns.barplot(x=cnt.index,y=cnt.values,alpha=0.8)

plt.xlabel("age groups")

plt.ylabel("No of occurences")

plt.title("barplot")

plt.show()
gender_per=df.Gender.value_counts()

gender_per
f, ax = plt.subplots( figsize=(16,8))

colors = ["#3791D7", "#D72626"]

labels ="Females", "Males"



plt.suptitle('Pie chart ', fontsize=20)



df["Gender"].value_counts().plot.pie(explode=[0,0.05], autopct='%1.2f%%', ax=ax, shadow=True, colors=colors, 

                                             labels=labels, fontsize=12, startangle=70)







ax.set_xlabel('% of Males and females', fontsize=14)
plt.figure(figsize=(12,8))

plt.scatter(x="Age",y="Annual Income (k$)",data=df)

plt.xlabel("Age groups",fontsize=17)

plt.ylabel("Annual income",fontsize=17)

plt.show()
plt.figure(figsize=(12,8))

plt.scatter(x="Age",y="Spending Score (1-100)",data=df)

plt.xlabel("Age",fontsize=17)

plt.ylabel("Spending score",fontsize=17)

plt.show()
plt.figure(figsize=(12,8))

plt.scatter(x="Annual Income (k$)",y="Spending Score (1-100)",data=df)

plt.xlabel("Annual Income (k$)",fontsize=17)

plt.ylabel("Spending score",fontsize=17)

plt.show()
def changeage(s):

    if(s=='Male'):

        return 0

    else:

        return 1

df.Gender=df.Gender.apply(changeage)

df.head()

from sklearn import datasets # we need the iris dataset from here

from sklearn.cluster import KMeans # here is our main function

from sklearn.preprocessing import StandardScaler # this will reduce our mean to 0 and standard deviation to 1 for each feature

from sklearn.pipeline import make_pipeline # this helps us to apply a standard scaler to our k-means
df.iloc[:,1:5].head()
plt.figure(figsize=(14,12))

inertias = []



for k in range(1, 10):

    model = KMeans(n_clusters=k)

    model.fit(df.iloc[:,3:5].values)

    inertias.append(model.inertia_)

    

plt.plot(list(range(1, 10)), inertias, '-o')

plt.title('K VS Inertia')

plt.xlabel('Number of Clusters',fontsize='17')

plt.ylabel('WCSS',fontsize='17')

plt.show()
model1=KMeans(n_clusters= 5, init='k-means++', random_state=0)

model1.fit(df.iloc[:,3:5].values)

labels=model1.predict(df.iloc[:,3:5].values)

model1.labels_


plt.figure(figsize=(16,12))

plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'], c=labels, cmap='rainbow')

plt.xlabel('Annual Income (k$)',fontsize=17)

plt.ylabel('Spending Score (1-100)',fontsize=17)

plt.title('Annual Income (k$) VS Spending Score (1-100)',fontsize=17)

plt.show()
import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering


df1=df.iloc[:,3:5].values

plt.figure(figsize=(10, 7))  

plt.title("Customer Dendograms")  

dend = shc.dendrogram(shc.linkage(df1, method='ward'))  

plt.ylabel("Distance between the clusters",fontsize=17)

plt.xlabel("cluster data points",fontsize=17)


model2 = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  

model2.fit_predict(df1)  
plt.figure(figsize=(10, 7))  

plt.scatter(df1[:,0], df1[:,1], c=model2.labels_, cmap='rainbow') 

plt.xlabel('Annual Income (k$)',fontsize=17)

plt.ylabel('Spending Score (1-100)',fontsize=17)

plt.title('Annual Income (k$) VS Spending Score (1-100)',fontsize=17)

plt.show()
final_df=pd.DataFrame({'Column1':model1.labels_,'Column2':model2.labels_})

final_df.columns=["y1","y2"]

final_df.head()
c=final_df[final_df["y1"]==final_df["y2"]]

accuracy=round(c.shape[0]/final_df.shape[0],2)*100

accuracy
