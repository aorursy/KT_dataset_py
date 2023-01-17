import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
filepath="../input/mall-customers/Mall_Customers.csv"

df=pd.read_csv(filepath)

df.head()
df.rename(columns={"Genre":"Gender"}, inplace = True)

df
df.describe()
df_gen = df["Gender"]

df_gen.value_counts().plot(kind="bar", legend =1)
df_gen.value_counts().plot(kind="pie",legend =1)
df_age = df["Age"]

df_age.describe()
df_age.hist(grid = 0)
df.head()
df.rename(columns={"Annual Income (k$)":"Salary"}, inplace=1)

df.rename(columns={"Spending Score (1-100)":"Score"}, inplace=1)

df.head()
df["Salary"].describe()
h0=df["Salary"].hist(grid=0)

h0.set_title("Histogram of Annual Income of Customers")

h0.set_xlabel("Dollars(in Thousands)")

h0.set_ylabel("Frequency")
sns.distplot(df['Salary'])
df.head()
sps=sns.boxplot(df["Score"], color="grey")

sps.set_title("Distribution of Spending Scores")
h1=df['Score'].hist(grid = 0, color= "grey")

h1.set_title("Histogram of Spending Scores")

h1.set_xlabel("Spending Scores")
df.head()

#df.drop(index="CustomerID")
plt.scatter(df["Salary"],df["Score"])
k_range=[1,2,3,4,5,6,7,8,9,10]

sse=[]

for k in k_range:

    km=KMeans(n_clusters=k)

    km.fit(df[["Age","Salary","Score"]])

    sse.append(km.inertia_)
sse
plt.xlabel('k')

plt.ylabel('sum of squared error')

plt.plot(k_range,sse)
km= KMeans(n_clusters=4)

km
y_pred=km.fit_predict(df[["Salary","Score"]])

y_pred
df['cluster']=y_pred

df.tail()
df0=df[df.cluster==0]

df1=df[df.cluster==1]

df2=df[df.cluster==2]

df3=df[df.cluster==3]



plt.scatter( df0.Salary, df0.Score,color="grey")

plt.scatter( df1.Salary, df1.Score,color="red")

plt.scatter( df2.Salary, df2.Score,color="green")

plt.scatter( df3.Salary, df3.Score,color="blue")



plt.legend()
km1=KMeans(n_clusters=5)

y_pred1=km.fit_predict(df[["Salary","Score"]])

y_pred1
df0=df[df.cluster==0]

df1=df[df.cluster==1]

df2=df[df.cluster==2]

df3=df[df.cluster==3]

df4=df[df.cluster==4]



plt.scatter( df0.Salary, df0.Score,color="grey")

plt.scatter( df1.Salary, df1.Score,color="red")

plt.scatter( df2.Salary, df2.Score,color="green")

plt.scatter( df3.Salary, df3.Score,color="blue")

plt.scatter( df4.Salary, df4.Score,color="black")
scaler= MinMaxScaler()

scaler.fit(df[['Salary']])

df["Salary"]= scaler.transform(df[["Salary"]])

df
scaler= MinMaxScaler()

scaler.fit(df[['Score']])

df["Score"]= scaler.transform(df[["Score"]])

df
km = KMeans(n_clusters=5)

y_pred=km.fit_predict(df[["Salary", "Score"]])

y_pred
df.head()
df['cluster']=y_pred

df.head()
df0=df[df.cluster==0]

df1=df[df.cluster==1]

df2=df[df.cluster==2]

df3=df[df.cluster==3]

df4=df[df.cluster==4]



plt.scatter( df0.Salary, df0.Score,color="grey")

plt.scatter( df1.Salary, df1.Score,color="red")

plt.scatter( df2.Salary, df2.Score,color="green")

plt.scatter( df3.Salary, df3.Score,color="blue")

plt.scatter( df4.Salary, df4.Score,color="black")

plt.legend()