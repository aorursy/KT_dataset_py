import numpy as np #linear algebra

import pandas as pd #data processing

import seaborn as sns #data visualization

import matplotlib.pyplot as plt
df=pd.read_csv('../input/Mall_Customers.csv')

df.info()
df.head()
df.describe()
df.isnull().sum()
del df['CustomerID']
print("Mean of Annual Income (k$) of Female:",df['Annual Income (k$)'].loc[df['Gender'] == 'Female'].mean())

print("Mean of Annual Income (k$) of Male:",df['Annual Income (k$)'].loc[df['Gender'] == 'Male'].mean())
print("Mean of Spending Score (1-100) of Female:",df['Spending Score (1-100)'].loc[df['Gender'] == 'Female'].mean())

print("Mean of Spending Score (1-100) of Male:",df['Spending Score (1-100)'].loc[df['Gender'] == 'Male'].mean())
df.corr()

sns.heatmap(df.corr(), annot=True)

plt.show()
df.query('Gender == "Male"').Gender.count()
df.query('Gender == "Female"').Gender.count()


labels = ['Male','Female']

sizes = [df.query('Gender == "Male"').Gender.count(),df.query('Gender == "Female"').Gender.count()]

#colors

colors = ['#ffdaB9','#66b3ff']

#explsion

explode = (0.05,0.05)

plt.figure(figsize=(8,8)) 

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85,explode=explode)

p=plt.gcf()

plt.axis('equal')

p.gca().add_artist(my_circle)

plt.show()
p1=sns.kdeplot(df['Annual Income (k$)'].loc[df['Gender'] == 'Male'],label='Income Male', shade=True, color="r")

p1=sns.kdeplot(df['Annual Income (k$)'].loc[df['Gender'] == 'Female'],label='Income Female', shade=True, color="b")

plt.xlabel('Annual Income (k$)')

plt.show()
df.sort_values(['Age'])

plt.figure(figsize=(10,8))

plt.bar( df['Age'],df['Spending Score (1-100)'])

plt.xlabel('Age')

plt.ylabel('Spending Score')

plt.show()
sns.lmplot(x='Age', y='Spending Score (1-100)', data=df, fit_reg=True, hue='Gender')

plt.show()
sns.lmplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, fit_reg=True, hue='Gender')

plt.show()
p1=sns.kdeplot(df['Spending Score (1-100)'].loc[df['Gender'] == 'Male'],label='Density Male',bw=2, shade=True, color="r")

p1=sns.kdeplot(df['Spending Score (1-100)'].loc[df['Gender'] == 'Female'],label='Density Female',bw=2, shade=True, color="b")

plt.xlabel('Spending Score')

plt.show()
from sklearn.cluster import KMeans
Y = df[['Spending Score (1-100)']].values

X = df[['Annual Income (k$)']].values

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]

score

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
km = KMeans(n_clusters=5)

clusters = km.fit_predict(df.iloc[:,1:])



df["label"] = clusters



from mpl_toolkits.mplot3d import Axes3D

 



fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)

ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)

ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)

ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)

ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='purple', s=60)

ax.view_init(30, 185)

plt.xlabel("Age")

plt.ylabel("Annual Income (k$)")

ax.set_zlabel('Spending Score (1-100)')

plt.show()