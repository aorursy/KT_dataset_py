# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/2016.csv')

df.head()
df.shape
pd.DataFrame(df.isnull().sum()).T
df.info()
df_drop=df.loc[:,'Region':'Dystopia Residual']

df_drop.head()
df_drop.drop(df_drop.columns[[0,1,3,4]], axis=1,inplace=True)

df_drop.head()
corrmat = df_drop.corr()

sns.heatmap(corrmat)

plt.show()
df_Happiness_Category=df['Happiness Score']
i=0;

for x in df_Happiness_Category:

    if 7 < x < 8 :

        df.loc[i,'H_C']=7

    elif 6 < x < 7:

        df.loc[i,'H_C']=6

    elif 5 < x < 6:

        df.loc[i,'H_C']=5

    elif 4 < x < 5:

        df.loc[i,'H_C']=4

    elif 3 < x < 4:

        df.loc[i,'H_C']=3

    elif 2 < x < 3:

        df.loc[i,'H_C']=2

    elif 1 < x < 2:

        df.loc[i,'H_C']=1

    elif 0 < x < 1:

        df.loc[i,'H_C']=0

    i=i+1



df.head()        
from mpl_toolkits.mplot3d import Axes3D

#plt.scatter(df['Economy (GDP per Capita)'],df['Health (Life Expectancy)'],c=df['H_C'])

xs=df_drop['Economy (GDP per Capita)']

ys=df_drop['Family']

zs=df_drop['Health (Life Expectancy)']

#Axes3D.scatter(xs, ys, zs, zdir='z', s=20, c=df['H_C'], depthshade=True,*args, **kwargs)

fig=plt.figure(figsize=(30,20))

ax = fig.add_subplot(111, projection='3d')

p=ax.scatter(xs, ys, zs, c=df['H_C'],s=100)



ax.set_xlabel('Economy (GDP per Capita)')

ax.set_ylabel('Family')

ax.set_zlabel('Health (Life Expectancy)')



fig.colorbar(p)



plt.show()
sns.distplot(df['Happiness Score'])

plt.show()
fig = plt.figure(figsize=(20, 8))

plt.subplot(1, 3,1)

plt.scatter(df['Economy (GDP per Capita)'], df['Happiness Score'], s=30)

plt.xlabel("Economy (GDP per Capita)")

plt.ylabel("Happiness Score")





plt.subplot(1, 3,2)

plt.scatter(df['Family'], df['Happiness Score'], s=30)

plt.xlabel("Family")

plt.ylabel("Happiness Score")



plt.subplot(1, 3,3)

plt.scatter(df['Health (Life Expectancy)'], df['Happiness Score'], s=30)

plt.xlabel("Health (Life Expectancy))")

plt.ylabel("Happiness Score")



plt.show()
df1_drop=df.loc[:,'Region':'Dystopia Residual']

df1_drop.drop(df1_drop.columns[[0,1,2,3,4]], axis=1,inplace=True)

df1_drop.head()
import seaborn as sns

u=df['H_C'].to_frame()

T = pd.concat([df1_drop, u],axis=1)

T.head()
F=T[['Economy (GDP per Capita)','Family','Health (Life Expectancy)','H_C']]

g = sns.pairplot(F, hue="H_C")

plt.show()
from sklearn import linear_model

from sklearn.model_selection import train_test_split

X=df.loc[:,'Economy (GDP per Capita)':'Dystopia Residual']

Y=df.loc[:,'Happiness Score']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

reg = linear_model.LinearRegression()

reg.fit(X, Y)

reg.score(X_test, Y_test)
Y_pred = reg.predict(X_test)

Y_pred
from sklearn.decomposition import PCA as sklearnPCA

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

X1=df.loc[:,'Economy (GDP per Capita)':'Dystopia Residual']

X_std = StandardScaler().fit_transform(X1)

sklearn_pca = sklearnPCA(n_components=3)

Y_sklearn = sklearn_pca.fit_transform(X_std)

t=sklearn_pca.explained_variance_

x = [i + 0.1 for i, _ in enumerate(t)]

plt.ylabel("explained variance")

plt.title("Info")

l=np.arange(28)

plt.xticks([i + 0.5 for i, _ in enumerate(t)], l )

plt.bar(x,t)

plt.show()
I=pd.DataFrame(Y_sklearn)

I.head()
I.shape
from sklearn.cluster import KMeans

X1=I

Y1=df.loc[:,'H_C']

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1)

clusterer = KMeans(n_clusters=6).fit(X1_train)

prediction=clusterer.predict(X1_train)

prediction
from mpl_toolkits.mplot3d import axes3d

fig=plt.figure(figsize=(30,20))

ax = fig.add_subplot(111, projection='3d')

p=ax.scatter(X1_train.iloc[:,1],X1_train.iloc[:,2],X1_train.iloc[:,0], c=prediction,s=300) 



ax.set_xlabel('Axe1')

ax.set_ylabel('Axe2')

ax.set_zlabel('Axe0')



fig.colorbar(p)



for angle in range(0, 360):

    ax.view_init(30, angle)

    plt.draw()

    plt.pause(.001)



fig = plt.figure(figsize=(30, 20))

plt.subplot(1, 3,1)

plt.scatter(X1_train.iloc[:,0], X1_train.iloc[:,1],c=prediction, s=300)

plt.xlabel("Axe0")

plt.ylabel("Axe1")



plt.subplot(1, 3,2)

plt.scatter(X1_train.iloc[:,0], X1_train.iloc[:,2],c=prediction, s=300)

plt.xlabel("Axe0")

plt.ylabel("Axe2")



plt.subplot(1, 3,3)

plt.scatter(X1_train.iloc[:,1], X1_train.iloc[:,2],c=prediction, s=300)

plt.xlabel("Axe1")

plt.ylabel("Axe2")









plt.show()