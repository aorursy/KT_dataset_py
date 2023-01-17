import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def dropquestion(x):

    if x == "?":

        x=0

    return x



def distance(x,y):

    return np.sqrt(np.dot(x-y,x-y))



def find_maximum(df, number):

    maximum = []

    for i in range(number):

        subset = df[df["cluster"]==i].values

        cluster_maximum=0

        for x in subset:

            for y in subset:

                cluster_maximum=max(cluster_maximum, distance(x,y))

        maximum.append(cluster_maximum)

    return maximum

    

   



df = pd.read_csv("/kaggle/input/wiki4HE.csv", sep=";")

df = df.applymap(dropquestion)

df = df.applymap(lambda x: float(x))

from sklearn.cluster import KMeans

maximum_metric = np.zeros([3, 15])

for i in range(2,17):

    model = KMeans(i)

    new_df = pd.concat([df, pd.DataFrame(model.fit_predict(df))], axis=1)

    new_df["cluster"]=new_df[0]

    del new_df[0]

    loc_max = find_maximum(new_df, i)

    maximum_metric[0,i-2]=np.max(loc_max)

    maximum_metric[1,i-2]=np.mean(loc_max)

    maximum_metric[2,i-2]=np.min(loc_max)

from  matplotlib import pyplot as plt



plt.plot(range(2,17), maximum_metric[0,:], label="maximum of cluster diameters")

plt.plot(range(2,17), maximum_metric[1,:], label="mean of cluster diameters")

plt.plot(range(2,17), maximum_metric[2,:], label="minumum of cluster diameters")

plt.legend()

plt.show()
df = pd.read_csv("/kaggle/input/wiki4HE.csv", sep=";")

df = df.applymap(dropquestion)

df = df.applymap(lambda x: float(x))

from sklearn.cluster import Birch

maximum_metric = np.zeros([3, 15])

for i in range(2,17):

    model = Birch(i)

    new_df = pd.concat([df, pd.DataFrame(model.fit_predict(df))], axis=1)

    new_df["cluster"]=new_df[0]

    del new_df[0]

    loc_max = find_maximum(new_df, i)

    maximum_metric[0,i-2]=np.max(loc_max)

    maximum_metric[1,i-2]=np.mean(loc_max)

    maximum_metric[2,i-2]=np.min(loc_max)
from  matplotlib import pyplot as plt



plt.plot(range(2,17), maximum_metric[0,:], label="maximum of cluster diameters")

plt.plot(range(2,17), maximum_metric[1,:], label="mean of cluster diameters")

plt.plot(range(2,17), maximum_metric[2,:], label="minumum of cluster diameters")

plt.legend()

plt.show()
df = pd.read_csv("/kaggle/input/wiki4HE.csv", sep=";")

df = df.applymap(dropquestion)

df = df.applymap(lambda x: float(x))

from sklearn.cluster import AgglomerativeClustering

maximum_metric = np.zeros([3, 15])

for i in range(2,17):

    model = AgglomerativeClustering(i)

    new_df = pd.concat([df, pd.DataFrame(model.fit_predict(df))], axis=1)

    new_df["cluster"]=new_df[0]

    del new_df[0]

    loc_max = find_maximum(new_df, i)

    maximum_metric[0,i-2]=np.max(loc_max)

    maximum_metric[1,i-2]=np.mean(loc_max)

    maximum_metric[2,i-2]=np.min(loc_max)
from  matplotlib import pyplot as plt



plt.plot(range(2,17), maximum_metric[0,:], label="maximum of cluster diameters")

plt.plot(range(2,17), maximum_metric[1,:], label="mean of cluster diameters")

plt.plot(range(2,17), maximum_metric[2,:], label="minumum of cluster diameters")

plt.legend()

plt.show()
model = AgglomerativeClustering(6)

new_df = pd.concat([df, pd.DataFrame(model.fit_predict(df))], axis=1)

new_df["cluster"]=new_df[0]

del new_df[0]

new_df[["AGE","GENDER","DOMAIN","PhD","YEARSEXP", "cluster"]].hist()
new_df[new_df["cluster"]==0][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==1][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==3][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==4][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==5][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
for i in range(6):

    loc_df = new_df[new_df["cluster"]==i][["PhD"]]

    print(i, "cluster PhD total number", loc_df.sum()[0], "and procentage", loc_df.sum()/len(loc_df))
model = KMeans(6)

new_df = pd.concat([df, pd.DataFrame(model.fit_predict(df))], axis=1)

new_df["cluster"]=new_df[0]

del new_df[0]

new_df[["AGE","GENDER","DOMAIN","PhD","YEARSEXP", "cluster"]].hist()
new_df[new_df["cluster"]==0][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==1][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==2][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==3][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==4][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
new_df[new_df["cluster"]==5][["AGE","GENDER","DOMAIN","PhD","YEARSEXP"]].hist()
for i in range(6):

    loc_df = new_df[new_df["cluster"]==i][["PhD"]]

    print(i, "cluster PhD total number", loc_df.sum()[0], "and procentage", loc_df.sum()/len(loc_df))