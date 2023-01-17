

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df.info()
df.columns
df.head(10)
df.drop(["Id"],axis = 1 , inplace = True)
df.Species.unique()
plt.plot(df.index[df.Species == "Iris-setosa"] , df.SepalLengthCm[df.Species == "Iris-setosa"],color = "red" , alpha = 0.5)

plt.plot(df.index[df.Species == "Iris-versicolor"] , df.SepalLengthCm[df.Species == "Iris-versicolor"],color = "cyan" , alpha = 0.8)

plt.plot(df.index[df.Species == "Iris-virginica"] , df.SepalLengthCm[df.Species == "Iris-virginica"],color = "green" , alpha = 0.5)

plt.grid()

plt.xlabel("indexes of all 3 species of iris")

plt.ylabel("SepalLengthCm")

plt.show()
plt.plot(df.index[df.Species == "Iris-setosa"] , df.SepalWidthCm[df.Species == "Iris-setosa"],color = "red" , alpha = 0.5)

plt.plot(df.index[df.Species == "Iris-versicolor"] , df.SepalWidthCm[df.Species == "Iris-versicolor"],color = "cyan" , alpha = 0.8)

plt.plot(df.index[df.Species == "Iris-virginica"] , df.SepalWidthCm[df.Species == "Iris-virginica"],color = "green" , alpha = 0.5)

plt.grid()

plt.xlabel("indexes of all 3 species of iris")

plt.ylabel("SepalWidthCm")

plt.show()
plt.plot(df.index[df.Species == "Iris-setosa"] , df.PetalLengthCm[df.Species == "Iris-setosa"],color = "red" , alpha = 0.5)

plt.plot(df.index[df.Species == "Iris-versicolor"] , df.PetalLengthCm[df.Species == "Iris-versicolor"],color = "cyan" , alpha = 0.8)

plt.plot(df.index[df.Species == "Iris-virginica"] , df.PetalLengthCm[df.Species == "Iris-virginica"],color = "green" , alpha = 0.5)

plt.grid()

plt.xlabel("indexes of all 3 species of iris")

plt.ylabel("PetalLengthCm")

plt.show()
plt.plot(df.index[df.Species == "Iris-setosa"] , df.PetalWidthCm[df.Species == "Iris-setosa"],color = "red" , alpha = 0.5)

plt.plot(df.index[df.Species == "Iris-versicolor"] , df.PetalWidthCm[df.Species == "Iris-versicolor"],color = "cyan" , alpha = 0.8)

plt.plot(df.index[df.Species == "Iris-virginica"] , df.PetalWidthCm[df.Species == "Iris-virginica"],color = "green" , alpha = 0.5)

plt.grid()

plt.xlabel("indexes of all 3 species of iris")

plt.ylabel("PetalWidthCm")

plt.show()
df_c= df[df.Species != "Iris-versicolor"]

df_c.Species.unique()
df_c.Species = [1 if each == "Iris-setosa" else 0 for each in df_c["Species"]]

y = df_c.Species

x_df = df_c.drop(["Species"],axis = 1)

x = (x_df - np.min(x_df)) / (np.max(x_df) - np.min(x_df))

x.describe()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
tsz = 0.95

tsz = float(tsz)

accuracy = []

for each in range(1,15):

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = tsz,random_state = 42)

    lr = LogisticRegression()

    lr.fit(x_train,y_train)

    print("accuracy on step {} = {} ".format(each,lr.score(x_test,y_test)))

    tsz = tsz-0.01

    accuracy.append(lr.score(x_test,y_test))
plt.plot(range(1,15),accuracy,color = "red")

plt.grid()

plt.xlabel("steps")

plt.ylabel("value of accuracy")

plt.show
tsz1 = 0.95

tsz1 = float(tsz)

accuracy1 = []

for each1 in range(1,82):

    x_train, x_test, y_train, y_test = train_test_split(x_df,y,test_size = tsz1,random_state = 42)

    lr1 = LogisticRegression()

    lr1.fit(x_train,y_train)

    print("accuracy on step {} = {} ".format(each1,lr.score(x_test,y_test)))

    tsz1 = tsz1-0.01

    accuracy1.append(lr.score(x_test,y_test))
accuracy1 = pd.DataFrame(accuracy1)

plt.plot(accuracy1.index,accuracy1[0],color = "red")

plt.grid()

plt.xlabel("Steps")

plt.ylabel("Value of accuracy")

plt.show()
plt.scatter(df.PetalWidthCm[df.Species == "Iris-setosa"] , df.PetalLengthCm[df.Species == "Iris-setosa"],color = "red" , alpha = 0.5)

plt.scatter(df.PetalWidthCm[df.Species == "Iris-versicolor"] , df.PetalLengthCm[df.Species == "Iris-versicolor"],color = "cyan" , alpha = 0.5)

plt.scatter(df.PetalWidthCm[df.Species == "Iris-virginica"] , df.PetalLengthCm[df.Species == "Iris-virginica"],color = "green" , alpha = 0.5)

plt.grid()

plt.xlabel("PetalWidthCm")

plt.ylabel("PetalLengthCm")

plt.show()
plt.scatter(df.PetalWidthCm , df.PetalLengthCm, color = "black")

plt.xlabel("PetalWidthCm")

plt.ylabel("PetalLengthCm")

plt.grid()

plt.show()
df_new = pd.DataFrame({"PW" : df.PetalWidthCm , "PL" :df.PetalLengthCm })

df_new.head(10)
from sklearn.cluster import KMeans

wcss = []

for each in range(1,15):

    k_m = KMeans(n_clusters = each)

    k_m.fit(df_new)

    wcss.append(k_m.inertia_)

plt.plot(wcss)

plt.xlabel("cluster")

plt.ylabel("wcss")

plt.grid()

plt.show()
k_m2=KMeans(n_clusters = 3)

clusters = k_m2.fit_predict(df_new)

df_new["cls"] = clusters

df_new.cls.unique()

plt.scatter(df_new.PW[df_new.cls == 0],df_new.PL[df_new.cls == 0] , color = "green" ,alpha = 0.5)

plt.scatter(df_new.PW[df_new.cls == 1],df_new.PL[df_new.cls == 1] , color = "cyan" ,alpha = 0.5)

plt.scatter(df_new.PW[df_new.cls == 2],df_new.PL[df_new.cls == 2] , color = "red" ,alpha = 0.5)

plt.grid()

plt.xlabel("Petal Weight")

plt.ylabel("Petal Length")

plt.show
df_new2 = df_new.copy()

df_new2.drop(["cls"],axis=1,inplace = True)

from scipy.cluster.hierarchy import linkage , dendrogram

merg = linkage(df_new2,method = "ward" )

dendrogram(merg)

plt.show()

from sklearn.cluster import AgglomerativeClustering

h_c = AgglomerativeClustering(n_clusters = 3 , affinity = "euclidean" , linkage = "ward")

cluster2 = h_c.fit_predict(df_new2)

df_new2["cls"] = cluster2
plt.scatter(df_new2.PW[df_new2.cls == 0],df_new.PL[df_new2.cls == 0] , color = "green" ,alpha = 0.5)

plt.scatter(df_new2.PW[df_new2.cls == 1],df_new.PL[df_new2.cls == 1] , color = "red" ,alpha = 0.5)

plt.scatter(df_new2.PW[df_new2.cls == 2],df_new.PL[df_new2.cls == 2] , color = "cyan" ,alpha = 0.5)

plt.xlabel("Petal Weight")

plt.ylabel("Petal Length")

plt.show()