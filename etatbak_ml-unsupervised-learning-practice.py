# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/2016.csv")
data=data.loc[:,"Region":"Dystopia Residual"]
data.drop(["Happiness Score","Happiness Rank","Lower Confidence Interval","Upper Confidence Interval"],axis=1,inplace=True)
data.Region.value_counts()
f1=data["Region"]=="Sub-Saharan Africa"
f2=data["Region"]=="Western Europe"
f3=data["Region"]=="Latin America and Caribbean"
data=data[f1 | f2 | f3]
data=data.loc[:,["Region","Economy (GDP per Capita)","Family","Freedom","Health (Life Expectancy)"]]
import warnings
warnings.filterwarnings("ignore")

sns.pairplot(data=data,hue="Region",palette="dark")
plt.show()
df=data.drop(["Region"],axis=1)
from sklearn.cluster import KMeans
#finding k values
wcss=[]
for i in range(1,15):
    km=KMeans(n_clusters=i)
    km.fit(df)
    wcss.append(km.inertia_)
#plot
plt.plot(range(1,15),wcss,"-o")
plt.grid(True)
plt.xlabel("k values")
plt.ylabel("wcss values")
plt.show()
#find clusters
df2=df.copy()
km2=KMeans(n_clusters=2)
clusters=km2.fit_predict(df)
df2["clusters"]=clusters

df3=df.copy()
km3=KMeans(n_clusters=3)
clusters=km3.fit_predict(df)
df3["clusters"]=clusters

df4=df.copy()
km4=KMeans(n_clusters=4)
clusters=km4.fit_predict(df)
df4["clusters"]=clusters

#plot
plt.subplot(1,2,1)
plt.scatter(df2["Economy (GDP per Capita)"][df2.clusters==0],
            df2["Health (Life Expectancy)"][df2.clusters==0],color="b")

plt.scatter(df2["Economy (GDP per Capita)"][df2.clusters==1],
            df2["Health (Life Expectancy)"][df2.clusters==1],color="r")
plt.xlabel("K=2 chart")

plt.subplot(1,2,2)

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Western Europe"],
            data["Health (Life Expectancy)"][data.Region=="Western Europe"],color="r")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Latin America and Caribbean"],
            data["Health (Life Expectancy)"][data.Region=="Latin America and Caribbean"],color="b")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Sub-Saharan Africa"],
            data["Health (Life Expectancy)"][data.Region=="Sub-Saharan Africa"],color="g")
plt.xlabel("Real chart")
plt.show()

plt.subplot(1,2,1)
plt.scatter(df2["Economy (GDP per Capita)"][df2.clusters==0],
            df2["Health (Life Expectancy)"][df2.clusters==0],color="b")

plt.scatter(df3["Economy (GDP per Capita)"][df3.clusters==1],
            df3["Health (Life Expectancy)"][df3.clusters==1],color="g")

plt.scatter(df3["Economy (GDP per Capita)"][df3.clusters==2],
            df3["Health (Life Expectancy)"][df3.clusters==2],color="r")
plt.xlabel("K=3 chart")

plt.subplot(1,2,2)
plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Western Europe"],
            data["Health (Life Expectancy)"][data.Region=="Western Europe"],color="r")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Latin America and Caribbean"],
            data["Health (Life Expectancy)"][data.Region=="Latin America and Caribbean"],color="b")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Sub-Saharan Africa"],
            data["Health (Life Expectancy)"][data.Region=="Sub-Saharan Africa"],color="g")
plt.xlabel("Real chart")
plt.show()

plt.subplot(1,2,1)
plt.scatter(df2["Economy (GDP per Capita)"][df2.clusters==0],
            df2["Health (Life Expectancy)"][df2.clusters==0],color="b")

plt.scatter(df3["Economy (GDP per Capita)"][df3.clusters==1],
            df3["Health (Life Expectancy)"][df3.clusters==1],color="g")

plt.scatter(df3["Economy (GDP per Capita)"][df3.clusters==2],
            df3["Health (Life Expectancy)"][df3.clusters==2],color="r")

plt.scatter(df4["Economy (GDP per Capita)"][df4.clusters==3],
            df4["Health (Life Expectancy)"][df4.clusters==3],color="orange")
plt.xlabel("K=4 chart")

plt.subplot(1,2,2)
plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Western Europe"],
            data["Health (Life Expectancy)"][data.Region=="Western Europe"],color="r")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Latin America and Caribbean"],
            data["Health (Life Expectancy)"][data.Region=="Latin America and Caribbean"],color="b")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Sub-Saharan Africa"],
            data["Health (Life Expectancy)"][data.Region=="Sub-Saharan Africa"],color="g")
plt.xlabel("Real chart")
plt.show()

#Hierarchical Clustering
from scipy.cluster.hierarchy import linkage,dendrogram
merg=linkage(df,method="ward")
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidean dist.")
plt.show()

from sklearn.cluster import AgglomerativeClustering

hie_cl2=AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="ward")
cluster2=hie_cl2.fit_predict(df)
df_hie2=df.copy()
df_hie2["label"]=cluster2

hie_cl3=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
cluster3=hie_cl3.fit_predict(df)
df_hie3=df.copy()
df_hie3["label"]=cluster3
#plot
plt.subplot(1,2,1)
plt.scatter(df_hie2["Economy (GDP per Capita)"][df_hie2.label==0],
            df_hie2["Health (Life Expectancy)"][df_hie2.label==0],color="b")

plt.scatter(df_hie2["Economy (GDP per Capita)"][df_hie2.label==1],
            df_hie2["Health (Life Expectancy)"][df_hie2.label==1],color="r")
plt.xlabel("K=2 chart")

plt.subplot(1,2,2)

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Western Europe"],
            data["Health (Life Expectancy)"][data.Region=="Western Europe"],color="g")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Latin America and Caribbean"],
            data["Health (Life Expectancy)"][data.Region=="Latin America and Caribbean"],color="r")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Sub-Saharan Africa"],
            data["Health (Life Expectancy)"][data.Region=="Sub-Saharan Africa"],color="b")
plt.xlabel("Real chart")
plt.show()

plt.subplot(1,2,1)
plt.scatter(df_hie3["Economy (GDP per Capita)"][df_hie3.label==0],
            df_hie3["Health (Life Expectancy)"][df_hie3.label==0],color="b")

plt.scatter(df_hie3["Economy (GDP per Capita)"][df_hie3.label==1],
            df_hie3["Health (Life Expectancy)"][df_hie3.label==1],color="g")

plt.scatter(df_hie3["Economy (GDP per Capita)"][df_hie3.label==2],
            df_hie3["Health (Life Expectancy)"][df_hie3.label==2],color="r")
plt.xlabel("K=3 chart")

plt.subplot(1,2,2)
plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Western Europe"],
            data["Health (Life Expectancy)"][data.Region=="Western Europe"],color="g")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Latin America and Caribbean"],
            data["Health (Life Expectancy)"][data.Region=="Latin America and Caribbean"],color="r")

plt.scatter(data["Economy (GDP per Capita)"][data.Region=="Sub-Saharan Africa"],
            data["Health (Life Expectancy)"][data.Region=="Sub-Saharan Africa"],color="b")
plt.xlabel("Real chart")
plt.show()


#Standardization

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 3)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(df)
labels = pipe.predict(df)
dfS = pd.DataFrame({'labels':labels,"Region":data['Region']})
ct = pd.crosstab(dfS['labels'],dfS['Region'])
ct
data2=pd.read_csv("../input/2016.csv")
#elimination some columns
data2.drop(["Country","Happiness Rank","Happiness Score",
            "Lower Confidence Interval","Upper Confidence Interval"],axis=1,inplace=True)
#preparation of data
f1=data2["Region"]=="Sub-Saharan Africa"
f2=data2["Region"]=="Western Europe"
f3=data2["Region"]=="Latin America and Caribbean"
data2=data2[f1 | f2 | f3]
data2.Region=[0 if i=="Western Europe"
              else 1 if i=="Latin America and Caribbean"
              else 2 for i in data2.Region]
y=data2.Region.values
x_data=data2.drop(["Region"],axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train-test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print("test accuracy: ",lr.score(x_test,y_test))

#knn model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
print("3nn score: ",knn.score(x_test,y_test))

#let's check best k value for accuracy with for loop
score_knn=[]
for i in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_knn.append(knn2.score(x_test,y_test))
#plot
plt.plot(range(1,15),score_knn)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
##**We have found best accuracy values between 1-15**
##Random Forest Class.
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,random_state=42)
rf.fit(x_train,y_train)
print("score: ",rf.score(x_test,y_test))
#Confusion matrix
y_pred=rf.predict(x_test)
y_true=y_test

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_pred)
#cm visualization
#f,ax=plt.subplots(figsize=(5,5))
#sns.heatmap(cm,annot=True,linewidths=0.5,linecolor='r',fmt=".0f",ax=ax)
#plt.show()