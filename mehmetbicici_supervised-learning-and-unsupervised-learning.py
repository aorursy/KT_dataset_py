# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings('ignore') 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
data.head()
data.describe()
print(data.shape)
data["class"].unique()
data.info()
data.isnull().sum()
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidths=0.5,fmt=".1f")

plt.show()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
plt.scatter(data["pelvic_radius"],data["lumbar_lordosis_angle"],c=color_list,alpha=0.7)

plt.xlabel("pelvic radius")

plt.ylabel("lumbar_lordosis_angle")

plt.show()
sns.countplot(x="class",data=data)

data.loc[:,"class"].value_counts()
data["class"].unique()
n=data[data["class"] =="Abnormal"]

ab=data[data["class"] =="Normal"]
n.describe()
ab.describe()
n["class"]=[1 for i in n["class"]]

ab["class"]=[0 for i in ab["class"]]
plt.scatter(n.lumbar_lordosis_angle,n.degree_spondylolisthesis,color="green",label="normal")

plt.scatter(ab.lumbar_lordosis_angle,ab.degree_spondylolisthesis,color="red",label="abnormal")

plt.xlabel("lumbar_lordosis_angle")

plt.ylabel("degree_spondylolisthesis")

plt.legend()
data["class"] = [ 0 if i=="Abnormal" else 1 for i in data["class"] ]
data["class"].unique()
y=data["class"].values

x_data=data.drop(["class"],axis=1)

x= (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)
train_accuracy=[]

test_accuracy=[]

from sklearn.neighbors import KNeighborsClassifier

for i,k in enumerate(range(1,25)):

    knn2=KNeighborsClassifier(n_neighbors=k)

    knn2.fit(x_train,y_train)

    test_accuracy.append(knn2.score(x_test,y_test))

    train_accuracy.append(knn2.score(x_train,y_train))

plt.figure(figsize=(10,8))

plt.plot(range(1,25),test_accuracy,label="test accuracy")

plt.plot(range(1,25),train_accuracy,label="train accuracy")

plt.legend()

plt.xlabel("k values")

plt.ylabel("accuarcy")

plt.xticks(range(1,25))

plt.show()

print("best accuracy values : {},K: {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
y_pred=knn2.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,fmt=".1f",linewidths=0.5,ax=ax)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr=lr.fit(x_train,y_train)

print("test accuracy: {}".format(lr.score(x_test,y_test)))
y_pred=lr.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,fmt=".1f",linewidths=0.5,ax=ax)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.svm import SVC

svm=SVC(random_state=1)

svm.fit(x_train,y_train)

print("accuracy score: {}".format(svm.score(x_test,y_test)))
y_pred=svm.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,fmt=".1f",linewidths=0.5,ax=ax)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

steps = [('scalar', StandardScaler()),

         ('SVM', SVC())]

pipeline = Pipeline(steps)

parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)

cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)

cv.fit(x_train,y_train)



y_pred = cv.predict(x_test)



print("Accuracy: {}".format(cv.score(x_test, y_test)))

print("Tuned Model Parameters: {}".format(cv.best_params_))
y=data["class"].values

x_data=data.drop(["class"],axis=1)

x= (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state = 1)
from sklearn.naive_bayes import GaussianNB

nb= GaussianNB()

nb.fit(x_train,y_train)

print("print accuracy of Naive Bayes algo:",nb.score(x_test,y_test))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=1)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("Test accuracy score: {}".format(dt.score(x_test,y_test)))
y_pred=dt.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,ax=ax,fmt=".1f",linewidths=0.5)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=1)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(x_train,y_train)



print("score: ",rf.score(x_test,y_test))
y_pred=rf.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,ax=ax,fmt=".1f",linewidths=0.5)
result=pd.DataFrame({"method":["KNN","Logistic_Regression","SWM","NB","Decision_Tree_Classification","Random_Forest_Clasification"],"score":[0.8172043010752689,0.7419354838709677,0.8548387096774194,0.8172043010752689, 0.8085106382978723,0.8936170212765957]})
sns.barplot(data=result,x="score",y="method")
data=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

plt.scatter(data["pelvic_radius"],data["degree_spondylolisthesis"])

plt.xlabel("pelvic radius")

plt.ylabel("degree spondy lolisthesis")

plt.show()
data2=data.loc[:,["degree_spondylolisthesis","pelvic_radius"]]

from sklearn.cluster import KMeans

k_means=KMeans(n_clusters=2)

k_means.fit(data2)

labels=k_means.predict(data2)

plt.scatter(data["pelvic_radius"],data["degree_spondylolisthesis"],c=labels)

plt.xlabel("pelvic radius")

plt.ylabel("degree spondylolisthesis")

plt.show()
df=pd.DataFrame({"labels":labels,"class":data["class"]})

ct=pd.crosstab(df["labels"],df["class"])

print(ct)
inertia_list=np.empty(8)

for i in range(1,8):

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(data2)

    inertia_list[i]=kmeans.inertia_

plt.plot(range(0,8),inertia_list,"-o")

plt.xlabel("Number of cluster")

plt.ylabel("Inertia")

plt.show()
data=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data3=data.drop("class",axis=1)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar=StandardScaler()

kmeans=KMeans(n_clusters=2)

pipe=make_pipeline(scalar,kmeans)

pipe.fit(data3)

labels=pipe.predict(data3)

df=pd.DataFrame({"labels":labels,"class":data["class"]})

ct=pd.crosstab(df["labels"],df["class"])

print(ct)

from scipy.cluster.hierarchy import linkage,dendrogram

merg=linkage(data3.iloc[200:220,:],method="single")

dendrogram(merg,leaf_rotation=90,leaf_font_size=6)

plt.show()
plt.scatter(data["pelvic_radius"],data["degree_spondylolisthesis"])

plt.xlabel("pelvic radius")

plt.ylabel("degree spondylolisthesis")

plt.show()
from sklearn.manifold import TSNE

model=TSNE(learning_rate=100)

transformed=model.fit_transform(data2)

x=transformed[:,0]

y=transformed[:,1]

color_list=[ "blue" if i == "Abnormal" else "green" for i in data.loc[:,"class"]]

plt.scatter(x,y,c=color_list)

plt.xlabel("pelvic radius")

plt.ylabel("degree spondylolisthesis")

plt.show()
from sklearn.decomposition import PCA

model=PCA()

model.fit(data3)

transformed=model.transform(data3)

print("Principle components",model.components_)
scaler=StandardScaler()

pca=PCA()

pipeline=make_pipeline(scaler,pca)

pipeline.fit(data3)



plt.bar(range(pca.n_components_),pca.explained_variance_)

plt.xlabel("PCA Feature")

plt.ylabel("variance")

plt.show()

pca=PCA(n_components=2)

pca.fit(data3)

transformed=pca.transform(data3)

x=transformed[:,0]

y=transformed[:,1]

plt.scatter(x,y,c=color_list)

plt.show()