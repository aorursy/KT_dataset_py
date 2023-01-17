# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/iris/Iris.csv")
df.head()
df.info()
sns.countplot(df.Species,palette="icefire")

plt.show()

df.Species.value_counts()
df1=df.drop(["Id"],axis=1)
labels=df.Species.value_counts().index

colors=["grey","blue","red"]

explode=[0,0,0]

sizes=df.Species.value_counts().values

plt.figure(figsize=(7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Distribution of Species',color = 'blue',fontsize = 15)

plt.show()
setosa = df1[df1.Species == "Iris-setosa"]

virginica = df1[df1.Species == "Iris-virginica"]

versicolor = df1[df1.Species == "Iris-versicolor "]
setosa.SepalLengthCm.plot(kind="line",grid=True,color="purple",label="SepalLengthCm",linewidth=2)

plt.legend()

plt.xlabel("Number of Setosa")

plt.ylabel("Sepal Lenght of Setosa as cm")

plt.show()
setosa.SepalLengthCm.plot(kind="hist",bins=15,grid=True,figsize=(10,10),label="SepalLengthCm")

plt.legend()

plt.xlabel("Different Values of Sepal Length as cm")

plt.ylabel("Frequency")

plt.show()

print("Sepal Length Cm of Setosa has different value =", len(setosa.SepalLengthCm.value_counts()))

setosa.SepalWidthCm.plot(kind="line",grid=True,color="darkblue",label="SepalWidthCm",linewidth=2)

plt.legend()

plt.xlabel("Number of Setosa")

plt.ylabel("Sepal Width Cm of Setosa")

plt.show()
setosa.SepalWidthCm.plot(kind="hist",bins=16,grid=True,label="SepalWidthCm",figsize=(10,10))

plt.legend()

plt.xlabel("Different Values of Sepal Width as cm")

plt.ylabel("Frequency")

plt.show()

print("Sepal Width Cm of Setosa has different value =", len(setosa.SepalWidthCm.value_counts()))
setosa.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm",label="Setosa")

plt.legend()

plt.xlabel("Sepal Length Cm")

plt.ylabel("Sepal Width Cm")

plt.title("Correlation by Scatter")

plt.show()
sns.jointplot("SepalLengthCm","SepalWidthCm",data=setosa,size=5,ratio=3,color="cyan")
setosa.PetalLengthCm.plot(kind="line",grid=True,color="darkgreen",label="PetalLengthCm",linewidth=2)

plt.legend()

plt.xlabel("Number of Setosa")

plt.ylabel("Petal Length Cm")

plt.show()
setosa.PetalLengthCm.plot(kind="hist",bins=9,grid=True,label="PetalLengthCm",figsize=(10,10))

plt.legend()

plt.xlabel("Different Values of Petal Legenth as cm")

plt.ylabel("Frequency")

plt.show()

print("Petal Length Cm of Setosa has different value =", len(setosa.PetalLengthCm.value_counts()))
setosa.PetalWidthCm.plot(kind="line",grid=True,color="darkred",linewidth=2,label="PetalWidthCm")

plt.legend()

plt.xlabel("Number of Setosa")

plt.ylabel("Petal Width Cm")

plt.show()
setosa.PetalWidthCm.plot(kind="hist",bins=6,grid=True,label="PetalWidthCm",figsize=(10,10))

plt.legend()

plt.xlabel("Different Values of Petal Width as cm")

plt.ylabel("Frequency")

plt.show()

print("Petal Width Cm of Setosa has different value =", len(setosa.PetalWidthCm.value_counts()))
setosa.plot(kind="scatter",x="PetalLengthCm",y="PetalWidthCm",label="Setosa")

plt.legend()

plt.xlabel("Petal Length Cm")

plt.ylabel("Petal Width Cm")

plt.title("Correlation by Scatter")

plt.show()
sns.jointplot("PetalLengthCm","PetalWidthCm",data=setosa,size=5,ratio=3,color="red")

plt.show()
f,ax=plt.subplots(figsize=(12,12))

sns.heatmap(setosa.corr(),annot=True,cmap="coolwarm",linewidth=2,linecolor="black",fmt=".1f",ax=ax)

plt.show()
sns.pairplot(data=setosa)

plt.show()
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

setosa.SepalLengthCm.plot(kind="line",linewidth=2,label="SepalLenghtCm",color="red",grid=True)

plt.legend()

plt.subplot(2,2,2)

setosa.SepalWidthCm.plot(kind="line",linewidth=2,label="SepalWidthCm",color="blue",grid=True)

plt.legend()

plt.subplot(2,2,3)

setosa.PetalLengthCm.plot(kind="line",linewidth=2,label="PetalLengthCm",color="blue",grid=True)

plt.legend()

plt.subplot(2,2,4)

setosa.PetalWidthCm.plot(kind="line",linewidth=2,label="PetalWidthCm",color="purple",grid=True)

plt.legend()

plt.show()
virginica.SepalLengthCm.plot(kind="line",grid=True,linewidth=2,color="brown",label="SepalLengthCm")

plt.legend()

plt.xlabel("Number of Virginica")

plt.ylabel("Sepal Length Cm")

plt.show()
virginica.SepalLengthCm.plot(kind="hist",bins=21,grid=True,label="SepalLengthhCm",figsize=(10,10))

plt.legend()

plt.xlabel("Different Values of Sepal Length as cm")

plt.ylabel("Frequency")

plt.show()

print("Sepal Length Cm of Setosa has different value =", len(virginica.SepalLengthCm.value_counts()))
virginica.SepalWidthCm.plot(kind="line",grid=True,color="skyblue",linewidth=2,label="SepalWidthCm")

plt.legend()

plt.xlabel("Number of Virginica")

plt.ylabel("Sepal Width Cm")

plt.show()
virginica.SepalWidthCm.plot(kind="hist",bins=13,grid=True,label="SepalWidthCm",figsize=(10,10))

plt.legend()

plt.xlabel("Different Values of Sepal Width as cm")

plt.ylabel("Frequency")

plt.show()

print("Sepal Width Cm of Setosa has different value =", len(virginica.SepalWidthCm.value_counts()))
virginica.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")

plt.legend()

plt.xlabel("Sepal Length Cm")

plt.ylabel("Sepal Width Cm")

plt.title("Correlation by Scatter")

plt.show()
sns.jointplot("SepalLengthCm","SepalWidthCm",data=virginica,ratio=3,color="darkgreen")

plt.show()
virginica.PetalLengthCm.plot(kind="line",grid=True,linewidth=2,label="PetalLengthCm")

plt.legend()

plt.xlabel("Number of virginica")

plt.ylabel("Petal Length Cm")

plt.show()
virginica.PetalLengthCm.plot(kind="hist",bins=20,grid=True,label="PetalLengthCm",figsize=(10,10))

plt.legend()

plt.xlabel("Different Values of Petal Length as cm")

plt.ylabel("Frequency")

plt.show()

print("Petal Length Cm of Setosa has different value =", len(virginica.PetalLengthCm.value_counts()))
virginica.PetalWidthCm.plot(kind="line",linewidth=2,grid=True,color="black",label="PetalWidthCm")

plt.legend()

plt.xlabel("Number of Virginica")

plt.ylabel("Petal Width Cm")

plt.show()
virginica.PetalWidthCm.plot(kind="hist",bins=12,grid=True,label="PetalWidthCm",figsize=(10,10))

plt.legend()

plt.xlabel("Different Values of Petal Width as cm")

plt.ylabel("Frequency")

plt.show()

print("Petal Width Cm of Setosa has different value =", len(virginica.PetalWidthCm.value_counts()))
virginica.plot(kind="scatter",x="PetalLengthCm",y="PetalWidthCm")

plt.legend()

plt.xlabel("Petal Length Cm")

plt.ylabel("Petal Width Cm")

plt.title("Correlation by Scatter")

plt.show()
sns.jointplot("PetalLengthCm","PetalWidthCm",data=virginica,ratio=3,color="darkblue")

plt.show()
f,ax=plt.subplots(figsize=(12,12))

sns.heatmap(virginica.corr(),annot=True,cmap="coolwarm",linewidths=2,linecolor="black",fmt=".1f",ax=ax)

plt.show()
sns.pairplot(data=virginica)

plt.show()
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

virginica.SepalLengthCm.plot(kind="line",linewidth=2,label="SepalLenghtCm",color="red",grid=True)

plt.legend()

plt.subplot(2,2,2)

virginica.SepalWidthCm.plot(kind="line",linewidth=2,label="SepalWidthCm",color="blue",grid=True)

plt.legend()

plt.subplot(2,2,3)

virginica.PetalLengthCm.plot(kind="line",linewidth=2,label="PetalLengthCm",color="blue",grid=True)

plt.legend()

plt.subplot(2,2,4)

virginica.PetalWidthCm.plot(kind="line",linewidth=2,label="PetalWidthCm",color="purple",grid=True)

plt.legend()

plt.show()
x_=df1.drop(["Species"],axis=1)

y=df1.Species.values.reshape(-1,1)
#normalization

x=(x_-np.min(x_))/(np.max(x_)-np.min(x_))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=42)

print("x_train shape : ",x_train.shape)

print("y_train shape : ",y_train.shape)

print("x_test shape : ",x_test.shape)

print("y_test shape : ",y_test.shape)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

list1=[]

list1.append(lr.predict(x_test))

print("train accuracy: {} ".format(lr.score(x_train,y_train)))

print("test accuracy: {} ".format(lr.score(x_test,y_test)))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,lr.predict(x_test))

sns.heatmap(cm,annot=True,cmap="coolwarm",linewidths=2,linecolor="black")

plt.show()