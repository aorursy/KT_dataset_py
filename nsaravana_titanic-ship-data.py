import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv("../input/titanic_data.csv")
data.head()
data.shape
data.info
data.columns
data.isnull().sum()
data1=data.dropna(how="any",axis=0)
data1.head()
data1.shape
data1.describe()
data1["Sex"].unique()
data1["Pclass"].unique()
data1["Sex"].value_counts()
data1.groupby(data["Pclass"])["Sex"].value_counts()
data1.groupby(data["Sex"])["Pclass"].value_counts()

data1.groupby(data["Sex"])["Pclass"].sum()
data1["Survived"].unique()
data1["Survived"].value_counts()
sns.barplot(x="Pclass",y="Survived",data=data)
plt.show()
pclass_sex=data.groupby(['Pclass','Sex'])['Pclass'].count().unstack('Sex')
ax=pclass_sex.plot(kind='bar',stacked=False,alpha=1.0)
plt.xlabel("class")
plt.ylabel("Count")
plt.show()
labels='Male','Female'
colors = ['red','green']
g=data1.Sex.value_counts()
plt.pie(g,labels=labels,colors=colors,autopct='%1.1f%%', shadow=False)
plt.axis('equal')
plt.xticks(rotation=0)
plt.show()

data["Pclass"].value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.show()
colors=["red",'orange']
data1['Sex'].value_counts().plot(kind="pie",colors=colors,autopct="%1.1f%%")
plt.axis('equal')
plt.show()
data["Survived"].value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.axis('equal')
plt.show()
sns.countplot(data1["Sex"])
plt.show()
fig, ax = plt.subplots(nrows=1,ncols=3)
labels='Male','Female'
colors = ['pink','blue']
plt.subplot(1,3,1)
class1=data[data["Pclass"]==1].Sex.value_counts()
plt.pie(class1,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.title('class1',fontsize=15,fontweight='bold')
plt.xticks(rotation=0)
plt.subplot(1,3,2)
class2=data[data["Pclass"]==1].Sex.value_counts()
plt.pie(class2,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.title('class2',fontsize=15,fontweight='bold')
plt.xticks(rotation=0)
plt.subplot(1,3,3)
class3=data[data["Pclass"]==1].Sex.value_counts()
plt.pie(class2,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.title('class3',fontsize=15,fontweight='bold')
plt.xticks(rotation=0)
plt.show()


fig, ax = plt.subplots(nrows=1,ncols=2)
labels='Died','Live'
colors = ['yellow','green']
plt.subplot(1,2,1)
class1=data1[data1["Survived"]==1].Sex.value_counts()
plt.pie(class1,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.title('Died',fontsize=15,fontweight='bold')
plt.xticks(rotation=0)
plt.show()
plt.subplot(1,2,2)
class1=data1[data1["Survived"]==1].Sex.value_counts()
plt.pie(class1,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.title('Live',fontsize=15,fontweight='bold')
plt.xticks(rotation=0)
plt.show()