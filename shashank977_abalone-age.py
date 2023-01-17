

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline

data= pd.read_csv("../input/abalone-dataset/abalone.csv")
data.head()
data.info()
feature_with_null= [feature for feature in data.columns if data[feature].isnull().sum() >1 ]

feature_with_null
data.describe()
data["Sex"].unique()
data.insert(9, "Age",value= data["Rings"] +1.5)

data.head()
# ACROSS RINGS



plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

sns.distplot(data["Rings"],kde=False,bins=range(0,31,2))



plt.subplot(2,2,2)

sns.distplot(data["Rings"])



plt.subplot(2,2,3)

sns.boxplot(data["Rings"])
# HEIGHT, LENGTH, DIAMETER



plt.figure(figsize=(12,12))

color= sns.color_palette()



plt.subplot(3,3,3)

sns.distplot(data["Height"])



plt.subplot(3,3,1)

sns.distplot(data["Length"])



plt.subplot(3,3,2)

sns.distplot(data["Diameter"])



plt.subplot(3,3,6)

sns.distplot(data["Height"], kde=False, bins=10)



plt.subplot(3,3,4)

sns.distplot(data["Length"], kde=False,bins=10)



plt.subplot(3,3,5)

sns.distplot(data["Diameter"], kde=False, bins=10)



plt.subplot(3,3,9)

sns.boxplot(data["Height"])





plt.subplot(3,3,7)

sns.boxplot(data["Length"])





plt.subplot(3,3,8)

sns.boxplot(data["Diameter"])
data= data[data["Height"] < 0.4]
plt.scatter(data["Height"], data["Age"])
# HEIGHT, LENGTH, DIAMETER



plt.figure(figsize=(12,12))

color= sns.color_palette()



plt.subplot(3,3,3)

sns.distplot(data["Height"])



plt.subplot(3,3,1)

sns.distplot(data["Length"])



plt.subplot(3,3,2)

sns.distplot(data["Diameter"])



plt.subplot(3,3,6)

sns.distplot(data["Height"], kde=False, bins=10)



plt.subplot(3,3,4)

sns.distplot(data["Length"], kde=False,bins=10)



plt.subplot(3,3,5)

sns.distplot(data["Diameter"], kde=False, bins=10)



plt.subplot(3,3,9)

sns.boxplot(data["Height"])





plt.subplot(3,3,7)

sns.boxplot(data["Length"])





plt.subplot(3,3,8)

sns.boxplot(data["Diameter"])
data.head()
plt.figure(figsize=(12,12))



plt.subplot(3,4,1)

sns.distplot(data["Whole weight"])

plt.subplot(3,4,2)

sns.distplot(data["Shucked weight"])

plt.subplot(3,4,3)

sns.distplot(data["Viscera weight"])

plt.subplot(3,4,4)

sns.distplot(data["Shell weight"])



plt.subplot(3,4,5)

sns.distplot(data["Whole weight"],kde=False,bins=14)

plt.subplot(3,4,6)

sns.distplot(data["Shucked weight"], kde= False, bins=14)

plt.subplot(3,4,7)

sns.distplot(data["Viscera weight"], kde= False, bins=14)

plt.subplot(3,4,8)

sns.distplot(data["Shell weight"], kde= False, bins=14)



plt.subplot(3,4,9)

sns.boxplot(data["Whole weight"])

plt.subplot(3,4,10)

sns.boxplot(data["Shucked weight"])

plt.subplot(3,4,11)

sns.boxplot(data["Viscera weight"])

plt.subplot(3,4,12)

sns.boxplot(data["Shell weight"])

#corr= correlation



corr= data.corr()

corr
plt.figure(figsize=(8,6))

sns.heatmap(corr, annot=True)
sns.countplot(data["Sex"])

plt.title("Count of each sex of abalone")
sns.jointplot(data=data, x="Age",y="Shell weight",kind= "reg")



sns.jointplot(data=data, x="Age", y="Height",kind="reg")
plt.figure(figsize=(16,16))



plt.subplot(3,3,1)

plt.scatter(data["Length"],data["Age"])

plt.xlabel("Length")

plt.ylabel("Age")



plt.subplot(3,3,2)

plt.scatter(data["Height"],data["Age"])

plt.xlabel("Height")

plt.ylabel("Age")



plt.subplot(3,3,3)

plt.scatter(data["Diameter"],data["Age"])

plt.xlabel("Diameter")

plt.ylabel("Age")

features=["Length","Height","Diameter","Shell weight","Rings"]

sns.pairplot(data[features],size=2,kind="scatter")