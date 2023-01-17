# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Heart_Disease_Data.csv",na_values="?")
data.info()
data.head()
data.tail()
data["pred_attribute"].replace(inplace=True, value=[1, 1, 1, 1], to_replace=[1, 2, 3, 4])
columns=data.columns[:14]
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    data[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()

dataset_copy=data[data['pred_attribute']==1]
columns=data.columns[:13]
plt.subplots(figsize=(20,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    dataset_copy[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()

features_continuous=["age", "trestbps", "chol", "thalach", "oldpeak", "pred_attribute"]
sns.pairplot(data=data[features_continuous],hue='pred_attribute',diag_kind='kde')
#plt.gcf().set_size_inches(20,15)
plt.show()
# Visualization of age and thalach(max heart rate) with different style of seaborn code
# joint kernel density
# Show the joint distribution using kernel density estimation 
g = sns.jointplot(data.age,data.thalach,kind="kde", size=7)
g = sns.jointplot(data.age, data.thalach, data=data,size=5, ratio=3, color="r")
# Visualization of the predicted attribute and exercise induced angina with different style of seaborn code
# lmplot 
# Show the results of a linear regression within each dataset
g=sns.lmplot(x='pred_attribute', y='exang', data=data)
plt.show()
sns.heatmap(data[data.columns[:14]].corr(),annot=True,cmap='RdYlGn')
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.title('Correlation of Features', y=1.05, size=25)
plt.show()
# Plot the orbital period with horizontal boxes
sns.boxplot(x=data.sex, y=data.age, hue=data.pred_attribute, data=data, palette="PRGn")
plt.show()
ax = sns.violinplot(x="sex", y="chol", data=data)
plt.show()
f,ax1 = plt.subplots(figsize =(20,10))
plt.title('The change of age-cholesterol and age-thalach(max heart rate achieved)', fontsize=20, fontweight='bold')
sns.pointplot(x='age',y='chol',data=data,color='red',alpha=0.8)
sns.pointplot(x='age',y='thalach',data=data,color='green',alpha=0.8)
plt.xlabel('Age',fontsize = 20,color='blue')
plt.ylabel('Values',fontsize = 20,color='blue')
plt.grid()
sns.swarmplot(x="sex", y="age",hue="pred_attribute", data=data)
plt.show()
sns.countplot(x=data.pred_attribute,data=data)
plt.show()
sns.countplot(data.sex)
plt.title("gender",color = 'blue',fontsize=15)