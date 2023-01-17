import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


breed_labels = pd.read_csv("../input/breed_labels.csv")
color_labels = pd.read_csv("../input/color_labels.csv")
state_labels = pd.read_csv("../input/state_labels.csv")

test = pd.read_csv("../input/test/test.csv")
train = pd.read_csv("../input/train/train.csv")
breed_labels.info()
color_labels.info()
state_labels.info()
test.info()
train.info()
breed_labels.head()
color_labels.head()
state_labels.head()
test.head()
train.head()
len(test["Breed1"].unique())
test.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(test.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
list1 = test['Gender']
list2 = test['Vaccinated']
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
test.plot(kind='scatter', x='Color1', y='Color2',alpha = 1,color = 'red')
plt.xlabel('Color1')    
plt.ylabel('Color2')
plt.title('Color1 - Color2') 
test.Breed1.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
test[(test['Breed1']<300) & (test['Fee']>100)]
test[["Breed1","Fee"]]
sns.lmplot(x='Breed1', y='Breed2', data=test)
plt.figure(figsize=(10,5))
sns.swarmplot(x='Type', y='Age', data=test)
plt.figure(figsize=(10,5))
sns.violinplot(x='Type',y='Age', data=test, inner=None)
sns.swarmplot(x='Type', y='Age', data=test, color='k', alpha=0.7) 
plt.title('Age by Type')
sns.pairplot(test, hue = 'Breed1')

labels = 'breed_labels', 'color_labels', 'state_labels', 'test'
sizes = [215, 130, 245, 210]
colors = ['gold', 'lightskyblue', 'red', 'lightcoral']
explode = (0.1, 0, 0, 0) 

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
