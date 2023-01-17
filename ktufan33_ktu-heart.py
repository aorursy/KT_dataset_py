# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/heart.csv")
data.head()
#data.target.unique
data["target"].value_counts()
data.info()
data.describe()
# data = (data - data.mean()) / (data.max() - data.min())
# bunu sınıflamada kullan
data.head()
plt.figure(figsize=(15,10)) #burda matplotlib kullanılarak yeni bir figure açılıyor
sns.barplot(x=data['age'], y=data['cp'],hue=data["target"]) #burasu seaborn
plt.xticks(rotation= 45)
plt.xlabel('Age')
plt.ylabel('CP')
plt.title('CP given Age')
plt.figure(figsize=(15,10)) #burda matplotlib kullanılarak yeni bir figure açılıyor
sns.barplot(x=data['sex'], y=data['ca'],hue=data["target"]) #burasu seaborn
plt.xticks(rotation= 45)
plt.xlabel('sex')
plt.ylabel('CA')
plt.title('CA given Sex')
data.head(3)
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='age',y='trestbps',hue='target', data=data,color='lime',alpha=0.8)
sns.pointplot(x='age',y='thalach',hue='target', data=data,color='red',alpha=0.8)
#plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')
#plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('sex',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')
plt.grid()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='oldpeak',y='trestbps',hue='ca', data=data,color='lime',alpha=0.8)
#sns.pointplot(x='oldpeak',y='thalach',hue='ca', data=data,color='red',alpha=0.8)
#plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')
#plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('oldpeak',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')
plt.grid()
g = sns.jointplot(data.thalach, data.chol, kind="kde", size=7)
#plt.savefig('graph.png')
plt.show()
# Race rates according in kill data - bu örnek anlaşılmadı tekrar bak
# kill.race.dropna(inplace = True)
labels = data.target.value_counts().index
colors = ['grey','blue']
explode = [0,0]
sizes = data.target.value_counts().values
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Target (Sick or Healthy)',color = 'blue',fontsize = 15)
data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x="trestbps", y="chol", data=data)
plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# cubehelix plot
sns.kdeplot(data.trestbps, data.chol, shade=True, cut=3)
plt.show()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3) #google et bir sürü geliyor birini seç
sns.violinplot(data=data, palette=pal, inner="points") #inner ="points" diyerek içerde noktalarıgöster diyoruz
plt.show()
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla
# gender cinsiyet
# age: yas
# Plot the orbital period with horizontal boxes
sns.boxplot(x="sex", y="age", hue="target", data=data, palette="PRGn")
plt.show()
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla
# gender cinsiyet
# age: yas
# Plot the orbital period with horizontal boxes
sns.boxplot(x="sex", y="age", hue="target", data=data, palette="PRGn")
plt.show()
# swarm plot
# manner of death(olum sekli) : ates edilerek, ates edilerek ve sok tabancasiyla
# gender cinsiyet
# age: yas
sns.swarmplot(x="sex", y="age",hue="target", data=data)
plt.show()
# pair plot
sns.pairplot(data)
plt.show()
corr = data.corr()
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})










