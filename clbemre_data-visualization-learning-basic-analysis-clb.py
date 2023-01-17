import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf-8"))
data = pd.read_csv("../input/Iris.csv")
data.info()
data.count()
data.describe()
data.groupby('Species').size()
data.corr()
data.head()
data.tail()
data.columns
data.shape
species = iter(data['Species'])
print(*species)
print(data['Species'].value_counts(dropna=False))
data.Species.unique()
Counter(data.Species)
plt.figure(figsize=(5,5))
sns.barplot(x=data['Species'], y=data['SepalLengthCm'])
plt.xticks(rotation = 90)
plt.xlabel('Species')
plt.ylabel('Sepal Length Cm')
plt.title('Species for Sepal Length Cm')
plt.show()
PetalLengthCm = data.PetalLengthCm[data.PetalLengthCm > 1.0]
setosaAndVirgina = data.Species[(data.Species == 'Iris-setosa') | (data.Species == 'Iris-virginica')]
plt.figure(figsize=(5,5))
sns.barplot(x=setosaAndVirgina, y=PetalLengthCm,palette = sns.cubehelix_palette(2))
plt.xticks(rotation = 90)
plt.xlabel('Setosa And Virgina')
plt.ylabel('PetalLengthCm')
plt.show()
# SepalLengthCm ,PetalLengthCm
newdata = pd.concat([data['SepalLengthCm'],data['PetalLengthCm'],data['Species']],axis=1)
newdata.sort_values('SepalLengthCm')

f,ax =  plt.subplots(figsize=(15,5))
sns.pointplot(x='Species',y='SepalLengthCm',data=newdata,color='lime',alpha=0.9)
sns.pointplot(x='Species',y='PetalLengthCm',data=newdata,color='red',alpha=0.7)
plt.text(1,5.5,' Species Sepal Length Cm', color='lime', fontsize=17,style='italic')
plt.text(1,3,' Species Petal Length Cm', color='red', fontsize=18, style='italic')
plt.xlabel('')
plt.ylabel('')
plt.title('Sepal Length Petal Length', fontsize=20,color='blue')
plt.grid()
g = sns.jointplot(newdata.SepalLengthCm,newdata.PetalLengthCm, kind="kde", size=5)
plt.show()
g2 = sns.jointplot('SepalLengthCm','PetalLengthCm',data=newdata,size=5,ratio=3,color='r')
plt.show()
labels = data.Species.value_counts().index
colors = ['red','blue','yellow']
explode = [0,0,0]
sizes = data.Species.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.2f%%')
plt.title('Species Type', color='green',fontsize=15)
plt.show()
sns.lmplot(x = "SepalLengthCm" , y = "PetalLengthCm" , data=newdata)
plt.show()
sns.kdeplot(newdata.PetalLengthCm,newdata.SepalLengthCm, shade=True,cut=2)
plt.show()
pal = sns.cubehelix_palette(2,rot=-5,dark=.3)
sns.violinplot(data=newdata,palette=pal,inner="points")
plt.show()
f,ax = plt.subplots(figsize=(3,3))
sns.heatmap(newdata.corr(),annot=True,linewidths=1.5, linecolor='red',fmt='.1f',ax=ax)
plt.show()
newdatahead = newdata.head(10)
sns.boxplot(x="SepalLengthCm",y="PetalLengthCm", hue="Species", data=newdatahead,palette="PRGn")
plt.show()
sns.swarmplot(x="SepalLengthCm",y="PetalLengthCm", hue="Species",data=newdata)
plt.show()
sns.pairplot(newdata)
plt.show()
sns.countplot(newdata.SepalLengthCm)
plt.title("Species", color="blue",fontsize=15)
plt.show()



