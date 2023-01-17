
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#bar plot
mydata=pd.read_csv("../input/Iris.csv")
list_iris = list(mydata["Species"].unique())
SepalLength_ratio = []
for i in list_iris:
    x = mydata[mydata["Species"]==i]
    Sepallenght_rate = sum(x.SepalLengthCm)/len(x)
    SepalLength_ratio.append(Sepallenght_rate)
data = pd.DataFrame({"list_iris":list_iris,"SepalLength_ratio":SepalLength_ratio})
new_index = (data["SepalLength_ratio"].sort_values(ascending=False)).index.values
# we will see our datas from bigest to smalest one because we choose False at ascending=False
#if we choose ascending=True, we will see from smalest to bigest data
sorted_data1 = data.reindex(new_index)

plt.figure(figsize=(6,6))
sns.barplot(x = sorted_data1["list_iris"],y = sorted_data1["SepalLength_ratio"])
plt.xticks(rotation= 45)
plt.xlabel("Species")
plt.ylabel("SepalLengthCm")
plt.title("iris species")
#we can check counts of our values
mydata.Species.value_counts()
mydata.head()
#bar plot
list_iris = list(mydata["Species"].unique())
SepalWidth_ratio = []
for i in list_iris:
    x = mydata[mydata["Species"]==i]
    SepalWidth_rate = sum(x.SepalWidthCm)/len(x)
    SepalWidth_ratio.append(SepalWidth_rate)
data = pd.DataFrame({"list_iris":list_iris,"SepalWidth_ratio":SepalWidth_ratio})
new_index = (data["SepalWidth_ratio"].sort_values(ascending=True)).index.values
sorted_data2 = data.reindex(new_index)

plt.figure(figsize=(7,6))
sns.barplot(x = sorted_data2["list_iris"],y = sorted_data2["SepalWidth_ratio"])
plt.xticks(rotation= 45)
plt.xlabel("Species")
plt.ylabel("SepalWidthCm")
plt.title("iris species")
mydata.info()
#horizontal bar ploting with seaborn
mydata_list = list(mydata["Species"].unique())
SepalLength = []
SepalWidth = []
PetalLength = []
PetalWidth = []

for i in mydata_list:
    x = mydata[mydata["Species"]==i]
    SepalLength.append(sum(x.SepalLengthCm)/len(x))
    SepalWidth.append(sum(x.SepalWidthCm)/len(x))
    PetalLength.append(sum(x.PetalLengthCm)/len(x))
    PetalWidth.append(sum(x.PetalWidthCm)/len(x))

f,ax =plt.subplots(figsize = (7,6))
sns.barplot(x = SepalLength, y = mydata_list,color="yellow",alpha=0.5,label="SepalLength")
sns.barplot(x = SepalWidth, y = mydata_list,color="red",alpha=0.6,label="SepalWidth")
sns.barplot(x = PetalLength, y = mydata_list,color="green",alpha=0.7,label="PetalLength")
sns.barplot(x = PetalWidth, y = mydata_list,color="blue",alpha=0.8,label="PetalWidth")

ax.legend(loc="upper right",frameon=True)
ax.set(xlabel="Rate of the Species",ylabel="Species",title="iris Species")
#we found this sorted data when we plot the barplot
sorted_data1.head()
#we found this sorted data when we plot the barplot
sorted_data2.head()
#SepalLength_ratio vs SepalWidth_ratio
#to compare our datas we will make basiz normalization (divided with max value)
sorted_data1["SepalLength_ratio"]=sorted_data1["SepalLength_ratio"]/max(sorted_data1["SepalLength_ratio"])
sorted_data2["SepalWidth_ratio"] = sorted_data2["SepalWidth_ratio"]/max(sorted_data2["SepalWidth_ratio"])

# we can combine our datas, which are SepalLength_ratio and SepalWidth_ratio
data = pd.concat([sorted_data1,sorted_data2["SepalWidth_ratio"]],axis=1)
data.sort_values("SepalLength_ratio",inplace=True)

f,ax= plt.subplots(figsize = (20,16))
sns.pointplot(x="list_iris",y="SepalLength_ratio",data=data,color="red",alpha=0.4)
sns.pointplot(x="list_iris",y="SepalWidth_ratio",data=data,color="yellow",alpha=0.7)
plt.text(40,0.6,"mydata",color="green",fontsize=12,style="italic")
plt.text(40,0.5,"SepalLength",color="green",fontsize=12,style="italic")
plt.xlabel("aaa",fontsize=15,color="blue")
plt.ylabel("bbb",fontsize=15,color="blue")
plt.title("SepalLength_ratio vs SepalWidth_ratio")
plt.grid()
#combine of SepalLength_ratio and SepalWidth_ratio
data.head()
#joint plot
g=sns.jointplot(data.SepalLength_ratio,data.SepalWidth_ratio,kind = "kde",size=7)
#kind="scatter","reg","kde","resid","hex"
#kde=kernel densty estimation
plt.savefig("iris.png")
plt.show()
# pearsonr value give us correlation between our datas (SepalLength_ratio and SepalWidth_ratio)
#different joint plot
s =sns.jointplot("SepalLength_ratio","SepalWidth_ratio",data=data,size=5,ratio=3,color="b")
#pie graph
#piecharm graph belongs to matplotlib
labels = mydata.Species.value_counts().index
colors = ["red","green","blue"]
explode = [0,0,0]
sizes = mydata.Species.value_counts().values

plt.figure(figsize = (7,8))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct="%1.1f%%")
plt.title("iris species rate",color="blue",fontsize=14)

#Lm ploting
#Lm plot shows the results of the linear regression between our datas

sns.lmplot(x="SepalLength_ratio",y="SepalWidth_ratio",data=data)

#Cube helix plot
#kde=kernel density estimation
sns.kdeplot(data.SepalLength_ratio,data.SepalWidth_ratio,shade=True,cut=3)
#when we make shade=False, our ploting will be just line without fulling inside graph(blue region)
#cut is size of graph when we decrease number of cut, graph is getting bigger.
#Violin Plot
pl=sns.cubehelix_palette(2,rot=-5,dark= .3)
sns.violinplot(data=data,palette=pl,inner="points")
#inner="points" is our data points
#in fat place,in this plot we can see density of our data where is more
plt.show()
#we can see correlation in our datas with this command
data.corr()
# heatmap Plot
#we use this ploting method to see corelation between our datas
# we will see correlation between SepalLength_ratio and SepalWidth_ratio
f,ax=plt.subplots(figsize =(7,7))
sns.heatmap(data.corr(),annot=True,linewidth=1,linecolor="red",fmt=".1f",ax=ax)
#with annot =True, we can see correlation numbers(1,-0.7)
plt.show()
data.head()
#box plot
sns.boxplot(x="PetalLengthCm",y="SepalLengthCm",hue="Species",data=mydata,palette="PRGn")
plt.show()
#swarm plot
sns.swarmplot(x="Species",y="PetalLengthCm",hue="Species",data=mydata)
plt.show()
#pair plot
#this plotting techniqeu gives correlations graphs with scatter and bar 
sns.pairplot(mydata)
plt.show()
#count plot, this tecnique works like .value_counts(), but we can see this result with graph
sns.countplot(mydata.Species)
sns.countplot(mydata.PetalLengthCm)
plt.title("Count plot",color="green",fontsize=15)
plt.show()
mydata.head()
above= ["above1.4" if i>=1.4 else "below1.4" for i in mydata.PetalLengthCm]
df= pd.DataFrame({"Species":above})
sns.countplot(x=df.Species)
plt.ylabel("number of species")
plt.title("Species to Petallength",color="green",fontsize=14)
