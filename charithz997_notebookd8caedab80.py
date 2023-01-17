# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#read data 

data_df=pd.read_csv("../input/Iris.csv")
data_df.head()
data_df.describe()
#size of the data frame

data_df.shape
data_df.columns
data_df['Species'].value_counts()
data_df[data_df['Species']=='Iris-setosa'].describe()
sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

#individual plot by id

#g=sns.barplot(x='Species',y='SepalLengthCm',hue='Id',palette=sns.color_palette(flatui),data=data_df,ci=None)



#mean plot of the graphs

g=sns.barplot(x='Species',y='SepalLengthCm',palette=sns.color_palette(flatui),data=data_df,ci=None)



#set style of the graph

g.figure.set_size_inches(12,7)



# remove the top and right line in graph

sns.despine()





#set the title of the graph from here

g.axes.set_title("Irsis",fontsize=34,color="b",alpha=0.3)



#set axis titles

g.set_xlabel("Type of Species",size=20,color='g',alpha=0.6)

g.set_ylabel("Sepal Length",size=20,color='g',alpha=0.6)



#set the ticklabel color and size 

g.tick_params(labelsize=14,labelcolor="black")

#create a pairplot(plot across different attributes)

#size decides the size of the graph

#palette decides the colour

a=sns.pairplot(data_df,hue='Species',palette="muted",size=5,vars=['SepalWidthCm','SepalLengthCm','PetalLengthCm','PetalWidthCm'],kind='scatter',markers=['o','x','+'])



#to change the size of scatterpoints

a=a.map_offdiag(plt.scatter,s=35,alpha=0.9)



#remove the top and the right lines

sns.despine()



#additional line to adjust some appearance issues

plt.subplots_adjust(top=0.9)



#set the title of the graph

a.fig.suptitle('Relation between Sepal Width and Sepal Length',fontsize=20,color='b',alpha=0.5)
b=sns.jointplot(x='SepalWidthCm', y='SepalLengthCm', data=data_df, size=8, alpha=.6,color='k', marker='x')
c=sns.FacetGrid(data_df,col='Species')

c.map(plt.scatter,'SepalLengthCm','SepalWidthCm')
e=sns.factorplot(x="SepalWidthCm", y="SepalLengthCm",col="Species", data=data_df, kind="swarm");

#sns.axes_style('axes.linewidth': 1.0)

#plt.xlim(0,8)

#e.set_ticks(np.arange(1,4,1))

#hue="Species"

e.set(xticklabels=[])
e=sns.violinplot(x='Species',y='SepalLengthCm',data=data_df)
f=sns.violinplot(x='Species',y='SepalWidthCm',data=data_df)
g=sns.violinplot(x='Species',y='PetalLengthCm',data=data_df)
h=sns.violinplot(x='Species',y='PetalWidthCm',data=data_df)
#for index,row in data_df.iterrows():

    #length=row['PetalLengthCm']

    #width=row['PetalWidthCm']

data_df['Length_ratio']=data_df['PetalLengthCm']/data_df['SepalWidthCm']

data_df['Width_ratio']=data_df['SepalLengthCm']/data_df['PetalWidthCm']
data_df['Petal_ratio']
sns.violinplot(x='Species',y='Length_ratio',data=data_df)
sns.jointplot(x='Length_ratio',y='Width_ratio',data=data_df)