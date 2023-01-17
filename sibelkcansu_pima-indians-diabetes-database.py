# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#matplotlib
import matplotlib.pyplot as plt

#seaborn
import seaborn as sns

from collections import Counter
%matplotlib inline

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/diabetes.csv")
df.head()
df.info()
df.describe()
df.corr()
#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
df.Glucose.plot(kind="line",color="r",label = 'Glucose',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize=(10,10))
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.legend(loc="upper right")
plt.show()
plt.subplots(figsize=(10,10))
plt.plot(df.Glucose[0:100],linestyle="-.")
plt.plot(df.BloodPressure[0:100],linestyle="-")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Glucose-BloodPressure Plot")
plt.legend(loc="upper right")
plt.show()

glucose=df.Glucose
blood_pressure=df.BloodPressure
skin_thickness=df.SkinThickness
age=df.Age
plt.subplots(figsize=(10,10))

plt.subplot(4,1,1)
plt.title("glucose-blood_pressure-skin_thickness-age subplot")
plt.plot(glucose,color="r",label="glucose")
plt.legend()
plt.grid()

plt.subplot(4,1,2)
plt.plot(blood_pressure,color="b",label="blood_pressure")
plt.legend()
plt.grid()

plt.subplot(4,1,3)
plt.plot(skin_thickness,color="g",label="skin_thickness")
plt.legend()
plt.grid()

plt.subplot(4,1,4)
plt.plot(age,color="purple",label="age")
plt.legend()
plt.grid()

plt.show()



#pregnancy histogram

df.Pregnancies.plot(kind="hist",bins=50,figsize=(10,10),color="b",grid="True")
plt.xlabel("pregnancies")
plt.legend(loc="upper right")
plt.title("Pregnancy Histogram")
plt.show()
#age histogram

df.Age.plot(kind="hist",bins=20,figsize=(10,10),color="b",grid="True")
plt.xlabel("age")
plt.legend(loc="upper right")
plt.title("Age Histogram")
plt.show()
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
df.plot(kind="hist",y="BloodPressure",bins = 50,range= (0,100),normed = True,ax = axes[0])
df.plot(kind = "hist",y = "BloodPressure",bins = 50,range= (0,100),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
df.plot(kind="hist",y="Age",bins = 20,range= (20,80),normed = True,ax = axes[0])
df.plot(kind = "hist",y = "Age",bins = 20,range= (20,80),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
# Pregnancies-age bar plot
plt.subplots(figsize=(15,10))
plt.bar(df.Pregnancies,df.Age,color="r")
plt.xlabel("Pregnancies")
plt.ylabel("Age")
plt.title("Pregnancies-Age bar plot")
plt.show()
# bar plot in seaborn
# age average given by pregnancy
p_list=list(df.Pregnancies.unique())
age_avg=[]
for i in p_list:
    x=df[df.Pregnancies==i]
    age=sum(x.Age)/len(x)
    age_avg.append(age)
    
data1=pd.DataFrame({"Pregnancies":p_list,"age_avg":age_avg})

# visualization
plt.figure(figsize=(15,10)) #create a new figure with size (15,10). 
sns.barplot(x=data1['Pregnancies'], y=data1['age_avg'])
plt.xticks(rotation= 360)
plt.xlabel('Pregnancies')
plt.ylabel('age_avg')
plt.title('Age Avg Given Pregnancies')
plt.show()
df.Age.unique()
# First 20 Ages(Decreasing)
Age=df.Age.value_counts()
Age.index=np.arange(len(Age))
plt.figure(figsize=(10,7))
sns.barplot(x=Age[:20].index,y=Age[:20].values)
plt.xticks(rotation=90)
plt.ylabel("Ages")
plt.xlabel("Indexes")
plt.title('First 20 Ages(Decreasing)',color = 'blue',fontsize=15)
plt.show()

df.head()
# Scatter Plot 
# x = pregnancy, y = age
df.plot(kind='scatter', x='Pregnancies', y='Age',alpha = 0.5,color = 'red')
plt.xlabel('Pregnancies')              # label = name of label
plt.ylabel('Age')
plt.title('Pregnancies Age Scatter Plot')
plt.show()
# x = Glucose, y = age
df.plot(kind='scatter', x='Glucose', y='Age',alpha = 0.5,color = 'green')
plt.xlabel('Glucose')              # label = name of label
plt.ylabel('Age')
plt.title('Glucose Age Scatter Plot')
plt.show()
# x = BMI, y = Age
df.plot(kind='scatter', x='BMI', y='Age',alpha = 0.6,color = 'blue')
plt.xlabel('BMI')              # label = name of label
plt.ylabel('Age')
plt.title('BMI Age Scatter Plot')
plt.show()
# Age and Glucose of the people in terms of Pregnancy

data4=df.copy()
new_index=(data4.Pregnancies.sort_values(ascending=False)).index.values
sorted_data2=data4.reindex(new_index)

#creating trace1
trace1 =go.Scatter(
                    x = sorted_data2.Pregnancies[0:100],
                    y = sorted_data2.Age[0:100],
                    mode = "markers",
                    name = "age",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= sorted_data2.Pregnancies[0:100])
# creating trace2
trace2 =go.Scatter(
                    x = sorted_data2.Pregnancies[0:100],
                    y = sorted_data2.Glucose[0:100],
                    mode = "markers",
                    name = "Glucose",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= sorted_data2.Pregnancies[0:100])

data = [trace1, trace2]
layout = dict(title = 'Age vs Glucose for the maximum number of pregnancies',
              xaxis= dict(title= 'Pregnancies',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
#compare pregnancy in terms of outcome
df.boxplot(column="Pregnancies", by="Outcome")
plt.show()
#compare age in terms of outcome
df.boxplot(column="Age", by="Outcome")
plt.show()
#Compare pregnancy and age in terms of outcome
sns.boxplot(x="Pregnancies", y="Age", hue="Outcome", data=df, palette="PRGn") #hue=class.
plt.show()
# BMI vs Age in terms of pregnancy

data2=df.loc[:,["Pregnancies","BMI","Age"]]
data2.BMI=data2.BMI/max(data2.BMI)
data2.Age=data2.Age/max(data2.Age)
data2.sort_values('BMI',inplace=True)

# visualize
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Pregnancies',y='BMI',data=data2,color='lime',alpha=0.8)
sns.pointplot(x='Pregnancies',y='Age',data=data2,color='red',alpha=0.8)
plt.text(0.6,0.55,'BMI',color='lime',fontsize = 18,style = 'italic')
plt.text(0.62,0.6,'Age',color='red',fontsize = 17,style = 'italic')

plt.xlabel('Number of Pregnancy',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('BMI  vs  Age',fontsize = 20,color='blue')
plt.grid()
plt.show()
df.head()
#BMI vs Age in terms of pregnancy

# import graph objects as "go"
import plotly.graph_objs as go

#preapering data
data4=df.copy()
new_index=(data4.Pregnancies.sort_values(ascending=False)).index.values
sorted_data2=data4.reindex(new_index)

# Creating trace1
trace1 = go.Scatter(
                    x = sorted_data2.Pregnancies[0:50],
                    y = sorted_data2.Age[0:50],
                    mode = "lines",
                    name = "Age",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= sorted_data2.Pregnancies)
# Creating trace2
trace2 = go.Scatter(
                    x = sorted_data2.Pregnancies[0:50],
                    y = sorted_data2.BMI[0:50],
                    mode = "lines+markers",
                    name = "BMI",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= sorted_data2.Pregnancies[0:50])
data = [trace1, trace2]
layout = dict(title = 'Age and BMI vs Pregnancy',
              xaxis= dict(title= 'Pregnancy',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

g = sns.jointplot(df.Pregnancies, df.Age, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
g = sns.jointplot("Pregnancies", "Age", data=df,size=5, ratio=3, color="r")
g = sns.jointplot(df.BloodPressure, df.Age, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
g = sns.jointplot("BloodPressure", "Age", data=df,size=7, ratio=3, color="r")
g = sns.jointplot(df.BMI, df.Age, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
g = sns.jointplot("BMI", "Age", data=df,size=7, ratio=3, color="r")
df.Pregnancies.value_counts()
# Maximum Number of Pregnancies in Data
df.Pregnancies.dropna(inplace=True)
labels=df.Pregnancies.value_counts().index[0:7]
colors=['grey','blue','red','yellow','green','brown',"lime"]
explode=[0,0,0,0,0,0,0]
sizes=df.Pregnancies.value_counts().values[0:7]

plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Maximum Number of Pregnancies in Data',color = 'blue',fontsize = 15)
plt.show()
# Minumum Number of Pregnancies in Data
df.Pregnancies.dropna(inplace=True)
labels=df.Pregnancies.value_counts().index[10:]
colors=['grey','blue','red','yellow','green','brown',"lime"]
explode=[0,0,0,0,0,0,0]
sizes=df.Pregnancies.value_counts().values[10:]

plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Minumum Number of Pregnancies in Data',color = 'blue',fontsize = 15)
plt.show()
sns.lmplot(x="Pregnancies", y="Age", data=df)
plt.show()
sns.lmplot(x="BloodPressure", y="Age", data=df)
plt.show()
sns.lmplot(x="BMI", y="Age", data=df)
plt.show()
sns.kdeplot(df.Pregnancies, df.Age, shade=True, cut=3) #shade=True: grafikteki sekillerin ici dolu olsun. cut=3: cıkan sekillerin buyuklugunu ayarlar.
plt.show()
data1.head()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data1, palette=pal, inner="points")
plt.show()
# add a new column to data1 as BMI_avg
p_list=list(df.Pregnancies.unique())
bmi_avg=[]
for i in p_list:
    x=df[df.Pregnancies==i]
    bmi=sum(x.BMI)/len(x)
    bmi_avg.append(bmi)
    
data1["BMI_avg"]=bmi_avg

sns.pairplot(data1)
plt.show()
df.tail()
sns.pairplot(df.loc[0:100,["Pregnancies","Age"]])
plt.show()

sns.countplot(df.Pregnancies)
plt.title("Pregnancies",color="red",fontsize=15)
plt.show()
sns.countplot(df.Age)
plt.title("Ages",color="red",fontsize=15)
above35=["above35" if i>=35 else "below35" for i in df.Age]
data3=pd.DataFrame({"age":above35})
sns.countplot(x=data3.age)
plt.ylabel("Number of People")
plt.title("Age of People in Data",color="blue",fontsize=18)
plt.show()

sns.countplot(df.Outcome)
plt.title("Outcomes",color="blue",fontsize=18)
plt.show()
