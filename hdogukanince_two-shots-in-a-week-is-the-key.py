# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/student-mat.csv')
data.head()
data.tail()
data.columns
data.shape
assert data["Walc"].notnull().all()
data["guardian"].value_counts(dropna=False)
data["total_alcohol"] = data["Dalc"] + data["Walc"]
data["total_alcohol"] = data["total_alcohol"].astype(int)

threshold = sum(data.total_alcohol)/len(data.total_alcohol)
data["alcohol_level"] = ["high" if i > threshold else "low" for i in data.total_alcohol]
print(data.loc[:3, "alcohol_level"])
fig_size=plt.rcParams["figure.figsize"]
print("Current Size:" , fig_size)
plt.figure(figsize = (13,13))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cbar=True)
plt.show()
sns.countplot(data.sex)
plt.title("Gender", color="blue")
labels=data.age.value_counts().index
colors=["lime","orange","blue","yellow","purple","red","black","grey"]
explode=[0,0,0,0,0,0,0,0]
sizes=data.age.value_counts().values
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode,autopct='%2.2f%%')
plt.title("Age of Empires -Just Kidding- Age of Students", color="lime",fontsize=15)
plt.show()
school_list = (data["school"].unique())
final_grade = []
for i in school_list:
    x = data[data["school"]==i]
    general_final_grade = sum(x.G3)/len(x)
    final_grade.append(general_final_grade)
    
df = pd.DataFrame({"school_list":school_list, "final_grade":final_grade})
new_data = (df["final_grade"].sort_values(ascending=True)).index.values
sorted_data = df.reindex(new_data)

sns.pointplot(x='school_list',y='final_grade',data=sorted_data,color='lime',alpha=0.8)
plt.xlabel("Schols")
plt.ylabel("Final Grade")
plt.title("Avarage Success Rate for Schools")
plt.show()
sns.set(style="whitegrid")
sns.boxplot(x="school",y="total_alcohol",hue="alcohol_level", data=data)
plt.show()
data.loc[:, ["Dalc", "Walc", "total_alcohol"]].plot()
plt.show()

list = []
for i in range(11):
    list.append(len(data[data.total_alcohol == i]))
ax = sns.barplot(x = [0,1,2,3,4,5,6,7,8,9,10], y = list)
plt.ylabel('Number of Students')
plt.xlabel('Total alcohol consumption')
plt.show()
g1= data["G1"]
g2= data["G2"]
g3= data["G3"]
totAlc= data["total_alcohol"]
newdf = pd.concat([g1,g2,g3,totAlc],axis=1)
sns.pairplot(data=newdf)
plt.show()
sns.jointplot(data.G3, data.total_alcohol, kind="kde", height=5)
plt.show()
sns.lmplot(x="total_alcohol", y="G3", data=data)
plt.show()