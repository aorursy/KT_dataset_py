# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/titanic_data.csv")

data.head()
data.shape
data['Pclass'].unique()
data['Sex'].unique()
data['Embarked'].unique()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data['Age'].hist()
pclass_sex=data.groupby(['Pclass','Sex'])['Pclass'].count().unstack('Sex')

ax=pclass_sex.plot(kind='bar',stacked=True, alpha=0.7)

plt.xticks(rotation=0)

labels='Male','Female'

colors = ['gold','lightgreen']

g=data.Sex.value_counts()

plt.pie(g,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.xticks(rotation=0)

plt.show()

labels='3rd class','1st class','2nd class'

colors=['gold','lightpink','lightblue']

ax=data.Pclass.value_counts()

plt.pie(ax,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.xticks(rotation=0)

plt.show()

labels='S','c','Q'

colors=['gold','lightgreen','lightblue']

em_count=data.Embarked.value_counts()

plt.pie(em_count,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.xticks(rotation=0)

plt.show()



fig, ax = plt.subplots(nrows=1,ncols=3)

#sex count in class1

labels='Male','Female'

colors = ['gold','lightgreen']

plt.subplot(1,3,1)

class1=data[data["Pclass"]==1].Sex.value_counts()

plt.pie(class1,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('class1',fontsize=15,fontweight='bold')

plt.xticks(rotation=0)

#sex count in class2

plt.subplot(1,3,2)

class2=data[data["Pclass"]==2].Sex.value_counts()

plt.pie(class2,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title('class2',fontsize=15,fontweight='bold')

plt.xticks(rotation=0)

#sex count in class3

plt.subplot(1,3,3)

class3=data[data["Pclass"]==3].Sex.value_counts()

plt.pie(class3,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title("class3",fontsize=15,fontweight='bold')

plt.xticks(rotation=0)

plt.show()

fig, ax = plt.subplots(nrows=1,ncols=2)

#sex count in class1

labels='Male','Female'

colors = ['gold','lightgreen']

plt.subplot(1,2,1)

not_survived=data[data["Survived"]==0].Sex.value_counts()

plt.pie(not_survived,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('not_survived',fontsize=15,fontweight='bold')

plt.xticks(rotation=0)



labels='Female','Male'

colors = ['gold','lightgreen']

plt.subplot(1,2,2)

survived=data[data["Survived"]==1].Sex.value_counts()

plt.pie(survived,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.title('survived',fontsize=15,fontweight='bold')

plt.xticks(rotation=0)

plt.show()
#kde distribution pclass vs. age

for pclass in data['Pclass']:

    data.Age[data.Pclass == pclass].plot(kind='kde',stacked=True)

plt.xlabel('Age')

plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')
