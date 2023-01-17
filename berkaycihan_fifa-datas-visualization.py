# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/fifa19/data.csv")

data.columns
data.shape
mdata=data.loc[:,["Name","Age","Nationality","Overall","Club","Release Clause"]]
mdata.head()
mdata.info()
mdata["superstar"]=["+" if i>89 else "-" for i in mdata.Overall]

mdata.head(5)
print(mdata["Nationality"].value_counts().head())
mdata=mdata.head(100)

sns.countplot(x="Nationality" ,data = mdata)

plt.xticks(rotation = 90)

plt.show()
i="€127.5K"

if("." in i):

    b=i.split(".")[1].replace("€","").replace("M","00000").replace(".","").replace("K","00")

    i=i.split(".")[0].replace("€","")+b

    print(i)

else:

    i=i.replace("€","").replace("M","000000").replace("K","000")

    print(i)

    

rate=[]

mdata["Release Clause"]=mdata["Release Clause"].fillna("0")



for i in list(mdata["Release Clause"]):

    if("." in i):

        b=i.split(".")[1].replace("€","").replace("M","00000").replace(".","").replace("K","00")

        i=i.split(".")[0].replace("€","")+b

        int(i)

        rate.append(i)

    else:

        i=i.replace("€","").replace("M","000000").replace("K","000")

        int(i)

        rate.append(i)

mdata["Release Clause(€)"]=rate

mdata.head(15)
mdata["Release Clause(€)"]=mdata["Release Clause(€)"].astype(int)

mdata.info()
mdata=mdata.head(55)

plt.figure(figsize=(15,5))

sns.barplot(x=mdata['Name'], y=mdata['Release Clause(€)'])

plt.xticks(rotation= 90)

plt.xlabel('Players')

plt.ylabel('Rate coef')

plt.title('Rate of the best players')

plt.show()
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x=mdata['Name'],y=mdata['Release Clause(€)'],data=mdata,color='lime',alpha=0.8)

plt.text(40,0.55,'Release Clause(€)',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('Players',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('Release Clause(€)',fontsize = 20,color='blue')

plt.grid()
mdata.tail()
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(mdata.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
g = sns.jointplot(mdata["Overall"], mdata["Release Clause(€)"], kind="kde", size=7)

plt.show()
g = sns.jointplot(mdata["Age"], mdata["Release Clause(€)"], kind="kde", size=7)

plt.show()
sns.pairplot(mdata)

plt.show()
sns.countplot(mdata.superstar)



plt.title(" overall >= 90  ",color = 'blue',fontsize=15)
import plotly.express as px

fig = px.scatter_3d(mdata[mdata.Club=="Real Madrid"], x='Age', y='Overall', z='Release Clause(€)',

                    color='Release Clause(€)', symbol='Release Clause(€)')

fig.show()