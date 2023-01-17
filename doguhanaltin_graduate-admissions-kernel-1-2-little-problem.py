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
data=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
data.head()

#we can see initial 5 row
data.tail()

#we can see last 5 row
data.info()

#we learn that dataframe has 9 columns(not including index),500 row,in addition dataframe's datatypes are float(4) or integer(5)
data.corr()

#we can see correlation between all variables easily.
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(data.corr(),annot=True, fmt=".2f", cmap="Purples", ax=ax,linewidths=2,)

plt.title("Correlation HEATMAP")

plt.show()

#we can see correlation between all variables in heatmap .

   
f,ax= plt.subplots(figsize=(10,10))

data["SOP"].plot(kind="line",color="blue",alpha=0.5,linestyle=":",linewidth=3,grid=True, label="SOP",ax=ax,)

data["University Rating"].plot(color="red", alpha=0.3, linestyle="--",label="University Rating",grid= True)

plt.legend(loc="upper left")

plt.title("Line PLOT")

plt.show()

data2=data.loc[:,["SOP","University Rating"]]

data2.plot(subplots=True)

plt.show()
f,ax= plt.subplots(figsize=(10,10))

data.plot(kind="scatter",x="GRE Score",y="TOEFL Score",color="green",ax=ax)

plt.show()

#we can analyze correlation between to variables(in detail)

data["University Rating"].plot(kind="hist",bins=50,)

plt.title("Uni Rating Frequency")

plt.show()
data.columns
gm=data["GRE Score"].mean()

tm=data["TOEFL Score"].mean()
conf=data["GRE Score"]>gm

cons=data["TOEFL Score"]>tm
data[np.logical_and(conf,cons)]


data["Status Gre"]=[ "Sucss"if each >= gm else "uns" for each in data["GRE Score"]]

data["Status TOEFL"]=["Sucss"if each >= tm else "uns" for each in data["TOEFL Score"]]
print(data["Status Gre"].value_counts(dropna=False))

print(data["Status TOEFL"].value_counts(dropna=False))
data.describe()

data.boxplot(column="TOEFL Score",by="Status Gre")

plt.show()

#we can detect Q3,Q1,mean 
data_new=data.head()
melted= pd.melt(frame=data_new, id_vars="Serial No.",value_vars=["TOEFL Score","University Rating"])

print(melted)
melted.pivot(index="Serial No.",columns="variable",values="value")
data3=data.set_index("Serial No.")
data4= pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

data5=data4.set_index(["University Rating","Research"])

data5
swapdata=data5.swaplevel(0,1)

print(swapdata)
bydata=data4.groupby("University Rating")

bydata.min()
