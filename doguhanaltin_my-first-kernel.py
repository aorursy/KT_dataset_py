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
data1= pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
data1.describe()
data1.info()

data1.columns

#correlation_map:



data1.corr()

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data1.corr(),annot=True,fmt=".2f",ax=ax)

plt.show()

plt.scatter(data1["GRE Score"],data1["TOEFL Score"])

plt.ylabel("GRE Scr")

plt.xlabel("TOEFL SCR")

plt.show()

data1["SOP"].plot(kind="line",color="blue",alpha=0.5,linestyle="-",grid=True,label="SOP",figsize=(15,15))

data1.CGPA.plot(color="black",alpha=0.8,linestyle="-.",label="CGPA")

plt.ylabel("value")

plt.xlabel("index")

plt.legend()

plt.show()

data1["University Rating"].plot(kind="hist" ,bins=50,figsize=(10,10))

plt.show()
