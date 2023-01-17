# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=[10,5]

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
Data=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

# Any results you write to the current directory are saved as output.
Data.head()
Data.columns=['Serial No', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR', 'CGPA', 'Research', 'Chance of Admit']
Data.describe()
Data.head()
GRE_Score=Data["GRE Score"]
GRE_Score.describe()

sns.distplot(GRE_Score,hist_kws={"color":"orange","edgecolor":"black"})
plt.show()
TOEFL_Score=Data["TOEFL Score"]
TOEFL_Score.describe()
sns.distplot(TOEFL_Score,hist_kws={"color":"blue","edgecolor":"black"})
plt.show()
with sns.axes_style('darkgrid'):
    sns.jointplot(TOEFL_Score,GRE_Score, kind='scatter')
UR=Data["University Rating"]
UR1=UR.value_counts()
UR1=UR1.reset_index()
plt.bar(UR1["index"],UR1["University Rating"])
plt.show()
SOP=Data["SOP"]
plt.rcParams['figure.figsize']=[10,5]
SOP.unique()
sns.countplot(SOP)
plt.show()

LOR=Data["LOR"]
plt.rcParams['figure.figsize']=[10,5]
LOR.unique()
sns.countplot(LOR)
plt.show()
CGPA=Data["CGPA"]
plt.hist(CGPA,edgecolor="black")
plt.show()
sns.countplot(Data["Research"])
plt.show()
sns.boxplot(Data["University Rating"],y=Data["CGPA"])
plt.show()
sns.scatterplot(Data["GRE Score"],Data["TOEFL Score"],hue=Data["Research"])
plt.show()
plt.rcParams['figure.figsize'] = [25, 15]
Data["University Rating"]=Data["University Rating"].astype("category")
#sns.swarmplot(Data["GRE Score"],Data["TOEFL Score"],hue=Data["University Rating"])
sns.stripplot(Data["GRE Score"],Data["TOEFL Score"],hue="University Rating",data=Data,size=20)
#Data["University Rating"].unique()
plt.show()
CGPA=Data["CGPA"]
bins=[6,6.5,7,7.5,8,8.5,9,9.5,10]
CGPA=pd.cut(CGPA,bins)
CGPA=CGPA.to_frame()
CGPA.columns=["range"]
CGPA["range"].unique()
plt.rcParams['figure.figsize']=[25,15]
sns.stripplot(Data["GRE Score"],Data["TOEFL Score"],hue=CGPA["range"],data=Data,size=15)
plt.show()


Data["University Rating"]=Data["University Rating"].astype("float")
Data1=Data.loc[:,"GRE Score":]
Data_x=Data1.loc[:,:"Research"]
Data_y=Data1.loc[:,"Chance of Admit"]
from sklearn.preprocessing import StandardScaler
#Data_x["SOP"]=Data_x["SOP"].astype("category")
#Data_x["LOR"]=Data_x["LOR"].astype("category")
Data_x.info()
scaler = StandardScaler()
Data_x=scaler.fit_transform(Data_x)
Data_x

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
Data_y=[1 if each > 0.8 else 0 for each in Data_y]
train_x,test_x,train_y,test_y=train_test_split(Data_x,Data_y,test_size=0.2,random_state=28)
model=DecisionTreeClassifier(max_depth=2,random_state=28)
model.fit(train_x,train_y)
prediction=model.predict(test_x)
from sklearn.metrics import f1_score
f1_score(prediction,test_y)
from sklearn.naive_bayes import GaussianNB
model1=GaussianNB()
model1.fit(train_x,train_y)
prediction=model1.predict(test_x)
f1_score(prediction,test_y)
