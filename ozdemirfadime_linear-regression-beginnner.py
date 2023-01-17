# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df =pd.read_csv("../input/column_2C_weka.csv")
df.info()
df.columns
df.describe()
df.isnull().sum()
df.head()
df.tail()
df.corr() #coorelation between features
cor=df.corr()
ax =sns.heatmap(cor,annot=True,fmt="0.1")
plt.show()
sns.countplot(x="class",data=df)
df.loc[:,"class"].value_counts()
color_list =["red" if i=="Abnormal" else "green" for i in df.loc[:,"class"]]
pd.plotting.scatter_matrix(df.loc[:,df.columns != "class"],
                           c =color_list,
                           figsize =[15,15],
                           diagonal ="kde",
                           alpha =0.5,
                           s=250,  #Bubble size
                           marker =";",
                           edgecolor="black")
plt.show()
data =df[df["class"] == "Abnormal"]
x =data.loc[:,"pelvic_incidence"].values.reshape(-1,1)
y =data.loc[:,"sacral_slope"].values.reshape(-1,1)
 #visiulasion
plt.figure(figsize=(10,10))
plt.scatter(x,y,color="red")
plt.xlabel("pelvic_incidence")
plt.ylabel("scaral_slope")
plt.show()
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import  r2_score
linear_reg =LinearRegression()
linear_reg.fit(x,y)
x_ = np.arange(min(x),max(x)).reshape(-1,1)
y_head =linear_reg.predict(x_)
#print("r squared:",linear_reg.score(x,x_))
print("r square:",linear_reg.score(x,y)) #if 1 axapproximate that is good
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.show()


#pelvinic and lumbar_lordosise_angle between linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
x1=data.loc[:,"pelvic_incidence"].values.reshape(-1,1)
y1=data.loc[:,"lumbar_lordosis_angle"].values.reshape(-1,1)
reg.fit(x1,y1)
x1_ =np.arange(min(x1),max(x1)).reshape(-1,1)
y1_head =reg.predict(x1_)
print("r square :",reg.score(x1,y1))
plt.scatter(x1,y1,color="green")
plt.plot(x1_,y1_head,color="red")
plt.show()



