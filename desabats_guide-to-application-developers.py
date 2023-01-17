# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Data=pd.read_csv("../input/AppleStore.csv")
Data=Data.iloc[0:,2:]
#print(Data.shape)
Data.head(6)
#Converted the size in MB
Data["size_bytes"]=Data["size_bytes"]*0.000001
plt.figure(figsize=(10,10))
sns.countplot(y=Data["prime_genre"])
plt.xlabel("Number of applications", fontsize=12, color='blue')
plt.ylabel("Genre",fontsize=12, color='blue')
plt.title("Figure 1: Number of applications present in each genre by the end of 2018",fontsize=14, color='blue')

Index=["Games","Education","Entertainment","Utilities","Photo & Video"]
X=["Games","Education","Entertainment","Utilities","Photo & Video"]
for i in range (0,5):
   X[i]=Data[Data["prime_genre"]==Index[i]]

Y=["Games_Notrated","Education_Notrated","Entertainment_Notrated","Utilities_Notrated","Photo_Video_Notrated"]
for i in range (0,5): 
    Y[i]=X[i][X[i]["rating_count_tot"]==0]

Z=["Games_rated","Education_rated","Entertainment_rated","Utilities_rated","Photo_Video_rated"]
for i in range (0,5): 
    Z[i]=X[i][X[i]["rating_count_tot"]>0]
K=[]
for i in range (0,5):
    K.append(len(Y[i]))

J=[]
for i in range (0,5):
    J.append(len(Z[i]))

A=[]
B=[]
C=[]
D=[]
E=[]
L=[A,B,C,D,E]

for i in range (0,5):
    L[i].append(K[i])
    L[i].append(J[i])
    
Labels=["Not rated",'Rated']  
fig=plt.figure(figsize=(12,12))
for i in range (1,6):
    ax = fig.add_subplot(3,2,i)
    plt.pie(L[i-1],labels=Labels,autopct='%.2f%%',explode=(0,0))
    plt.title(Index[i-1])
    plt.axis('equal')
fig.show()
#Eliminate the outliers.
Z[0]=Z[0][Z[0]["rating_count_tot"]<1000000]
Z[1]=Z[1][Z[1]["rating_count_tot"]<25000]
Z[2]=Z[2][Z[2]["rating_count_tot"]<150000]
Z[3]=Z[3][Z[3]["rating_count_tot"]<150000]
Z[4]=Z[4][Z[4]["rating_count_tot"]<500000]

#Generating correlation plots
T=["Corr1","Corr2","Corr3","Corr4","Corr5"]
for i in range (0,5):
    T[i]=Z[i].corr()
for i in range (1,6):
    plt.figure(figsize=(8,8))
    sns.heatmap(T[i-1], annot=True, fmt=".1f")
    plt.title(Index[i-1])
    plt.show()
    
fig4=plt.figure(figsize=(20,15))
plt.subplot(221)
plt.scatter(Z[0]["rating_count_tot"],Z[0]["price"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Games",fontsize=16)
plt.subplot(222)
plt.scatter(Z[0]["rating_count_tot"],Z[0]["lang.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of languages supported", fontsize=16)
plt.title("Games",fontsize=16)
plt.subplot(223)
plt.scatter(Z[0]["user_rating"],Z[0]["lang.num"],color='r')
plt.xlabel("Average user rating value", fontsize=16)
plt.ylabel("Number of languages supported",fontsize=16)
plt.title("Games", fontsize=16)

fig4=plt.figure(figsize=(20,28))
plt.subplot(421)
plt.scatter(Z[1]["rating_count_tot"],Z[1]["size_bytes"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(422)
plt.scatter(Z[1]["rating_count_tot"],Z[1]["lang.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(423)
plt.scatter(Z[1]["rating_count_tot"],Z[1]["sup_devices.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of supporting devices",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(424)
plt.scatter(Z[1]["user_rating"],Z[1]["size_bytes"], color='r')
plt.xlabel("Average user rating value",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(425)
plt.scatter(Z[1]["user_rating"],Z[1]["sup_devices.num"],color='r')
plt.xlabel("Average user rating value",fontsize=16)
plt.ylabel("Number of supporting devices",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(426)
plt.scatter(Z[1]["user_rating"],Z[1]["ipadSc_urls.num"],color='r')
plt.xlabel("Average user rating value",fontsize=16)
plt.ylabel("Number of screenshots showed for display",fontsize=16)
plt.title("Education",fontsize=16)
plt.subplot(427)
plt.scatter(Z[1]["user_rating"],Z[1]["lang.num"],color='r')
plt.xlabel("Average user rating value",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Education",fontsize=16)
fig45=plt.figure(figsize=(20,25))
plt.subplot(421)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["price"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(422)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["lang.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(423)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["ipadSc_urls.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of screenshots showed for display",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(424)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["sup_devices.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of supporting devices",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(425)
plt.scatter(Z[2]["rating_count_tot"],Z[2]["vpp_lic"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Vpp Device Based Licensing Enabled",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(426)
plt.scatter(Z[2]["user_rating"],Z[2]["lang.num"],color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Entertainment",fontsize=16)
plt.subplot(427)
plt.scatter(Z[2]["user_rating"],Z[2]["size_bytes"],color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Entertainment",fontsize=16)

fig45=plt.figure(figsize=(20,35))
plt.subplot(521)
plt.scatter(Z[3]["rating_count_tot"],Z[3]["price"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(522)
plt.scatter(Z[3]["rating_count_tot"],Z[3]["ipadSc_urls.num"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Number of screenshots showed for display",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(523)
plt.scatter(Z[3]["rating_count_tot"],Z[3]["lang.num"])
plt.xlabel("User rating counts")
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(524)
plt.scatter(Z[3]["rating_count_tot"],Z[3]["size_bytes"])
plt.xlabel("User rating counts",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(525)
plt.scatter(Z[3]["user_rating"],Z[3]["size_bytes"], color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(526)
plt.scatter(Z[3]["user_rating"],Z[3]["price"], color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(527)
plt.scatter(Z[3]["user_rating"],Z[3]["sup_devices.num"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel("Number of supporting devices",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(528)
plt.scatter(Z[3]["user_rating"],Z[3]["ipadSc_urls.num"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel(" Number of screenshots showed for display",fontsize=16)
plt.title("Utilities",fontsize=16)
plt.subplot(529)
plt.scatter(Z[3]["user_rating"],Z[3]["lang.num"], color='r')
plt.xlabel("Average User Rating value ",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Utilities",fontsize=16)
X[4]=X[4][X[4]["rating_count_tot"]<150000]
fig6=plt.figure(figsize=(20,20))
plt.subplot(321)
plt.scatter(Z[4]["rating_count_tot"],Z[4]["price"])
plt.xlabel("User Rating counts",fontsize=16)
plt.ylabel("Price",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(322)
plt.scatter(Z[4]["rating_count_tot"],Z[4]["size_bytes"])
plt.xlabel("User Rating counts",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(323)
plt.scatter(Z[4]["rating_count_tot"],Z[4]["lang.num"])
plt.xlabel("User Rating counts",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(324)
plt.scatter(Z[4]["user_rating"],Z[4]["ipadSc_urls.num"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel("Number of screenshots showed for display",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(325)
plt.scatter(Z[4]["user_rating"],Z[4]["size_bytes"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel("Size in MB",fontsize=16)
plt.title("Photo & Video",fontsize=16)
plt.subplot(326)
plt.scatter(Z[4]["user_rating"],Z[4]["lang.num"], color='r')
plt.xlabel("Average User Rating value",fontsize=16)
plt.ylabel("Number of supported languages",fontsize=16)
plt.title("Photo & Video",fontsize=16)
