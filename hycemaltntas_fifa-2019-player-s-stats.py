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
datf = pd.read_csv("/kaggle/input/fifa19/data.csv")
datf.head()

datf.info()
datf2 = datf.loc[0:25,["ID","Name","Age","Photo","Nationality","Flag","Overall","Potential","Value","Finishing","SprintSpeed"]]
data = datf2.drop(["ID","Photo"],axis = 1)
data.head()
corr_dat = data.corr()
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_dat,annot=True)
plt.show()

ptdata = data.sort_values("Age", axis = 0)
ptdata.plot(kind = 'bar',x='Name',y='SprintSpeed',figsize=(12, 10))
plt.xlabel("Age")
plt.ylabel("Sprint Speed")

plt.show()





ptdata.plot(kind ='scatter',x='Overall',y ='Value',color ='g',alpha = 1,figsize=(10, 8))
plt.show()
plt.subplot(2,1,1)
ptdata.Age.plot(kind = "hist",bins = 26, color ="b",figsize=(12, 10))
plt.xlabel("Age")




plt.subplot(2,1,2)
sns.countplot(ptdata['Nationality'],palette = 'bone')
plt.title("Nationality Analysis",fontsize = 12)


plt.show()

SpanishPlayers = ptdata[ptdata.Nationality == "Spain"]
print(SpanishPlayers.loc[:,["Name","Nationality"]])
avgAge = ptdata.Age.mean()
mvp = ptdata[ptdata.Overall == ptdata.Overall.max()]
print("Average Age  :",avgAge)
print("MVP:\n",mvp.loc[:,["Name","Overall"]])
wonderboys = ptdata[ptdata.Age < ptdata.Age.mean()]
wonderboy = wonderboys[wonderboys.Potential == wonderboys.Potential.max()]
print("\nWonderboys:\n",wonderboys.loc[:,["Name","Overall","Potential"]])
print("\nWonderboy:\n",wonderboy.loc[:,["Name","Overall","Potential"]])

mylist = ptdata["Value"]

temp = 0

for each in mylist:
    if len(each) >= 5:
        maxi = float(each[1:4])
        if maxi > temp:
            temp = maxi
            
            



dicto = mvp.loc[:,["Name","Overall","Value"]].to_dict()
for key,value in dicto.items():
    print(key," : ",value)
print(" ")

