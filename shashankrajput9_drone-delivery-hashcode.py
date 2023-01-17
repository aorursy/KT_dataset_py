# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
with open('/kaggle/input/hashcode-drone-delivery/busy_day.in') as data:

    df =data.read().splitlines()

df[0:7]
n =df[0].split()

print("number of rows ",n[0])

print("number of coluns ",n[1])

print("number of drones ",n[2])

print("Number of turns ",n[3]) 

print("Weight that drone can carry ",n[4])  

 
for i in df[0:5]:

    print("length of data =",len(i))
print("number of warhouse",str(df[3]))   

for i in range(4,24,2):

    

    m=str(df[i])

    print('delivery location is ' ,m)

      

      
a =[]

b =[]



for i in range(4,24,2):

    sp =str(df[i]).split()

    y=sp[0]

    a=np.append(a,y)

    z=sp[1]

    b =np.append(b,z)

    

    

warehouse = pd.DataFrame({'Latitude': a, 'Longitude': b}).astype(np.uint16)

warehouse
plt.scatter(x="Latitude",y="Longitude",data=warehouse)

plt.xlabel("Latitude") 

plt.ylabel("Longitude") 
product =[]

stuff =df[5:25:2]

for k in stuff:

    product.append(k.split())

warehouses=["warehouse "+str(i) for i in range(10)]  

df_new=pd.DataFrame(product).T 

df_new.columns=warehouses 

df_new
df_new["product_weight"]=df[2].split()

df_new
row =[]

col =[]



for i in range(25,3775,3):

    loct =str(df[i]).split()

    m=loct[0]

    row=np.append(row,m)

    n=loct[1]

    col=np.append(col,n)

    
location= pd.DataFrame({'delivered_row': row, 'delivered_col': col})

location
fig=plt.figure(figsize=(20,10)) 

ax1=fig.add_subplot(111)  



ax1.scatter(x="Latitude",y="Longitude",data=warehouse,marker="*",s=1000,c="red") 

ax1.scatter(x="delivered_row",y="delivered_col",data=location)
fig=plt.figure(figsize=(15,10)) 

sns.distplot(df_new["product_weight"])
ordr =[]

for i in range(27,3775,3):

    ordr.append(str(df[i]).split())


items= df[26:3775:3] 
fig=plt.figure(figsize=(15,10)) 

sns.distplot(items)