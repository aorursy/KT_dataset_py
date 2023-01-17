import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
south = pd.read_csv("../input/ap-south-1.csv",sep=",")

east = pd.read_csv("../input/ca-central-1.csv",sep=",")

west = pd.read_csv("../input/us-east-1.csv",sep=",")

central = pd.read_csv("../input/us-west-1.csv",sep=",")

south.info()
east.info()
south.head()
central.describe()
s=east['os'].value_counts().plot("bar",figsize=(12,4),fontsize=12)

s.set_title("AWS east-zone OS frequency",color='darkorange',fontsize=30)

s.set_xlabel("OS name",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=west['os'].value_counts().plot("bar",figsize=(12,4),fontsize=12)

s.set_title("AWS west-zone OS frequency",color='darkorange',fontsize=30)

s.set_xlabel("OS name",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=south['os'].value_counts().plot("bar",figsize=(12,4),fontsize=12)

s.set_title("AWS south-zone OS frequency",color='darkorange',fontsize=30)

s.set_xlabel("OS name",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=central['os'].value_counts().plot("bar",figsize=(12,4),fontsize=12)

s.set_title("AWS central-zone OS frequency",color='darkorange',fontsize=30)

s.set_xlabel("OS name",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=east.groupby('instance_type').size().plot("bar",figsize=(12,6),fontsize=12)

s.set_title("AWS east-zone instance-type",color='darkorange',fontsize=30)

s.set_xlabel("Instance type",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=west.groupby('instance_type').size().plot("bar",figsize=(12,6),fontsize=12)

s.set_title("AWS west-zone instance-type",color='darkorange',fontsize=30)

s.set_xlabel("Instance type",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=south.groupby('instance_type').size().plot("bar",figsize=(12,6),fontsize=12)

s.set_title("AWS South-zone instance-type",color='darkorange',fontsize=30)

s.set_xlabel("Instance type",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=central.groupby('instance_type').size().plot("bar",figsize=(12,6),fontsize=12)

s.set_title("AWS Central-zone instance-type",color='darkorange',fontsize=30)

s.set_xlabel("Instance type",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=east.groupby('region').sum().plot(kind="bar",fontsize=12,color='r',legend=False)

s.set_title("AWS east-zone region wise price collection",color='darkorange',fontsize=30)

s.set_xlabel("Region name",color='g',fontsize=20)

s.set_ylabel("Frequency",color='g',fontsize=20)
s=east.groupby('os').sum().plot(kind="bar",fontsize=12,color='g',legend=False)

s.set_title("AWS east-zone price per OS",color='darkorange',fontsize=30)

s.set_xlabel("OS name",color='g',fontsize=20)

s.set_ylabel("Price",color='g',fontsize=20)
d =east["instance_type"].unique()

s1=east.groupby('instance_type')['os'].value_counts()

s1.head()
l=[]

for p in d:

    type_per_os = s1[p]

    l.append(type_per_os.values)

    

df = pd.DataFrame(l,index=d,columns=['Linux/UNIX','Windows'])

s=df.plot(kind="bar",stacked=True,figsize=(12,6),fontsize=12)

s.set_title("AWS east-zone: OS wise instance type distribution",color='g',fontsize=30)

s.set_xlabel("Instance type",color='b',fontsize=20)

s.set_ylabel("Frequency",color='b',fontsize=20)
d =east["region"].unique()

s1=east.groupby('region')['os'].value_counts()

s1.head()
l=[]

for p in d:

    type_per_os = s1[p]

    l.append(type_per_os.values)

    

df = pd.DataFrame(l,index=d,columns=['Linux/UNIX','Windows'])

s=df.plot(kind="bar",stacked=True,figsize=(12,6),fontsize=12)

s.set_title("AWS east-zone: Region wise os distribution",color='g',fontsize=30)

s.set_xlabel("Region name",color='b',fontsize=20)

s.set_ylabel("Frequency",color='b',fontsize=20)
d =east["instance_type"].unique()

s1=east.groupby('instance_type')['region'].value_counts()

s1.head()
l=[]

for p in d:

    type_per_os = s1[p]

    l.append(type_per_os.values)

    

df = pd.DataFrame(l,index=d,columns=['ca-central-1a','ca-central-1b'])

s=df.plot(kind="bar",stacked=True,figsize=(12,6),fontsize=12)

s.set_title("AWS east-zone: Region wise instance type distribution",color='g',fontsize=30)

s.set_xlabel("Region name",color='b',fontsize=20)

s.set_ylabel("Frequency",color='b',fontsize=20)