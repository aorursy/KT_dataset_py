

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
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data_south = pd.read_csv("../input/ap-south-1.csv",sep=",")

data_south.fillna(0)

data_south.head()
data_central = pd.read_csv("../input/ca-central-1.csv",sep=",")

data_central.fillna(0)

data_central.head()
data_central = pd.read_csv("../input/ca-central-1.csv",sep=",")

data_central.fillna(0)

data_central.head()
data_east = pd.read_csv("../input/us-east-1.csv",sep=",")

data_east.fillna(0)

data_east.head()
data_west = pd.read_csv("../input/us-west-1.csv",sep=",")

data_west.fillna(0)

data_west.head()

#os usage in east region

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('os usage in east region')

sns.countplot(data_east['os'])
#os usage in west region

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('os usage in west region')

sns.countplot(data_west['os'])
#os usage in central region

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('os usage in west region')

sns.countplot(data_central['os'])
#os usage in south region

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('os usage in south region')

sns.countplot(data_south['os'])
#instance type in east region

data=data_east.groupby('instance_type').size().plot("bar",figsize=(12,6),fontsize=12)

data.set_title("east region instance type",color='b',fontsize=30)

data.set_xlabel("Instance type",color='g',fontsize=20)

data.set_ylabel("count",color='g',fontsize=20)
#instance type in west region

data=data_west.groupby('instance_type').size().plot("bar",figsize=(12,6),fontsize=12)

data.set_title("west region instance type",color='b',fontsize=30)

data.set_xlabel("Instance type",color='g',fontsize=20)

data.set_ylabel("count",color='g',fontsize=20)
#instance type in south region

data=data_south.groupby('instance_type').size().plot("bar",figsize=(12,6),fontsize=12)

data.set_title("south region instance type",color='b',fontsize=30)

data.set_xlabel("Instance type",color='g',fontsize=20)

data.set_ylabel("count",color='g',fontsize=20)
#instance type in central region

data=data_central.groupby('instance_type').size().plot("bar",figsize=(12,6),fontsize=12)

data.set_title("south region instance type",color='b',fontsize=30)

data.set_xlabel("Instance type",color='g',fontsize=20)

data.set_ylabel("count",color='g',fontsize=20)
#central region usage details

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('central region usage details')

sns.countplot(data_central['region'])
#west region usage details

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('west region usage details')

sns.countplot(data_west['region'])
#west region usage details

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('south region usage details')

sns.countplot(data_south['region'])
#east region usage details

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('east region usage details')

sns.countplot(data_east['region'])
inst_type =data_west["instance_type"].unique()

data=data_west.groupby('instance_type')['region'].value_counts()

df=[]

for dat in inst_type:

    data_os = data[dat]

    df.append(data_os.values)

    

df = pd.DataFrame(df,index=inst_type,columns=['Linux/UNIX','Windows'])

s=df.plot(kind="bar",stacked=True,figsize=(12,6),fontsize=12)

s.set_title("West Region wise os distribution",color='g',fontsize=30)

s.set_xlabel("Region name",color='b',fontsize=20)

s.set_ylabel("Frequency",color='b',fontsize=20)