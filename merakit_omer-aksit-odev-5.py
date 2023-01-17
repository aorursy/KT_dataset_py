# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
#indexing using square brackets

data["age"]
#indexing using square brackets

data["age"][0:2]
#using column attribute and row label

data.age
#using column attribute and row label

data.age[0:2]
#using loc accessor

#selecting only some columns

data.loc[0:2,["age","sex","cp"]]
#difference between series and dataframes

print(data["age"][0:2])   #series

print(type(data["age"][0:2]))

print("\n")

print(data[["age"]][0:2])   #dataframe

print(type(data[["age"]][0:2]))
data.head()
#slicing and indexing series

print(data.loc[10:20,"age":"trestbps"])

print("\n")

print(data.loc[[10,15,20],"age":"trestbps"])

print("\n")

data1=data.loc[10:20,"age":"trestbps"].reset_index(drop=True)

print(data1)
#reverse slicing

data.loc[20:10:-1,"age":"trestbps"]
#from something to end

data.loc[10:15,"trestbps":]
#filtering

print(data[data.age>67])

print("\n")

print(data[data["age"]>67]["age"].mean())
#combining filters

filter1=data.age>65

filter2=data.thalach>160

print(data[filter1 & filter2])
data.info()
#filtering columns based others

print(data["age"][data.thalach>180])
#plain python function

def multi(x):

    return 2*x

data.age.apply(multi)
#lambda function

data.age.apply(lambda x:x*2)
#defining new column using other columns

data["new_column"]=2*data.age

data
#transforming

dict1=dict(data.loc[0:5,"age":"trestbps"])

print(dict1.keys())

print(dict1.values())

print(dict1)
#transforming data

data2=pd.DataFrame(dict1)

print(data2)
data.head()
#finding index name

print(data.index.name)



#let's change it

data.index.name="new_name"

print(data.index.name)

data.head()
data8=data.copy()

data8.index=range(100,403)

data8
#index objects and labeled data

data3=data.set_index("trestbps")

print(data3)
#hierarchical indexing

data4=data.set_index(["trestbps","chol"])

print(data4.index)

print(data4.loc[145,233])
print(data5)
#pivoting data frames

data5=data.head(5)

data5.pivot(index='age',columns='slope',values='target')
dict2={"name":["ali","veli","hilal","ayse"],

       "age":[15,16,17,18],

       "sex":['male','male','female','female'],

       "treatment":['A','B','A','B'],

       "response":['positive','negative','negative','negative']}

data6=pd.DataFrame(dict2)

data6
data6.pivot(index='treatment',columns='sex')
data6.pivot(index='treatment',columns='sex').columns
#stacking and unstacking data frames

data6=data6.set_index(['treatment','sex'])

data6
data6.unstack()
data6.unstack(level=0)
print(data6)

data10=data6.swaplevel()

data10
#melting data frames

dict2={"name":["ali","veli","hilal","ayse"],

       "age":[15,16,17,18],

       "sex":['male','male','female','female'],

       "treatment":['A','B','A','B'],

       "response":['positive','negative','negative','negative']}

data7=pd.DataFrame(dict2)

data7
melted=pd.melt(data7,id_vars='treatment',value_vars=['sex','response'])

melted
data7
#categoricals and groupby

data7.groupby("treatment").mean()
data7.groupby("treatment").max()