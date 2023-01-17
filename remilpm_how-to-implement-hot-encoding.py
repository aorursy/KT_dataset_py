import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

Data1=pd.read_csv("../input/Titanic.csv")

Data1.head()

#Dropping Name, ticket and Cabin as there are so many values to be categorised

Data2=Data1.drop("Name",axis=1)

Data2=Data2.drop("Ticket",axis=1)

Data2=Data2.drop("Cabin",axis=1)

Data2.head()

 #After performing hot encoding 

Data3 = pd.get_dummies(Data2)

Data3.head()