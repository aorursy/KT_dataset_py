#1 - Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
print(os.listdir("../input"))
#2 - Importing data and taking basic information
# We need to clean data
data = pd.read_csv("../input/countries of the world.csv")
data.info()
data.head()
#3 Cleaning Data
data.columns = [re.sub(r"[\)\(%./]","",each).lower().strip() for each in data.columns]
data.columns = [re.sub(r"[ ]","_",each) for each in data.columns]
data.region = [re.sub(r"[\)\(%./]","",each).lower().strip() for each in data.region]
data.region = [re.sub(r"[ ]","_",each) for each in data.region]
data.country = [re.sub(r"[\)\(%./]","",each).lower().strip() for each in data.country]
data.country = [re.sub(r"[ ]","_",each) for each in data.country]

data.head()

#4 - Prepearing for changing types
for each in data.loc["population":]:
    data["{}".format(each)] = [each.replace(",",".") 
                               if type(each) == str 
                               else each for each in data["{}".format(each)]]
#5 - Changing types
lis = list(data.columns)
for each in lis[2:]:
    data["{}".format(each)] = data["{}".format(each)].astype('float')
data.info()

data.head()
data.corr()
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(), annot = True, fmt = ".1f", linewidth = .5)
