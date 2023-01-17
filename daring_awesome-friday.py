# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/BlackFriday.csv")
data.info()
data.describe()
data.columns
data.head(15)
data.corr()
f,ax = plt.subplots( figsize = (15,15))
sns.heatmap( data.corr(), annot = True , linewidth = 1, fmt = (".1f"), ax=ax)
plt.show()


#average purchase = ort_purchase
ort_purchase = sum(data.Purchase)/len(data.Purchase)
print("ort purchase: ",ort_purchase)
data["ort_purchase"] = [ "High" if i > ort_purchase else "low" for i in data.Purchase ]
data.loc[:15 , ("ort_purchase","Age" , "Purchase")]