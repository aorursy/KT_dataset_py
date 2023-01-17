# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/winequality-red.csv")
tail = data.tail(3)
head = data.head(3)
#import data,look up data head and data tail
data.columns
data.info
data.describe()
data.dtypes
#look columns,look data info,describe data info,look data types
sulphates = data["sulphates"]
alcohol = data["alcohol"]
#we access columns
sulphates_filter = sulphates > 0.75
alcohol_filter = alcohol >= 10
filtered_Data = data[sulphates_filter & alcohol_filter]
#filtered data
data.plot(kind="scatter",x = "citric acid",y="pH",alpha=0.5,color="red")
plt.xlabel("citric acid")
plt.ylabel("pH")
plt.title("pH and citric acid")
plt.show()
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(),annot=True,linewidth=4,fmt=".1f",ax=ax)
plt.show()