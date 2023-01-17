# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv("../input/procurement-notices.csv")
data.head()
sns.heatmap(data.isna(),cmap = "viridis")
data.info()
data["Notice Type"].unique()
plt.figure(figsize=(15,8))
sns.countplot("Notice Type", data = data)
data["Notice Type"].isna().sum()
for i in data["Notice Type"].unique():
    data[i] = data["Notice Type"].apply(lambda x: True if x == i else False)
dashb = data.groupby("Country Name")[list(data.columns[-5:])].sum()
dashb["TOTAL"] = dashb[data["Notice Type"].unique()[0]]+dashb[data["Notice Type"].unique()[1]]+dashb[data["Notice Type"].unique()[2]]+dashb[data["Notice Type"].unique()[3]]+dashb[data["Notice Type"].unique()[4]]
dashb.head()
dashb = dashb.append(dashb.agg(['sum']))
dashb.tail()
dashboard = dashb.sort_values(by ="TOTAL", ascending = False)
dashboard
