# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/flavors_of_cacao.csv")
# Any results you write to the current directory are saved as output.
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
data.head()
data["Cocoa\nPercent"] = data["Cocoa\nPercent"].apply(lambda x: float(float(x.split("%")[0])/100))
data.head()
data.columns[data.isnull().any()].tolist()
sns.heatmap(data.corr(), annot=True)
data.columns
# data[data.columns[8]].unique().tolist()
plt.figure(figsize=(8,6))
sns.distplot(data['Rating'],bins=5,color='red')
plt.figure(figsize=(8,6))
sns.distplot(data['Cocoa\nPercent'],bins=5,color='red')
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x = data['Rating'], y = data['Cocoa\nPercent'], s=15)
plt.ylabel('Cocoa\nPercent', fontsize=13)
plt.xlabel('Rating', fontsize=13)
plt.show()
data[data["Rating"] == 5.0]
data[data[data.columns[0]] == "Amedei"]
data[data[data.columns[1]].isin(["Chuao", "Toscano Black"])]
data_bean_type = data[data[data.columns[7]].isin(["Trinitario", "Blend", "Criollo"])]
data_bean_type[data_bean_type["Rating"] < 2.50]
data[data[data.columns[0]] == "Hotel Chocolat"]
data[data[data.columns[4]] > 0.9]