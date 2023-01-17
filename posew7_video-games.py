# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
vgsales = pd.read_csv("../input/vgsales.csv")
vgsales.info()
vgsales.head(11)
vgsales.Platform.unique()
vgsales.Platform.value_counts()
data_platform_unique = list(vgsales.Platform.unique())
data_globalsales_ratio = []
for i in data_platform_unique:
    x = vgsales[vgsales.Platform == i]
    rate = sum(x.Global_Sales)/len(x)
    data_globalsales_ratio.append(rate)
dataframe = pd.DataFrame({"Platform":data_platform_unique, "Global_Sales":data_globalsales_ratio})
new_index = dataframe["Global_Sales"].sort_values(ascending=False).index.values
redata = dataframe.reindex(new_index)

plt.figure(figsize=(23,11))
sns.barplot(x=redata.Platform, y=redata.Global_Sales)
plt.title("Global Sales of Platforms")
plt.show()
vgsales.Publisher.value_counts()
publisher_list = list(vgsales.Publisher)
publisher_count = Counter(publisher_list)
most_common = publisher_count.most_common(30)
x,y = zip(*most_common)
x,y = list(x), list(y)

plt.figure(figsize=(23,15))
sns.barplot(x=y,y=x,palette=sns.cubehelix_palette(len(x)))
plt.title("Global Sales of Publishers")
plt.show()
genre_unique = list(vgsales.Genre.unique())
ratio_sales = []
for i in genre_unique:
    x = vgsales[vgsales.Genre == i]
    avarage = sum(x.Global_Sales)/len(x)
    ratio_sales.append(avarage)
dic = pd.DataFrame({"Genre":genre_unique,"Global_Sales":ratio_sales})
new_index = dic["Global_Sales"].sort_values(ascending=False).index.values
redic = dic.reindex(new_index)

plt.figure(figsize=(23,11))
sns.barplot(x=redic.Genre, y=redic.Global_Sales)
plt.title("Global Sales of Genre")
plt.show()
vgsales.head()
data1 = vgsales.loc[:500,["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]]
data1.plot(subplots=True,  figsize=(23,13))
plt.show()
vgsales.Year.plot(kind="hist", bins=30, figsize=(21,9))
plt.show()
vgsales.plot(kind="scatter", x="NA_Sales",y="EU_Sales", color="g",figsize=(21,9), alpha=0.3)
plt.show()













