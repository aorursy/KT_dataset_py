
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/bookserror-correction/books(error_correction).csv",encoding="utf8")
data.head()
plt.figure(figsize=(16,6))
sns.distplot(data.average_rating,rug_kws={"color":"green"},rug=True)

plt.figure(figsize=(16,6))
sns.countplot(data.language_code)
plt.figure(figsize=(16,6))
sns.boxplot(data["# num_pages"])

plt.figure(figsize=(16,6))
sns.pointplot(x="ratings_count",y="text_reviews_count",data=data)
plt.xticks(range(0,7000,700),(600000*i for i in range(10)))

corrArray = data.corr()
plt.figure(figsize=(6,6))
sns.heatmap(corrArray)

plt.figure(figsize=(16,6))
sns.pointplot(x="average_rating",y="# num_pages",data=data)
plt.figure(figsize=(16,16))
sns.pairplot(data)