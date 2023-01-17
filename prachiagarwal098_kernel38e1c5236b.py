# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.head()
df.info()
df.describe()
import warnings

warnings.filterwarnings("ignore")

g=sns.FacetGrid(df,hue="Gender",palette="coolwarm",size=5,aspect=2)

g=g.map(plt.hist,"Annual Income (k$)",bins=20)
sns.boxplot(x="Gender",y="Age",data=df,palette="rainbow")

plt.figure(figsize=(12,8))

sns.countplot(x="Gender",data=df)

sns.heatmap(df.corr(),cmap="coolwarm",annot=True)

plt.figure("correlation")
sns.violinplot(df['Gender'], df['Annual Income (k$)'],data=df, palette = 'muted')

plt.title('Gender vs Spending Score', fontsize = 15)

plt.show()
import seaborn as sns

sns.set(style="ticks")

g = sns.catplot(x="Annual Income (k$)",y="Spending Score (1-100)",col="Gender" ,data=df,aspect=2,height=6,dodge=True)
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=2)
kmeans.fit(df.drop("Gender",axis=1))

kmeans.cluster_centers_
def converter(Gender):

    if Gender=="Female":

        return 1

    else:

        return 0

    
df["cluster"]=df["Gender"].apply(converter)
df.head()
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df["cluster"],kmeans.labels_))

print("\n")

print(classification_report(df["cluster"],kmeans.labels_))