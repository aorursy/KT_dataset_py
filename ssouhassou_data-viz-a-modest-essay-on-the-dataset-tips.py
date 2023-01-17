# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tips = pd.read_csv("/kaggle/input/seaborn-tips-dataset/tips.csv")

tips.head()
tips.shape
tips.info()
tips.describe()
tips.describe(include="all")
tips.isna().sum()
tips.isnull().sum()
corr = tips.corr()

corr
import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
plt.figure(figsize=(8,8))

sns.heatmap(corr, annot=True, )

plt.title("HeatMap of correlation matrix of tips dataset")



plt.show()
plt.figure(figsize=(16,14))



plt.subplot(321)

sns.barplot(tips.sex, tips.total_bill)

plt.title("Barplot of Total Bill Vs Sexe")



plt.subplot(322)

sns.barplot(tips.smoker, tips.total_bill)

plt.title("Barplot of Total Bill Vs Smoker")



plt.subplot(323)

sns.barplot(tips.sex, tips.total_bill, hue=tips.smoker)

plt.title("Barplot of Total Bill Vs Sexe and Smoker")



plt.subplot(324)

sns.barplot(tips.day, tips.total_bill, order=["Thur", "Fri", "Sat", "Sun"])

plt.title("Barplot of Total Bill Vs Days")



plt.subplot(313)

sns.barplot(tips.day, tips.total_bill, order=["Thur", "Fri", "Sat", "Sun"], hue=tips.sex)

plt.title("Barplot of Total Bill Vs Days")



plt.tight_layout()

plt.show()
plt.figure(figsize=(16,6))



plt.subplot(121)

sns.distplot(tips.total_bill, kde=True)

plt.title("Distplot Total Bill (kde=True)")



plt.subplot(122)

sns.distplot(tips.total_bill, kde=False)

plt.title("Distplot Total Bill (kde=False)")





plt.show()
sns.jointplot(data=tips, x="total_bill", y="tip", height=7)



plt.show()
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg", height=7)



plt.show()
sns.jointplot(data=tips, x="total_bill", y="tip", kind="resid", height=7)



plt.show()
sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde", height=7)



plt.show()
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex", height=7)



plt.show()
sns.pairplot(tips)



plt.show()
sns.pairplot(tips, hue="sex")



plt.show()
sns.pairplot(tips, hue="smoker")



plt.show()
sns.pairplot(tips, hue="day", hue_order=["Thur", "Fri", "Sat", "Sun"])



plt.show()
num_tips = tips.select_dtypes(exclude="object").copy()

num_tips.head()
sns.clustermap(num_tips, metric="correlation")



plt.show()
sns.clustermap(num_tips, standard_scale=1)



plt.show()
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(15,10))



plt.subplot(211)

dendrogram(linkage(num_tips, method='ward'))



plt.subplot(212)

dendrogram(linkage(num_tips, method='single'))



plt.tight_layout()

plt.show()