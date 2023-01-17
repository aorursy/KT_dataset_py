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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')
df = pd.read_csv("../input/Mall_Customers.csv")

df.head()
print("Shape:",df.shape)

print(df.describe())
#Check null value(if any)

df.isnull().any()
#Divide age into groups

df["Age_Group"] = np.where(df.Age<25,"Age below 25 year",(np.where(df.Age<35, "Age 25 to 34 year" ,

                                                                   (np.where(df.Age<45, "Age 35 to 44 year" ,

                                                                             (np.where(df.Age<55, "Age 45 to 54 year" ,"Age more then 55 year")))))))



df.head()
plt.rcParams["figure.figsize"] =(18,6)



plt.subplot(221)

sns.countplot(df.Gender)



plt.subplot(222)

sns.countplot(df.Age)



plt.subplot(223)

sns.countplot(df["Annual Income (k$)"])



plt.subplot(224)

sns.countplot(df["Spending Score (1-100)"])
plt.rcParams['figure.figsize'] = (20,10)

plt.subplot(221)

sns.scatterplot(x="Age",y="Annual Income (k$)",data=df)



plt.subplot(222)

sns.scatterplot(x="Age",y="Spending Score (1-100)",data=df)



plt.subplot(223)

sns.scatterplot(x="Age",y="Annual Income (k$)",data=df,hue="Gender")



plt.subplot(224)

sns.scatterplot(x="Age",y="Spending Score (1-100)",data=df,hue="Gender")
#residplot

plt.rcParams['figure.figsize'] = (14,6)



plt.subplot(121)

sns.residplot(y="Annual Income (k$)", x="Age",data=df, lowess=True, color="g")



plt.subplot(122)

sns.residplot(y="Spending Score (1-100)", x="Age",data=df, lowess=True, color="b")
plt.rcParams['figure.figsize'] = (20,10)



plt.subplot(221)

sns.violinplot(x="Age_Group",y="Annual Income (k$)",data=df)



plt.subplot(222)

sns.violinplot(x="Age_Group",y="Spending Score (1-100)",data=df)



plt.subplot(223)

sns.violinplot(x="Age_Group",y="Annual Income (k$)",data=df,hue="Gender")



plt.subplot(224)

sns.violinplot(x="Age_Group",y="Spending Score (1-100)",data=df,hue="Gender")
# Check how Age wise Data is distributed

plt.rcParams['figure.figsize'] = (20,10)



plt.subplot(221)

sns.boxplot(x="Age_Group",y="Annual Income (k$)",data=df)

sns.swarmplot(x="Age_Group", y="Annual Income (k$)", data=df, color=".25")



plt.subplot(222)

sns.boxplot(x="Age_Group",y="Spending Score (1-100)",data=df)

sns.swarmplot(x="Age_Group", y="Spending Score (1-100)", data=df, color=".25")



plt.subplot(223)

sns.boxplot(x="Age_Group",y="Annual Income (k$)",data=df,hue="Gender")



plt.subplot(224)

sns.boxplot(x="Age_Group",y="Spending Score (1-100)",data=df,hue="Gender")
#plt.rcParams['figure.figsize'] = (30,30)

sns.pairplot(df.drop("CustomerID",axis=1), hue="Age_Group",size=3)
#corelation matrix



corr = df.drop("CustomerID",axis=1).corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(6, 10))

cmap = sns.diverging_palette(30, 0, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,square=True, linewidths=.5,linecolor='yellow', cbar_kws={"shrink": .4})

# there is Negative weak corelatin between Age and Spending
sns.heatmap(df.drop("CustomerID",axis=1).corr(),annot=True)

plt.title("Heatmap of Data",fontsize=20)