# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

%matplotlib inline



import matplotlib.pyplot as plt



import seaborn as sns #http://ipython.readthedocs.io/en/stable/interactive/plotting.html



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run |or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# get training & test csv files as a DataFrame

train_df = pd.read_csv("../input/train.csv" )

test_df    = pd.read_csv("../input/test.csv")



# preview the data

train_df.head()
train_df.info()

train_df.dtypes
train_df.describe()
aixs1 = plt.subplots(1,1,figsize=(10,5))

sns.countplot(x='Response',data=train_df)
facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")

facet.map(sns.kdeplot,'Ins_Age')

facet.set(xlim=(0, train_df['Ins_Age'].max()))

facet.add_legend()
#큰 의미는 없고, 막대그래프로 분포 확인

fig, axis1 = plt.subplots(1,1,figsize=(15,5))

sns.countplot(x="Ins_Age",data=train_df, ax=axis1)
sns.distplot(train_df["Ins_Age"],bins=10,kde=True)
#employment_info_2,3,5

fig, axis1 = plt.subplots(1,1,figsize=(15,5))

sns.countplot(x='Employment_Info_3', hue="Response", data=train_df, ax=axis1, hue_order=[1,2,3,4,5,6,7,8])


fig, axis1 = plt.subplots(1,1,figsize=(15,5))

sns.countplot(x='Employment_Info_5', hue="Response", data=train_df, ax=axis1, hue_order=[1,2,3,4,5,6,7,8])
g = sns.FacetGrid(train_df, col="Response", col_wrap=2, 

                  size=5, )

g.map(sns.countplot, "Employment_Info_3")
#Ht FacetGrid

facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")

facet.map(sns.kdeplot,'Ht')

facet.set(xlim=(0.4, train_df['Ht'].max())) #x축 범위 조정 

facet.add_legend() #범례
#Wt FacetGrid

facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")

facet.map(sns.kdeplot,'Wt')

facet.set(xlim=(0, train_df['Wt'].max()))

facet.add_legend()
#BMI FacetGrid

facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")

facet.map(sns.kdeplot,'BMI')

facet.set(xlim=(0, 1.0))

facet.add_legend()
#BMI boxplot

ax = sns.boxplot(x="Response", y="BMI", data=train_df, 

                 order=[1,2,3,4,5,6,7,8], palette="RdBu")

ax.set(ylim=(0, 1.1))
#Age ~ BMI

fig, ax = plt.subplots(1,1,figsize=(14,8))

cm = plt.cm.get_cmap('RdBu')

ax = plt.scatter(train_df["Ins_Age"], train_df["BMI"], c=train_df["Response"], alpha=0.5, cmap=cm)

plt.xlim=(0, 1.0)

plt.ylim=(0, 1.0)



plt.title("Age vs BMI")

plt.xlabel("Age")

plt.ylabel("BMI")



cbar=plt.colorbar(ax)

cbar.ax.set_ylabel('Response', rotation=270)
fig, axis1 = plt.subplots(1,1,figsize=(20,5))

sns.countplot(x='InsuredInfo_6', hue="Response", data=train_df, 

              ax=axis1, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet = sns.FacetGrid(train_df, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")

facet.map(sns.kdeplot,'Insurance_History_5')

facet.set(xlim=(0, 0.01))

facet.add_legend()