# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats











heart = pd.read_csv('../input/heart.csv')

sns.set()

# getting the shape

heart.info()

heart.head(10)

heart.describe()

heart.shape

heart.dtypes


heart.isnull().sum()
sns.countplot(x="target", data=heart, palette="bwr")

plt.title('Distibution Patient and Non-Patient')

plt.show()
heart.sex[heart.sex == 1] = 'male'

heart.sex[heart.sex == 0] = 'female'



plt.title('Distibution by age: blue=heart disease')

heart[heart.target==1].age.hist(bins=20);

heart[heart.target==0].age.hist(bins=20);
heart[heart.sex==1].age.hist(bins=10);

heart[heart.sex==0].age.hist(bins=10);
heart.hist(column=["trestbps", "age","oldpeak","chol"],bins=40)  # Plot specific columns





fig, ax = plt.subplots(2,2, figsize=(32,32))

sns.distplot(heart.age, bins = 20, ax=ax[0,0]) 

sns.distplot(heart.oldpeak, bins = 20, ax=ax[0,1]) 

sns.distplot(heart.trestbps, bins = 20, ax=ax[1,0]) 

sns.distplot(heart.chol, bins = 20, ax=ax[1,1]) 







 
# Add labels

plt.title('Correlation between Oldpeak and Cholesteral')

plt.xlabel('OldPeak')

plt.ylabel('CHOL')

# Custom the histogram:

sns.jointplot(x=heart["oldpeak"], y=heart["chol"], kind='hex', marginal_kws=dict(bins=5, rug=True))



#hset.boxplot()

# Use a color palette

#hset.boxplot( x=hset["target"], y=hset["sex"])

#sns.plt.show()

sns.boxplot(y='trestbps', x='sex', 

                 data=heart, 

                 palette="colorblind",

                 hue='target')



sns.violinplot(y='trestbps', x='sex', 

                 data=heart, 

                 palette="colorblind",

                 hue='target',

                 linewidth=1,

                 width=0.5)



sns.countplot(heart['target'],label="Count Distribution")
sns.catplot(x="sex", y="thalach", hue="target", inner="quart", kind="violin", split=True, data=heart)



sns.catplot(x="sex", y="age", hue="target", inner="quart", kind="violin", split=True, data=heart)



sns.catplot(x="sex", y="chol", hue="target", inner="quart", kind="violin", split=True, data=heart)



sns.catplot(x="sex", y="trestbps", hue="target", inner="quart", kind="violin", split=True, data=heart)



sns.catplot(x="sex", y="cp", hue="target", inner="quart", kind="violin", split=True, data=heart)




from sklearn.model_selection import train_test_split

# Split our data



features = heart[heart.columns[0:13]]

target = heart['target']

#features_train, features_test, target_train, target_test = train_test_split(features,

                                                                           # target, test_size = 0.25, random_state = 10)

train, test = train_test_split(heart, test_size=0.25)



# getting the shape

train.head(10)

train.describe()

train.shape

train.dtypes
#corelation matrix

plt.figure(figsize=(11,7))

sns.heatmap(cbar=False,annot=True,data=heart.corr()*100,cmap='coolwarm')

plt.title('% Corelation Matrix')

plt.show()
#boxplot of 

plt.figure(figsize=(10,6))

sns.boxplot(data=heart,x='slope',y='thalach',palette='viridis')

plt.plot()
plt.figure(figsize=(10,6))

sns.boxplot(data=heart,x='cp',y='chol',palette='viridis')

plt.plot()
plt.figure(figsize=(10,6))

sns.boxplot(data=heart,x='target',y='chol',palette='viridis')

plt.plot()
# basic plot

plt.xlabel(ax.set_xlabel(), rotation=90)

heart.boxplot()





# Draw a graph with pandas and keep what's returned

ax = df.plot(kind='boxplot', x='target', y='chol')



# Set the x scale because otherwise it goes into weird negative numbers

ax.set_xlim((0, 1000))



# Set the x-axis label

ax.set_xlabel("Target")



# Set the y-axis label

ax.set_ylabel("Cholesterol")
p.xaxis.major_label_orientation = "vertical"

heart.plot(kind='bar',alpha=0.75)
