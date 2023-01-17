from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Import data

df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')

df.head()
df.info()
#Initialize an empty array to collect the sum of missing values per column

mvs = []



#Count the missing values

for x in df.columns:

    mvs.append(df[x].isin(["?"]).sum())



#Build the plot

fig, ax =  plt.subplots(figsize=(10,3))

index   = np.arange(df.shape[1])



ax.bar(index, mvs, alpha = 0.4, color = 'g')

ax.set_ylabel('Missing Values')

ax.set_xticks(index)

ax.set_xticklabels((df.columns))



fig.tight_layout()



plt.xticks(rotation=45)

plt.show()
#Only three features contain missing values

#To see if the missing values have an significant effect on the dataset

#Visualize the missing values compared with the complete dataset to see the effect



#Build the plot

yvalues = [df.shape[0], mvs[1], mvs[6], mvs[13]]



fig, ax = plt.subplots()

index   = np.arange(4)



ax.bar(index, yvalues, alpha = 0.4, color = 'y')

ax.set_ylabel('Data count')

ax.set_xticks(index)

ax.set_xticklabels(('dataset size','workclass','occupation','native.country'))



fig.tight_layout()



plt.xticks(rotation=45)

plt.show()
df = df.replace('?', np.NaN)

df.head()
hmap = df.corr()

plt.subplots(figsize=(12, 9))

sns.heatmap(hmap, vmax=.8,annot=True, square=True)
#Before we can begin to model are dataset, we first have to drop any categorical data and convert the one's we want to keep into binary:: Yes (1) or No (0)

df["marital.status"] = df["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')

df["marital.status"] = df["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')

df["marital.status"] = df["marital.status"].map({"Married":0, "Single":1})

df["marital.status"] = df["marital.status"]

df['income']=df['income'].map({'<=50K': 0, '>50K': 1})

df.drop(labels=["sex","workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)



df.head()
plt.figure(figsize=(7,5))

sns.countplot(df["income"])

plt.xlabel("İncome Case",fontsize=15)

plt.ylabel("Count",fontsize=15)

print(">50K  rate : %{:.2f}".format(sum(df["income"])/len(df["income"])*100))

print("<=50K rate : %{:.2f}".format((len(df["income"])-sum(df["income"]))/len(df["income"])*100))
from sklearn.utils import resample

positive = df[df.income==1]

negative = df[df.income==0]



positive_increase = resample(positive,

                              replace = True,

                              n_samples = len(negative),

                              random_state = 111)

increase_df = pd.concat([negative,positive_increase])

increase_df.income.value_counts()
x = df.iloc[ : ,:-1]

y = df[['income']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)