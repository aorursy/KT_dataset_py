import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt

import time



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/data.csv')
data.head()
# feature names as a list

col = data.columns       # .columns gives columns names in data 

print(col)
# y includes our labels and x includes our features

y = data.diagnosis                          # M or B 

list = ['Unnamed: 32','id','diagnosis']

x = data.drop(list,axis = 1 )

x.head()
ax = sns.countplot(y,label="Count")

B, M = y.value_counts()

print('Number of Benign: ',B)

print('Number of Malignant : ',M)
x.describe()
# first ten features

data_dia = y

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=45);
# Second ten features

data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=45);
# Third ten features

data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=45);
# As an alternative of violin plot, box plot can be used

# box plots are also useful in terms of seeing outliers

# I do not visualize all features with box plot

# In order to show you lets have an example of box plot

# If you want, you can visualize other features as well.

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=45);
sns.jointplot(x.loc[:,'concavity_worst'],

              x.loc[:,'concave points_worst'],

              kind="regg",

              color="#ce1414");
sns.set(style="white")

df = x.loc[:,['radius_worst','perimeter_worst','area_worst']]

g = sns.PairGrid(df, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3);
sns.set(style="whitegrid", palette="muted")

data_dia = y

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=45);
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=45);
data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=45);
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);