import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.Age.fillna(train.Age.median(),inplace=True)
plt.figure(figsize= (10,10))

ax = sns.boxplot(y="Age", x="Survived", data=train, whis=np.inf)

ax = sns.stripplot(y="Age", x="Survived", data=train, color=".3")
survived = train[train['Survived'] == 1]

deaths = train[train['Survived'] == 0]



new = [np.array(survived['Age']), np.array(deaths['Age'])]

categorical = ['Survived','Deaths']



vals, names, xs = [],[],[]



for i, target in enumerate(categorical):

    vals.append(new[i])

    names.append(target)

    xs.append(np.random.normal(i + 1, 0.04, len(new[i])))  # adds jitter to the data points - can be adjusted
plt.figure(figsize=(10,10))

palette = ['b', 'r']

plt.boxplot(vals, labels=names);

for x, val, c in zip(xs, vals, palette):

    plt.scatter(x, val, alpha=0.4, color=c)



plt.show()
##### Set style options here #####

sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"

boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')

flierprops = dict(marker='o', markersize=1,

                  linestyle='none')

whiskerprops = dict(color='#00145A')

capprops = dict(color='#00145A')

medianprops = dict(linewidth=1.5, linestyle='-', color='#01FBEE')



plt.figure(figsize=(10,10))

palette = ['#FF2709', '#09FF10', '#0030D7', '#FA70B5']

plt.boxplot(vals, labels=names, notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=False); 



for x, val, c in zip(xs, vals, palette):

    plt.scatter(x, val, alpha=0.4, color=c)







plt.xlabel("Species", fontweight='normal', fontsize=14)

plt.ylabel("sepal length (cm)", fontweight='normal', fontsize=14)

sns.despine(bottom=True) # removes right and top axis lines

plt.axhline(y=5, color='#ff3300', linestyle='--', linewidth=1, label='Threshold Value')

plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)



plt.show()

new2 = [np.array(survived.Fare),np.array(deaths.Fare)]



vals2, names2, xs2 = [],[],[]



for i, target in enumerate(categorical):

    vals2.append(new2[i])

    names2.append(target)

    xs2.append(np.random.normal(i + 1, 0.04, len(new2[i])))  # adds jitter to the data points - can be adjusted
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

bplot1 = ax1.boxplot(vals, labels=names, notch=False,     showmeans=False)

bplot2 = ax2.boxplot(vals2, labels=names2, notch=False, 

            showmeans=False)



palette = ['#33FF3B', '#3379FF', '#FFD633', '#33FFF1']



for xA, xB, valA, valB, c in zip(xs, xs2, vals, vals2, palette):

    ax1.scatter(xA, valA, alpha=0.4, color=c)

    ax2.scatter(xB, valB, alpha=0.4, color=c)

    

ax1.set_xlabel("Survived", fontweight='normal', fontsize=14)

ax1.set_ylabel("Age", fontweight='normal', fontsize=14)



ax2.set_xlabel("Survived", fontweight='normal', fontsize=14)

ax2.set_ylabel("Fare", fontweight='normal', fontsize=14)

plt.show()
from sklearn import datasets

iris = datasets.load_iris()



x = iris.data

y = iris.target

df = pd.DataFrame(x, columns = iris.feature_names)

df['Species'] = y

df.head()
plt.figure(figsize= (10,10))

ax = sns.boxplot(y="sepal length (cm)", x="Species", data=df, whis=np.inf)

ax = sns.stripplot(y="sepal length (cm)", x="Species", data=df, color=".3")
sepal_0 = df[df['Species'] == 0]

sepal_1 = df[df['Species'] == 1]

sepal_2 = df[df['Species'] == 2]



new = [np.array(sepal_0['sepal length (cm)']),np.array(sepal_1['sepal length (cm)']),np.array(sepal_2['sepal length (cm)'])]

categorical = ['Setosa','Versicolor','Virginica']



vals, names, xs = [],[],[]



for i, target in enumerate(categorical):

    vals.append(new[i])

    names.append(target)

    xs.append(np.random.normal(i + 1, 0.04, len(new[i])))  # adds jitter to the data points - can be adjusted
plt.figure(figsize=(10,10))

palette = ['r', 'g', 'b', 'y']

plt.boxplot(vals, labels=names);

for x, val, c in zip(xs, vals, palette):

    plt.scatter(x, val, alpha=0.4, color=c)



plt.show()
##### Set style options here #####

sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')

flierprops = dict(marker='o', markersize=1,

                  linestyle='none')

whiskerprops = dict(color='#00145A')

capprops = dict(color='#00145A')

medianprops = dict(linewidth=1.5, linestyle='-', color='#01FBEE')



plt.figure(figsize=(10,10))

palette = ['#FF2709', '#09FF10', '#0030D7', '#FA70B5']

plt.boxplot(vals, labels=names, notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=False); 



for x, val, c in zip(xs, vals, palette):

    plt.scatter(x, val, alpha=0.4, color=c)







plt.xlabel("Species", fontweight='normal', fontsize=14)

plt.ylabel("sepal length (cm)", fontweight='normal', fontsize=14)

sns.despine(bottom=True) # removes right and top axis lines

plt.axhline(y=5, color='#ff3300', linestyle='--', linewidth=1, label='Threshold Value')

plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)



plt.show()

new2 = [np.array(sepal_0['sepal width (cm)']),np.array(sepal_1['sepal width (cm)']),np.array(sepal_2['sepal width (cm)'])]



vals2, names2, xs2 = [],[],[]



for i, target in enumerate(categorical):

    vals2.append(new2[i])

    names2.append(target)

    xs2.append(np.random.normal(i + 1, 0.04, len(new2[i])))  # adds jitter to the data points - can be adjusted
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

bplot1 = ax1.boxplot(vals, labels=names, notch=False,     showmeans=False)

bplot2 = ax2.boxplot(vals2, labels=names2, notch=False, 

            showmeans=False)



palette = ['#33FF3B', '#3379FF', '#FFD633', '#33FFF1']



for xA, xB, valA, valB, c in zip(xs, xs2, vals, vals2, palette):

    ax1.scatter(xA, valA, alpha=0.4, color=c)

    ax2.scatter(xB, valB, alpha=0.4, color=c)

    

ax1.set_xlabel("Species", fontweight='normal', fontsize=14)

ax1.set_ylabel("sepal length (cm)", fontweight='normal', fontsize=14)



ax2.set_xlabel("Species", fontweight='normal', fontsize=14)

ax2.set_ylabel("sepal width (cm)", fontweight='normal', fontsize=14)



plt.show()