import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
data_path = '/kaggle/input/abalone-dataset/abalone.csv'

raw_df = pd.read_csv(data_path)
raw_df.head()
print(raw_df.info())
raw_df['Age'] = raw_df['Rings']+1.5

df = raw_df #just a copy to be safe

df.columns
val_labl = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',]

trgt_labl = ['Rings', 'Age']

catg_lbl = ['Sex']
df.info()
def plot_graph(var, color='r'):

    plt.figure(figsize=(10,5))

    plt.scatter(x = df[var], y = df['Age'], marker='+', c=color, s=60, alpha=0.7)

    plt.ylabel('Age of Abalone')

    plt.xlabel(var+' of Abalone')

    plt.grid(True)
def data_removal(var, para1, para2):

    df.drop(df[(df[var]> para1)].index, inplace=True)

    df.drop(df[(df[var]< para2)].index, inplace=True)
plot_graph(val_labl[0])
data_removal(val_labl[0], 0.8, 0.1)

df.drop(df[(df[val_labl[0]]==0.8)].index, inplace=True)
plot_graph(val_labl[1])
data_removal(val_labl[1],0.65,0.1)
plot_graph(val_labl[2], 'c')
data_removal(val_labl[2],0.4,0)
plot_graph(val_labl[2], 'c')
data_removal(val_labl[2],0.25,0.01)
plot_graph(val_labl[3], 'b')
data_removal(val_labl[3],2.52,0)
plot_graph(val_labl[4])
data_removal(val_labl[4],1.20,0)
plot_graph(val_labl[5])
data_removal(val_labl[5],0.55,0)
plot_graph(val_labl[6], 'g')
data_removal(val_labl[6],0.75,0)
df[df['Age'] >= 25]
data_removal('Age',25,0)
fig = plt.figure(figsize=[15,10])

axes = fig.subplots(2, 4)

k = 0

for i in range(2):

    for j in range(4):

        if k == 7: break

        axes[i][j].scatter(df[val_labl[k]], df['Age'], alpha=0.7, marker='+',s= 40)

        axes[i][j].set(xlabel=val_labl[k]+' of Abalone', ylabel='Age of Abalone')

        k+=1
df.describe()
f,ax = plt.subplots(figsize=(10, 8))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
sns.pairplot(df[df.select_dtypes(include=[np.number]).columns])
df.skew(axis = 0, skipna = True) 
fig = plt.figure(figsize=[15,5])

axes = fig.subplots(1, 2)



axes[0].axvline(df['Height'].mean(), ls='--', color='r')

axes[1].axvline(df['Age'].mean(), ls='--', color='r')



sns.distplot(df['Height'], bins=50, ax=axes[0], color='c')

sns.distplot(df['Age'], bins=20, ax=axes[1], color='c')

fig = plt.figure(figsize=[20,5])

axes = fig.subplots(1, 2)

axes[0].pie(df.groupby(['Sex']).size(), explode=(0,0.1,0), colors=['gold', 'r', 'c'], shadow=True, autopct='%1.1f%%', labels=['Immature','Female','Male'])

sns.violinplot(x = 'Sex', y = 'Age', data = df, ax=axes[1], colors=['c', 'r', ''])

sns.swarmplot(x = 'Sex', y = 'Age', data = df, hue = 'Sex', ax=axes[1])
X = df[val_labl]

y = df['Age']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.7)
lr_model = LinearRegression()

clf = lr_model.fit(X_train, y_train)
print("LinearRegression Train Score: ",  clf.score(X_train, y_train))

print("LinearRegression Test Score: ",  clf.score(X_test, y_test))