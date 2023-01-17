import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

print(train.shape)
train.head(5)
train.dtypes
train.describe()
print(train.isnull().sum())
#The distribution of "target"

plt.figure(figsize = (5, 5))

ax = sns.countplot(x="target", data=train, palette="Set2")

ax.set_title('Target Distribution', weight = 'bold',fontsize = 15)

plt.xlabel('Target', fontsize = 12)

plt.ylabel('Count', fontsize = 12)
fig, ax = plt.subplots(3,2,figsize = (15,20))

for i in range(5): 

    sns.countplot(f'bin_{i}', data= train,ax = ax[i//2][i%2])

    ax[i//2][i%2].set_ylim([0, 550000])

fig.suptitle("Binary Feature Distribution", fontsize=20, weight = 'bold')

plt.show()
fig, ax = plt.subplots(3,2, figsize=(15, 20))

for i in range(5): 

    sns.countplot(f'nom_{i}', data= train, ax=ax[i//2][i%2],

                 order=train[f'nom_{i}'].value_counts().index)

    ax[i//2][i%2].set_ylim([0, 350000])

fig.suptitle("Nominal Feature Distribution", fontsize = 20, weight = 'bold')

plt.show()
fig, ax = plt.subplots(3,2, figsize=(15, 20))

for i in range(5): 

    sns.countplot(f'ord_{i}', data= train, ax=ax[i//2][i%2],

                 order=train[f'ord_{i}'].value_counts().index)

    ax[i//2][i%2].set_ylim([0, 250000])

fig.suptitle("Ordinal Feature Distribution", fontsize = 20, weight = 'bold')

plt.show()
fig, ax = plt.subplots(2,1, figsize=(24, 16))



sns.countplot('day', hue='target', data= train, ax=ax[0], palette="Set2")

ax[0].set_ylim([0, 100000])



sns.countplot('month', hue='target', data= train, ax=ax[1], palette="Set3")

ax[1].set_ylim([0, 70000])



fig.suptitle("Time Distribution", fontsize=20, weight = 'bold')

plt.show()
train.groupby('bin_3').count()
train.groupby('bin_4').count()
train['bin_3']=np.where(train['bin_3'] =='F', '0', train['bin_3'])

train['bin_3']=np.where(train['bin_3'] =='T', '1', train['bin_3'])



train['bin_4']=np.where(train['bin_4'] =='N', '0', train['bin_4'])

train['bin_4']=np.where(train['bin_4'] =='Y', '1', train['bin_4'])



bin = train[['id','bin_0','bin_1','bin_2','bin_3','bin_4','target']]

bin["bin_3"]= bin["bin_3"].astype(float) 

bin["bin_4"]= bin["bin_4"].astype(float) 

bin.head(5)
bin = bin.interpolate()

print(bin.isnull().sum())



correlation_b = bin.corr()

plt.figure(figsize = (10, 10))

sns.set(font_scale = 1.5)

sns.heatmap(correlation_b, annot = True, annot_kws = {'size': 10}, cmap = 'Blues')
train.groupby('nom_0').count()
train['nom_0']=np.where(train['nom_0'] =='Blue', '1', train['nom_0'])

train['nom_0']=np.where(train['nom_0'] =='Green', '2', train['nom_0'])

train['nom_0']=np.where(train['nom_0'] =='Red', '3', train['nom_0'])
train.groupby('nom_1').count()
train['nom_1']=np.where(train['nom_1'] =='Circle', '1', train['nom_1'])

train['nom_1']=np.where(train['nom_1'] =='Polygon', '2', train['nom_1'])

train['nom_1']=np.where(train['nom_1'] =='Square', '3', train['nom_1'])

train['nom_1']=np.where(train['nom_1'] =='Star', '4', train['nom_1'])

train['nom_1']=np.where(train['nom_1'] =='Trapezoid', '5', train['nom_1'])

train['nom_1']=np.where(train['nom_1'] =='Triangle', '6', train['nom_1'])
train.groupby('nom_2').count()
train['nom_2']=np.where(train['nom_2'] =='Axolotl', '1', train['nom_2'])

train['nom_2']=np.where(train['nom_2'] =='Cat', '2', train['nom_2'])

train['nom_2']=np.where(train['nom_2'] =='Dog', '3', train['nom_2'])

train['nom_2']=np.where(train['nom_2'] =='Hamster', '4', train['nom_2'])

train['nom_2']=np.where(train['nom_2'] =='Lion', '5', train['nom_2'])

train['nom_2']=np.where(train['nom_2'] =='Snake', '6', train['nom_2'])
train.groupby('nom_3').count()
train['nom_3']=np.where(train['nom_3'] =='Canada', '1', train['nom_3'])

train['nom_3']=np.where(train['nom_3'] =='China', '2', train['nom_3'])

train['nom_3']=np.where(train['nom_3'] =='Costa Rica', '3', train['nom_3'])

train['nom_3']=np.where(train['nom_3'] =='Finland', '4', train['nom_3'])

train['nom_3']=np.where(train['nom_3'] =='India', '5', train['nom_3'])

train['nom_3']=np.where(train['nom_3'] =='Russia', '6', train['nom_3'])
train.groupby('nom_4').count()
train['nom_4']=np.where(train['nom_4'] =='Bassoon', '1', train['nom_4'])

train['nom_4']=np.where(train['nom_4'] =='Oboe', '2', train['nom_4'])

train['nom_4']=np.where(train['nom_4'] =='Piano', '3', train['nom_4'])

train['nom_4']=np.where(train['nom_4'] =='Theremin', '4', train['nom_4'])
nom = train[['id','nom_0','nom_1','nom_2','nom_3','nom_4','target']]

nom["nom_0"]= nom["nom_0"].astype(float) 

nom["nom_1"]= nom["nom_1"].astype(float) 

nom["nom_2"]= nom["nom_2"].astype(float) 

nom["nom_3"]= nom["nom_3"].astype(float) 

nom["nom_4"]= nom["nom_4"].astype(float)

nom.head(5)
nom = nom.interpolate()

print(nom.isnull().sum())
correlation_n = nom.corr()

plt.figure(figsize = (10, 10))

sns.set(font_scale = 1.5)

sns.heatmap(correlation_n, annot = True, annot_kws = {'size': 10}, cmap = 'Reds')
train.groupby('ord_2').count()
train['ord_2']=np.where(train['ord_2'] =='Boiling Hot', '1', train['ord_2'])

train['ord_2']=np.where(train['ord_2'] =='Cold', '2', train['ord_2'])

train['ord_2']=np.where(train['ord_2'] =='Freezing', '3', train['ord_2'])

train['ord_2']=np.where(train['ord_2'] =='Hot', '4', train['ord_2'])

train['ord_2']=np.where(train['ord_2'] =='Lava Hot', '5', train['ord_2'])

train['ord_2']=np.where(train['ord_2'] =='Warm', '6', train['ord_2'])
train.groupby('ord_3').count()
train['ord_3']=np.where(train['ord_3'] =='a', '1', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='b', '2', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='c', '3', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='d', '4', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='e', '5', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='f', '6', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='g', '7', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='h', '8', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='i', '9', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='j', '10', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='k', '11', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='l', '12', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='m', '13', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='n', '14', train['ord_3'])

train['ord_3']=np.where(train['ord_3'] =='o', '15', train['ord_3'])
ord = train[['id','ord_0','ord_2','ord_3','month','day','target']]

ord["ord_0"]= ord["ord_0"].astype(float) 

ord["ord_2"]= ord["ord_2"].astype(float) 

ord["ord_3"]= ord["ord_3"].astype(float)

ord.head(5)
ord = ord.interpolate()

print(ord.isnull().sum())
correlation_o = ord.corr()

plt.figure(figsize = (10, 10))

sns.set(font_scale = 1.5)

sns.heatmap(correlation_o, annot = True, annot_kws = {'size': 10}, cmap = 'Greens')