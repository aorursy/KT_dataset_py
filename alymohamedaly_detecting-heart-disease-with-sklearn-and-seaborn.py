import numpy as np

import pandas as pd
df = pd.read_csv("../input/cardio_train.csv", sep=';', index_col=0)

for col in df.columns:

    df.rename(columns={col:col.capitalize()}, inplace=True)   #I just like it capitalized

df.head()
df.describe()
df.isnull().sum()
from matplotlib import pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(15,7))

counts, bins, patches = ax.hist(df["Age"], bins=15, edgecolor='g')

ax.set_xticks(bins)

ax.set_xlim(left=13381)



bin_centers = 0.5 * np.diff(bins[4:]) + bins[4:-1]

for count, x in zip(counts[4:], bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    

plt.title("Age")

ax.grid(axis='y')
fig, ax = plt.subplots(figsize=(15,7))

counts, bins, patches = ax.hist(df["Height"], bins=15, edgecolor='g')

ax.set_xticks(bins)

ax.set_xlim(left=107, right=211)



bin_centers = 0.5 * np.diff(bins[5:-4]) + bins[5:-5]

for count, x in zip(counts[5:-4], bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    

plt.title("Height")

ax.grid(axis='y')
fig, ax = plt.subplots(figsize=(15,7))

counts, bins, patches = ax.hist(df["Weight"], bins=15, edgecolor='g')

ax.set_xticks(bins)

ax.set_xlim(left=22.7, right=187.3)



bin_centers = 0.5 * np.diff(bins[2:-2]) + bins[2:-3]

for count, x in zip(counts[2:-2], bin_centers):

    percent = '%0.0f%%' % (100 * float(count) / counts.sum())

    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),

        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    

plt.title("Weight")

ax.grid(axis='y')


fig, axes = plt.subplots(2, 3, figsize=(20,15))

fig.delaxes(axes[1][2])

for counter, index in enumerate(['Gender', 'Cholesterol', 'Gluc', 'Smoke', 'Active']):

    sns.countplot(x=index, hue='Cardio', data=df[[index, 'Cardio']], ax=axes[counter//3, counter%3])
# Classification

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



# Modelling Helpers :

from sklearn.model_selection import train_test_split
x = df.copy()

y = x.pop('Cardio')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 37)
RFC = RandomForestClassifier(n_estimators=200, random_state=82)

KNN = KNeighborsClassifier(n_neighbors = 25, algorithm='brute')

BAG = BaggingClassifier(random_state = 222, n_estimators=140)

GradBost = GradientBoostingClassifier(random_state = 15)

ADA = AdaBoostClassifier(random_state = 741, n_estimators=70)

DT = DecisionTreeClassifier(random_state=12, criterion='entropy')
ADA.fit(x_train,y_train)

ADA_pred = ADA.predict(x_test)

print("accuracy: {0:.2f} %".format((ADA.score(x_test,y_test)*100)))
RFC.fit(x_train,y_train)

RFC_pred = RFC.predict(x_test)

print("accuracy: {0:.2f} %".format((RFC.score(x_test,y_test)*100)))
BAG.fit(x_train,y_train)

BAG_pred = BAG.predict(x_test)

print("accuracy: {0:.2f} %".format((BAG.score(x_test,y_test)*100)))
KNN.fit(x_train,y_train)

KNN_pred = KNN.predict(x_test)

print("accuracy: {0:.2f} %".format((KNN.score(x_test,y_test)*100)))
DT.fit(x_train,y_train)

DT_pred = DT.predict(x_test)

print("accuracy: {0:.2f} %".format((DT.score(x_test,y_test)*100)))