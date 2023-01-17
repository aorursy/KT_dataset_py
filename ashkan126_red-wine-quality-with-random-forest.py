import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report



wines = pd.read_csv("..//input//winequality-red.csv")

wines.info()
sns.countplot('quality', data = wines)
fig , axs = plt.subplots(3,4, figsize=(20,15))

counter = 0

for col in wines.columns:

    sns.barplot(x='quality' ,y = col , data = wines , ax= axs[counter%3][int(counter/3)] )

    counter = counter + 1
plt.subplots(figsize=(20,15))

corrmat = wines.corr()

corrmat = np.abs(corrmat)

cm = np.corrcoef(wines.values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=wines.columns.values, xticklabels=wines.columns.values)

plt.show()
wines['q2'] = pd.cut(wines['quality'],[0,5,10],labels=[0,1])

sns.countplot('q2', data = wines)
wines.q2 = pd.Categorical(wines.q2).codes

print(wines.columns[:-2])
X = wines[wines.columns[:-2]].values

Y = wines['q2'].values



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)



print('X train size: ', X_train.shape)

print('y train size: ', Y_train.shape)

print('X test size: ', X_test.shape)

print('y test size: ', Y_test.shape)
random_forest = RandomForestClassifier(n_estimators=100,max_depth= 10)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print(classification_report(Y_test,Y_pred))