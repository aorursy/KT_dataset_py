import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import shuffle



wine = pd.read_csv('../input/wine.csv')



wine.describe()
#wine = shuffle(wine, random_state = 0)



wine.head()
fig = wine[wine.Type==1].plot(kind='scatter', x='Alco', y='Mal', color='orange', label='Wine 1')

wine[wine.Type==2].plot(kind='scatter', x='Alco', y='Mal', color='violet', ax=fig, label='Wine 2')

wine[wine.Type==3].plot(kind='scatter', x='Alco', y='Mal', color='green', ax=fig, label='Wine 3')



fig.set_xlabel('Alcohol Content')

fig.set_ylabel('Malic Acid content')

fig.set_title('Alc v/s Mal')



fig = plt.gcf()

fig.set_size_inches(10,6)



plt.show()

plt.figure(figsize=(15,15))

sns.heatmap(wine.corr(), annot=True, cmap = 'cubehelix_r')



plt.show()
fig = wine[wine.Type==1].plot(kind='scatter', x='Phe', y='Flav', label='Wine 1', color='orange')

wine[wine.Type==2].plot(kind='scatter', x='Phe', y='Flav', label='Wine 2', color='blue', ax=fig)

wine[wine.Type==3].plot(kind='scatter', x='Phe', y='Flav', label='Wine 3', color='green', ax=fig)



fig.set_xlabel('Phenol Conc.')

fig.set_ylabel('Flavinoids Conc.')

fig.set_title('Flav v/s Phen')



fig = plt.gcf()

fig.set_size_inches(10,6)



plt.show()
fig = wine[wine.Type==1].plot(kind='scatter', x='OD', y='Flav', label='Wine 1', color='orange')

wine[wine.Type==2].plot(kind='scatter', x='OD', y='Flav', label='Wine 2', color='blue', ax=fig)

wine[wine.Type==3].plot(kind='scatter', x='OD', y='Flav', label='Wine 3', color='green', ax=fig)



fig.set_xlabel('OD Conc.')

fig.set_ylabel('Flavinoids Conc.')

fig.set_title('Flav v/s OD')



fig = plt.gcf()

fig.set_size_inches(10,6)



plt.show()
fig = wine[wine.Type==1].plot(kind='scatter', x='Phe', y='OD', label='Wine 1', color='orange')

wine[wine.Type==2].plot(kind='scatter', x='Phe', y='OD', label='Wine 2', color='blue', ax=fig)

wine[wine.Type==3].plot(kind='scatter', x='Phe', y='OD', label='Wine 3', color='green', ax=fig)



fig.set_xlabel('Phenol Conc.')

fig.set_ylabel('OD Conc.')

fig.set_title('OD v/s Phen')



fig = plt.gcf()

fig.set_size_inches(10,6)



plt.show()
# GEt impo features

feats = ['Type','Flav', 'OD', 'Alco', 'ProAn']



# split into test and train

train, test = train_test_split(wine, test_size = 0.3, random_state=0)



# assign the datas appropriately

trX = train[feats]

trY = train.Type



sns.pairplot(trX, hue='Type')
# GEt impo features

feats = ['Type','Flav', 'OD', 'Alco', 'ProAn']



# split into test and train

train, test = train_test_split(wine, test_size = 0.3, random_state=0)



# assign the datas appropriately

trX = train[feats]

trY = train.Type



#sns.pairplot(trX, hue='Type')



teX = test[feats]

teY = test.Type



#model = svm.SVC()

#model.fit(trX, trY)

#pred = model.predict(teX)

#print("Accu svm = ", metrics.accuracy_score(pred, teY))



for i in range(20):

    model = LogisticRegression(C = 10**(i-10))

    #model = svm.SVC(C = 10**(i-10))

    #model = KNeighborsClassifier(n_neighbors=i+1)

    model.fit(trX, trY)

    pred = model.predict(teX)

    print("Accu logi = ", metrics.accuracy_score(pred, teY))