# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/titanic/train.csv")

df.head()
classe3=df[df.Pclass==3]

classe3_survivant=classe3[classe3.Survived==1]

n1=classe3_survivant[classe3_survivant.Sex=='male'].shape[0]

print(n1)

classe3_mort=classe3[classe3.Survived==0]

n2=classe3[classe3.Sex=='male'].shape[0]

print(n2)

Pjack=n1/n2

print(Pjack) ## probabilité de survie de Jack
classe1=df[df.Pclass==1]

classe1_survivant=classe1[classe1.Survived==1]

m1=classe1_survivant[classe1_survivant.Sex=='female'].shape[0]

print(m1)

classe1_mort=classe1[classe1.Survived==0]

m2=classe1[classe1.Sex=='female'].shape[0]

print(m2)

Prose=m1/m2

print(Prose) ## probabilité de survie de Rose
import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=sns.load_dataset("iris")

data.head()
sns.jointplot("sepal_length", "sepal_width",data, kind='kde')
seto=data.species=='setosa'

virgi=data.species=='virginica'

versi=data.species=='versicolor'

plt.figure(figsize=(12,12))

sns.kdeplot(data[seto].sepal_length, data[seto].sepal_width, cmap="Reds",  shade=True, alpha=0.3, shade_lowest=False)

sns.kdeplot(data[virgi].sepal_length, data[virgi].sepal_width, cmap="Greens",  shade=True, alpha=0.3, shade_lowest=False)

sns.kdeplot(data[versi].sepal_length, data[versi].sepal_width, cmap="Blues",  shade=True, alpha=0.3, shade_lowest=False)
sns.boxplot(x="species", y="sepal_length", data=data)
sns.violinplot(x="species", y="sepal_length", data=data)
fig = sns.FacetGrid(data, hue="species", aspect=3, palette="Set2") 

fig.map(sns.kdeplot, "sepal_length", shade=True)

fig.add_legend()
sns.lmplot(x="sepal_length", y="sepal_width", data=data, fit_reg=False, hue='species')
data_train = data.sample(frac=0.8, random_state=1)          

data_test = data.drop(data_train.index)
X_train = data_train.drop(['species'], axis=1)

y_train = data_train['species']

X_test = data_test.drop(['species'], axis=1)

y_test = data_test['species']
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)
fig = sns.FacetGrid(data, hue="species", aspect=3, palette="Set2") 

fig.map(sns.kdeplot, "petal_length", shade=True)

fig.add_legend()
from sklearn import tree

from sklearn.metrics import accuracy_score, confusion_matrix

dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train,y_train)

y_dtc = dtc.predict(X_test)

print(accuracy_score(y_test, y_dtc)) #résultat de l'arbre de décision
plt.figure(figsize=(30,30))

tree.plot_tree(dtc, feature_names=X_train.columns, class_names=['setosa','virginica','versicolor'], fontsize=14, filled=True)
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_

indices = np.argsort(importances)
plt.figure(figsize=(12,8))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), data.columns[indices])

plt.title('Importance des caracteristiques')