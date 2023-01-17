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
# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
# SeaBorn : librairie de graphiques avancés
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
df.head()
sns.pairplot(df, hue="species")
df.shape
df.describe()
df.columns
df['species'].value_counts()
Irisvirginica = df.species=='Iris-virginica'
Irissetosa = df.species=='Iris-setosa'
Irisversicolor = df.species=='Iris-versicolor'
sns.jointplot("sepal_length", "sepal_width", df, kind='kde');
plt.figure(figsize=(12,12))
sns.kdeplot(df.sepal_length, df.sepal_width,  shade=True)
plt.figure(figsize=(12,12))
sns.kdeplot(df[Irisvirginica].sepal_length, df[Irisvirginica].sepal_width, cmap="Reds",  shade=True, alpha=0.3, shade_lowest=False)
sns.kdeplot(df[Irissetosa].sepal_length, df[Irissetosa].sepal_width, cmap="Greens", shade=True, alpha=0.3, shade_lowest=False)
sns.kdeplot(df[Irisversicolor].sepal_length, df[Irisversicolor].sepal_width, cmap="Blues", shade=True, alpha=0.3, shade_lowest=False)
sns.boxplot(x="species", y="sepal_length", data=df)
sns.violinplot(x="species", y="sepal_length", data=df)
fig = sns.FacetGrid(df, hue="species", aspect=3, palette="Set2")
fig.map(sns.kdeplot, "sepal_length", shade=True)
fig.add_legend()
sns.lmplot(x="sepal_length", y="sepal_width", data=df, fit_reg=False, hue='species')
sns.pairplot(df, hue="species")
data_train = df.sample(frac=0.8, random_state=1)
data_test = df.drop(data_train.index)
X_train = data_train.drop(['species'], axis=1)
y_train = data_train['species']
X_test = data_test.drop(['species'], axis=1)
y_test = data_test['species']
plt.figure(figsize=(9,9))

logistique = lambda x: np.exp(x)/(1+np.exp(x))   

x_range = np.linspace(-10,10,50)       
y_values = logistique(x_range)

plt.plot(x_range, y_values, color="red")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
lr_score = accuracy_score(y_test, y_lr)
print(lr_score)
cm = confusion_matrix(y_test, y_lr)
print(cm)
pd.crosstab(y_test, y_lr, rownames=['Reel'], colnames=['Prediction'], margins=True)
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
plt.yticks(range(len(indices)), df.columns[indices])
plt.title('Importance des caracteristiques')