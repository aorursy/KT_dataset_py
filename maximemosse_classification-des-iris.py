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
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
df = sns.load_dataset("iris")
df.head()
sns.pairplot(df, hue="species")
setosa=df.species=="setosa"
versicolor=df.species=="versicolor"
virginica=df.species=="virginica"
plt.figure(figsize=(12,12))
sns.kdeplot(df[setosa].petal_length, df[setosa].petal_width, cmap="Blues",  shade=True, alpha=0.3, shade_lowest=False)
sns.kdeplot(df[versicolor].petal_length, df[versicolor].petal_width, cmap="Reds",  shade=True, alpha=0.3, shade_lowest=False)
sns.kdeplot(df[virginica].petal_length, df[virginica].petal_width, cmap="Greens",  shade=True, alpha=0.3, shade_lowest=False)
data_train = df.sample(frac=0.7, random_state=1)          # 70% des données avec frac=0.7
data_test = df.drop(data_train.index)     # le reste des données pour le test
X_train = data_train.drop(['species'], axis=1)
y_train = data_train['species']
X_test = data_test.drop(['species'], axis=1)
y_test = data_test['species']
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_dtc = dtc.predict(X_test)
print(accuracy_score(y_test, y_dtc))
print(confusion_matrix(y_test, y_dtc))
data_test_setosaless =data_test[data_test['species']!="setosa"]
X_test_setosaless = data_test_setosaless.drop(['species'], axis=1)
y_test_setosaless = data_test_setosaless['species']
y_dtc_setosaless = dtc.predict(X_test_setosaless)
print(accuracy_score(y_test_setosaless, y_dtc_setosaless))
print(confusion_matrix(y_test_setosaless, y_dtc_setosaless))
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf_setosaless = rf.predict(X_test_setosaless)
print(accuracy_score(y_test_setosaless, y_rf_setosaless))
pd.crosstab(y_test_setosaless, y_rf_setosaless, rownames=['Reel'], colnames=['Prediction'], margins=True)
y_rf=rf.predict(X_test)
print(accuracy_score(y_test, y_rf))
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12,8))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), df.columns[indices])
plt.title('Importance des caracteristiques')