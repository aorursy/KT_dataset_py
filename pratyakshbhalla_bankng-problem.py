# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import mlxtend.plotting as mlp


df = pd.read_csv("../input/svm-classification/UniversalBank.csv")
df.head()
df.shape
df.info()

df.describe()
X = df.iloc[:,1:13].values

y = df.iloc[:, -1].values
import missingno as msno

p = msno.bar(df)
p = sns.lineplot(x =X[:, 0], y = y, data = df)

p = sns.lineplot(x =X[:, 1], y = y, data = df)

p = sns.lineplot(x =X[:, 2], y = y, data = df)

p = sns.lineplot(x =X[:, 3], y = y, data = df)

p = sns.lineplot(x =X[:, 4], y = y, data = df)

p = sns.lineplot(x =X[:, 5], y = y, data = df)

p = sns.lineplot(x =X[:, 6], y = y, data = df)

p = sns.lineplot(x =X[:, 7], y = y, data = df)

p = sns.lineplot(x =X[:, 8], y = y, data = df)

p = sns.lineplot(x =X[:, 9], y = y, data = df)

p = sns.lineplot(x =X[:, 10], y = y, data = df)

p = sns.lineplot(x =X[:, 11], y = y, data = df)

p = sns.pairplot(data = df)

cols = ['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Personal Loan', 'Securities Account', 'CD Account', 'Online']
cm = np.corrcoef(df[cols].values.T)
from mlxtend.plotting import heatmap

hm = heatmap(cm, column_names=cols, row_names=cols, figsize = (25, 25))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()
from sklearn.metrics import confusion_matrix

cmatrix = confusion_matrix(y_test, y_pred)
from mlxtend.plotting import plot_confusion_matrix as pcm

p = pcm(conf_mat = cmatrix, cmap = 'winter_r', figsize = (5, 5))
((871+56)/1250)*100
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)