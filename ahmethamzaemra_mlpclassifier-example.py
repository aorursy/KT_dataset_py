# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns
df = pd.read_csv('../input/Dataset_spine.csv')
df = df.drop(['Unnamed: 13'], axis=1)
df.head()
df.describe()
df = df.drop(['Col7','Col8','Col9','Col10','Col11','Col12'], axis=1)
df.head()
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
y = df['Class_att']

x = df.drop(['Class_att'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)
clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,

                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

cm
sns.heatmap(cm, center=True)

plt.show()