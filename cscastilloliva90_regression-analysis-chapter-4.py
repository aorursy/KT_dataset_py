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
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification



X, y = make_classification(n_samples = 100, n_features=2, n_informative=2,n_redundant=0, n_clusters_per_class=1, class_sep=2, random_state=101)

plt.scatter(X[:,0], X[:,1], marker='o', c=y, linewidths=0, edgecolors=None)

plt.show()
y_orig=[0,0,0,0,0,0,1,1,1,1]

y_pred=[0,0,0,0,1,1,1,1,1,0]
from sklearn.metrics import confusion_matrix

confusion_matrix(y_orig,y_pred)
import seaborn as sns

sns.heatmap(confusion_matrix(y_orig,y_pred), linecolor='white', linewidths=3, annot=True)

plt.title('Confusion Matrix', fontsize=20)

plt.show()
from mlxtend.plotting import plot_confusion_matrix

plot_confusion_matrix(confusion_matrix(y_orig,y_pred))

plt.show()
from sklearn.metrics import accuracy_score

#ACC = (TP/(P+N))  TP = 3, TN = 4, P = 4, N = 6, 

acc=accuracy_score(y_orig, y_pred)

acc
from sklearn.metrics import precision_score

precision_score (y_orig, y_pred)
from sklearn.metrics import recall_score

# recall score or hit rate

recall_score(y_orig, y_pred)
from sklearn.metrics import f1_score

f1_score (y_orig, y_pred)
from sklearn.metrics import classification_report

print(classification_report (y_orig, y_pred))
X.shape
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y.astype(float), test_size=0.33, random_state = 101)
from sklearn.linear_model import LinearRegression

regr=LinearRegression()

regr.fit(X_train, y_train)

regr.predict(X_train)
import numpy as np



def model(x):

    return 1/(1+np.exp(-x))
plt.style.use('seaborn')

X_vals = np.linspace(-10,10, 1000)

plt.plot(X_vals, model(X_vals), color='red', linewidth=4)

plt.ylabel('sigmoid(t)')

plt.xlabel("t")

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report



clf=LogisticRegression()

clf.fit(X_train, y_train.astype(int))

y_clf=clf.predict(X_test)



print(classification_report(y_test, y_clf))
y_clf
h=0.02

plt.style.use('ggplot')

x_min, x_max=X[:,0].min()- 0.5, X[:,0].max() + 0.5

y_min, y_max=X[:,1].min()- 0.5, X[:,1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min,y_max,h))

Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])



Z=Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.autumn)

plt.contour(xx,yy,Z)





plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', linewidth = 0, cmap=plt.cm.Paired)

plt.xticks(())

plt.yticks(())

plt.show()
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification



X, y = make_classification(n_samples=200, n_features=2, 

                          n_classes=3, n_informative=2, 

                          n_redundant=0, n_clusters_per_class= 1,

                          class_sep=2.0, random_state=101)
plt.style.use('default')

plt.scatter(X[:,0],X[:,1], marker='o', c=y)

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y.astype(float), test_size=0.33, random_state = 101)





from sklearn.linear_model import LinearRegression

clf=LogisticRegression()

clf.fit(X_train, y_train.astype(int))

y_clf=clf.predict(X_test)
print(classification_report(y_test, y_clf))
h=0.01

plt.style.use('default')

x_min, x_max=X[:,0].min()- 0.5, X[:,0].max() + 0.5

y_min, y_max=X[:,1].min()- 0.5, X[:,1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min,y_max,h))

Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])



Z=Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.autumn)





plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', linewidth = 1, cmap=plt.cm.Paired)

plt.xticks(())

plt.yticks(())

plt.show()