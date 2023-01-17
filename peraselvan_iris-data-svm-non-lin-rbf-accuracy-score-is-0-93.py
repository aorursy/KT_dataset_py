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
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from matplotlib import style 
from sklearn.svm import SVC  
  
# Data visualization libraires
import seaborn as sns
import matplotlib.pyplot as plt

# show plot in the notebook
%matplotlib inline
plt.style.use('fivethirtyeight')
style.use('fivethirtyeight') 

# create mesh grids 
def make_meshgrid(x, y, h =.02): 
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    return xx, yy 
# plot the contours 
def plot_contours(ax, clf, xx, yy, **params): 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    out = ax.contourf(xx, yy, Z, **params) 
    return out 
color = ['r', 'b', 'g', 'k'] 

iris = pd.read_csv("../input/iris/Iris.csv") 

iris.head()
iris.info()

iris['Species'].value_counts()

sns.pairplot(iris.drop(['Id'], axis=1),hue='Species')

iris = pd.read_csv("../input/iris/Iris.csv").values
features = iris[0:150, 2:4] 
level1 = np.zeros(150) 
level2 = np.zeros(150) 
level3 = np.zeros(150) 
# level1 contains 1 for class1 and 0 for all others. 
# level2 contains 1 for class2 and 0 for all others. 
# level3 contains 1 for class3 and 0 for all others. 
for i in range(150): 
    if i>= 0 and i<50: 
        level1[i] = 1
    elif i>= 50 and i<100: 
        level2[i] = 1
    elif i>= 100 and i<150: 
        level3[i]= 1
# create 3 svm with rbf kernels 
svm1 = SVC(kernel ='rbf') 
svm2 = SVC(kernel ='rbf') 
svm3 = SVC(kernel ='rbf') 

# fit each svm's 
svm1.fit(features, level1) 
svm2.fit(features, level2) 
svm3.fit(features, level3) 
fig, ax = plt.subplots() 
X0, X1 = iris[:, 2], iris[:, 3] 
xx, yy = make_meshgrid(X0, X1) 
# plot the contours 
plot_contours(ax, svm1, xx, yy, cmap = plt.get_cmap('hot'), alpha = 0.8) 
plot_contours(ax, svm2, xx, yy, cmap = plt.get_cmap('hot'), alpha = 0.3) 
plot_contours(ax, svm3, xx, yy, cmap = plt.get_cmap('hot'), alpha = 0.5) 
  
color = ['r', 'b', 'g', 'k'] 
  
for i in range(len(iris)): 
    plt.scatter(iris[i][2], iris[i][3], s = 30, c = color[int(iris[i][4])]) 
plt.show() 


iris = pd.read_csv("../input/iris/Iris.csv")

sns.pairplot(iris.drop(['Id'], axis=1),hue='Species')
iris.head(10)
# Split data into a training set and a testing set.
# train_test_split shuffle the data before the split (shuffle=True by default)
from sklearn.model_selection import train_test_split
X=iris.drop(['Species', 'Id'], axis=1)
y=iris['Species']
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.5, shuffle=True,random_state=100)
from sklearn.svm import SVC
model=SVC(C=1, kernel='rbf', tol=0.001)
model.fit(X_train, y_train)

###We get predictions from the model now and create a confusion matrix and a classification report.
pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
print('\n')
print('Accuracy score using RBF is: ', accuracy_score(y_test, pred))

from sklearn.svm import SVC
model=SVC(C=1, kernel='poly', tol=0.001)
model.fit(X_train, y_train)

###We get predictions from the model now and create a confusion matrix and a classification report.
pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
print('\n')
print('Accuracy score using Poly is: ', accuracy_score(y_test, pred))
from sklearn.svm import SVC
model=SVC(C=1, kernel='linear', tol=0.001)
model.fit(X_train, y_train)
###We get predictions from the model now and create a confusion matrix and a classification report.
pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
print('\n')
print('Accuracy score Linear kernal is: ', accuracy_score(y_test, pred))