# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data = pd.read_csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/diabetes.csv")
data.head()
data.tail()
data.info()
data.describe()
data.groupby('class').size()
dataset = data.drop('class', axis=1)

import matplotlib.pyplot as plt
dataset['age'].plot(kind='box', subplots=True,

layout=(1,1), sharex=False, sharey=False)

plt.figure()

plt.show()
dataset['age'].hist()

plt.show()
# dataset['insu'].plot(kind='box', subplots=True,

# layout=(1,1), sharex=False, sharey=False)

# dataset['mass'].plot(kind='box', subplots=True,

# layout=(1,1), sharex=False, sharey=False)

# dataset['pedi'].plot(kind='box', subplots=True,

# layout=(1,1), sharex=False, sharey=False)

# dataset['plas'].plot(kind='box', subplots=True,

# layout=(1,1), sharex=False, sharey=False)

# dataset['preg'].plot(kind='box', subplots=True,

# layout=(1,1), sharex=False, sharey=False)

# dataset['pres'].plot(kind='box', subplots=True,

# layout=(1,1), sharex=False, sharey=False)

# dataset['skin'].plot(kind='box', subplots=True,

# layout=(1,1), sharex=False, sharey=False)

# plt.show()
# dataset['insu'].hist()

# dataset['mass'].hist()

# dataset['pedi'].hist()

# dataset['plas'].hist()

# dataset['preg'].hist()

# dataset['pres'].hist()

dataset['skin'].hist()

plt.show()
pd.plotting.scatter_matrix(dataset,diagonal='hist')

plt.show()
from sklearn.model_selection import train_test_split

array = data.values

x = array[:,0:8]

y = array[:,8]

x_train,x_validation,y_train,y_validation = train_test_split(x,y,test_size=0.20,random_state=1)



print(len(x_train))

print(len(x_validation))

print(len(y_train))

print(len(y_validation))
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC 
models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('DT',DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))
results =[]

names = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
plt.boxplot(results,labels=names)

plt.title('Algorithm Comparison')

plt.show()
model = SVC(gamma='auto')

model.fit(x_train, y_train)	

predictions = model.predict(x_validation)

print(accuracy_score(y_validation, predictions))

print(confusion_matrix(y_validation, predictions))

print(classification_report(y_validation,predictions))