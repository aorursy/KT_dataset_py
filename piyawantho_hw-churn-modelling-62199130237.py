# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np

import matplotlib.pyplot as plt

import pandas as pd







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import dataset

dataset = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv', dtype={'CreditScore': 'float64','Age': 'float64','Tenure': 'float64','NumOfProducts': 'float64','HasCrCard': 'float64','IsActiveMember': 'float64','EstimatedSalary': 'float64','Exited': 'float64'})
#check dataset

print(dataset.shape)

print(dataset.info())
df = dataset[['Geography','Gender','HasCrCard','IsActiveMember','Balance','NumOfProducts','CreditScore','Tenure','Age','EstimatedSalary','Exited']]

df.dtypes
#dtype={'Customer Number': 'int'}
df.isnull().any()
X = df.iloc[:, 8:10].values

y = df.iloc[:, 10].values
print(X)

print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 11, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(y_pred)
#	Visualising the	Training	set	results

from	matplotlib.colors import	ListedColormap

X_set,	y_set =	X_train,	y_train

X1,	X2	=	np.meshgrid(np.arange(start	=	X_set[:,	0].min()	- 1,	stop	=	X_set[:,	0].max()	+	1,	step	=	0.01),np.arange(start	=	X_set[:,	1].min()	- 1,	stop	=	X_set[:,	1].max()	+	1,	step	=	0.01))

plt.contourf(X1,	X2,	classifier.predict(np.array([X1.ravel(),	X2.ravel()]).T).reshape(X1.shape),	alpha	=	0.75,	cmap = ListedColormap(('yellow',	'green')))

plt.xlim(X1.min(),	X1.max())

plt.ylim(X2.min(),	X2.max())

for	i,	j	in	enumerate(np.unique(y_set)):				

    plt.scatter(X_set[y_set ==	j,	0],	X_set[y_set ==	j,	1],	c	=	ListedColormap(('red',	'blue'))(i),	label	=	j)

plt.title('KNN	(Training	set)')

plt.xlabel('Age')

plt.ylabel('Estimated	Salary')

plt.legend()

plt.show()
from	matplotlib.colors import	ListedColormap

X_set,	y_set =	X_test,	y_test

X1,	X2	=	np.meshgrid(np.arange(start	=	X_set[:,	0].min()	- 1,	stop	=	X_set[:,	0].max()	+	1,	step	=	0.01), np.arange(start	=	X_set[:,	1].min()	- 1,	stop	=	X_set[:,	1].max()	+	1,	step	=	0.01))

plt.contourf(X1,	X2,	classifier.predict(np.array([X1.ravel(),	X2.ravel()]).T).reshape(X1.shape),	alpha	=	0.75,	cmap =	ListedColormap(('yellow',	'green')))

plt.xlim(X1.min(),	X1.max())

plt.ylim(X2.min(),	X2.max())

for	i,	j	in	enumerate(np.unique(y_set)):				

    plt.scatter(X_set[y_set ==	j,	0],	X_set[y_set ==	j,	1],	c	=	ListedColormap(('red',	'blue'))(i),	label	=	j)

plt.title('KNN	(Test	set)')

plt.xlabel('Age')

plt.ylabel('Estimated	Salary')

plt.legend()

plt.show()
import	numpy as	np

import	matplotlib.pyplot as	plt

import	pandas	as	pd

#	Importing	the	dataset

# dataset	=	pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

X	=	df.iloc[:,	8:10].values

y	=	df.iloc[:,	10].values

#	Splitting	the	dataset	into	the	Training	set	and	Test	set

from	sklearn.model_selection import	train_test_split

X_train,	X_test,	y_train,	y_test =	train_test_split(X,	y,	test_size =	0.25,	random_state =	0)

#	Feature	Scaling

from	sklearn.preprocessing import	StandardScaler

sc_X =	StandardScaler()

X_train =	sc_X.fit_transform(X_train)

X_test =	sc_X.transform(X_test)
#	Fitting	SVM	to	the	Training	Set

from	sklearn.svm import	SVC

classifier	=	SVC(kernel	=	'linear',	degree	=	10,	random_state =	1000)

classifier.fit(X_train,	y_train)	

#	Predicting	the	Test	Set	results

y_pred =	classifier.predict(X_test)

#	Making	the	Confusion	Matrix

from	sklearn.metrics import	confusion_matrix

cm	=	confusion_matrix(y_test,	y_pred)
from	matplotlib.colors import	ListedColormap

X_set,	y_set =	X_train,	y_train

X1,	X2	=	np.meshgrid(np.arange(start	=	X_set[:,	0].min()	- 1,	stop	=	X_set[:,	0].max()	+	1,	step	=	0.01),np.arange(start	=	X_set[:,	1].min()	- 1,	stop	=	X_set[:,	1].max()	+	1,	step	=	0.01))

plt.contourf(X1,	X2,	classifier.predict(np.array([X1.ravel(),	X2.ravel()]).T).reshape(X1.shape),	alpha	=	0.75,	cmap = ListedColormap(('yellow',	'green')))

plt.xlim(X1.min(),	X1.max())

plt.ylim(X2.min(),	X2.max())

for	i,	j	in	enumerate(np.unique(y_set)):				

    plt.scatter(X_set[y_set ==	j,	0],	X_set[y_set ==	j,	1],	c	=	ListedColormap(('red',	'blue'))(i),	label	=	j)

    plt.title('SVM	(Training	set)')

plt.xlabel('Age')

plt.ylabel('Estimated	Salary')

plt.legend()

plt.show()
from	matplotlib.colors import	ListedColormap

X_set,	y_set =	X_test,	y_test

X1,	X2	=	np.meshgrid(np.arange(start	=	X_set[:,	0].min()	- 1,	stop	=	X_set[:,	0].max()	+	1,	step	=	0.01),np.arange(start	=	X_set[:,	1].min()	- 1,	stop	=	X_set[:,	1].max()	+	1,	step	=	0.01))

plt.contourf(X1,	X2,	classifier.predict(np.array([X1.ravel(),	X2.ravel()]).T).reshape(X1.shape),	alpha	=	0.75,	cmap = ListedColormap(('yellow',	'green')))

plt.xlim(X1.min(),	X1.max())

plt.ylim(X2.min(),	X2.max())

for	i,	j	in	enumerate(np.unique(y_set)):				

    plt.scatter(X_set[y_set ==	j,	0],	X_set[y_set ==	j,	1],	c	=	ListedColormap(('red',	'blue'))(i),	label	=	j)

    plt.title('SVM	(Test	set)')

plt.xlabel('Age')

plt.ylabel('Estimated	Salary')

plt.legend()

plt.show()
#	Fitting	Naive	Bayes	to	the	Training	Set

from	sklearn.naive_bayes import	GaussianNB

classifier	=	GaussianNB()

classifier.fit(X_train,	y_train)

#	Predicting	the	Test	Set	results

y_pred =	classifier.predict(X_test)

#	Making	the	Confusion	Matrix

from	sklearn.metrics import	confusion_matrix

cm	=	confusion_matrix(y_test,	y_pred)
from	matplotlib.colors import	ListedColormap

X_set,	y_set =	X_train,	y_train

X1,	X2	=	np.meshgrid(np.arange(start	=	X_set[:,	0].min()	- 1,	stop	=	X_set[:,	0].max()	+	1,	step	=	0.01),np.arange(start	=	X_set[:,	1].min()	- 1,	stop	=	X_set[:,	1].max()	+	1,	step	=	0.01))

plt.contourf(X1,	X2,	classifier.predict(np.array([X1.ravel(),	X2.ravel()]).T).reshape(X1.shape),	alpha	=	0.75,	cmap =	ListedColormap(('red',	'green')))

plt.xlim(X1.min(),	X1.max())

plt.ylim(X2.min(),	X2.max())

for	i,	j	in	enumerate(np.unique(y_set)):				

    plt.scatter(X_set[y_set ==	j,	0],	X_set[y_set ==	j,	1],	c	=	ListedColormap(('yellow',	'green'))(i),	label	=	j)

    plt.title('Naïve	Bayes	(Training	set)')

plt.xlabel('Age')

plt.ylabel('Estimated	Salary')

plt.legend()

plt.show()
from	matplotlib.colors import	ListedColormap

X_set,	y_set =	X_test,	y_test

X1,	X2	=	np.meshgrid(np.arange(start	=	X_set[:,	0].min()	- 1,	stop	=	X_set[:,	0].max()	+	1,	step	=	0.01),np.arange(start	=	X_set[:,	1].min()	- 1,	stop	=	X_set[:,	1].max()	+	1,	step	=	0.01))

plt.contourf(X1,	X2,	classifier.predict(np.array([X1.ravel(),	X2.ravel()]).T).reshape(X1.shape),	alpha	=	0.75,	cmap =	ListedColormap(('red',	'green')))

plt.xlim(X1.min(),	X1.max())

plt.ylim(X2.min(),	X2.max())

for	i,	j	in	enumerate(np.unique(y_set)):				

    plt.scatter(X_set[y_set ==	j,	0],	X_set[y_set ==	j,	1],	c	=	ListedColormap(('yellow',	'green'))(i),	label	=	j)

    plt.title('Naïve	Bayes	(Training	set)')

plt.xlabel('Age')

plt.ylabel('Estimated	Salary')

plt.legend()

plt.show()