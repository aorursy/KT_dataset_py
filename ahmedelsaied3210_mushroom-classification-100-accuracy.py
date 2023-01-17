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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')

df.head()
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for column in df.columns:

    df[column]=labelencoder.fit_transform(df[column])

    

df.head()

df.isnull().sum()
print(df.groupby('class').size())
df.columns
X=df[df.columns]

X=X.drop(columns=['class'])

X.head()
Y=df['class']

Y.head()
from sklearn.preprocessing import StandardScaler

stand_x=StandardScaler()

X=stand_x.fit_transform(X)
X[0]
#histograms the dataset

fig = plt.figure(figsize = (15,20))

ax = fig.gca()

pd.DataFrame(X).hist(ax = ax)
models=[]

models.append(('LR',LogisticRegression()))

models.append(('DT',DecisionTreeClassifier()))

models.append(('KN',KNeighborsClassifier()))

models.append(('NB',GaussianNB()))

models.append(('SVC',SVC()))
results=[]

names=[]

scoring='accuracy'
for name,model in models:

    kfold=KFold(n_splits=10,random_state=7)

    cv_result=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)

    results.append(cv_result)

    names.append(name)

    msg=("%s: %f (%f)" % (name,cv_result.mean(),cv_result.std()))

    print(msg)
import matplotlib.pyplot as plt

fig=plt.figure()

fig.suptitle('Algorithms Coparison')

ax=fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
my_model=DecisionTreeClassifier()

my_model.fit(x_train,y_train)
result=my_model.score(x_test,y_test)

print('Accuracy : ' ,(result*100))