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

from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import matplotlib.pyplot as plt
data=pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')

data.head()
data.columns
data.isnull().sum()
data.size
data=data.dropna()

data.isnull().sum()
data.corr()
print(data.groupby('TenYearCHD').size())
X=data[data.columns]

X=X.drop(columns=['TenYearCHD'])

X.head()
Y=data['TenYearCHD']

Y.head()
from sklearn.feature_selection import SelectKBest,chi2

test=SelectKBest(score_func=chi2,k=10)

fit=test.fit(X,Y)

print(fit.scores_)
X.columns
X=data[['age','cigsPerDay','totChol','sysBP','diaBP','glucose']]

X.head()
from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

X=sc_x.fit_transform(X)
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
my_model=LogisticRegression()

my_model.fit(x_train,y_train)
result=my_model.score(x_test,y_test)

print('Accuracy : ' ,(result*100))