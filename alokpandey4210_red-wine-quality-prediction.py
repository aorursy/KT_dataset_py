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
df=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.columns = df.columns.str.replace(' ', '')

df.columns

df['quality'].value_counts()
df.corr(method='pearson')
X=df.drop(['residualsugar','freesulfurdioxide','pH','quality'],axis=1)
y=df['quality']
X.isnull().sum()
X['fixedacidity'].hist(bins=50)
X.boxplot(column='fixedacidity')
X['fixedacidity'].plot('density')
X['volatileacidity'].hist(bins=40)
X.boxplot(column='volatileacidity')
X['volatileacidity'].plot('density')
X['citricacid'].hist(bins=50)
X.boxplot(column='citricacid')
X['citricacid'].plot('density')
X['chlorides'].hist(bins=50)
X.boxplot(column='chlorides',by='fixedacidity')
X['chlorides'].plot('density')
X['totalsulfurdioxide'].hist(bins=50)
X.boxplot(column='totalsulfurdioxide',by='citricacid')
X['totalsulfurdioxide'].plot('density')
X['density'].hist(bins=50)
X.boxplot(column='density',by='totalsulfurdioxide')
X['density'].plot('density',color='Red')
X['sulphates'].hist(bins=50)
X.boxplot(column='sulphates',by='density')
X['sulphates'].plot('density')
X['alcohol'].hist(bins=50)
X.boxplot(column='alcohol',by='density')
X['density'].plot('density')
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40)
from sklearn.linear_model import LinearRegression

ks=LinearRegression()

ks.fit(X_train,y_train)

ks.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor

pa=DecisionTreeRegressor(max_depth=3)

pa.fit(X_train,y_train)

pa.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression

qb=LogisticRegression()

qb.fit(X_train,y_train)

qb.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

yx=RandomForestRegressor()

yx.fit(X_train,y_train)

yx.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesRegressor

qm=ExtraTreesRegressor()

qm.fit(X_train,y_train)

qm.score(X_test,y_test)
from sklearn.linear_model import SGDRegressor

qp=SGDRegressor()

qp.fit(X_train,y_train)

qp.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

tv=KNeighborsClassifier()

tv.fit(X_train,y_train)

tv.score(X_test,y_test)
from sklearn.neighbors import KNeighborsRegressor

ez=KNeighborsRegressor()

ez.fit(X_train,y_train)

ez.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

yx=RandomForestClassifier()

yx.fit(X_train,y_train)

yx.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor

oc=DecisionTreeRegressor()

oc.fit(X_train,y_train)

oc.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

qx=RandomForestRegressor()

qx.fit(X_train,y_train)

qx.score(X_test,y_test)
y_test.value_counts()
from sklearn.metrics import confusion_matrix

ypqs=yx.predict(X_test)

results=confusion_matrix(y_test,ypqs)



print(results)

from sklearn.metrics import confusion_matrix

ypqs1=qb.predict(X_test)

results1=confusion_matrix(y_test,ypqs1)



print(results1)

from sklearn.metrics import confusion_matrix

ypqs2=tv.predict(X_test)

results2=confusion_matrix(y_test,ypqs2)



print(results2)

a=359/640

print(a)
X_test.describe()
# save model

import pickle 

file_name='wine_qual.sav'

tuples=(yx,X)

pickle.dump(tuples,open(file_name,'wb'))