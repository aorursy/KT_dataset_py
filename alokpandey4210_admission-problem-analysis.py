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
df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
df.corr(method='pearson')
df.columns = df.columns.str.replace('\s+', '_') # in case there are multiple white spaces
df.head()
X=df.drop(['Serial_No.','Chance_of_Admit_'],axis=1)
y=df['Chance_of_Admit_']
from sklearn.preprocessing import MinMaxScaler

scalerX=MinMaxScaler(feature_range=(0,1))

X_train[X_train.columns]=scalerX.fit_transform(X_train[X_train.columns])

X_test[X_test.columns]=scalerX.transform(X_test[X_test.columns])
X.head()

X.isnull().sum()
X['GRE_Score'].hist(bins=50)
X['TOEFL_Score'].hist(bins=50)
X.boxplot(column='GRE_Score',by='TOEFL_Score')
X['GRE_Score'].plot('density',color='Red')
X['TOEFL_Score'].plot('density',color='Black')
X['SOP'].hist(bins=50)
X.boxplot(column='SOP',by='TOEFL_Score')
X['SOP'].plot('density',color='Green')
X.boxplot(column='SOP',by='GRE_Score')
X['LOR_'].hist(bins=50)
X.boxplot(column='LOR_',by='TOEFL_Score')
X['LOR_'].plot('density',color='Pink')
X.boxplot(column='LOR_',by='GRE_Score')
X['CGPA'].hist(bins=50)
X.boxplot(column='CGPA',by='GRE_Score')
X['CGPA'].plot('density')
X['Research'].hist(bins=50)
X.boxplot(column='Research',by='CGPA')
X['Research'].plot('density',color='Blue')
X=pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.tree import DecisionTreeRegressor

q=DecisionTreeRegressor()

q.fit(X_train,y_train)

q.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

r=RandomForestRegressor()

r.fit(X_train,y_train)

r.score(X_test,y_test)
from sklearn.linear_model import LinearRegression

o=LinearRegression()

o.fit(X_train,y_train)

o.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesRegressor

e=ExtraTreesRegressor()

e.fit(X_train,y_train)

e.score(X_test,y_test)
from sklearn.neighbors import KNeighborsRegressor

n=KNeighborsRegressor()

n.fit(X_train,y_train)

n.score(X_test,y_test)
# save model

import pickle

file_name='Admission.sav'

tuples=(o,X)

pickle.dump(tuples,open(file_name,'wb'))
print(o.coef_)
print(o.intercept_)
c=print(o.predict(X_test))

c
print(y_test,' ',c)