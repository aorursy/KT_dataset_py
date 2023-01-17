%load_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from datetime import datetime
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df_train.shape,df_test.shape
df_train.head()
df_test.head()
np.sum(df_train.isnull())
to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Age']
df_train.drop(to_drop, axis=1, inplace=True)
df_train.head()
df_train.Sex = df_train.Sex.astype('category')
train_categories = df_train.Sex.cat.categories
df_train.Sex = df_train.Sex.cat.codes
df_train.head()
train_categories
x_train = df_train.drop('Survived', axis=1)
y_train = df_train.Survived
x_train.shape,y_train.shape
mu = x_train.mean()
sigma  = x_train.std()
x_train = (x_train - mu)/sigma
x_train.mean(),x_train.std()
# All paramaters to their default values
m = LogisticRegression()
m.fit(x_train,y_train)
x_test = df_test.drop(to_drop, axis=1)
x_test.Sex = x_test.Sex.astype('category')
#This code comes from fastai library (apply_cats function)
x_test.Sex.cat.set_categories(train_categories, ordered=True, inplace=True)
x_test.Sex = x_test.Sex.cat.codes
x_test.head()
x_test.isnull().sum()
x_test.Fare[df_test.Fare.isnull()] = x_test.Fare.mean()
x_test.isnull().sum()
x_test = (x_test-mu)/sigma
pred = m.predict(x_test);pred
output = pd.concat([df_test.PassengerId,pd.Series(pred,name="Survived")],axis=1)
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d-%Hh%Mm%Ss")
output.to_csv(index=False,path_or_buf=f'output{dt_string}.csv')
