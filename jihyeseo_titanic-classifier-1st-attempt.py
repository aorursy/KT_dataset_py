# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

filenames = check_output(["ls", "../input"]).decode("utf8").strip()
df = pd.read_csv("../input/train.csv")

dg = pd.read_csv("../input/test.csv")

pId = dg['PassengerId']

print(df.dtypes)

df.head()



df.columns.values
varnames = df.columns.values



for varname in varnames:

    if varname not in ['Name', 'Ticket', 'Cabin'] and df[varname].dtype == 'object':

        lst = df[varname].unique()

        print(varname + " : " + str(len(lst)) + " values such as " + str(lst))
import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
sns.barplot(x="Parch", y="Survived", hue="SibSp", data=df);
sns.barplot(x="Pclass", y="Survived", hue="SibSp", data=df);
sns.barplot(x="Sex", y="Survived", hue="Embarked", data=df);
df.columns.values
df.groupby('Survived').mean()
df.groupby('Survived')['Age'].describe()
df.plot.scatter(x = 'Age', y ='Survived')
df.plot.scatter(x = 'Fare', y ='Survived')
sns.pairplot(x_vars = ['Age'], y_vars = ['Fare'], data = df, hue = 'Survived', size = 5)  

 
df.dtypes
sns.pairplot(x_vars = ['PassengerId'], y_vars = ['Age'], data = df, hue = 'Survived', size = 5)  



from sklearn.preprocessing import Imputer
df.describe()
df['Age'].fillna(df['Age'].mean(), inplace = True)
df.describe()
dg.describe()
dg['Age'].fillna(dg['Age'].mean(), inplace = True)

dg['Fare'].fillna(dg.groupby('Pclass')['Fare'].transform("mean"), inplace = True)
dg.describe()
del df['Name']

del df['PassengerId']

del df['Cabin']
del df['Ticket']
del dg['Name']

del dg['PassengerId']

del dg['Cabin']

del dg['Ticket']
df.dtypes
df = pd.get_dummies(df)
dg.describe()
dg = pd.get_dummies(dg)
#df
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
y = df['Survived']
del df['Survived']

X = df
X.head()
y.head()
X.describe()
log_reg.fit(X,y)
y_pred = log_reg.predict(dg)
dfout = pd.DataFrame({'PassengerId': pId, 'Survived': y_pred})
dfout.head()
dfout.to_csv('output.csv', index= False)