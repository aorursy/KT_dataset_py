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
#Reading and Analysis

df = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df.head()
import matplotlib.pyplot as plt

import seaborn as sns
df.info()
df.describe()
df.info()
sns.pairplot(df, hue = 'Survived')

plt.show()
#Dropping useful columns

df.drop(columns = ['PassengerId','Name','Ticket','Cabin','Embarked'], inplace = True)

df_test.drop(columns = ['PassengerId','Name','Ticket','Cabin','Embarked'], inplace = True)
#Checking missing values

df.isnull().sum()
df.Age.median()
df.Age.fillna(28, inplace = True)
df.Survived.value_counts()
df.isnull().sum()
df_test.isnull().sum()
df_test.Age.fillna(28, inplace = True)
df_test.Fare.describe()
df[df.Pclass==3].median()
df_test.Fare.fillna(8.05, inplace = True)
df_test.isnull().sum()
#Getting the label in a different variable

y = df['Survived']
y
df.drop(columns=['Survived'], inplace = True)
df.head()
df.Sex.replace('female', 1, inplace=True)

df.Sex.replace('male', 0, inplace=True)

df_test.Sex.replace('female', 1, inplace=True)

df_test.Sex.replace('male', 0, inplace=True)
df.head()
#from sklearn.preprocessing import StandardScaler



#scaler = StandardScaler()
#scaler.fit(df.iloc[:,[2,5]])
#df.iloc[:,[2,5]] = scaler.transform(df.iloc[:,[2,5]])

#df.head()
#df_test.iloc[:,[2,5]] = scaler.transform(df_test.iloc[:,[2,5]])

#df_test.head()
df['Fare'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])

df.head()
df['Age'] = pd.qcut(df['Age'], 4, labels=[0, 1, 2, 3])

df.head()
#importing and fitting the model

#from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



#lr = LogisticRegression()

#parameters = {

#    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

#}



#clf = GridSearchCV(lr, parameters, cv = 5, n_jobs = -1)

#clf.fit(df, y)
from sklearn.ensemble import RandomForestClassifier



RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=10, scoring="accuracy", n_jobs=-1, verbose = 1)



gsRFC.fit(df,y)
#Make predictions

y_pred = gsRFC.predict(df_test)
#random_forest = RandomForestClassifier(n_estimators=100)

#random_forest.fit(df, y)

#y_pred = random_forest.predict(df_test)
Id = np.array(range(892,1310))
y_pred = pd.DataFrame(y_pred, columns = ['Survived'])
Id = pd.DataFrame(Id, columns = ['PassengerId'])
my_sub = pd.concat([Id, y_pred], axis = 1)

my_sub
my_submission = pd.DataFrame(my_sub)

my_submission.to_csv('submission.csv', index = False)