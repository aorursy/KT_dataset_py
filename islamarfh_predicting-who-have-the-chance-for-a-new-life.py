# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for ploting 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('/kaggle/input/titanic/train.csv')

dataset.head()
dataset.info()
dataset.describe()
# we have 12 column one dependent variable (Survived) and the remaining will be the independent variables 

# there passengerid will not be benfit in building model also the name but from the name we can extract a new columns we will fetch that passenger is mr or miss or mrs

dataset['Name']= dataset['Name'].str.lower()

for i in range (0 ,900): 

    if dataset['Name'].str.contains('mr.').sum()>0:

        dataset['title']='Married'

    elif dataset['Name'].str.contains('miss.').sum()>0:

        dataset['title']='UNMarried'        

    elif dataset['Name'].str.contains('ms.').sum()>0:

        dataset['title']='unknown'

    elif dataset['Name'].str.contains('mrs.').sum()>0:

        dataset['title']='Married'      
dataset.head()
dataset.isnull().any()
# we need to handle the null values in columns that have null 

# we have three columns have null value Age,cabin,Embarked  -- for age we can get mean and replace null by it but for other two the category we can get most frequent value 

# and apllied it instead of null
dataset['Age'].mean()
dataset=dataset.fillna(dataset.mean())
dataset.isnull().any()
dataset.head()
dataset.groupby(dataset['Embarked']).count()

# most frequently in column Embarked is vale S
dataset['Embarked']=dataset['Embarked'].fillna('S')
dataset.groupby(dataset['Cabin']).count()

# most frequently in column Embarked is value S
#THE COLUMN Cabin is same as Embarked so no need to use it to not duplicate variable and this will introduce bad quality in model  

dataset.head()
# we will change the dataframe to array to apply our model random forest and take columns only used in model

X=dataset.iloc[:, [2,4,5,6,7,9,11,12]].values

Y=dataset.iloc[:, 1].values
#  Encoding the categorical

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#  Encoding the categorical

labelencoder_X_2 = LabelEncoder()

X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6])
#  Encoding the categorical

labelencoder_X_3 = LabelEncoder()

X[:, 7] = labelencoder_X_3.fit_transform(X[:, 7])
onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()
#REMOVE ONE OF DUMMY COLUMN TO AVOID 

X_train=X[:, [0,1,2,3,4,5,6,8] ]
#feature scalling 

from sklearn.preprocessing import StandardScaler

sc =StandardScaler()

X_train=sc.fit_transform(X_train)
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

classifier.fit(X_train,Y)
pred_y=classifier.predict(X_train)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y,pred_y)
cm
#31 wrong prediction from total  so the acuuracy = 31/(545+315)=96.39%
test=pd.read_csv('/kaggle/input/titanic/test.csv')

test['Name']= test['Name'].str.lower()

for i in range (0 ,900): 

    if test['Name'].str.contains('mr.').sum()>0:

        test['title']='Married'

    elif test['Name'].str.contains('miss.').sum()>0:

        test['title']='UNMarried'        

    elif test['Name'].str.contains('ms.').sum()>0:

        test['title']='unknown'

    elif test['Name'].str.contains('mrs.').sum()>0:

        test['title']='Married'    

test['Age']=test['Age'].fillna(29.6)

test['Embarked']=test['Embarked'].fillna('S')
test.isnull().any()

dataset['Fare'].mean()
test['Fare']=test['Fare'].fillna(32.2)
X_test=test.iloc[:, [1,3,4,5,6,8,10,11]].values
#  Encoding the categorical

labelencoder_X_4 = LabelEncoder()

X_test[:, 1] = labelencoder_X_4.fit_transform(X_test[:, 1])

labelencoder_X_5 = LabelEncoder()

X_test[:, 6] = labelencoder_X_5.fit_transform(X_test[:, 6])

labelencoder_X_6 = LabelEncoder()

X_test[:, 7] = labelencoder_X_6.fit_transform(X_test[:, 7])

onehotencoder = OneHotEncoder(categorical_features = [1])

X_test = onehotencoder.fit_transform(X_test).toarray()

X_test=X_test[:, [0,1,2,3,4,5,6,8] ]

X_test=sc.transform(X_test)
pred_y=classifier.predict(X_test)
pred =pd.DataFrame (pred_y,columns=['Pred_Y'])
test['Pred_Y']=pred['Pred_Y']
test.head()