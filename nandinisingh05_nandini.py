# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df1 = pd.read_csv("/kaggle/input/titanic/train.csv")
df1.head()
df1.info()
df1.describe()
df1.Survived.value_counts().plot(kind='bar',alpha=0.8)
plt.subplot2grid((2,3),(1,0),colspan=2)

for x in [1,2,3]:

    df1.Age[df1.Pclass == x].plot(kind = 'kde')

plt.title('class wrt age')

plt.legend(("1st","2nd","3rd"))
df1.Embarked.value_counts().plot(kind="bar",alpha=0.5)
df1.isnull().sum()
df1.drop(['PassengerId','Name','Ticket','Embarked','SibSp','Parch','Cabin'],axis = 1,inplace=True)
df1.head()
df1['Age'].describe()
df1['Age'].fillna(df1['Age'].mean(),inplace=True)
df1.isnull().sum()
male = pd.get_dummies(df1['Sex'],drop_first=True)

df1 = pd.concat([df1,male],axis = 1)
df1.head()
df1.drop(['Sex'],axis=1,inplace = True)
from sklearn.preprocessing import StandardScaler

sts = StandardScaler()
feature_scale = ['Age','Fare']

df1[feature_scale] = sts.fit_transform(df1[feature_scale])
df1.head()
x = df1.drop(['Survived'],axis = 1)

y = df1['Survived']
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
model_param = {

    'DecisionTreeClassifier':{

        'model':DecisionTreeClassifier(),

        'param':{

            'criterion': ['gini','entropy']

        }

    },

        'KNeighborsClassifier':{

        'model':KNeighborsClassifier(),

        'param':{

            'n_neighbors': [5,10,15,20,25]

        }

    },

        'SVC':{

        'model':SVC(),

        'param':{

            'kernel':['rbf','linear','sigmoid'],

            'C': [0.1, 1, 10, 100]

         

        }

    }

}
scores =[]

for model_name, mp in model_param.items():

    model_selection = GridSearchCV(estimator=mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)

    model_selection.fit(x,y)

    scores.append({

        'model': model_name,

        'best_score': model_selection.best_score_,

        'best_params': model_selection.best_params_

    })



df_model_score = pd.DataFrame(scores,columns=['model','best_score','best_params'])

df_model_score
model_svc = SVC( C= 100,kernel='rbf')
model_svc.fit(x, y)
df2 = pd.read_csv('/kaggle/input/titanic/test.csv')


df2.head()
df3=df2.drop(['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'], axis=1 )

df3.isnull().sum()
df3['Age'].fillna(df3['Age'].mean(),inplace=True)

df3['Fare'].fillna(df3['Fare'].mean(),inplace=True)
male=pd.get_dummies(df3['Sex'],drop_first=True)

df3= pd.concat([df3,male],axis=1)

df3.drop(['Sex'], axis=1, inplace=True )

df3.head()
df3[feature_scale] = sts.fit_transform(df3[feature_scale])

df3.head()
y_predicted = model_svc.predict(df3)
submission = pd.DataFrame({

        "PassengerId": df2['PassengerId'],

        "Survived": y_predicted

    })
submission.to_csv('titanic_submission_v02.csv', index=False)