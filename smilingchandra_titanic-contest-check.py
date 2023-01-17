# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

#test = pd.read_csv('../input/test.csv')

#dataset = pd.concat([train,test])

#dataset.reset_index(drop=True)

dataset =train
dataset.info()
#plot bar chart on all column to check relationship with survival

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



def bar_chart(feature):

    live = dataset.groupby([feature,'Survived'])["Survived"].count()

    live.unstack().plot(kind='bar', stacked=True)



#bar_chart('Sex')

#bar_chart('Pclass')

for col in dataset.columns.values.tolist()[1:]:

    bar_chart(col)



   # Pclass=1

   # Sex =Female

   # sibSp =[0,1,2]

   # Parch =[0,1]

   # Embarked =C
dataset.head()
#convert string columns  into int for model & selecting columns that has impact on survival 

datasetnew = pd.get_dummies(dataset,columns = ['Sex','Embarked'])

X = pd.DataFrame (datasetnew, columns = ['Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S','Sex_female','Sex_male'])

Y = dataset['Survived']

X.info()
#fill nan values with columnmean 

X['Age'].fillna(X['Age'].mean(),inplace=True)

X.info()
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1,random_state=0)

X_train.head()
#DecisionTreeRegression Model 

from sklearn.tree import DecisionTreeRegressor

logreg = DecisionTreeRegressor()

logreg.fit(X_train, Y_train)

logreg.score(X_test,Y_test)

#SVC Model 

from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train, Y_train)

svc.score(X_test,Y_test)

#RandomForestRegressor model



from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X_train, Y_train)

forest_model.score(X_test,Y_test)
#GaussianNB model

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

gaussian.score(X_test,Y_test)
test_df = pd.read_csv('../input/test.csv')



test_df = pd.get_dummies(test_df,columns = ['Sex','Embarked'])

X_test = pd.DataFrame (test_df, columns = ['Age','SibSp','Parch','Embarked_C','Embarked_Q','Embarked_S','Sex_female','Sex_male'])



X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)

X_test.info()
prediction = svc.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": prediction

    })

submission.to_csv('submission_titanic.csv', index=False)