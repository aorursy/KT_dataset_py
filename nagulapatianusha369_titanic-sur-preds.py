# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
# reding training data:

tr_titan = pd.read_csv('../input/train.csv')

te_titan = pd.read_csv('../input/test.csv')

te_titan.tail()
sns.heatmap(tr_titan.isnull())
sns.boxplot(x = 'Pclass',y = 'Age',data = tr_titan)





means =  round(tr_titan.groupby(tr_titan.Pclass)['Age'].mean())

means

#means.iloc[0]
def ageNAfiller(cols):

    Age = cols[0]

    Pclass=cols[1]

    if pd.isnull(Age): 

        if Pclass ==1:

            return  means.iloc[0]

        elif Pclass == 2:

            return  means.iloc[1]

        else:

            return  means.iloc[2]

    else:

        return Age

    
tr_titan['Age']=tr_titan[['Age','Pclass']].apply(ageNAfiller,axis = 1)

tr_titan['Age'].head()
sns.heatmap(tr_titan.isnull())
tr_titan.drop('Cabin',axis = 1,inplace = True )

tr_titan.dropna(inplace = True)
sns.heatmap(tr_titan.isnull())
tr_titan.head()
sex = pd.get_dummies(tr_titan['Sex'],drop_first = True)




embark = pd.get_dummies(tr_titan['Embarked'],drop_first= True)

tr_titan = pd.concat([tr_titan,sex,embark],axis = 1)

tr_titan.drop(['PassengerId','Name','Sex','Ticket','Embarked'],axis = 1, inplace = True)

tr_titan.head()
te_titan['Age']=te_titan[['Age','Pclass']].apply(ageNAfiller,axis = 1)
te_titan.drop('Cabin',axis = 1,inplace = True )



sex = pd.get_dummies(te_titan['Sex'],drop_first = True)






embark = pd.get_dummies(te_titan['Embarked'],drop_first= True)

te_titan = pd.concat([te_titan,sex,embark],axis = 1)

te_titan.drop(['PassengerId','Name','Sex','Ticket','Embarked'],axis = 1, inplace = True)

te_titan.head()

from sklearn.model_selection import train_test_split

X = tr_titan[['Pclass','Age','SibSp','Parch','Fare','male','Q','S']]

y = tr_titan['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33)





X_test = te_titan



models = []

models.append(('Random Forest', RandomForestClassifier(n_estimators=200)))

models.append(('KNN', KNeighborsClassifier()))

models.append(('LOGR', LogisticRegression()))

models.append(('DTC', DecisionTreeClassifier()))



best_model = None

best_model_name = ""

best_valid = 0

for name, model in models:

    model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    accuracy = model.score(X_valid,y_valid)

    if accuracy > best_valid:

        best_valid = accuracy

        best_model = model

        best_model_name = name



print(f"Best model is {best_model_name}")

best_valid
len(X_test)

col_mask=X_test.isnull().any(axis=0)

col_mask

X_test = X_test.fillna(0)

predictions = best_model.predict(X_test)



len(predictions)
sub_titan = pd.read_csv('../input/test.csv')

sub_titan.head()

len(sub_titan)

P_Id = sub_titan['PassengerId']

len(P_Id)

predict_sur=pd.Series(predictions, name='Survived')

#predict_sur.tail()

result = pd.concat([P_Id, predict_sur], axis=1)

#result['Survived'].fillna(0, inplace = True)



result.to_csv('Submission.csv',index=False)



result.tail()



              




