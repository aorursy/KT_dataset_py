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
import pandas as pd
def testNumber(number):
    g=0
    if 1<=number<12:
        g=4
    elif 12<=number<18:
        g=3
    elif 18<=number<50:
        g=1
    if number>50:
        g=2
    return g
tabela_original = pd.read_csv("../input/train.csv")
tabela_original=tabela_original.drop(columns=["Name","Ticket","Fare","Cabin"])
tabela_original['Sex'].replace('male', -1,inplace=True)
tabela_original['Sex'].replace('female',1,inplace=True)
tabela_original['Embarked'].replace('Q', 38,inplace=True)
tabela_original['Embarked'].replace('C',55,inplace=True)
tabela_original['Embarked'].replace('S',33,inplace=True)
tabela_original['Embarked'].fillna(value=0, inplace=True)
tabela_original['Age'].fillna(value=0, inplace=True)
tabela_original["Ageclass"]=tabela_original.Age.apply(testNumber)
tabela_original["Sexclass"]=4*tabela_original["Sex"]-2*tabela_original["Pclass"]
tabela_original=tabela_original.drop(columns=["Sex","Pclass"])
tabela_original


tabela_teste = pd.read_csv("../input/test.csv")
tabela_teste=tabela_teste.drop(columns=["Name","Ticket","Fare","Cabin"])
tabela_teste['Sex'].replace('male', -1,inplace=True)
tabela_teste['Sex'].replace('female',1,inplace=True)
tabela_teste['Embarked'].replace('Q', 38,inplace=True)
tabela_teste['Embarked'].replace('C',55,inplace=True)
tabela_teste['Embarked'].replace('S',33,inplace=True)
tabela_teste['Embarked'].fillna(value=0, inplace=True)
tabela_teste['Age'].fillna(value=0, inplace=True)
tabela_teste["Ageclass"]=tabela_teste.Age.apply(testNumber)
tabela_teste["Sexclass"]=4*tabela_teste["Sex"]-2*tabela_teste["Pclass"]
tabela_teste=tabela_teste.drop(columns=["Sex","Pclass"])



from sklearn.ensemble import AdaBoostClassifier

X_train = tabela_original.drop(columns = ['PassengerId','Survived']).copy()
Y_train = tabela_original['Survived'].copy()

adab = AdaBoostClassifier(n_estimators=500)

adab.fit(X_train,Y_train)

X_teste = tabela_teste.drop(columns= ['PassengerId']).copy()

Y_teste  = adab.predict(X_teste)

tmp = {"PassengerId":tabela_teste['PassengerId'], "Survived":Y_teste}

resultado = pd.DataFrame(tmp)
resultado.to_csv('submission7.csv', index=False)

tabela_original








        
        