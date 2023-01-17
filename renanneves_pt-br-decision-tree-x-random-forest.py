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
df_test = pd.read_csv('../input/test.csv')
df_train = pd.read_csv('../input/train.csv')
df_train.head()

del df_train['PassengerId']
del df_train['Name']
df_id = df_test.pop('PassengerId')#vamos aproveitar essa coluna para concatenar nosso resultado da predição com os ids dos passageiros.
del df_test['Name']
df_test.head()

df_train.head()
del df_train['Ticket']
del df_test['Ticket']
df_train.head()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
print(df_train.isnull().sum())
df_train.isnull().sum().plot(kind='Bar')
del df_train['Cabin']
del df_test['Cabin']
df_train.head()
sns.boxplot(x='Pclass',y='Age',hue='Sex', data=df_train )

def test_class(df):
    #indice 0- classe 1, indice 1-classe 2 ,indice 2-classe 3.
    media_homem =[]
    media_mulher=[]
    
    #Vamos criar as variáveis com os valores das médias dos homens e mulheres por classe:
    for x in range(1,4):
        media_homem.append(int(df['Age'][(df['Sex']=='male')&(df['Pclass']==x)].mean()))
        media_mulher.append(int(df['Age'][(df['Sex']=='female')&(df['Pclass']==x)].mean()))
    
    return(media_homem, media_mulher)


def input_age(df):
    age = df[0]
    pclass=df[1]
    sex=df[2]
    
    if pd.isnull(age):
        if pclass == 1:
            if sex == 'male':
                return med_men[0]
            else:
                return  med_women[0]
        elif pclass == 2:
            if sex == 'male':
                return  med_men[1]
            else:
                return med_women[1]
        else:
            if sex == 'male':
                return  med_men[2]
            else:
                return med_women[2]
    else:
        return age   
med_men,med_women = test_class(df_train)
df_train['Age'] = df_train[['Age','Pclass','Sex']].apply(input_age, axis=1)

#agora iremos fazer o mesmo com o dataset test
med_men,med_women = test_class(df_test)
df_test['Age'] = df_test[['Age','Pclass','Sex']].apply(input_age, axis=1)
print(df_train.isnull().sum())
df_train.isnull().sum().plot(kind='Bar', title='NaN df_train')


df_train.dropna(inplace=True)
print(df_train.isnull().sum())
df_train.isnull().sum().plot(kind='Bar', title='NaN df_train')
print(df_test.isnull().sum())
df_test.isnull().sum().plot(kind='Bar', title='NaN df_test')

df_test[df_test['Fare'].isnull()==True]
df_test.fillna(value=(df_test['Fare'][df_test['Pclass']==3].mean()),inplace=True)
print(df_test.isnull().sum())
df_test.isnull().sum().plot(kind='Bar')
df_train = pd.get_dummies(df_train, drop_first=True)
df_train.head()
df_test = pd.get_dummies(df_test, drop_first=True)
df_test.head()
from sklearn.tree import DecisionTreeClassifier
X= df_train.drop(labels='Survived', axis=1)
y= df_train['Survived']
dtc = DecisionTreeClassifier()
dtc.fit(X,y)
dtc.score(X,y)
pred = dtc.predict(df_test)
df_result = pd.DataFrame(data={'PassengerId':df_id.values , 'Survived':pred})
df_result.to_csv(path_or_buf='./sub_decision_tree.csv',index=False)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X,y)
rfc.score(X,y)
pred_rf = rfc.predict(df_test)

df_result_rf = pd.DataFrame(data={'PassengerId':df_id.values , 'Survived':pred_rf})
df_result_rf.to_csv(path_or_buf='./sub_random_forest.csv',index=False)
