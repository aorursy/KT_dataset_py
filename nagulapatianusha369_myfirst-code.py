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
# reding training data:
tr_titan = pd.read_csv('../input/train.csv')
te_titan = pd.read_csv('../input/test.csv')
tr_titan.head()
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
te_titan.dropna(inplace = True)
sex = pd.get_dummies(te_titan['Sex'],drop_first = True)




embark = pd.get_dummies(te_titan['Embarked'],drop_first= True)
te_titan = pd.concat([te_titan,sex,embark],axis = 1)
te_titan.drop(['PassengerId','Name','Sex','Ticket','Embarked'],axis = 1, inplace = True)
te_titan.head()

X_train = tr_titan[['Pclass','Age','SibSp','Parch','Fare','male','Q','S']]
y_train = tr_titan['Survived']
X_test = te_titan


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
predictions = logreg.predict(X_test)
len(predictions)
accuracy = logreg.score(X_train,y_train)
accuracy

sub_titan = pd.read_csv('../input/test.csv')
sub_titan.head()

P_Id = sub_titan['PassengerId']
P_Id.head()
predict_sur=pd.Series(predictions, name='Survived')
#predict_sur.tail()
result = pd.concat([P_Id, predict_sur], axis=1)
result['Survived'].fillna(0, inplace = True)

result.to_csv('my_fisrt_submit2.csv',index=False)

result.tail()

              




