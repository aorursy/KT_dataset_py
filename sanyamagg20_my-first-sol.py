# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1=pd.read_csv("../input/titanic/train.csv")
df2=pd.read_csv('../input/titanic/test.csv')

D=[df1,df2]
for df in D :
    df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
df2
D
embark=pd.get_dummies(df1['Embarked'],drop_first=True)
sex=pd.get_dummies(df1['Sex'],drop_first=True)
df1=pd.concat([df1,sex,embark],axis=1)
df1.drop(['Sex','Embarked'],axis=1,inplace=True)

    
    
    
    


embark=pd.get_dummies(df2['Embarked'],drop_first=True)
sex=pd.get_dummies(df2['Sex'],drop_first=True)
df2=pd.concat([df2,sex,embark],axis=1)
df2.drop(['Sex','Embarked'],axis=1,inplace=True)

    
    
    
    


df2
sns.boxplot(x='Pclass',y='Age',data=train)
def age(cols) :
    Age=cols[0]
    Pclass=cols[1]
    if(pd.isnull(Age)):
        
        if Pclass==1 :
            return 39
        elif Pclass==2 :
            return 30
        elif Pclass==3 :
            return 25
        
    else :
        return Age
    
df1['Age']=df1[['Age','Pclass']].apply(age,axis=1)
df2['Age']=df2[['Age','Pclass']].apply(age,axis=1)                  
df2.loc[152,'Fare']=8.6539
X=df1.drop("Survived",axis=1)
y=df1["Survived"]
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,y)
predictions=model.predict(df2)
predictions
submit=pd.DataFrame(predictions)
submit=pd.concat([df2['PassengerId'],submit],axis=1)
submit
submit.to_csv('submission.csv',index=False)
fg=pd.read_csv("submission.csv")
fg