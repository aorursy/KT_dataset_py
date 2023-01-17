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
# loading the dataset
titanic=pd.read_csv('../input/titanic/train.csv')
# display the dataset
titanic
# display the columns of the dataset
titanic.columns
# analyse the columns so that which columns are not related to the target variable
titanic.describe().transpose()
# dropping the unecessary attributes from the dataset
# PassengerId,Name,Ticket,Cabin,Embarked
titanic=titanic.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1)
titanic
titanic.dtypes
# we have an attribute od sex as object hence we need to change the calues of the attribute to the number
# for male the risk factor is high hence for male we are choosing 1 
titanic['Sex']=titanic['Sex'].replace({'female':0,'male':1})
titanic
titanic.dtypes
# independent variables and dependent variable association
y=titanic['Survived']
x=titanic.drop(['Survived'],axis=1)
print(y)
print(x)
print(y.isna().sum())
print(x.isna().sum())
x['Age']=x['Age'].fillna(x['Age'].mean())
print(x.isna().sum())
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y)
test=pd.read_csv('../input/titanic/test.csv')
test
test=test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
test
test['Sex']=test['Sex'].replace({'male':1,'female':0})
test
test['Age']=test['Age'].fillna(test['Age'].mean())
test.isna().sum()
test['Fare']=test['Fare'].fillna(test['Fare'].median())
print(test)
test.isna().sum()
testsamp=test.iloc[:,1:]
testsamp
predict=model.predict(testsamp)
predict
predicted=pd.DataFrame(predict)
predicted['PassengerId']=test['PassengerId']
predicted=predicted.rename(columns={'PassengerId':'PassengerId',0:'Survived'})
column_titles=['PassengerId','Survived']
predicted
predicted=predicted.reindex(columns=column_titles)
predicted
ex=pd.read_csv('../input/titanic/gender_submission.csv')
ex
from sklearn import metrics
print(metrics.confusion_matrix(predicted['Survived'],ex['Survived']))
print(model.score(testsamp,ex.iloc[:,1]))
print("Performance:",model.score(testsamp,ex.iloc[:,1])*100)
predicted.to_csv('submission.csv',index=False)
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(predicted, title = "Download CSV file", filename = "submission.csv"):  
    csv = predicted.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predicted['Survived'],
    })
print(submission)
create_download_link(submission)
sample=pd.read_excel('../input/submission2/submission.xlsx')
sample
