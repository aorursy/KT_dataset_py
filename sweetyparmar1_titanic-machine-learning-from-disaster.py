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
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
df1=pd.read_csv("../input/train.csv")
df2=pd.read_csv("../input/test.csv")
df1.head(5)

def discrete(x):
    if x=='male':
        return 1
    else :
        return 0
df1['Sex']=df1['Sex'].apply(discrete)
df2['Sex']=df2['Sex'].apply(discrete)

x=df1.iloc[:,[2,4]]
y=df1.iloc[:,1]
x_test=df2.iloc[:,[1,3]]

x_train,xtest,y_train,ytest=train_test_split(x,y,test_size=0.3)

svm=SVC()
svm.fit(x_train,y_train)
prediction=svm.predict(xtest)
print(accuracy_score(ytest,prediction))

y_pred=svm.predict(x_test)

submission = pd.DataFrame({'PassengerId':df2['PassengerId'],'Survived':y_pred})
submission.head()

filename = 'TitanicPredictions.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)



