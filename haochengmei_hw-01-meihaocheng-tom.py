# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Get data

train_alldata=pd.read_csv('/kaggle/input/titanic/train.csv')

test_alldata=pd.read_csv('/kaggle/input/titanic/test.csv')

#train_alldate.head()

#test_alldate.head()
train_X=train_alldata[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

train_Y=train_alldata[['Survived']]

test_X=test_alldata[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

train_X.head()

#train_Y.head()

#test_X.head()

#test_Y.head()

#print(train_X.shape,train_Y.shape)
sum2=0

sum_num2=0

for i in range(0,len(train_X)) :

    if(train_X.iloc[i,1]=='male'):

        train_X.iloc[i,1]=1

    else:

        train_X.iloc[i,1]=2

    if(train_X.iloc[i,6]=='S'):

        train_X.iloc[i,6]=1 

    if(train_X.iloc[i,6]=='C'):

        train_X.iloc[i,6]=2

    if(train_X.iloc[i,6]=='Q'):

        train_X.iloc[i,6]=3

    if np.isnan(train_X.iloc[i,6]):

        train_X.iloc[i,6]=1

    if(train_X.iloc[i,2]==train_X.iloc[i,2]):

        sum_num2=sum_num2+1

        sum2=sum2+train_X.iloc[i,2]

    

for i in range(0,len(train_X)) :

    if(train_X.iloc[i,2]!=train_X.iloc[i,2]):

         train_X.iloc[i,2]=round(sum2/sum_num2)
sum2=0

sum_num2=0

test_X.iloc[152,5]=8

for i in range(0,len(test_X)) :

    if(test_X.iloc[i,1]=='male'):

        test_X.iloc[i,1]=1

    else:

        test_X.iloc[i,1]=2

    if(test_X.iloc[i,6]=='S'):

        test_X.iloc[i,6]=1 

    if(test_X.iloc[i,6]=='C'):

        test_X.iloc[i,6]=2

    if(test_X.iloc[i,6]=='Q'):

        test_X.iloc[i,6]=3

    if np.isnan(test_X.iloc[i,6]):

        test_X.iloc[i,6]=1

    if(test_X.iloc[i,2]==train_X.iloc[i,2]):

        sum_num2=sum_num2+1

        sum2=sum2+train_X.iloc[i,2]

    

for i in range(0,len(test_X)) :

    if(test_X.iloc[i,2]!=test_X.iloc[i,2]):

         test_X.iloc[i,2]=round(sum2/sum_num2)
test_X[test_X.isnull().values==True] 

train_X[train_X.isnull().values==True] 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(train_X,train_Y)


re=rf.predict(test_X)

rf.score(test_X,re)

re=pd.DataFrame(re)

re.columns=['Survived']

test_Y=test_alldata[['PassengerId']]

submission=pd.concat([test_Y.PassengerId,re.Survived],axis=1)

submission
filename = 'Titanic Predictions 6.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)