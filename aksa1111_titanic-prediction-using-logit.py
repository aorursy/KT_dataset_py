# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
frame = pd.read_csv("/kaggle/input/titanic/train.csv")
frame
print(sum(frame.Age >65))

print(sum(frame.Sex.isnull()))

print(sum(frame.SibSp.isnull()))

print(sum(frame.Parch.isnull()))

print(sum(frame.Pclass.isnull()))

print(sum(frame.Fare.isnull()))

frame.loc[frame.Age.isnull(),'Age'] = np.random.randint(0,65)

frame.loc[frame.Fare.isnull(),'Fare'] = np.random.randint(7.75,100.00)

frame.loc[frame.Cabin.isnull(),'Cabin'] = 'U'



frame.Cabin = frame.Cabin.astype(str).str.translate({ord(i): None for i in '1234567890 '})

print(frame.loc[frame.Cabin ==""])

frame.head()

for i in range(len(frame)):

    frame.loc[i,'Cabin'] = ord(str(frame.loc[i,'Cabin'])[0])

    

for i in range(len(frame)):

    if frame.loc[i,'Sex']=='female':

        frame.loc[i,'Sex'] = 1

     

    else:

         frame.loc[i,'Sex'] = 0

for i in range(len(frame)):

    if frame.loc[i,'Embarked']=='S':

        frame.loc[i,'Embarked'] = 0

     

    elif frame.loc[i,'Embarked']=='C':

         frame.loc[i,'Embarked'] = 1

    else:

         frame.loc[i,'Embarked'] = 2

    #frame.loc[i,'Age'] = frame.loc[i,'Age']//10



    

    
frame
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
logrg = LogisticRegression(max_iter=1000)

min_max_scaler = preprocessing.MinMaxScaler()

features = ['Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']

target = ['Survived']
#X = pd.DataFrame(min_max_scaler.fit_transform(frame[features]),columns = features)



y = np.array(frame[target])

y = np.hstack(y)

logrg.fit(frame[features],y)
test_set = pd.read_csv("/kaggle/input/titanic/test.csv")

test_set
test_set.loc[test_set.Age.isnull(),'Age'] = np.random.randint(0,65)

test_set.loc[test_set.Fare.isnull(),'Fare'] = np.random.randint(7.75,100)

test_set.loc[test_set.Cabin.isnull(),'Cabin'] = 'U'

test_set.Cabin = test_set.Cabin.astype(str).str.translate({ord(i): None for i in '1234567890 '})

test_set['Rel'] = test_set['SibSp']

for i in range(len(test_set)):

    test_set.loc[i,'Cabin'] = ord(str(test_set.loc[i,'Cabin'])[0])



for i in range(len(test_set)):

    if test_set.iloc[i,3]=='female':

        test_set.iloc[i,3] = 1

    elif test_set.iloc[i,3]=='male':

        test_set.iloc[i,3] = 0

for i in range(len(test_set)):

    if test_set.loc[i,'Embarked']=='S':

        test_set.loc[i,'Embarked'] = 0

    elif test_set.loc[i,'Embarked']=='C':

        test_set.loc[i,'Embarked'] = 1

    else:

        test_set.loc[i,'Embarked'] = 2

for i in range(len(test_set)):

    if test_set.loc[i,'SibSp']==0 and test_set.loc[i,'Parch']==0:

        test_set.loc[i,'Rel'] = 0

    else:

        test_set.loc[i,'Rel'] = 1

test_set



#frame_test = pd.DataFrame(min_max_scaler.fit_transform(test_set[features]),columns = features)
frame_test = test_set[features]
prediction = logrg.predict(frame_test)

prediction
test_predict = pd.DataFrame(prediction,index = test_set.PassengerId)

test_predict.columns = ['Survived']

test_predict
test_predict.to_csv("/kaggle/working/titanic_pred_scaled_with_combined_rel")