# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

train_data.head()

Survials_By_Age= train_data.groupby('Age')['Survived'].sum().reset_index()

Survials_By_Age_Segment = []

age_difference = 5

max_age = 70

for i in range( max_age // age_difference):

    s=0

    for j in range (age_difference):

    

        s= s + Survials_By_Age.loc[ [i * age_difference + j , 'Age'],'Survived'][0]

    Survials_By_Age_Segment.append(s)

Survials_By_Age_Segment = pd.Series(Survials_By_Age_Segment, 

                                        index = list(range(0,max_age,age_difference)))

sns.barplot(y=Survials_By_Age_Segment, x = Survials_By_Age_Segment.index)

        

    

print(Survials_By_Age_Segment)

#Data visualization

boolean_Survivals = train_data ['Survived' ] == 1

Survivals = train_data[ boolean_Survivals ] 



# I wil use a decesion tree regressor

from sklearn.ensemble import RandomForestClassifier

titanic_model = RandomForestClassifier(random_state=1)

X=pd.DataFrame(train_data['Age'].fillna(0))

y=train_data['Survived']

titanic_model.fit(X, y)

X_test = pd.DataFrame(test_data['Age'].fillna(0))

prediction= titanic_model.predict(X_test)

L=[]

for i in range(len(prediction)):

    if prediction[i] > 0.5:

        L.append(1)

    else:

        L.append(0)



ayu= pd.DataFrame({'PassengerId' : test_data['PassengerId'],'Survived' : L})

ayu.reset_index()

ayu.to_csv('Test_1.csv', index = False)

print(pd.read_csv('Test_1.csv'))