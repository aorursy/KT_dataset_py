# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train/train.csv')

train_data = train_data.drop(['PetID','Name','Breed1','Breed2','Color1','Color2','Color3','RescuerID','VideoAmt','Description','PhotoAmt','State','Quantity'],axis=1)

print(list(train_data))

train_data['Type'] = train_data['Type'].map({1:'Dog',2:'Cat'})

train_data['Gender'] = train_data['Gender'].map({1 : 'Male', 2 : 'Female', 3 : 'Mixed'})

train_data['MaturitySize'] = train_data['MaturitySize'].map({1:'Small',2:'Medium',3:'Large',4:'Extra Large',0:'Not Specified'})

train_data['FurLength'] = train_data['FurLength'].map({1 : 'Male', 2 : 'Female', 3 : 'Mixed',0 : 'Not Specified'})

train_data['Vaccinated'] = train_data['Vaccinated'].map({1: 'Yes', 2 : 'No', 3 : 'Not Sure'})

train_data['Dewormed'] = train_data['Dewormed'].map({1 :'Yes', 2: 'No', 3: 'Not Sure'})

train_data['Sterilized'] = train_data['Sterilized'].map({1 :'Yes', 2: 'No', 3: 'Not Sure'})

train_data['Health'] = train_data['Health'].map({1 :'Healthy', 2: 'Minor Injury', 3: 'Serious Injury',0:'Not Specified'})

train_data['AdoptionSpeed'] = train_data['AdoptionSpeed'].map({0:'A',1:'B',2:'C',3:'D',4:'E'})

X = train_data.drop(['AdoptionSpeed'],axis=1)

X = pd.get_dummies(X)

y = train_data['AdoptionSpeed']
my_model = RandomForestClassifier()

my_model.fit(X, y)
test_data1 = pd.read_csv('../input/test/test.csv')

test_data = test_data1.drop(['PetID','Name','Breed1','Breed2','Color1','Color2','Color3','RescuerID','VideoAmt','Description','PhotoAmt','State','Quantity'],axis=1)

print(list(test_data))

test_data['Type'] = test_data['Type'].map({1:'Dog',2:'Cat'})

test_data['Gender'] = test_data['Gender'].map({1 : 'Male', 2 : 'Female', 3 : 'Mixed'})

test_data['MaturitySize'] = test_data['MaturitySize'].map({1:'Small',2:'Medium',3:'Large',4:'Extra Large',0:'Not Specified'})

test_data['FurLength'] = test_data['FurLength'].map({1 : 'Male', 2 : 'Female', 3 : 'Mixed',0 : 'Not Specified'})

test_data['Vaccinated'] = test_data['Vaccinated'].map({1: 'Yes', 2 : 'No', 3 : 'Not Sure'})

test_data['Dewormed'] = test_data['Dewormed'].map({1 :'Yes', 2: 'No', 3: 'Not Sure'})

test_data['Sterilized'] = test_data['Sterilized'].map({1 :'Yes', 2: 'No', 3: 'Not Sure'})

test_data['Health'] = test_data['Health'].map({1 :'Healthy', 2: 'Minor Injury', 3: 'Serious Injury',0:'Not Specified'})

test_data = pd.get_dummies(test_data)



predictions = my_model.predict(test_data)

predictions = pd.Series(predictions)

predictions = predictions.map({'A':0, 'B':1, 'C':2, 'D':3 ,'E':4})

print(np.unique(predictions))

# predictions = np.random.randint(5,size=len(test_data))

my_submission = pd.DataFrame({'PetID': test_data1.PetID, 'AdoptionSpeed': predictions})

my_submission.to_csv('submission.csv', index=False)