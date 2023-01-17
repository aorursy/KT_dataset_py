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
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

labels = train['target'].values

train_id = train.pop("id")

test_id = test.pop("id")
print("Train Shape:",train.shape)

print("\n Column in train:",train.columns)

print("\n Test Shape:",test.shape)

print("\n Column in test:",test.columns)

print("\n Submission shape:",submission.shape)

print("\n Column in submission:",submission.columns)
type(labels)
data = pd.concat([train.drop('target',axis=1), test])

totaal=train.append(test)
data.shape
data.columns
totaal.columns
totaal.shape
totaal[499999:]
train.shape,test.shape,data.shape,totaal.shape
data.isna().sum()
data = data.drop(columns=['ord_3', 'ord_4','ord_5'])
data['bin_3'].unique(),data['bin_4'].unique()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['bin_3']=le.fit_transform(data['bin_3'].values)

data['bin_4']=le.fit_transform(data['bin_4'].values)
data['bin_3'].unique(),data['bin_4'].unique()
data = data.drop(columns=['nom_5', 'nom_6','nom_7','nom_8','nom_9'])
print(data['nom_0'].unique())

print(data['nom_1'].unique())

print(data['nom_2'].unique())

print(data['nom_3'].unique())

print(data['nom_4'].unique())
data.columns
data = pd.get_dummies(data=data,columns=['nom_0','nom_1','nom_2','nom_3','nom_4','day','month'], drop_first=True)
data.shape
print(data['ord_0'].unique())

print(data['ord_1'].unique())

print(data['ord_2'].unique())
level_mapping = {

    'Novice':0,

    'Contributor':1,

    'Expert':2,

    'Master':3,

    'Grandmaster':4

}
data['ord_1'] = data['ord_1'].map(level_mapping)
data['ord_1'].unique()
train['ord_2'].unique()
hot_mapping = {

    'Freezing':0,

    'Cold':1,

    'Warm':2,

    'Hot':3,

    'Boiling Hot':4,

    'Lava Hot':5

}
data['ord_2'] = data['ord_2'].map(hot_mapping)
data.ord_2.unique()
print(list(data.columns))
data.shape
trainX = data.iloc[:300000, :]

testX = data.iloc[300000:, :]
trainX.shape,testX.shape
trainY = pd.DataFrame()
trainY['target'] =labels
trainY.shape
trainY.head()
trainY.target.unique()
from sklearn.linear_model import LogisticRegression
# create logistic regression object 

reg = LogisticRegression() 

   

# train the model using the training sets 

reg.fit(trainX, trainY) 

  

# making predictions on the testing set 

y_pred = reg.predict(testX)
y_pred.shape
y_pred
submissionlr = pd.DataFrame()
submissionlr["id"] = test_id

submissionlr["target"] = y_pred
submissionlr.head()
submissionlr.to_csv("submissionlr.csv", index=False)