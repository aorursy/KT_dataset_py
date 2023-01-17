# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

#Making a new column Family Size from Sibsp and Parch

dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

test_data['FamilySize']= test_data['SibSp'] + test_data['Parch']



"""

#TO FIND WHICH COLUMNS HAVE MISSING VALUES

for column in dataset:

    print(column, dataset[column].isnull().sum())

    

print('-------------------------------------------------------------------------')

    

for column in test_data:

    print(column, test_data[column].isnull().sum())

"""





#We can see that Embarked has 2 Nan values which we will fill using the highest occured data (mode of that column)

freq=dataset.Embarked.dropna().mode()[0]

dataset['Embarked']=dataset['Embarked'].fillna(freq)



#Filling the missing value in Fare column in the test_data:

test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].dropna().median())



#Filling the missing values in age column

#if age column in used



#Label Encoding

le_x=LabelEncoder()

dataset['Sex']=le_x.fit_transform(dataset['Sex'])

test_data['Sex']=le_x.fit_transform(test_data['Sex'])



#Getting dummy variables for Embarked column anf also getting rid of one column to avoid dummy variable trap

Emb1=pd.get_dummies(dataset['Embarked'], prefix='Emb', drop_first=True) #drop_first to avoid dummy variable trap

dataset=pd.concat([dataset, Emb1], axis=1)



Emb2=pd.get_dummies(test_data['Embarked'], prefix='Emb', drop_first=True)

test_data=pd.concat([test_data, Emb2], axis=1)



#print(dataset.columns.values)

#print(test_data.columns.values)



#possible feature scaling
features=['Pclass', 'Sex', 'Fare', 'FamilySize', 'Emb_Q', 'Emb_S']

X=dataset[features]

y=dataset['Survived']

test_set=test_data[features]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
tf.logging.set_verbosity(tf.logging.ERROR)

#Creating the layers



layer_0=tf.keras.layers.Dense(units=1, input_shape=[6])

#model.add(Dense(output_dim=1))

model=tf.keras.Sequential([layer_0])



#Compiling the model

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))



#Fitting the model

model.fit(X_train, y_train, epochs=550, verbose=False)



print('its done')



y_prediction = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

cm=confusion_matrix(y_test, y_prediction)

res=accuracy_score(y_test, y_prediction)

print(cm, res)
model.fit(X, y, epochs=500, verbose=False)



#predicting for the test_set

test_pred=model.predict_classes(test_set)

print(len(test_set))

#to make the test_pred "1 Dimensional" as required in the submission csv file as test_pred is here

#in the form of [[0], [1], [0], ......]. We are converting it into [0, 1, 0, ....] form for submission.

flat_list = []

for sublist in test_pred:

    for item in sublist:

        flat_list.append(item)



        

#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':flat_list})

submission.to_csv('submission.csv',index=False)








