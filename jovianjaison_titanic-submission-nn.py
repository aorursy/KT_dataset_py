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
import pandas as pd

from sklearn import preprocessing

import numpy as np

from sklearn.impute import SimpleImputer

train_data = pd.read_csv('../input/titanic/train.csv')



train_data = train_data.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=0)



imp = SimpleImputer(missing_values=np.nan,strategy="mean")

temp_df = np.array(train_data[['Pclass','Age','SibSp','Parch','Fare']])

train_data[['Pclass','Age','SibSp','Parch','Fare']] = pd.DataFrame(imp.fit_transform(temp_df))





imp = SimpleImputer(missing_values=np.nan,strategy="most_frequent")

temp_df = np.array(train_data[['Sex','Embarked']])

train_data[['Sex','Embarked']] = pd.DataFrame(imp.fit_transform(temp_df))



train_data = train_data.set_axis(['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'], axis=1, inplace=False)



train_data_Y = train_data['Survived']

train_data = train_data.drop(columns=['Survived'],axis=0)



input_labels = ['male','female']

encoder = preprocessing.LabelEncoder()

encoder.fit(input_labels)



train_data['Sex'] = encoder.transform(train_data['Sex'])



input_labels1 = ['S','C','Q']

encoder1 = preprocessing.LabelEncoder()

encoder1.fit(input_labels1)



train_data['Embarked'] = encoder1.transform(train_data['Embarked'])



print(train_data)



'''train_data = preprocessing.normalize(train_data,norm='l1')

print(train_data)

'''

#Normalizing the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train_data = sc.fit_transform(train_data)



train_data = pd.DataFrame(imp.fit_transform(train_data))

train_data = train_data.set_axis(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'], axis=1, inplace=False)

train_data.insert(7,"Survived",train_data_Y,True)

print(train_data)

train_data = np.array(train_data)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection as ms

from sklearn import metrics



X,y = train_data[:,:-1],train_data[:,-1]

print(X.shape)

X = X.astype('float64') 

y = y.astype('int') 





# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

from keras.utils.np_utils import to_categorical

y = to_categorical(y, num_classes = 2)



X_train,X_test,y_train,y_test = ms.train_test_split(X,y,test_size=0.2,random_state=4)



#Dependencies

import keras

from keras.models import Sequential

from keras.layers import Dense



# Neural network

model = Sequential()

model.add(Dense(16, input_dim=7, activation='relu'))

model.add(Dense(12, activation='relu'))

model.add(Dense(2, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=64)
test_data = pd.read_csv('../input/titanic/test.csv')



pid = test_data['PassengerId']

test_data = test_data.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=0)





imp = SimpleImputer(missing_values=np.nan,strategy="mean")

temp_df = np.array(test_data[['Pclass','Age','SibSp','Parch','Fare']])

test_data[['Pclass','Age','SibSp','Parch','Fare']] = pd.DataFrame(imp.fit_transform(temp_df))





imp = SimpleImputer(missing_values=np.nan,strategy="most_frequent")

temp_df = np.array(test_data[['Sex','Embarked']])

test_data[['Sex','Embarked']] = pd.DataFrame(imp.fit_transform(temp_df))



test_data = test_data.set_axis(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'], axis=1, inplace=False)







test_data['Sex'] = encoder.transform(test_data['Sex'])



test_data['Embarked'] = encoder1.transform(test_data['Embarked'])



print(test_data)



'''test_data = preprocessing.normalize(test_data,norm='l1')

print(test_data)'''



sc = StandardScaler()

test_data = sc.fit_transform(test_data)
X = test_data[:,:]

print(X.shape)



X = X.astype('float64') 



y_pred = model.predict(X)

print(y_pred.shape)

#Converting predictions to label

pred = list()

for i in range(len(y_pred)):

    pred.append(np.argmax(y_pred[i]))

print(pred)

y_pred = pred



y_pred = pd.DataFrame(y_pred)

y_pred = y_pred.set_axis(['Survived'],axis=1,inplace=False)

y_pred.insert(0,'PassengerId',pid,True)



print(y_pred)

y_pred.to_csv('gender_submission.csv',index=False)




