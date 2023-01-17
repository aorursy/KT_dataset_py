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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv").fillna(value=0)

test_data = pd.read_csv("/kaggle/input/titanic/test.csv").fillna(value=0)



#Filter the Cabin Number out and get only the class

cabins = [list(str(x)) for x in train_data["Cabin"]]

train_data["Cabin"] = [c[0] for c in cabins]



cabins = [list(str(x)) for x in test_data["Cabin"]]

test_data["Cabin"] = [c[0] for c in cabins]

                      

train_data.head()

import tensorflow as tf

import numpy as np



features = ["Pclass", "Sex", "SibSp", "Parch","Age","Sex","Fare","Embarked","Cabin"]

x_train = pd.get_dummies(train_data[features])

x_train.head()



x_test = pd.get_dummies(test_data[features])

from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

x_train_scaled = min_max_scaler.fit_transform(x_train)

#convert back to pd dataframe

x_train_scaled = pd.DataFrame(x_train_scaled)

x_train_scaled.head()
#Normalize Test data aswell

x_test_scaled  = min_max_scaler.fit_transform(x_test)

#convert back to pd dataframe

x_test_scaled = pd.DataFrame(x_test_scaled)

x_test_scaled.head()

print(x_test_scaled.max)
#Creating the model Object

model = tf.keras.models.Sequential()

np.shape(x_test_scaled)
model.add(tf.keras.layers.Dense(units = 640, activation = 'relu', input_shape=(22,)))

model.add(tf.keras.layers.Dense(units = 1280, activation = 'relu', input_shape=(640,)))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units = 1280, activation = 'relu', input_shape=(640,)))

model.add(tf.keras.layers.Dropout(0.2))        

model.add(tf.keras.layers.Dense(units = 640, activation = 'relu', input_shape=(640,)))



#Does relu really fit here? maybe change it?

#Output Layer

model.add(tf.keras.layers.Dense(units  = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'BinaryCrossentropy', metrics = ['accuracy'])

model.summary()

np.shape(x_train_scaled)
model.fit(x_train_scaled,train_data['Survived'], epochs = 10, batch_size = 1)

#x_train_scaled = data to train

#y_train = result if he survived or not
#test_loss, test_accuracy = model.evaluate(x_test_scaled,test_data['Survived'])

np.shape(x_test_scaled)
#Zero Paddin for x_test_scaled

x_test_scaled.head()

x_test_scaled['20'] = 0

x_test_scaled['21'] = 0

x_test_scaled.head()
prediction = model.predict(x_test_scaled)

#print(prediction)

#test_data.loc['Survived'] = prediction



output = []#test_data['PassengerId']

for c,s in enumerate(prediction):

    if(s > 0.5):

        output.append(1)

        

    else:

        output.append(0)



print(output)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': output})

print(output)

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")       