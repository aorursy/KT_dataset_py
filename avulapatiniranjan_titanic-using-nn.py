import tensorflow as tf

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

label_encoder_sex = LabelEncoder()
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
y = train_data[['Survived']]

X = train_data.drop(['Name','Ticket','Fare','Cabin','Embarked','Age','Survived'],axis =1)#['Survived','Name','PassengerId','SibSp','Parch','Ticket','Embarked'],1)



X.head()
#X = train_data.drop(['Name','Ticket','Fare','Cabin','Embarked','Age'],axis =1)#['Survived','Name','PassengerId','SibSp','Parch','Ticket','Embarked'],1)

#X.loc[X['Sex'] == 'male', 'sex'] = 0

#X.loc[X['Sex'] == 'female', 'sex'] = 1



#X.loc[X['Cabin'].isnull(),'cabin' ]=1

#X.loc[X['Cabin'].notnull(),'cabin' ]=1

#X = X.drop(['Sex','Cabin'],1)

X.iloc[:,2]  = label_encoder_sex.fit_transform(X.iloc[:,2])

X.head()
X_train = X[:700]

X_test = X[701:]

y_train = y[:700]

y_test = y[701:]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

X = sc.fit_transform(X)

#X_train.head()
#from keras.layers import Dense

model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(5, activation='relu',input_dim = 5),

        #tf.keras.layers.Dense(4, activation='relu'),

        tf.keras.layers.Dense(3, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')

    ])

model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Train the NN

history = model.fit(X, y,validation_split = 0.1, batch_size = 10, epochs = 100,shuffle=True)#validation_data=(X_test, y_test)
X_test = test_data.drop(['Name','Ticket','Fare','Cabin','Embarked','Age'],axis =1)

X_test.iloc[:,2]  = label_encoder_sex.fit_transform(X_test.iloc[:,2])

X_test = sc.fit_transform(X_test)

from matplotlib import pyplot as plt

#history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
predictions = np.round(model.predict(X_test))

final = pd.DataFrame({'PassengerId' : test_data.PassengerId,'Survived' : predictions[:,0]})
final.head()
final.to_csv('NN31.csv', index=False)

print("Your submission was successfully saved!")