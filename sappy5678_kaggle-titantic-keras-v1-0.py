!mkdir data

!kg download -c titanic

!mv *.csv data/
import pandas as pd

train = pd.read_csv("data/train.csv")

test  = pd.read_csv("data/test.csv")
train.head(5)
test.head(5)
# 刪除不需要的特徵

features = list(train.columns.values)

# Remove unwanted features

features.remove('Name')

features.remove('PassengerId')

features.remove('Survived')

print(features)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['Sex'] = le.fit_transform(train['Sex'])

test['Sex'] = le.fit_transform(test['Sex'])



train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

test['Fare'] = train['Fare'].fillna(train['Fare'].mean())



train['Age'] = train['Age'].fillna(train['Age'].mean())

test['Age'] = train['Age'].fillna(train['Age'].mean())



train['Embarked'] = train['Embarked'].fillna("S")

test['Embarked'] = test['Embarked'].fillna("S")

train['Embarked'] = le.fit_transform(train['Embarked'])

test['Embarked'] = le.fit_transform(test['Embarked'])



train['Cabin'] = train['Cabin'].fillna("None")

test['Cabin'] = test['Cabin'].fillna("None")

train['Cabin'] = le.fit_transform(train['Cabin'])

test['Cabin'] = le.fit_transform(test['Cabin'])



train['Ticket'] = le.fit_transform(train['Ticket'])

test['Ticket'] = le.fit_transform(test['Ticket'])
# 拉出特徵資料

y = train['Survived']

x = train[features]

test_x = test[features]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=32)
from keras.models import Sequential

import keras

model = Sequential()
from keras.layers import Dense

model.add(Dense(units=64, activation="relu", input_shape=(9,)))

model.add(keras.layers.core.Dropout(0.2))

model.add(Dense(units=64, activation='relu'))

model.add(keras.layers.core.Dropout(0.2))

model.add(Dense(units=2, activation='softmax'))
# to_categorical 是為了讓 int 能夠編碼成 float

from keras.utils import to_categorical

y_train = to_categorical(y_train)



# 建立模型

# model.compile(loss=keras.losses.categorical_crossentropy,

#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 訓練模型

model.fit(X_train,  y_train , epochs=500, batch_size=32)
# 評估模型

y_test = to_categorical(y_test)

loss_and_metrics = model.evaluate(X_test,  y_test, batch_size=128)
loss_and_metrics
# classes = model.predict(test_x, batch_size=32)

classes = model.predict_classes(test_x, batch_size=32)
print(classes)
submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": classes})

print(submission)



submission.to_csv('titanic_lin.csv', index=False)
!kg submit titanic_lin.csv -c titanic -m "My First Titanic Keras output"  
from keras.utils import plot_model

plot_model(model, to_file='model.png',show_shapes=True)
from IPython.display import Image

Image("model.png")