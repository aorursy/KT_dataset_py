import pandas as pd

import numpy as np
# import train data

df1 = pd.read_csv("../input/train.csv")
# quality data to quantity

df1 = df1.replace(["male", "female"], [0,1])

df1 = df1.replace(["S", "C", "Q"], [0,1,2])

df1= df1.fillna(0)
y = df1[["Survived"]]

X = df1[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
# convert to numpy array for NN

X = X.astype(np.float32).values

y = y.astype(np.float32).values
# import test data

df2 = pd.read_csv("../input/test.csv")
df2 = df2.replace(["male", "female"], [0,1])

df2 = df2.replace(["S", "C", "Q"], [0,1,2])

df2= df2.fillna(0)
X_Test = df2[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]

X_Test = X_Test.astype(np.float32).values
# Keras

from keras.models import Sequential

from keras.layers import Dense

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras.wrappers.scikit_learn import KerasRegressor
seed = 42

np.random.seed(seed)
# data split

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)
# Model 

model = Sequential()

#input layer

model.add(Dense(8, input_shape=(8,)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(0.4))



# hidden layers

model.add(Dense(8))

model.add(BatchNormalization())

model.add(Activation("sigmoid"))

model.add(Dropout(0.4))

    

model.add(Dense(4))

model.add(BatchNormalization())

model.add(Activation("sigmoid"))

model.add(Dropout(0.4))

    

model.add(Dense(2, activation="sigmoid"))

    

# output layer

model.add(Dense(1, activation='linear'))
# model compile for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# learning

model.fit(X, y, nb_epoch=300, batch_size=30)
# float to [0,1]

predictions = np.round(model.predict(X_Test))
predictions = pd.DataFrame(predictions)
# result

result = pd.concat([df1[["PassengerId"]], predictions], axis = 1)
predictions.to_csv("result.csv", index=False)