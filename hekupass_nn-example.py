# Please bear with my awful English ^^;

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn
train_data = pd.read_csv("../input/train.csv") # Load training data

test_data = pd.read_csv("../input/test.csv") # Load testing data

train_data # Check

train_data.isnull().sum()
# =-=-= Filling NaN value in Age =-=-=

plt.hist(train_data['Age'].dropna(), color = "blue")

plt.show()



mode_train = train_data['Age'].dropna().mode()[0]

train_data['Age'] = train_data['Age'].fillna(mode_train) # Replace NaN for mode

mode_test = test_data['Age'].dropna().mode()[0]

test_data['Age'] = test_data['Age'].fillna(mode_test)

#train_data['Age']
# =-=-= Filling Nan value in Fare =-=-=

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].dropna().mode()[0])
# =-=-= Filling Nan value in Cabin =-=-=

train_data.isnull().sum()['Cabin'] / len(train_data['Cabin'].index)



"""

As you see, 77% of value in Cabin is NaN.

I decide to ignore Cabin columns because predicting using only 23% of entire Cabin data is not effective.

"""



train_data = train_data.drop('Cabin', axis = 1)

test_data = test_data.drop('Cabin', axis = 1)
# =-=-= Filling Nan value in Embarked =-=-=

seaborn.countplot(train_data['Embarked'].dropna())

plt.show()



"""

As you see the figure below, Filling NaN value with S seems good.

"""



train_data['Embarked'] = train_data['Embarked'].fillna("S")

test_data['Embarked'] = test_data['Embarked'].fillna("S")
train_data.isnull().sum()
dummy_sex = pd.get_dummies(train_data['Sex']) # Dummy variables for sex

dummy_embarked = pd.get_dummies(train_data['Embarked']) # Dummy variables for Embarked



train_data = pd.merge(train_data, dummy_sex, left_index=True, right_index = True)

train_data = pd.merge(train_data, dummy_embarked, left_index = True, right_index = True)



dummy_sex_test = pd.get_dummies(test_data['Sex'])

dummy_embarked_test = pd.get_dummies(test_data['Embarked'])



test_data = pd.merge(test_data, dummy_sex_test, left_index = True, right_index = True)

test_data = pd.merge(test_data, dummy_embarked_test, left_index = True, right_index = True)
train_data
# Now that all NaN is replaced. Let's move to train phase.

# =-=-= Train Phase =-=-=

from keras.models import load_model, Sequential

from keras.layers import Dense, Activation

from keras.callbacks import EarlyStopping, ModelCheckpoint



import h5py



X = train_data.loc[:, ["Pclass", "Age", "SibSp", "Parch", "Fare", "female", "male", "C", "Q", "S"]]

Y = train_data.loc[:, ["Survived"]]



X_test = test_data.loc[:, ["Pclass", "Age", "SibSp", "Parch", "Fare", "female", "male", "C", "Q", "S"]]



m = len(X.index)

n = len(X.columns)



early_stopping = EarlyStopping(monitor = 'val_loss', patience = 0, verbose = 1, mode = 'min')

model_check_point = ModelCheckpoint("ModelCheckPoint.h5",monitor = 'val_acc', save_best_only = True, mode = 'max')



model = Sequential()

model.add(Dense(64, input_dim = n))

model.add(Activation("sigmoid"))

model.add(Dense(128))

model.add(Activation("sigmoid"))

model.add(Dense(128))

model.add(Activation("sigmoid"))

model.add(Dense(1))

model.add(Activation("sigmoid"))



model.compile(optimizer = "adam", loss = "mse", metrics = ["accuracy"])

model.summary()



model.fit(X.values, Y.values, epochs = 50000, verbose = 1, validation_split = 0.05, callbacks = [early_stopping, model_check_point])



model.save("FinalModel.h5")
# =-=-= Predict Phase =-=-=

del model

model = load_model("FinalModel.h5")

#model = load_model("ModelCheckpoint.h5")

prediction = model.predict(X_test.values, verbose = 1)
# Decision Boundary

threshold = 0.5



prediction[prediction >= threshold] = 1

prediction[prediction < threshold] = 0
prediction.astype(int)



passenger_id = np.c_[test_data["PassengerId"].values]

prediction = np.c_[prediction]



prediction = np.append(passenger_id, prediction, axis = 1).astype(int)
np.savetxt("prediction.csv", prediction, fmt = "%.0f", delimiter = ",", header = "PassengerId,Survived")