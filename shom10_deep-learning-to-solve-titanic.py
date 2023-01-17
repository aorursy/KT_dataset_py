import numpy as np

import pandas as pd

%matplotlib inline 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler



from keras.utils.np_utils import to_categorical

from keras.utils import plot_model

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPool1D, BatchNormalization

from keras.layers.advanced_activations import PReLU



from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train
test
print(train.isnull().sum())
print(test.isnull().sum())
train['Title'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False) 

train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')

Title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 

train['Title'] = train['Title'].map(Title_mapping) 

train['Title'] = train['Title'].fillna(0)

del train['Name']

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Sex"] = train["Sex"].map({"male" : 0, "female" : 1})

train["Embarked"] = train["Embarked"].fillna("S")

train["Embarked"] = train["Embarked"].map({"S" : 0, "Q" : 1, "C" : 2})

train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])

train['Ticket_Lett'] = train['Ticket_Lett'].apply(lambda x: str(x)) 

train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'], np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 

train['Ticket_Lett']=train['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)

del train['Ticket'] 

train['Cabin_Lett'] = train['Cabin'].apply(lambda x: str(x)[0]) 

train['Cabin_Lett'] = train['Cabin_Lett'].apply(lambda x: str(x)) 

train['Cabin_Lett'] = np.where((train['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),train['Cabin_Lett'], np.where((train['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))

train['Cabin_Lett']=train['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1)

del train['Cabin'] 

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

train
test['Title'] = test.Name.str.extract(' ([A-Za-z]+).', expand=False) 

test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')

Title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 

test['Title'] = test['Title'].map(Title_mapping) 

test['Title'] = test['Title'].fillna(0)

del test['Name']

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"] = test["Sex"].map({"male" : 0, "female" : 1})

test["Embarked"] = test["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].map({"S" : 0, "Q" : 1, "C" : 2})

test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(x)[0])

test['Ticket_Lett'] = test['Ticket_Lett'].apply(lambda x: str(x)) 

test['Ticket_Lett'] = np.where((test['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test['Ticket_Lett'], np.where((test['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 

test['Ticket_Lett']=test['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)

del test['Ticket'] 

test['Cabin_Lett'] = test['Cabin'].apply(lambda x: str(x)[0]) 

test['Cabin_Lett'] = test['Cabin_Lett'].apply(lambda x: str(x)) 

test['Cabin_Lett'] = np.where((test['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),test['Cabin_Lett'], np.where((test['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))

test['Cabin_Lett']=test['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1)

del test['Cabin'] 

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

test["Fare"] = test["Fare"].fillna(test["Fare"].median())



test
train_x = train.drop(['PassengerId'], axis=1)

trainx_corr = train_x.corr()

plt.figure(figsize=(8, 8))

sns.heatmap(trainx_corr, cmap="Reds", annot=True, fmt="1.2f")

train_x = train_x.drop(['Survived'], axis=1)

train_y = train['Survived']



test_x= test.drop(['PassengerId'], axis=1)
stdsc = StandardScaler()

trainx_std = stdsc.fit_transform(train_x)

testx_std = stdsc.fit_transform(test_x)



X_train = np.reshape(trainx_std, (-1, 11, 1))

Y_train = to_categorical(train_y, num_classes = 2)

X_test = np.reshape(testx_std, (-1, 11, 1))
model = Sequential()

model.add(Conv1D(32, 5, padding='same', input_shape=(11, 1), activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Conv1D(32, 5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(MaxPool1D(2, padding='same'))

model.add(Conv1D(64, 3, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(2, activation='softmax'))



model.summary()
plot_model(model, to_file='model.png', show_layer_names=False, show_shapes=True)
model.compile(optimizer = "Adam" , loss='binary_crossentropy', metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
history = model.fit(X_train, Y_train, epochs = 30, verbose = 2, validation_split = 0.2, callbacks=[learning_rate_reduction])
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
pred_proba = model.predict(X_test)[:,1]

pred = np.where(pred_proba > 0.5,1,0)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})

submission.to_csv('submission_titanic.csv', index=False)