import numpy as np

import pandas as pd

from tqdm import tqdm, tqdm_notebook

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dropout, BatchNormalization, Dense

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))
base_dir = os.path.join("..", "input")

train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))

test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
train_df.head()
print(train_df['Pclass'].isnull().sum(axis=0)) # count number of NULL occurrences

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() # correlation with Survived
print(train_df['Sex'].isnull().sum(axis=0)) 

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean() 
train_df['SibSp'].isnull().sum(axis=0), train_df['Parch'].isnull().sum(axis=0)
train_df['Family'] = train_df['SibSp'] + train_df['Parch'] + 1 # creation of new feature

train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
print(train_df['Fare'].isnull().sum(axis=0))

train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean()
print("NULL Records: " + str(train_df['Cabin'].isnull().sum(axis=0)))

print("Total Number of Records: " + str(len(train_df)))
train_df['Embarked'].isnull().sum(axis=0)
features = ['Pclass', 'Sex', 'Family', 'Fare']

x = train_df[features]

y = train_df['Survived']
LE = LabelEncoder()

x['Sex'] = LE.fit_transform(x['Sex'])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, shuffle=True)
%load_ext tensorboard.notebook

%tensorboard --logdir logs
model = Sequential([

    Dense(units=512, activation='relu', input_shape=(4,)),

    BatchNormalization(),

    Dropout(0.3),

    Dense(units=1024, activation='relu'),

    BatchNormalization(),

    Dropout(0.3),

    Dense(units=1024, activation='relu'),

    BatchNormalization(),

    Dropout(0.3),

    Dense(units=512, activation='relu'),

    BatchNormalization(),

    Dropout(0.3),

    Dense(units=1, activation='sigmoid')

])



model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['acc'])
ckpt_path = 'titanic.hdf5'



earlystop = EarlyStopping(monitor='val_acc', patience=20, verbose=1, restore_best_weights=True)

reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=1e-7)

modelckpt = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

tb = TensorBoard()



callbacks = [earlystop, reducelr, modelckpt, tb]
history = model.fit(x_train, 

                    y_train, 

                    batch_size=128, 

                    validation_data = (x_test, y_test),

                    epochs=100,

                    callbacks=callbacks)
# Training plots

epochs = [i for i in range(1, len(history.history['loss'])+1)]



plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")

plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")

plt.legend(loc='best')

plt.title('loss')

plt.xlabel('epoch')

plt.show()



plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")

plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")

plt.legend(loc='best')

plt.title('accuracy')

plt.xlabel('epoch')

plt.show()
test_df['Family'] = test_df['SibSp'] + test_df['Parch'] + 1

test = test_df[features]

test['Sex'] = LE.fit_transform(test['Sex']) # encode gender as a numerical value
pred = model.predict(test)

pred = (pred > 0.5).astype(int).reshape(test.shape[0])

test_df['Survived'] = pred



# Submitting to competition

output = test_df[['PassengerId', 'Survived']]

output.to_csv('submission.csv', index=False)