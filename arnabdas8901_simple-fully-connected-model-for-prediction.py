import pandas as pd

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

print(train_df.shape)

print(test_df.shape)
train_df['Dependent'] = train_df.Parch + train_df.SibSp

test_df['Dependent'] = test_df.Parch + test_df.SibSp



train_df = train_df.replace(['female','male'],[20,10])

test_df = test_df.replace(['female','male'],[20,10])



test_PassengerId = test_df.PassengerId



train_df.drop(columns=['Ticket', 'Fare', 'Cabin', 'Embarked','Name','Parch','SibSp','PassengerId'], inplace=True)

test_df.drop(columns=['Ticket', 'Fare', 'Cabin', 'Embarked','Name','Parch','SibSp','PassengerId'], inplace=True)
dead_mean = train_df.loc[train_df['Survived']==0].Age.mean()

dead_std = train_df.loc[train_df['Survived']==0].Age.std()

live_mean = train_df.loc[train_df['Survived']==1].Age.mean()

live_std = train_df.loc[train_df['Survived']==1].Age.std()

print(dead_mean,dead_std)

print(live_mean, live_std)



test_age_mean = test_df.Age.mean()

test_age_std = test_df.Age.std()

print(test_age_mean,test_age_std)
import numpy as np

train_df.Age.fillna(train_df.Survived, inplace=True )

train_df.Age.replace({

    0.0 : round(np.random.normal(dead_mean,dead_std),1),

    1.0 : round(np.random.normal(live_mean,live_std),1)

}, inplace = True)



test_df.Age.fillna(round(np.random.normal(test_age_mean,test_age_std),1),inplace=True)



print(train_df.shape)

print(test_df.dropna().shape)
train_label = train_df.Survived.to_numpy()

train_label = train_label.reshape(train_label.shape[0],1)



train_df.drop(columns=['Survived'], inplace=True)
train_set = train_df.to_numpy()

test_set = test_df.to_numpy()



print ("Training data Size : ", train_set.shape)

print ("Test data Size : ", test_set.shape)
import keras

from keras import layers

from keras import regularizers

from keras.regularizers import l2

from keras import initializers

from keras.layers import Input, Dense, Flatten, BatchNormalization

from keras.models import Model

import keras.backend as K

K.set_image_data_format('channels_last')
def FCModel(input_shape):

   X_input = Input(input_shape)

   X = X_input

   X = Dense(32, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=0), kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(X)

   X = Dense(16, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=0), kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(X) 

   X = Dense(8, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=0))(X)

   X = Dense(4, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=0))(X)

   X = Dense(4, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=0))(X)

   X = Dense(2, activation='softmax', name='fc7')(X)



   model = Model(inputs = X_input, outputs = X, name='FCModel')



   return model
MyModel = FCModel(train_set.shape[1:])

MyModel.compile(optimizer = 'adam', loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

taining_result = MyModel.fit(x = train_set*0.01, y = train_label, epochs = 150, validation_split= 0.1, batch_size = 20)
import matplotlib.pyplot as plt

# %matplotlib inline



plt.plot(taining_result.history['accuracy'])

plt.plot(taining_result.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'Validation'], loc='upper left')

plt.show()
plt.plot(taining_result.history['loss'])

plt.plot(taining_result.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validaion'], loc='upper left')

plt.show()
MyModel.save('TitanicPredict_SimpleNeuralNetwork.h5')
predictions = np.argmax(MyModel.predict(test_set*0.01), axis = -1)
myTitanicPreiction_df = pd.DataFrame()

myTitanicPreiction_df['PassengerId'] = test_PassengerId

myTitanicPreiction_df['Survived'] = pd.DataFrame(predictions)

myTitanicPreiction_df.Survived.value_counts()
myTitanicPreiction_df.to_csv('MyPrediction_submission.csv', index=False)