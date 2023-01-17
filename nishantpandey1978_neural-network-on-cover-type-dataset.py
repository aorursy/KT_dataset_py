#Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
#read data from the file

dataset_columns=['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon',

'Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Rawah Wilderness Area','Neota Wilderness Area','Comanche Peak Wilderness Area','Cache la Poudre Wilderness Area',

'2702','2703','2704','2705','2706','2717','3501','3502','4201','4703','4704','4744','4758','5101','5151','6101','6102','6731','7101','7102','7103','7201','7202','7700',

'7701','7702','7709','7710','7745','7746','7755','7756','7757','7790','8703','8707','8708','8771','8772','8776','Cover_Type']

dataset=pd.read_csv('../input/cover-type-uci/covtype.data',sep=',',names=dataset_columns)
#segregate data into feature matrix X and Vector of Predictions y

X=dataset.drop('Cover_Type',axis=1)

y=dataset['Cover_Type']
#segregate the dataset into test and train data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#create NN model

model=keras.models.Sequential()

model.add(keras.layers.Dense(1024,input_shape=(54,),activation='relu'))

model.add(keras.layers.Dense(512,activation='relu'))

model.add(keras.layers.Dense(256,activation='relu'))

model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(64,activation='relu'))

model.add(keras.layers.Dense(8,activation='softmax'))
#compile the model

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
#train the model over 20 epochs

history = model.fit(X_train, y_train, epochs = 20)
pd.DataFrame(history.history['accuracy']).plot()

plt.gca().set_ylabel('Accuracy')

plt.gca().set_xlabel('Epochs')

plt.legend('Accuracy')

plt.show()
model.evaluate(X_test,y_test)
y_pred=np.argmax(model.predict(X_test),axis=1)

output_data=pd.DataFrame(y_test)

output_data.columns=['Actual']
output_data['Predicted']=y_pred
output_data.head(20)