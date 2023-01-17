#importing require libraries

import pandas            as pd

import numpy             as np

import matplotlib.pyplot as plt

from pandas         import DataFrame

from keras.layers   import Dense

from keras.layers   import Dropout

from keras.layers   import Flatten

from keras.layers   import LeakyReLU

from keras          import models

from keras.layers   import MaxPool2D

from keras.optimizers           import Adam

from keras.layers.convolutional import Conv2D





# load the downloaded dataset

dataset_train  =pd.read_csv('../input/mnist-digit-dataset/mnist_train.csv')

dataset_test   =pd.read_csv('../input/mnist-digit-dataset/mnist_test.csv')

dataset_Ptest  =pd.read_csv('../input/digit-recognizer/test.csv')

submission     =pd.read_csv('../input/digit-recognizer/sample_submission.csv')



#splitting the training data

dataset_train_X  =np.asarray(dataset_train.iloc[:,1:]).reshape([len(dataset_train), 28, 28, 1])

dataset_train_Y  =np.asarray(dataset_train.iloc[:,:1]).reshape([len(dataset_train), 1])



#splitting the test data

dataset_test_X  =np.asarray(dataset_test.iloc[:,1:]).reshape([len(dataset_test), 28, 28, 1])

dataset_test_Y  =np.asarray(dataset_test.iloc[:,:1]).reshape([len(dataset_test), 1])



#formatting the data to be predicted

dataset_Ptest  =np.asarray(dataset_Ptest).reshape([len(dataset_Ptest), 28, 28, 1])



#converting pixel value in the range 0 to 1

dataset_train_X  =dataset_train_X/255

dataset_test_X   =dataset_test_X/255

dataset_Ptest    =dataset_Ptest/255



#initilizing model

model = models.Sequential()



# Block 1

model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))

model.add(LeakyReLU())



model.add(Conv2D(32,3, padding  ="same"))

model.add(LeakyReLU())



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Block 2

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())



model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(10,activation="sigmoid"))



model.compile(Adam(lr=0.001), loss="sparse_categorical_crossentropy" ,metrics=['accuracy'])

model.summary()



#predicting the data

predicted_value=model.predict(dataset_Ptest)

classes=[0,1,2,3,4,5,6,7,8,9]

list1=[]

for index in range(0, len(predicted_value)):

    list1.append(classes[np.argmax(predicted_value[index])])

results= DataFrame(list1, columns=['Label'])

    

#Conveting the output to Survided_data.csv file 

submission['Label']=pd.DataFrame(results, columns=['Label'])

submission.to_csv('./submission.csv', index= False)







#visualtization

history_1 = model.fit(dataset_train_X,dataset_train_Y,batch_size=256,epochs=10,validation_data=[dataset_test_X,dataset_test_Y])



# Diffining Figure

f = plt.figure(figsize=(20,7))



#Adding Subplot 1 (For Accuracy)

f.add_subplot(121)



plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()



print('Completed')




























