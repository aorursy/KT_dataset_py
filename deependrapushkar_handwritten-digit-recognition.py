import numpy as np

import pandas as pd

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import keras
import os

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape)

print(test.shape)
train.head()
train['label'].unique()
columns=[]

for col in train.columns:

    if col != 'label':

        columns.append(col)
X_train = train[columns]

print(X_train.shape)

y_train = train['label']

print(y_train.shape)
test.head()
X_test = test[columns]

print(X_test.shape)
print('first 5 training labels : ',y_train[:5])



# convert into one-hot encoded vectors using to_categorical function

num =10

y_train = keras.utils.to_categorical(y_train,num)



print('first 5 training labels after one-hot encoding : ',y_train[:5])
from keras.layers import Dense # Dense -> fully conected layer

from keras.models import Sequential



image_size = 784

num_classes = 10



model = Sequential()



model.add(Dense(units = 32, activation = 'sigmoid',input_shape =(image_size,)))

model.add(Dense(units = num_classes,activation ='softmax'))

model.summary()
import matplotlib.pyplot as plt

%matplotlib inline

batch_size = 128

model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics =['accuracy'])

history = model.fit(X_train,y_train,batch_size =batch_size,epochs =3,verbose = False, validation_split =0.1)

pred_val1 = model.predict_classes(X_train)

pred_val1 = keras.utils.to_categorical(pred_val1,num)

print(classification_report(y_train, pred_val1))

acc_model1 = accuracy_score(y_train, pred_val1)

print('accuracy score : ',acc_model1)





plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['training', 'validation'], loc='best')

plt.show()
def create_dense(layer_size):

    model = Sequential()

    model.add(Dense(layer_size[0],activation = 'sigmoid',input_shape =(image_size,)))

    

    for s in layer_size[1:] :

        model.add(Dense(units = s, activation = 'sigmoid'))

    model.add(Dense(units = num_classes,activation ='softmax'))

    

    return model



def evaluate(model,batch_size =128, epochs =5) :

    model.summary()

    model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics =['accuracy'])

    history = model.fit(X_train,y_train, batch_size = batch_size, epochs = epochs,verbose =False,validation_split =.1)

    pred_val = model.predict_classes(X_train)

    pred_val = keras.utils.to_categorical(pred_val,num)

    print(classification_report(y_train, pred_val))

    acc_model = accuracy_score(y_train, pred_val)

    print('accuracy score : ',acc_model)

    

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['training', 'validation'], loc='best')

    plt.show()



    print()
for layers in range(1, 5):

    model = create_dense([32] * layers)

    evaluate(model)
model = create_dense([32]*3)

evaluate(model,epochs =40)
for nodes in [32, 64, 128, 256, 512, 1024, 2048]:

    model = create_dense([nodes])

    evaluate(model)
for nodes_per_layer in [32, 128, 512]:

    for layers in [3, 4, 5]:

        model = create_dense([nodes_per_layer] * layers)

        evaluate(model, epochs=10*layers)
model = create_dense([1024]*3)

model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics =['accuracy'])

model.fit(X_train,y_train, batch_size = 128, epochs = 30,verbose =False,validation_split =.1)

prediction = model.predict_classes(X_test,verbose =0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

submissions.to_csv("out.csv", index=False, header=True)


