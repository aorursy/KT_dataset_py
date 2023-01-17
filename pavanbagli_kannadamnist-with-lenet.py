import pandas            as pd

import numpy             as np

import matplotlib.pyplot as plt

import tensorflow        as tf

from tensorflow import keras
train=pd.read_csv('../input/Kannada-MNIST/train.csv')

print('The Train  dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
train.head(3)
label = train.pop('label')

label = tf.keras.utils.to_categorical(label, num_classes=10,dtype='float32')
train = train.values.reshape(-1,28,28,1)

train = train/255
model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(6, (5,5), activation='tanh', input_shape=(28, 28, 1))

       ,tf.keras.layers.AveragePooling2D(2,2)

       ,tf.keras.layers.Conv2D(16, (5,5), activation='tanh')

       ,tf.keras.layers.AveragePooling2D(2,2)

       ,tf.keras.layers.Flatten()                        # Flatten the results to feed into a DNN

       ,tf.keras.layers.Dense(120, activation='tanh')    # 120 neuron hidden layer

       ,tf.keras.layers.Dense(84,  activation='tanh')    # 84  neuron hidden layer

       ,tf.keras.layers.Dense(10,  activation='softmax')  

])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train, y=label, shuffle=True, validation_split=0.05, epochs=15)
acc         = history.history['acc']

val_acc     = history.history['val_acc']

loss        = history.history['loss']

val_loss    = history.history['val_loss']

epoch_count = range(1, len(acc)+1)



fig_1, axes = plt.subplots(1,2,figsize=(20,5))



axes[0].set_title("Train Accuracy vs Val Accuracy")

axes[0].set(xlabel="Epoch", ylabel="Accuarcy")

axes[0].plot(epoch_count, acc, epoch_count, val_acc)

axes[0].legend(['Train Acc', 'Val Acc'])



axes[1].set_title("Train Loss vs Val Loss")

axes[1].set(xlabel="Epoch", ylabel="Loss")

axes[1].plot(epoch_count, loss, epoch_count, val_loss)

axes[1].legend(['Train Loss', 'Val Loss'])



plt.show()
test=pd.read_csv('../input/Kannada-MNIST/test.csv')

test.head()
test_id = test.pop('id')

test = test.values.reshape(-1, 28, 28, 1)

test = test/255
Y_pred = model.predict(test)

Y_pred = np.argmax(Y_pred,axis=1)
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

print(sample_sub)
sample_sub['label']=Y_pred

sample_sub.to_csv('submission.csv',index=False)
print(sample_sub)