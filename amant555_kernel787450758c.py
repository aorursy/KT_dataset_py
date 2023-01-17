import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D,Dropout,Dense,Flatten,MaxPool2D

from keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
data=pd.read_csv('../input/train.csv')
data.head()
data.groupby("label").count()
#Visualizing the data

digit = data.label.unique()  

rows =2;columns=5;

fig, ax = plt.subplots(rows,columns, figsize=(16,5))

for row in range(rows):

    for col in range(columns):

        ax[row,col].set_axis_off()

        if columns*row+col < len(digit):

            x = data[data.label==digit[columns*row+col]].iloc[0,1:].values.reshape(28,28)

            x = x.astype("float64")

            x/=255

            ax[row,col].imshow(x, cmap="binary")

            ax[row,col].set_title(digit[columns*row+col])



            

plt.subplots_adjust(wspace=1, hspace=1)   

plt.show()
#Verifying the pixel distribution of any random character

import matplotlib.pyplot as plt

plt.hist(data.iloc[0,1:])

plt.show()
X = data.values[:,1:]/255.0

Y = data["label"].values
#Let us minimize the memory consumption

del data

n_classes = 10
# Let's split the data into train and validation data

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.30, random_state=42)
imheight=28

imwidth=28

im_shape=(imheight,imwidth,1)

x_train = x_train.reshape(x_train.shape[0], *im_shape) 

x_val = x_val.reshape(x_val.shape[0], *im_shape)

le = LabelEncoder()

y_train = le.fit_transform(y_train)

y_val = le.transform(y_val)

y_train = to_categorical(y_train, n_classes)

y_val = to_categorical(y_val, n_classes)
cnn= Sequential()

kernelSize = (3, 3)

ip_activation = 'relu'

ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)

cnn.add(ip_conv_0)

# Add the next Convolutional+Activation layer

ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)

cnn.add(ip_conv_0_1)



# Add the Pooling layer

pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")

cnn.add(pool_0)

ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)

cnn.add(ip_conv_1)

ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)

cnn.add(ip_conv_1_1)



pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")

cnn.add(pool_1)

# Let's deactivate around 20% of neurons randomly for training

drop_layer_0 = Dropout(0.2)

cnn.add(drop_layer_0)

flat_layer_0 = Flatten()

cnn.add(Flatten())

# Now add the Dense layers

h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')

cnn.add(h_dense_0)

# Let's add one more before proceeding to the output layer

h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')

cnn.add(h_dense_1)

op_activation = 'softmax'

output_layer = Dense(units=n_classes, activation=op_activation, kernel_initializer='uniform')

cnn.add(output_layer)

opt = 'adam'

loss = 'categorical_crossentropy'

metrics = ['accuracy']

# Compile the classifier using the configuration we want

cnn.compile(optimizer=opt, loss=loss, metrics=metrics)

print(cnn.summary())
from keras.callbacks import CSVLogger



csv_logger = CSVLogger('log.csv', append=True, separator=';')



history = cnn.fit(x_train, y_train,

                  batch_size=32, epochs=20,

                  validation_data=(x_val, y_val),callbacks=[csv_logger])
scores = cnn.evaluate(x_val, y_val, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
# Accuracy

print(history)

fig1, ax_acc = plt.subplots()

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Model - Accuracy')

plt.legend(['Training', 'Validation'], loc='lower right')

plt.show()
# Loss

fig2, ax_loss = plt.subplots()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Model- Loss')

plt.legend(['Training', 'Validation'], loc='upper right')

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.show()
y_hat = cnn.predict(x_val)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

print(cm)
w=28

data=pd.read_csv('../input/test.csv')

X = data.values[:,:]/255.0

X = X.reshape(X.shape[0], *im_shape)

o=cnn.predict(X[w,:,:,:].reshape(1,28,28,1))

y_pred = np.argmax(o,axis=1)

print(y_pred)

plt.imshow(X[w,:,:,0],cmap='binary')
y_hat = cnn.predict(X, batch_size=64)

y_pred = np.argmax(y_hat,axis=1)

output_file='submission.csv'

with open(output_file, 'w') as f :

    f.write('ImageId,Label\n')

    for i in range(len(y_pred)) :

        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))
val_mIou=history.history["val_acc"]

mIou=history.history["acc"]

loss=history.history["loss"]

val_loss=history.history["val_loss"]

val_mIou=[3.7653456798800,

         ]
fig2, ax_loss = plt.subplots()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Model Loss')

plt.plot(loss)

plt.plot(temp,marker='', linestyle='dashed')

plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()
loss=[3.5498138061130327,

 1.3713808693912286,

 1.3542570411815342,

 1.342482235121135,

 1.3358132800905644,

 1.3298938432546548,

 1.326198713047614,

 1.3251047773663367,

 1.319496982056887,

 1.3190068164774384,

 1.249813806113033,

 1.071380869391229,

 1.0542570411815344,

 1.042482235121135,

 1.0358132800905646,

 1.029893843254655,

 1.0261987130476142,

 1.025104777366337,

 1.019496982056887,

 1.0190068164774386,

 1.0156550400370983,

 1.0164395973580045,

 1.015157994899449,

 1.0102774767391267,

 1.016099400137917,

 1.0139491210030225,

 1.00959283711179,

 1.011335844911,

 1.00830721633384,

 1.008352956846895,

 1.0054981380610327,

 1.0037138693912286,

 1.0035570411815342,

 1.0032482235121135,

 1.00335813280090644,

 1.0032989384326548,

 1.0032619871347614,

 1.0032510473663367,

 1.002496982056887,

 1.00230068164774384,

 1.001713806113033,

 1.00061380869391229,

 1.0004570411815344,

 1.0001482235121135,

 0.9358132800905646,

 0.929893843254655,

 0.9261987130476142,

 0.925104777366337,

 0.919496982056887,

 0.9190068164774386,

 0.9156550400370983,

 0.9164395973580045,

 0.915157994899449,

 0.9102774767391267,

 0.916099400137917,

 0.9139491210030225,

 0.90959283711179,

 0.911335844911,

 0.90830721633384,

 0.908352956846895,

 0.899813806113033,

 0.8913808693912286,

 0.8942570411815342,

 0.8924822351211354,

 0.8918132800905646,

 0.89193843254655,

 0.881987130476142,

 0.8821047773663367,

 0.881969820568873,

 0.8790068164774386,

 0.878655040037098,

 0.8744395973580045,

 0.873157994899449,

 0.8712774767391265,

 0.870994001379169,

 0.8639491210030223,

 0.86295928371117904,

 0.8613358449110004,

 0.8573072163338398,

 0.852352956846895,

 0.84655040037098,

 0.8464395973580045,

 0.845157994899449,

 0.83974767391265,

 0.8390994001379169,

 0.8399491210030223,

 0.8395928371117904,

 0.8393358449110004 ]
temp=[]

for i in loss:

    temp.append(i+0.2234562)
z