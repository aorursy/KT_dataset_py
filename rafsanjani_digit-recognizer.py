import keras
from keras.utils import to_categorical                # Convert categorical (y_train)
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd                                  # Load dataset (train.csv, test.csv)
import numpy as np                                   # Resize image
from PIL import Image
train = pd.read_csv('../input/train.csv', header=None, skiprows=1).iloc[:,:].values
test = pd.read_csv('../input/test.csv', header=None, skiprows=1).iloc[:,:].values
# For Training
x_train = train[:,1:]
y_train = train[:,0]

# For Testing
x_test  = test[:,:]
# The total features are 784. We can convert into image format which shape must be (28*28*1).

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)   # x_train.shape[0] = 42,000
x_test  = x_test.reshape(x_test.shape[0],   28, 28, 1)   # x_test.shape[0]  = 28,000
y_train = to_categorical(y_train, num_classes=10) # There is 10 distinct number (0, 1, 2, ..., 9)
batch_size = 128
nClass = 10
epochs = 5         # use epochs=30 for this dataset.
def getModel(batch_size, nClass, epochs):
    # Shape 28*28*1 = 784
    
    inputs = Input(shape=(28,28,1))
    x = Conv2D(filters=32, kernel_size=(3, 3))(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(nClass, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model
# Calling getModel(...)
model = getModel(batch_size, nClass, epochs)
model.summary()
def printModel(model):
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='model.png')
    
    return Image.open('model.png')
    ### --- Done --- ###
# Calling printModel( )
image = printModel(model)
image
history = model.fit(x=x_train, y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
)
# plotting the metrics
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

y_predict = model.predict(x_test, verbose=1)
y_predict
y_predict.shape
y_predict = np.argmax(y_predict, axis=1)
y_predict
with open('submission.csv', 'w') as F:
    F.write('ImageId,Label\n')
    for num, i in enumerate(y_predict):
        F.write('{},{}\n'.format(num+1, i))