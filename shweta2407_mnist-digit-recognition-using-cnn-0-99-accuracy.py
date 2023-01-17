import pandas as pd
import numpy as np
train_data = pd.read_csv('../input/digit-recognizer/train.csv')
test_data = pd.read_csv('../input/digit-recognizer/test.csv')

train_data.head()
y_train = train_data['label']
x_train = train_data.drop(['label'], axis=1)

del train_data
import seaborn as sns

sns.set(style='white', context='notebook', palette='Paired')

sns.countplot(y_train)

y_train.value_counts()
x_train.isnull().any().describe()
test_data.isnull().any().describe()
x_train = x_train / 255.0
test_data = test_data / 255.0

x_train.head()
x_train = x_train.values.reshape(-1, 28 , 28, 1)
test_data = test_data.values.reshape(-1, 28 , 28, 1)
test_data.shape
x_train.shape
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 10)
y_train[0]
from sklearn.model_selection import train_test_split

random_seed = 4

x, x_val, y, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)
import matplotlib.pyplot as plt

# Some examples
plt.figure(figsize=(10, 10))

for i in range(6):  
    plt.subplot(3, 3, i+1)
    plt.imshow(x[i][:,:,0])
    
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.applications.xception import Xception
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.summary()

plot_model(model, show_shapes=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
filepath = './model-ep{epoch:02d}-acc{val_accuracy:.3f}.h5'
callbacks = [   
             ReduceLROnPlateau(monitor='val_acc', 
                                patience=3, 
                                verbose=1, 
                                factor=0.5, 
                                min_lr=0.00001),          
            ModelCheckpoint(filepath= filepath, save_best_only = True, monitor='val_loss', mode='min')
            ]
EPOCHS = 20

history = model.fit(x,  
                    y,              
                    verbose = 1,            
                    epochs = EPOCHS, 
                    validation_data=(x_val, y_val),
                   callbacks = callbacks)
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
rows = 5
cols = 5

plt.figure(figsize=(10,10))
for index in range(rows*cols):
    img = test_data[index].reshape(1, 28, 28, 1)
    pred = np.argmax(model.predict(img))
    plt.subplot(rows, cols, index+1)
    plt.imshow(test_data[index][:,:,0])
    plt.xlabel('Predicted : {}'.format(pred))

plt.tight_layout()
plt.show()
results =[]
for index in range(28000):
    img = test_data[index].reshape(1, 28, 28, 1)
    pred = np.argmax(model.predict(img))
    results.append(pred)
submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, 28001)]
submission['Label'] = results
submission.to_csv('./my_submission.csv', index=False)
