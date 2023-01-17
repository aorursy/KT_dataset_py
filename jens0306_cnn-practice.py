import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(5)
missing_data = train.isnull().sum()
print(missing_data[missing_data > 0])
train_y = train['label']
train_x = train.drop('label', axis = 1)
train_x = train_x / 255.0
test = test / 255.0
train_x = train_x.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
train_y = to_categorical(train_y)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 2)
plt.imshow(train_x[0][:,:,0], cmap = 'gray')
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
# model.add(AvgPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2), strides = (2,2)))
# model.add(AvgPool2D(pool_size = (2, 2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

print(model.summary())
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.75 ** x)
from keras.callbacks import ReduceLROnPlateau
# annealer = ReduceLROnPlateau(monitor='val_acc', 
#                             patience=3, 
#                             verbose=1, 
#                             factor=0.5, 
#                             min_lr=0.00001)
history = model.fit_generator(datagen.flow(train_x,train_y, batch_size=16),
                              epochs = 1, 
                              validation_data = (val_x,val_y),
                              verbose = 2, 
                              steps_per_epoch = 500, 
                              callbacks=[annealer])
final_loss, final_acc = model.evaluate(val_x, val_y, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
def show_train_history(train_history, title, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(title)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()
show_train_history(history, 'Accuracy', 'acc', 'val_acc')
show_train_history(history, 'Loss', 'loss', 'val_loss')

prediction = model.predict(test)
# select the indix with the maximum probability
prediction = np.argmax(prediction, axis = 1)
submission = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),
                         "Label": prediction})
submission.to_csv("submission.csv", index=False)