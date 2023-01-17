from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras import utils
%matplotlib inline 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
batch_size = 200
nb_classes = 10
nb_epoch = 40
img_rows, img_cols = 32, 32
img_channels = 3
X_train.shape
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = utils.to_categorical(y_train, nb_classes)
y_test = utils.to_categorical(y_test, nb_classes)
model = Sequential()
#
model.add(Conv2D(
    filters = 32,
    kernel_size = (3, 3),
    padding = 'same',
    input_shape = (img_rows, img_cols, img_channels),
    activation = 'relu'
))
#
model.add(Conv2D(
    filters = 32,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'
))
#
model.add(AveragePooling2D(pool_size = (2, 2)))
#
model.add(BatchNormalization())
#
model.add(GaussianDropout(0.25))
#
model.add(Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'
))
#
model.add(Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'
))
#
model.add(AveragePooling2D(pool_size = (2, 2)))
#
model.add(Conv2D(
    filters = 64,
    kernel_size = (4, 4),
    padding = 'same',
    activation = 'elu'
))
#
model.add(Conv2D(
    filters = 128,
    kernel_size = (3, 4),
    activation = 'relu'
))
#
model.add(BatchNormalization())
#
model.add(GaussianDropout(0.4))
#
model.add(Flatten())
#
model.add(Dense(1024, activation = 'relu'))
#
model.add(Dropout(0.6))
#
model.add(Dense(nb_classes, activation = 'softmax'))
model.summary()
callback_list = [
    EarlyStopping(monitor = 'val_loss', patience = 5),
    ModelCheckpoint(
        filepath = 'my_model.h5',
        monitor = 'val_loss',
        save_best_only = True
    ),
    ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience = 3
    )
]
#
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)
%%time
history = model.fit(
    x = X_train,
    y = y_train,
    batch_size = batch_size,
    epochs = nb_epoch,
    callbacks = callback_list,
    validation_split = 0.1,
    verbose = 1
)
scores = model.evaluate(
    X_test,
    y_test,
    verbose = 1
)
print(round(scores[1] * 100, 4))
plt.plot(history.history['accuracy'], 
         label='Доля правильных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label='Доля правильных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля правильных ответов')
plt.legend()
plt.show()
plt.plot(history.history['loss'], 
         label='Оценка потерь на обучающем наборе')
plt.plot(history.history['val_loss'], 
         label='Оценка потерь на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Оценка потерь')
plt.legend()
plt.show()