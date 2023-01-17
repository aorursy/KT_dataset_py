default_path = '../input/'
from keras.datasets import mnist
import pandas as pd
train_csv = pd.read_csv(default_path+'train.csv')
test_csv = pd.read_csv(default_path+'test.csv')
import numpy as np
from sklearn.model_selection import train_test_split
kaggle_training_data = np.array(train_csv)
kaggle_testing_data = np.array(test_csv)
print(kaggle_training_data.shape, kaggle_testing_data.shape)

kaggle_feature = kaggle_training_data[:, 1:]
kaggle_label = kaggle_training_data[:, :1]
# keras dataset
(keras_train_X, keras_train_y), (keras_test_X, keras_test_y) = mnist.load_data()
keras_train_X = keras_train_X.reshape(-1, 28*28)
keras_test_X = keras_test_X.reshape(-1, 28*28)
print(keras_train_X.shape)
# concat
all_feature = np.r_[kaggle_feature, keras_train_X, keras_test_X]
all_label = np.r_[kaggle_label.ravel(), keras_train_y, keras_test_y]
print(all_feature.shape, all_label.shape)
X_train, X_test, y_train, y_test = train_test_split(all_feature, all_label, test_size = 0.1, shuffle=True)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
import seaborn as sns
sns.set(style='white')
g = sns.countplot(train_csv['label'])
# check missing value
train_csv.isnull().any().any()
# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28,1)
# One-hot encoding
from keras.utils import to_categorical
y_train_oneHot = to_categorical(y_train, num_classes= 10)
y_test_oneHot = to_categorical(y_test, num_classes=10)
import matplotlib.pyplot as plt
plt.imshow(X_train[0][:, :, 0])
from keras.layers import Convolution2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', input_shape = (28, 28, 1), activation='relu'))
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))       
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))       
model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))       
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(units=256, activation='relu',))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))

model.summary()
# optimizer = 'adam'
#model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
# optimizer = 'RMSprop'
model.compile(optimizer='RMSprop', loss = 'categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau
# To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).
# With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy is not improved after 3 epochs.

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)
from keras.preprocessing.image import ImageDataGenerator
# I did not apply a vertical_flip nor horizontal_flip since it could have lead to misclassify symetrical numbers such as 6 and 9.

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)


# Fit the model
history = model.fit_generator(datagen.flow(X_train, y_train_oneHot, batch_size=512),
                              epochs = 30, validation_data=(X_test, y_test_oneHot),
                              verbose = 1, steps_per_epoch=X_train.shape[0]/512, 
                              callbacks=[learning_rate_reduction])
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
ax.plot(history.history['acc'], label='Train acc')
ax.plot(history.history['val_acc'], label='Val acc')
ax.legend(loc='best')

ax1.plot(history.history['loss'], label='Train loss')
ax1.plot(history.history['val_loss'], label='Val loss')
ax1.legend(loc='best')
model.evaluate(X_test, y_test_oneHot)
testing_data = scaler.transform(kaggle_testing_data).reshape(-1, 28, 28, 1)
CNN_prediction = model.predict_classes(testing_data)
result = pd.DataFrame(CNN_prediction)
result.index += 1
result.index.name = 'ImageId'
result.columns = ['Label']
result.to_csv('results_CNN_.csv',header=True)
result.head()
