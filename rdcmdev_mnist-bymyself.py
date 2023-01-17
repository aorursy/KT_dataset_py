import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, BatchNormalization, Dropout, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from  keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
import seaborn as sns
from matplotlib import pyplot as plt

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 0.5

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))
dataset = pd.read_csv('../input/train.csv')
dataset.head()
X, y = dataset.loc[:, ['pixel' + str(i) for i in range(784)]].as_matrix(), dataset[['label']]
X.shape
temp = []
for x in X:
    temp.append(x.reshape(28, 28) / 255.0)
temp = np.array(temp)
temp = np.expand_dims(temp, -1)
temp.shape
X = temp
X.shape
lb = LabelBinarizer()
lb.fit(y)
lb.classes_
y = np.array(lb.transform(y))
y = np.array(y)
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42)
rl = regularizers.l1(1e-6)

model = Sequential()
model.add(Convolution2D(32, (5, 5), kernel_initializer='he_normal', activation='relu', padding='same', activity_regularizer=rl, input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Convolution2D(32, (5, 5), kernel_initializer='he_normal', activation='relu', padding='same', activity_regularizer=rl))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', activity_regularizer=rl))
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', activity_regularizer=rl))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu', activity_regularizer=rl))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', activity_regularizer=rl))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[fbeta, 'accuracy'])
model.summary()
data_generator = ImageDataGenerator(
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
data_generator.fit(X_train)
patience = 10
batch_size = 86
num_epochs = 50

early_stop = EarlyStopping(monitor='val_fbeta', patience=patience, mode='max')
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=(patience//4))
callbacks = [early_stop, reduce_lr]
history = model.fit_generator(data_generator.flow(X_train, y_train, batch_size), steps_per_epoch=len(X_train) / batch_size,
                    epochs=num_epochs, verbose=True,
                    validation_data=(X_test, y_test))
dataset = pd.read_csv('../input/test.csv')
dataset.head()
X = dataset.loc[:, ['pixel' + str(i) for i in range(784)]].as_matrix()
temp = []
for x in X:
    temp.append(x.reshape(28, 28) / 255.0)
temp = np.array(temp)
temp = np.expand_dims(temp, -1)
temp.shape
X = temp
X.shape
y_pred = model.predict_classes(X)
y_pred = pd.Series(y_pred, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), y_pred], axis = 1)
submission.to_csv("cnn_mnist_datagen.csv", index=False)
# Plot a confusion matrix

y_pred = model.predict_classes(X_val)
y_true = np.asarray([np.argmax(i) for i in y_val])

cm = confusion_matrix(y_true, y_pred)
cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.set(font_scale=4.5) 
fig, ax = plt.subplots(figsize=(30,20))
ax = sns.heatmap(cm_normalised, annot=True, linewidths=2.5, square=True, linecolor="Green", 
                    cmap="Greens", yticklabels=range(10), xticklabels=range(10), vmin=0, vmax=np.max(cm_normalised), 
                    fmt=".3f", annot_kws={"size": 25})
ax.set(xlabel='Predicted label', ylabel='True label')
y_pred = model.predict_classes(X_val)
y_true = np.asarray([np.argmax(i) for i in y_val])
fbeta_score(y_true, y_pred, average='micro', beta=1)