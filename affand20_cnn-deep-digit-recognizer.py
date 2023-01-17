import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
train_data.head()
test_data.head()
train_data.isnull().any().describe()
test_data.isnull().any().describe()
def show_image(pixel, label, index):
    image2d = pixel.values.reshape(28,28)
    plt.subplot('33%d' % (index))
    plt.imshow(image2d, cmap=plt.cm.gray)
    plt.title(label)

plt.figure(figsize=(5,5))
sample_image = train_data.sample(9).reset_index(drop=True)

for index, image in sample_image.iterrows():
    label = image['label']
    pixel = image.drop('label')
    show_image(pixel, label, index)
    
plt.tight_layout()
print("Total all train data = %d " % len(train_data))
print("Total all test data = %d " % len(test_data), end='\n============================\n')

labels = train_data.sort_values('label', ascending=True).label.unique()

for i in labels:
    print('Total data train in class %d = %d' % (i, len(train_data[train_data['label'] == i])))

plt.figure(figsize=(10,5))
sns.countplot(train_data['label'], palette='icefire')
plt.show()
train_data.drop(columns=['label']).to_numpy().max()
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

X = train_data.drop(columns=['label']).values.reshape(train_data.shape[0], 28, 28, 1)
y = to_categorical(train_data['label'], num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=10,
    rescale=1./255,
    zoom_range=0.1,
#     shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(
#     rotation_range=20,
#     rescale=1./255,
#     zoom_range=0.2,
# #     shear_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1
# )

train_datagen.fit(X_train)
train_datagen.fit(X_test)

train_gen = train_datagen.flow(X_train, y_train, batch_size=32)
val_gen = val_datagen.flow(X_test, y_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# model = Sequential([
#     Conv2D(32, kernel_size=5, padding='Same', activation='relu', input_shape=(28,28,1)),
#     Conv2D(32, kernel_size=5, padding='Same', activation='relu'),
#     MaxPooling2D(2,2),
#     Dropout(0.25),
    
#     Conv2D(64, kernel_size=3, padding='Same', activation='relu'),
#     Conv2D(64, kernel_size=3, padding='Same', activation='relu'),
#     MaxPooling2D(pool_size=(2,2), strides=(2,2)),
#     Dropout(0.25),
    
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')
# ])

model = Sequential([
    Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(10, activation='softmax')
])
from tensorflow.keras.optimizers import RMSprop

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
#     optimizer=tf.optimizers.Adam(),
    metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# callbacks = [
#     EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10),
#     ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
#     ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
# ]

callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
history = model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // 32,
    validation_data=val_gen,
    validation_steps=len(X_test) // 32,
    epochs=100,
    verbose=1,
    callbacks=callbacks
)
train_loss, train_acc = model.evaluate(train_gen, verbose=0)
test_loss, test_acc = model.evaluate(val_gen, verbose=0)
print('Train acc: %.3f%%, Test acc: %.3f%%' % (train_acc*100, test_acc*100))
print('Train loss: %.3f%%, Test loss: %.3f%%' % (train_loss*100, test_loss*100))
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(0,callbacks[0].stopped_epoch+1)
print("Total epochs = {}".format(callbacks[0].stopped_epoch+1))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_class)
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, linewidths=0.01, fmt='0')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Confusion Matrix')
plt.show()
error = (y_pred_class - y_true != 0)
y_pred_class_error = y_pred_class[error]
y_pred_error = y_pred[error]
y_true_error = y_true[error]
X_test_error = X_test[error]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)), cmap=plt.cm.gray)
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
            
y_pred_error_prob = np.max(y_pred_error, axis=1)
true_prob_error = np.diagonal(np.take(y_pred_error, y_true_error, axis=1))
delta_pred_error = y_pred_error_prob - true_prob_error
sorted_delta_error = np.argsort(delta_pred_error)
most_important_error = sorted_delta_error[-6:]

plt.figure(figsize=(5,5))
display_errors(most_important_error, X_test_error, y_pred_class_error, y_true_error)
plt.tight_layout()
# predict result
test_processed = test_data.values.reshape(test_data.shape[0],28,28,1).astype("float32") / 255
results = model.predict(test_processed)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
plt.figure(figsize=(5, 5))
sample_test = test_data.head(9)
for index, image_pixels in sample_test.iterrows():
    label = results[index]
    show_image(image_pixels, label, index)
plt.tight_layout()
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("digit_output.csv",index=False)