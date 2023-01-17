import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras import datasets

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from keras.utils.vis_utils import plot_model
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(f"Missing values in the train set: {train_df.isnull().any(axis=None)}",

      f"Missing values in the test set: {test_df.isnull().any(axis=None)}", sep='\n')
y_train = pd.get_dummies(train_df['label']).values  # one-hot encoding

X_train = train_df.values[:,1:] / 255.0             # normalization

X_train = X_train.reshape((-1, 28, 28, 1))          # reshaping 



X_test = test_df.values / 255.0                     # normalization

X_test = X_test.reshape((-1, 28, 28, 1))            # reshaping
sns.countplot(np.argmax(y_train, axis=1))

plt.show()
def show_some_img(img_array: np.ndarray, label_array: np.ndarray, n: int):

    """

    Chooses n images randomly, plots them and prints their label

    img_array shape: (x1, x2, x3, 1)

    label_array shape: (x1, number_of_labels)

    """

    

    idexes_to_show = np.random.choice(len(img_array), n, replace=False)

    

    fig, axes = plt.subplots(1, n, figsize=(20, n))

    for i, sample_index in enumerate(idexes_to_show):

        axes[i].imshow(np.squeeze(img_array[sample_index]))

        axes[i].axis('off')

        axes[i].title.set_text(f"Label: {np.argmax(label_array[sample_index])}")
show_some_img(X_train, y_train, 10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.05, random_state=1)

X_train, X_local_test, y_train, y_local_test = train_test_split(X_train, y_train, test_size = 0.02, random_state=2)
print(f"Train set size: {len(X_train)}", f"Validation set size: {len(X_val)}", f"Test set size: {len(X_local_test)}", sep='\n')
img_rows, img_cols = 28, 28  # the shape of the images





model = Sequential()

model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same', input_shape=(img_rows, img_cols, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Dropout(0.2))



model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))





model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)
datagen = ImageDataGenerator(

    rotation_range=15,

    zoom_range = 0.15,

    width_shift_range=0.1,

    height_shift_range=0.1)



datagen.fit(X_train)
augmented_images, augmented_labels = datagen.flow(X_train, y_train, batch_size=10).next()  # get the data from the iterator

show_some_img(augmented_images, augmented_labels, 10)
batch_size = 400  # save the batch size to use later



training = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                               validation_data=(X_val, y_val), epochs=70, steps_per_epoch=X_train.shape[0] // batch_size)
def plot_training(training):

    plt.figure(figsize=(15, 6))

    

    plt.plot(training.history['accuracy'])

    plt.plot(training.history['val_accuracy'])



    plt.grid()

    plt.ylim((0.0, 1.0))

    

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')



    plt.show()
plot_training(training)
model.evaluate(X_local_test, y_local_test, batch_size=batch_size)
y_local_pred = model.predict(X_local_test)
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):

    conf_matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))



    plt.figure(figsize = (10, 6))

    sns.heatmap(conf_matrix, annot=True, cmap=plt.cm.copper)

    

    plt.title('Confusion matrix')

    plt.ylabel('True value')

    plt.xlabel('Predicted value')

    

    plt.show()
plot_confusion_matrix(y_local_test, y_local_pred)
def show_some_wrong_img(img_array: np.ndarray, true_label_array: np.ndarray, pred_label_array: np.ndarray, n: int):

    """

    Chooses n images with wrong predictions randomly, plots them and prints their true and predicted labels

    img_array shape: (x1, x2, x3, 1)

    true_label_array shape: (x1, number_of_labels)

    pred_label_array shape: (x1, number_of_labels)

    """

    

    wrong = (np.argmax(true_label_array, axis=1) - np.argmax(pred_label_array, axis=1)) != 0

    

    wrong_img_array = img_array[wrong]

    wrong_true_label_array = true_label_array[wrong]

    wrong_pred_label_array = pred_label_array[wrong]

    

    m = min(n, len(wrong_img_array))

    if m < n:

        print(f"There is only {m} misclassified image")

    

    idexes_to_show = np.random.choice(len(wrong_img_array), m, replace=False)

    

    fig, axes = plt.subplots(1, m, figsize=(20, m))

    for i, sample_index in enumerate(idexes_to_show):

        axes[i].imshow(np.squeeze(wrong_img_array[sample_index]))

        axes[i].axis('off')

        axes[i].title.set_text(f"True label: {np.argmax(wrong_true_label_array[sample_index])}\n" +

                               f"Pred label: {np.argmax(wrong_pred_label_array[sample_index])}")
show_some_wrong_img(X_local_test, y_local_test, y_local_pred, 10)
y_pred = np.argmax(model.predict(X_test), axis=1)

result = pd.DataFrame({'ImageId':range(1, len(y_pred)+1), 'Label':y_pred})



result.head()
result.to_csv('my_submission.csv', index=False)