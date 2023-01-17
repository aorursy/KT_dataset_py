import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob,string
path = '../input/coil-100/coil-100/*.png'
#list files
files=glob.glob(path)
import codecs
from tqdm import tqdm
def contructDataframe(file_list):
    """
    this function builds a data frame which contains 
    the path to image and the tag/object name using the prefix of the image name
    """
    data=[]
    for file in tqdm(file_list):
        data.append((file,file.split("/")[-1].split("__")[0]))
    return pd.DataFrame(data,columns=['path','label'])
df=contructDataframe(files)
df.tail(10)
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
counts=df.groupby(df.label).size().reset_index(name="counts")
counts.plot.bar()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.path, df.label, test_size=0.15,random_state=0,stratify= df.label)
X_train.groupby(y_train).size().reset_index(name="counts").plot.bar()
X_test.groupby(y_test).size().reset_index(name="counts").plot.bar()
from keras.preprocessing.image import load_img,img_to_array
import cv2
X_train=[img_to_array(cv2.imread(file).astype("float")/255.0) for file in tqdm(X_train.values)]
X_test=[img_to_array(cv2.imread(file).astype("float")/255.0) for file in tqdm(X_test.values)]
import matplotlib.pyplot as plt
img = X_train[0]
plt.imshow(img)
plt.show()
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y_train_categorical=encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_categorical=encoder.transform(y_test.values.reshape(-1, 1))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras.optimizers import Adam

def build(width, height, depth, classes):
    # initialize the model
    model = Sequential()
    # first set of convolutional layer.
    model.add(Conv2D(30, (5, 5), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # second set convolutional layer.
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #we will dropout 20% of the neurons to improve generalization.
    model.add(Dropout(0.2))
    # Flatten layer
    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # Output layer
    model.add(Dense(classes, activation='softmax'))
    return model
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
model=build(128,128,3,encoder.classes_.__len__())

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
sequential_model_to_ascii_printout(model)
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]
X_train=np.array(X_train)
X_test=np.array(X_test)
X_train, X_validation, y_train_categorical, y_validation_categorical = train_test_split(X_train, y_train_categorical, test_size=0.15,random_state=0,stratify= y_train_categorical)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
hist = model.fit_generator(aug.flow(X_train, y_train_categorical, batch_size=BS), steps_per_epoch=len(X_train) // BS,
                           epochs=EPOCHS,
                           validation_data=(X_validation, y_validation_categorical),
                           verbose=1,callbacks=callbacks_list)
loss, accuracy = model.evaluate(X_test,y_test_categorical, verbose=2)
print('Accuracy: %f' % (accuracy*100),'loss: %f' % (loss*100))
#generate plots
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.title('CNN COIL-100')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()


plt.figure()
plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
plt.title('CNN COIL-100')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
prediction_test_c=model.predict(X_test)
prediction_test=encoder.inverse_transform(prediction_test_c)

prediction_train_c=model.predict(X_train)
prediction_train=encoder.inverse_transform(prediction_train_c)

prediction_validation_c=model.predict(X_validation)
prediction_validation=encoder.inverse_transform(prediction_validation_c)
from sklearn.metrics import confusion_matrix
%config InlineBackend.figure_format = 'retina'
def plot_cm(y,y_predict,classes,name):
    plt.figure(figsize=(30, 30))
    sns.heatmap(confusion_matrix(y,y_predict), 
            xticklabels=classes,
            yticklabels=classes)
    plt.title(name)
    plt.show()
plot_cm(prediction_test,y_test.values,encoder.classes_,"Test accuracy")
plot_cm(prediction_validation,encoder.inverse_transform(y_validation_categorical),encoder.classes_,"Validation accuracy")
plot_cm(prediction_train,encoder.inverse_transform(y_train_categorical),encoder.classes_,"Train accuracy")