import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/fer2013csv/fer2013.csv')
#check data shape
data.shape
data.Usage.value_counts()
#check target labels
emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = [j for (i,j) in emotion_map.items()]
emotion_counts
def row2image(row):
    pixels, emotion = row['pixels'], emotion_map[row['emotion']]
    img = np.array(pixels.split())
    img = img.reshape(48,48)
    image = np.zeros((48,48,3))
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return np.array([image.astype(np.uint8), emotion])

plt.figure(0, figsize=(16,10))
for i in range(1,8):
    face = data[data['emotion'] == i-1].iloc[0]
    img = row2image(face)
    plt.subplot(2,4,i)
    plt.imshow(img[0])
    plt.title(img[1])

plt.show()
train_data = data[data['Usage'] == 'Training'].copy()
val_data = data[data['Usage'] == 'PublicTest'].copy()
test_data = data[data['Usage'] == 'PrivateTest'].copy()

print('Training Data shape : ' + str(train_data.shape))
print('Validation Data shape : ' + str(val_data.shape))
print('Test Data shape : ' + str(test_data.shape))
num_classes = 7
width, height = 48,48
epochs = 100
batch_size = 64
num_features = 64
#CRNO - Convert, Reshape, Normalize, One hot Encoding
def crno(df, dataName):
  df['pixels'] = df['pixels'].apply(lambda x: [int(px) for px in x.split()])
  data_X = np.array(df['pixels'].to_list(), dtype='float32').reshape(-1,width, height,1)/255.0   
  data_Y = to_categorical(df['emotion'], num_classes)  

  print(str(dataName)+" _X shape: "+str(data_X.shape), end ="")
  print(" _Y shape " + str(data_Y.shape))
  return data_X, data_Y

train_X , train_Y = crno(train_data, "train")
val_X, val_Y = crno(val_data, "Validation")
test_X, test_Y = crno(test_data, "Test")
model = Sequential()
#Layer1
model.add(Conv2D(64, (3, 3), input_shape=(width,height,1),activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Layer2
model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))
model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Layer3
model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Layer4
model.add(Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(512, (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#FCC
model.add(Flatten())
#model.add(Dense(1024))
#model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#Softmax
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

model.summary()
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)


es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)

history = model.fit_generator(data_generator.flow(train_X, train_Y, batch_size),
                                steps_per_epoch=len(train_X) / batch_size,
                                epochs=epochs,
                                verbose=2, 
                                callbacks = [es],
                                validation_data=(val_X, val_Y))
test_true = np.argmax(test_Y, axis=1)
test_pred = np.argmax(model.predict(test_X), axis=1)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))
plot_model(model)
def prediction(path):
    img = image.load_img(path, color_mode = 'grayscale', target_size=(48, 48,1))
    show_img=image.load_img(path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255
    pred = model.predict(x)
    plt.gray()
    plt.imshow(show_img)
    plt.show()

    ind = np.argmax(pred[0])

    print('Expression Prediction:',emotion_map[ind])
model.save('../output/')
#prediction('../input/images/sample1.png')
#prediction('../input/images/sample2.png')
prediction('../input/images/sample3.jpg')
prediction('../input/images/sample4.jpeg')
prediction('../input/images/sample6.jpg')
prediction('../input/images/sample7.jpg')
prediction('../input/images/sample5.jpg')
prediction('../input/images/sample8.jpg')
prediction('../input/images/sample9.jpeg')