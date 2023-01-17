import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import keras
import cv2
from keras.preprocessing.image import load_img,img_to_array
from skimage import io

#%matplotlib inline
pic_size = 48

base_path = "/kaggle/input/facial-recognition-dataset/"

plt.figure(0, figsize=(12,20))
cpt = 0

# for expression in os.listdir(base_path + "Training/Training/"):
#     print(expression)
#     for i in range(1,6):
#         cpt += 1
#         plt.subplot(7,5,cpt)
#         img = load_img(base_path+'Training/Training/'+expression+"/"+os.listdir(base_path+"Training/Training/"+expression)[i],target_size=(pic_size,pic_size))
#         plt.imshow(img,cmap="gray")
#         pass
#     pass

# plt.tight_layout()
# plt.show()
class_names = []

for expression in os.listdir(base_path+"Training/Training/"):
    class_names.append(expression)
    print(str(len(os.listdir(base_path+"Training/Training/"+expression)))+" "+expression+" images")
    pass
class_names
def load_class(class_names, index, num_examples, mode):
    images = []
    class_name = class_names[index]
    
    TRAIN_PATH = base_path + "Training/Training/" + class_name
    TEST_PATH = base_path + "Testing/Testing/" + class_name
    PATH = ''
    if mode == 0:
        PATH = TRAIN_PATH
    else:
        PATH = TEST_PATH
    
    img_names = os.listdir(PATH)
    img_names = np.random.RandomState(seed=69).permutation(img_names)
    
    for i in range(num_examples):
        img = io.imread(PATH + "/" + img_names[i])
        img = img.reshape([48, 48, 1])
        images.append(img)

    labels = np.empty(num_examples)
    labels.fill(index)
    
    return np.array(images), keras.utils.to_categorical(labels, len(class_names), dtype=int)
data_train = []
label_train = []
for i in range(len(class_names)):
    data_part, label_part = load_class(class_names, i, 420, 0)
    data_train.extend(data_part)
    label_train.extend(label_part)
    
X_train = np.array(data_train)
y_train = np.array(label_train)
data_test = []
label_test = []
for i in range(len(class_names)):
    data_part, label_part = load_class(class_names, i, 110, 1)
    data_test.extend(data_part)
    label_test.extend(label_part)
    
X_test = np.array(data_train)
y_test = np.array(label_train)
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128

datagen_train = ImageDataGenerator(rescale= 1/255., horizontal_flip= True)
datagen_test = ImageDataGenerator(rescale= 1/255., horizontal_flip= True)

train_generator = datagen_train.flow(X_train, y_train, batch_size = 64, shuffle = True, seed = 69)

test_generator = datagen_test.flow(X_test, y_test, batch_size = 64, shuffle = True, seed = 69)
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential,Model
from keras.optimizers import Adam

n_classes = 7

# Initialising the CNN
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32,(3,3), padding="same", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# 2nd convolutional Layer
model.add(Conv2D(64,(5,5), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# 3rd Convolutional Layer
model.add(Conv2D(128,(3,3),padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# 4th convolution layer
model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Flattening
model.add(Flatten())

# Connected to first layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Connected to second layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes,activation='softmax'))

opt = Adam(lr=0.003)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
epochs = 100
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model_weights.h5",monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list=[checkpoint]
history = model.fit_generator(generator=train_generator,
                             steps_per_epoch=train_generator.n//train_generator.batch_size,
                             epochs=epochs,
                             validation_data=test_generator,
                             validation_steps = test_generator.n//test_generator.batch_size,
                             callbacks=callbacks_list)
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.suptitle('Optimizer: Adam',fontsize=18)
plt.ylabel('Loss',fontsize=16)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Testing loss')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.ylabel('Accuracy',fontsize=16)
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Testing Accuracy')
plt.legend(loc='lower right')

plt.show()
predictions = model.predict_generator(generator=test_generator)
y_pred = [np.argmax(probas) for probas in predictions]
#y_test = test_generator.classes
#class_names = test_generator.class_indices.keys()

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,title='Confusion matirx',cmap=plt.cm.Blues):
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    pass

cnf_mat = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_mat, classes=class_names, title='Normalized Confusion Matrix')
plt.show()
