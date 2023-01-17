import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense

from keras.models import Sequential

#from keras.utils import to_categorical

from keras.optimizers import SGD

import glob, os, random, cv2 ,re, keras

print(os.listdir("../input"))
train_path = '../input/garbageclass/Train/Train/'

img_list = glob.glob(os.path.join(train_path, '*.jpg'))

valid_path = '../input/garbageclass/Validation/Validation/'

imgV_list = glob.glob(os.path.join(valid_path, '*.jpg'))
label_train = []

Train_img = []

#Images

for img in img_list:

    image = cv2.imread(img, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (300, 300))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    Train_img.append(image)

#label

for img in img_list:

    trash_label = img.split("/")[-1]

    trash_label = trash_label.split(".")[0]

    m = re.split('(\d+)',trash_label)[0]

    label_train.append(m)



print(len(Train_img))

print(len(label_train))
label_valid = []

Valid_img = []

#Images

for img in imgV_list:

    image = cv2.imread(img, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (300, 300))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    Valid_img.append(image)

#label

for img in imgV_list:

    trash_label = img.split("/")[-1]

    trash_label = trash_label.split(".")[0]

    m = re.split('(\d+)',trash_label)[0]

    label_valid.append(m)



print(len(Valid_img))

print(len(label_valid))
Train_img = np.array(Train_img)

label_train = np.array(label_train)

Valid_img = np.array(Valid_img)

label_valid = np.array(label_valid)
print(len(Train_img) == len(label_train))

print(len(Train_img),len(Valid_img),len(label_train),len(label_valid))
label_to_id_dict = {v:i for i,v in enumerate(np.unique(label_train))}

id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

id_to_label_dict
label_ids_t = np.array([label_to_id_dict[x] for x in label_train])

len(label_ids_t)
plt.imshow(Train_img[0])
plt.imshow(Valid_img[0])
print("Image dimensions", Train_img[1].shape)

print("shape of images:",Train_img.shape)

print("Label size", label_train.shape, label_ids_t.shape)
label_to_id_dict = {v:i for i,v in enumerate(np.unique(label_valid))}

id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

id_to_label_dict

label_ids_v = np.array([label_to_id_dict[x] for x in label_valid])

len(label_ids_v)
X_train, X_test = Train_img, Valid_img

Y_train, Y_test = label_ids_t, label_ids_v



#Normalize color values to between 0 and 1

X_train = X_train/255

X_test = X_test/255



#Make a flattened version for some of our models

X_flat_train = X_train.reshape(X_train.shape[0], 300*300*3)

X_flat_test = X_test.reshape(X_test.shape[0], 300*300*3)



print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

print('Flattened:', X_flat_train.shape, X_flat_test.shape)
print(X_train[0].shape)

plt.imshow(X_train[0])

plt.show()
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers import LSTM, Input, TimeDistributed

from keras.models import Model

from keras.optimizers import Adam, RMSprop



# Import the backend

from keras import backend as K
model = Sequential()





model.add(Conv2D(32, (3, 3), input_shape = (300, 300, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())



# Step 4 - Full connection

model.add(Dense(units = 64, activation = 'relu'))

model.add(Dropout(0.25))

model.add(Dense(units = 64, activation = 'relu'))

model.add(Dropout(0.25))

model.add(Dense(units = 1, activation = 'sigmoid'))



# Compiling the CNN

opt=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9)

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

history=model.fit(X_train, Y_train,

          batch_size=128,

          epochs=25,

          verbose=1,   

          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show
test_path = '../input/garbageclass/Test/*'

imgT_list = glob.glob(os.path.join(test_path, '*.jpg'))



label_test = []

Test_img = []

#Images

for img in imgT_list:

    image = cv2.imread(img, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (300, 300))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    Test_img.append(image)

#label

for img in imgT_list:

    trash_label = img.split("/")[-1]

    trash_label = trash_label.split(".")[0]

    m = re.split('(\d+)',trash_label)[0]

    label_test.append(m)



print(len(Test_img))

print(len(label_test))
Test_img = np.array(Test_img)

label_test = np.array(label_test)

print(len(Test_img))



for i in range(16):

    print(label_test[i])

preds = model.predict(Test_img)



plt.figure(figsize=(16, 16))

for i in range(16):

    if preds[i].astype(np.int) == 0:

        string = "paper"

    else:

        string = "plastic"

    plt.subplot(4, 4, i+1)

    plt.title('pred:%s / truth:%s' % (string, label_test[i]))

    plt.imshow(Test_img[i])