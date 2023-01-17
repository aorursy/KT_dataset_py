import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, AvgPool2D, MaxPool2D , Flatten , Dropout, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

import cv2

import os



print(tf.__version__)
labels = ['PNEUMONIA', 'NORMAL']

img_size = 150

def get_training_data(data_dir):

    data = [] 

    for label in labels: 

        path = os.path.join(data_dir, label)

        class_num = labels.index(label)

        diag= ""

        for img in os.listdir(path):

            try:

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size

                if "bacteria" in img:

                    diag = [1,0,0] 

                elif "virus" in img:

                    diag = [0,1,0] 

                else:

                    diag = [0,0,1]              

                data.append([resized_arr, class_num, diag])

            except Exception as e:

                print(img, e)

    return np.array(data)
train = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')

val = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')

test = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
def display_set(one_set, order = ["pneumonia bacteria", "pneumonia virus", "normal"]):

    l = []

    for i in one_set:

        if i[2] == [1,0,0]:

            diag = "pneumonia bacteria"

        elif i[2] == [0,1,0]:

            diag = "pneumonia virus"

        else:

            diag = "normal"

        l.append(diag)

    sns.set_style('darkgrid')

    sns.countplot(l, order = order)  
display_set(train)
display_set(val)
display_set(test)
x_data = []

y_data = []



for feature, label, diag in train:

    x_data.append(feature)

    y_data.append(diag)



for feature, label, diag in test:

    x_data.append(feature)

    y_data.append(diag)

    

for feature, label, diag in val:

    x_data.append(feature)

    y_data.append(diag)



y_class_num = [np.where(np.asarray(r)==1)[0][0] for r in y_data]

sns.countplot(y_class_num).set_title('All data')




samples = pd.DataFrame(y_class_num, columns = ['class_num'])

samples['diag'] = y_data

samples['img'] = x_data



smallest_diag_count = samples['class_num'].value_counts().min()

print("number of sampels per category", smallest_diag_count)



class0 = samples[samples['class_num'] == 0].sample(smallest_diag_count)

class1 = samples[samples['class_num'] == 1].sample(smallest_diag_count)

class2 = samples[samples['class_num'] == 2].sample(smallest_diag_count)



samples_under = pd.concat([class0, class1, class2], axis=0)



x = samples_under['img'].tolist()

y = samples_under['diag'].tolist()

sns.countplot(samples_under['class_num']).set_title('Final Test data')

# Lets shuffle the samples to have matching ditributions regarding the differents sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

sns.countplot([np.where(np.asarray(r)==1)[0][0] for r in y_train]).set_title('Training data')
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)

sns.countplot([np.where(np.asarray(r)==1)[0][0] for r in y_val]).set_title('Validation data')
sns.countplot([np.where(np.asarray(r)==1)[0][0] for r in y_test]).set_title('Test data')
# Normalize the data

x_train = np.array(x_train) / 255

x_val = np.array(x_val) / 255

x_test = np.array(x_test) / 255
# resize data for deep learning 

x_train = x_train.reshape(-1, img_size, img_size, 1)

y_train = np.array(y_train)



x_val = x_val.reshape(-1, img_size, img_size, 1)

y_val = np.array(y_val)



x_test = x_test.reshape(-1, img_size, img_size, 1)

y_test = np.array(y_test)
# With data augmentation to prevent overfitting and handling the imbalance in dataset



train_datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip = False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

train_datagen.fit(x_train)



valid_datagen = ImageDataGenerator()



test_datagen = ImageDataGenerator()





train_generator = train_datagen.flow(

    x_train,

    y_train,

    batch_size = 32)



validation_generator = valid_datagen.flow(

    x_val,

    y_val)



test_generator = test_datagen.flow(

    x_test,

    y_test)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)



model = Sequential()

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))

model.add(AvgPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())

model.add(Dense(units = 128 , activation = 'relu'))

model.add(Dense(units = 3 , activation = 'softmax'))

model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
# When publishing this notebook, there is a bug in the fit function which exists after the first epoch

# I workaround it with a simple loop



class_weight = {0: 3, 1: 3, 2: 1}

loops = 20

history = []



for x in range(loops):

    history.append(model.fit(

            train_generator,

            steps_per_epoch=97,

            epochs=1,

            validation_data=test_generator,

            validation_steps=100,

            class_weight=class_weight))
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0]*100)

print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
print(history[0].history)
epochs = [i for i in range(loops)]

fig , ax = plt.subplots(1,2)

train_acc = []

train_loss = []

val_acc = []

val_loss = []

for h in history:

    train_acc.append(h.history['accuracy'])

    train_loss.append(h.history['loss'])    

    val_acc.append(h.history['val_accuracy'])

    val_loss.append(h.history['val_loss'])

fig.set_size_inches(20,10)



ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')

ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')

ax[0].set_title('Training & Validation Accuracy')

ax[0].legend()

ax[0].set_xlabel("Epochs")

ax[0].set_ylabel("Accuracy")



ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')

ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')

ax[1].set_title('Testing Accuracy & Loss')

ax[1].legend()

ax[1].set_xlabel("Epochs")

ax[1].set_ylabel("Training & Validation Loss")

plt.show()
predictions = model.predict_classes(x_test)

y_test_num = [np.where(r==1)[0][0] for r in y_test]

names = ['Pneumonia Bacteria', 'Pneumonia Virus','Normal']





print(classification_report(y_test_num, predictions, target_names = names))
cm = confusion_matrix(y_test_num,predictions)

cm = pd.DataFrame(cm , index = names , columns = names)

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'



group_counts = ["{0:0.0f}".format(value) for value in cm.to_numpy().flatten()]

group_percentages = ["{0:.2%}".format(value) for value in cm.to_numpy().flatten()/np.sum(cm.to_numpy())]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]

labels = np.asarray(labels).reshape(3,3)



plt.figure(figsize = (10,10))

sns.heatmap(cm,

            annot=labels,

            cmap= "coolwarm",

            linecolor = 'black',

            linewidth = 1,

            fmt='')