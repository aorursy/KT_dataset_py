!pip install imutils
import cv2

import os, glob



import numpy as np

import seaborn as sns

from imutils import paths

import matplotlib.pyplot as plt

import sklearn.metrics as metrics



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report



from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16, ResNet101, Xception

from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout, BatchNormalization, Conv2D
LR = 0.001

EPOCHS = 20

BATCH_SIZE = 32

COVID_LEN = 218

INP_SIZE = (224,224,3)
def create_data(dir_name):

    temp_data = []

    img_list = glob.glob('../' + dir_name + '/*')

    for img in img_list[:COVID_LEN]:

        image = cv2.imread(img)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224, 224))

        temp_data.append(image)

    return temp_data



data = []

labels = []



covid_dir = 'input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19'

normal_dir = 'input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL'

pneumonia_dir = 'input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia'



data.extend(create_data(covid_dir))

data.extend(create_data(normal_dir))

data.extend(create_data(pneumonia_dir))



labels.extend([1] * COVID_LEN)

labels.extend([0]*2*COVID_LEN)



data = np.array(data)/255.0

labels = np.array(labels)



print(data.shape)

print(labels.shape)
lb = LabelBinarizer()

labels = lb.fit_transform(labels)

labels = to_categorical(labels)



(x_train, x_test, y_train, y_test) = train_test_split(

    data,

    labels,

    test_size=0.20,

    stratify=labels,

    random_state=42

)

trainAug = ImageDataGenerator(

    rotation_range=15,

    fill_mode="nearest"

)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
def generate_custom_model():

    

    model = Sequential()

    model.add(BatchNormalization(input_shape=INP_SIZE))

    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Dropout(0.35))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.35))

    model.add(Dense(2, activation='softmax'))

    

    return model



def generate_pretrained_model(model_name):

    if model_name == 'VGG16':

        model = VGG16(

            include_top = False,

            weights = 'imagenet',

            input_tensor = Input(shape=INP_SIZE)

        )

    elif model_name == 'ResNet101':

        model = ResNet101(

            include_top = False,

            weights = 'imagenet',

            input_tensor = Input(shape=INP_SIZE)

        )

    elif model_name == 'Xception':

        model = Xception(

            include_top = False,

            weights = 'imagenet',

            input_tensor = Input(shape=INP_SIZE)

        )

    else:

        model = None

        print('Invalid Choice!')

    

    return model
def fit_model(model, model_name):

    optim = Adam(lr = LR, decay = LR/EPOCHS)

    

    if model_name == 'Custom':

        model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

        history = model.fit_generator(

            trainAug.flow(x_train, y_train, batch_size = BATCH_SIZE),

            steps_per_epoch = len(x_train) // BATCH_SIZE,

            validation_data = (x_test, y_test),

            validation_steps = len(x_test) // BATCH_SIZE,

            epochs = EPOCHS

        )

    else :

        for layer in model.layers:

            layer.trainable = False

        # top layer for shaping output    

        headModel = model.output

        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)

        headModel = Flatten(name="flatten")(headModel)

        headModel = Dense(64, activation="relu")(headModel)

        headModel = Dropout(0.5)(headModel)

        headModel = Dense(2, activation="softmax")(headModel)

        model = Model(inputs=model.input, outputs=headModel)

        model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

        

        history = model.fit_generator(

            trainAug.flow(x_train, y_train, batch_size = BATCH_SIZE),

            steps_per_epoch = len(x_train) // BATCH_SIZE,

            validation_data = (x_test, y_test),

            validation_steps = len(x_test) // BATCH_SIZE,

            epochs = EPOCHS

        )

    

    return history, model
def display_history(history_):

    fig, ax = plt.subplots(1,2, figsize=(12, 3))

    ax[0].plot(history_.history['loss'], color='b', label="training_loss")

    ax[0].plot(history_.history['val_loss'], color='r', label="validation_loss",axes =ax[0])

    legend = ax[0].legend(loc='best', shadow=True)



    ax[1].plot(history_.history['accuracy'], color='b', label="training_accuracy")

    ax[1].plot(history_.history['val_accuracy'], color='r',label="validation_accuracy")

    legend = ax[1].legend(loc='best', shadow=True)



def plot_metrices(model_):

    

    plt.figure()

    ax = plt.subplot()

    ax.set_title('Confusion Matrix')

    

    pred = model_.predict(x_test, batch_size = BATCH_SIZE)

    pred = np.argmax(pred, axis = 1)

    cm = confusion_matrix(y_test.argmax(axis = 1), pred)

    classes=['normal', 'covid19']

    sns.heatmap(cm, annot = True, xticklabels = classes, yticklabels = classes, cmap = 'Reds')



    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.show

    

    print(classification_report(y_test.argmax(axis = 1), pred))

    total = sum(sum(cm))

    acc = (cm[0, 0] + cm[1, 1]) / total

    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    

    print("ACC: {:.4f}".format(acc))

    print("Sensitivity: {:.4f}".format(sensitivity))

    print("Specificity: {:.4f}".format(specificity))
custom_mod = generate_custom_model()

vgg_mod = generate_pretrained_model('VGG16')

resnet_mod = generate_pretrained_model('ResNet101')

xception_mod = generate_pretrained_model('Xception')
(cus_his, custom_mod) = fit_model(custom_mod, 'Custom')

display_history(cus_his)

plot_metrices(custom_mod)

custom_mod.save('custom.h5')
(vgg_his, vgg_mod)= fit_model(vgg_mod, 'VGG16')

display_history(vgg_his)

plot_metrices(vgg_mod)

vgg_mod.save('vgg.h5')
res_his, resnet_mod = fit_model(resnet_mod, 'ResNet101')

display_history(res_his)

plot_metrices(resnet_mod)

resnet_mod.save('resnet.h5')
xcep_his, xception_mod = fit_model(xception_mod, 'Xception')

display_history(xcep_his)

plot_metrices(xception_mod)

xception_mod.save('xception.h5')