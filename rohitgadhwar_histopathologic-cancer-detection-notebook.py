# Importing Libraries

import os

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras import layers as KL

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D as Convolution2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.preprocessing import image
TRAIN_PATH = '/kaggle/input/histopathologic-cancer-detection/train/'

TRAIN_LABELS = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'

SIZE_IMG = 96

EPOCHS = 10



model_path = '../input/resnet-cancer-detection/my_model.h5'

saved_model = os.path.isfile(model_path)
df = pd.read_csv(TRAIN_LABELS, dtype=str)



#remove unwanted data detected by other kaggle users

df = df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

df = df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']



print(df['label'].value_counts(), 

      '\n\n', df.describe(), 

      '\n\n', df.head())
def display_random_data(dataframe, path, rows):



    imgs = dataframe.sample(rows *2)

    fig, axarr = plt.subplots(2, rows, figsize=(rows*10, rows*4))



    for i in range(1,rows*2+1):

        img_path = path + imgs.iloc[i-1]['id'] + '.tif'

        img = image.load_img(img_path, target_size=(96, 96))

        img = image.img_to_array(img)/255

        axarr[i//(rows+1),i%rows].imshow(img)

        axarr[i//(rows+1),i%rows].set_title(imgs.iloc[i-1]['label'], fontsize=35)

        axarr[i//(rows+1),i%rows].axis('off')

        

display_random_data(df,TRAIN_PATH, 5)
#add .tif to ids in the dataframe to use flow_from_dataframe

df["id"]=df["id"].apply(lambda x : x +".tif")

df.head()
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)
train_generator=train_datagen.flow_from_dataframe(

    dataframe=df,

    directory=TRAIN_PATH,

    x_col="id",

    y_col="label",

    subset="training",

    batch_size=64,

    shuffle=True,

    class_mode="binary",

    target_size=(96,96))
valid_generator=train_datagen.flow_from_dataframe(

    dataframe=df,

    directory=TRAIN_PATH,

    x_col="id",

    y_col="label",

    subset="validation",

    batch_size=64,

    shuffle=True,

    class_mode="binary",

    target_size=(96,96))
def build_model():

    # Initialising the CNN

    classifier = Sequential()



    classifier.add(Convolution2D(32, (3, 3), input_shape = (96, 96, 3), activation = 'relu'))



    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))

    classifier.add(MaxPooling2D(pool_size = (2, 2)))



    classifier.add(Flatten())



    classifier.add(Dense(128, activation = 'relu'))

    classifier.add(Dense(1, activation = 'sigmoid'))



    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    

    return classifier
classifier = build_model()

classifier.summary()
history = classifier.fit_generator(train_generator,

                              steps_per_epoch=train_generator.n//train_generator.batch_size, 

                              validation_data=valid_generator,

                              validation_steps=valid_generator.n//valid_generator.batch_size,

                              epochs=EPOCHS)
def analyse_results(epochs):

    metrics = ['loss', "accuracy"]

        

    plt.style.use("ggplot")

    (fig, ax) = plt.subplots(1, 2, figsize=(30, 5))

    fig.subplots_adjust(hspace=0.1, wspace=0.3)



    for (i, l) in enumerate(metrics):

        title = "Loss for {}".format(l) if l != "loss" else "Total loss"

        ax[i].set_title(title)

        ax[i].set_xlabel("Epoch #")

        ax[i].set_ylabel(l.split('_')[-1])

        ax[i].plot(np.arange(0, epochs), history.history[l], label=l)

        ax[i].legend() 



if EPOCHS > 1 and saved_model == False:        

    analyse_results(EPOCHS)
test_path = '/kaggle/input/histopathologic-cancer-detection/test/'

df_test = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

df_test["id"]=df_test["id"].apply(lambda x : x +".tif")
test_datagen = ImageDataGenerator(rescale=1./255,

                                 samplewise_std_normalization= True)



test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=test_path,

    x_col="id",

    y_col=None,

    target_size=(96, 96),

    color_mode="rgb",

    batch_size=64,

    class_mode=None,

    shuffle=False,

)
test_generator.reset()

pred=classifier.predict_generator(test_generator,verbose=1).ravel()
results = dict(zip(test_generator.filenames, pred))



label = []

for i in range(len(df_test["id"])):

    label.append(results[df_test["id"][i]])

    

df_test["id"]=df_test["id"].apply(lambda x : x[:-4])
submission=pd.DataFrame({"id":df_test["id"],

                      "label":label})

submission.to_csv("submission.csv",index=False)

submission.head()