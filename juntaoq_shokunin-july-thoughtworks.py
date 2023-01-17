import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
!pip install git+https://github.com/qubvel/efficientnet
import matplotlib.pyplot as plt

%matplotlib inline



from keras.preprocessing import image

from keras_preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model, load_model

from keras.optimizers import Adam, RMSprop



from keras import layers as KL

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.losses import categorical_crossentropy

from efficientnet.keras import EfficientNetB5



from sklearn.preprocessing import LabelEncoder
BATCH_SIZE=256



LABELS = ['Luanda', 'HongKong', 'Zurich', 'Singapore', 'Geneva',

          'Beijing', 'Seoul', 'Sydney', 'Melbourne', 'Brisbane']

EPOCHS=16



DATA_ROOT="../input/synimg/"



SPLIT_AT=80000
styles_encoder = LabelEncoder().fit(LABELS)

print(styles_encoder.classes_)
df_train = pd.read_csv("../input/synimg/synimg/train/data.csv")



def display_random_data(dataframe, rows):

    """

    Display some images from dataframe for demostration

    """

    imgs = dataframe.sample(rows *2)

    fig, axarr = plt.subplots(2, rows, figsize=(rows*10, rows*4))



    for i in range(1, rows*2+1):

        img_path = "../input/synimg/" + imgs.iloc[i-1]['filepath']

        img = image.load_img(img_path, target_size=(32, 64))

        img = image.img_to_array(img)/255

        axarr[i//(rows+1),i%rows].imshow(img)

        axarr[i//(rows+1),i%rows].set_title(imgs.iloc[i-1]['style_name'], fontsize=35)

        axarr[i//(rows+1),i%rows].axis('off')

        

display_random_data(df_train, 8)
def create_train_data_sets():

    """

    define the generator for loading training data sets

    """



    df_train = pd.read_csv("../input/synimg/synimg/train/data.csv")

    df_train = df_train.sample(frac=1.0)



    train_data=df_train[:SPLIT_AT]

    validation_data=df_train[SPLIT_AT:]



    data_gen = image.ImageDataGenerator(

        rescale=1./255.,

        rotation_range=15,

        horizontal_flip = True,

        zoom_range = 0.2,

        width_shift_range = 0.2,

        height_shift_range=0.2)

    

    validation_gen = image.ImageDataGenerator(rescale=1./255.)



    train_generator = data_gen.flow_from_dataframe(

        dataframe=train_data,

        directory="../input/synimg/",

        x_col="filepath",

        y_col="style_name",

        batch_size=BATCH_SIZE,

        shuffle=True,

        class_mode="categorical",

        target_size=(32, 64))



    validation_generator = validation_gen.flow_from_dataframe(

        dataframe=validation_data,

        directory="../input/synimg/",

        x_col="filepath",

        y_col="style_name",

        batch_size=BATCH_SIZE,

        shuffle=True,

        class_mode="categorical",

        target_size=(32, 64))



    return (train_generator, validation_generator)
def build_model():

    input_shape = (32, 64, 3)

    

    efficient_net = EfficientNetB5(weights='imagenet', include_top=False, input_shape=input_shape)

    efficient_net.trainable = False

    

    x = efficient_net.output

    x = KL.Flatten()(x)

    x = KL.Dense(512, activation="relu")(x)

    x = KL.Dropout(0.5)(x)

    predictions = KL.Dense(len(LABELS), activation="softmax")(x)

    

    model = Model(inputs = efficient_net.input, outputs = predictions)

    

    model.compile(optimizer=RMSprop(lr=0.0001, decay=1e-6),

                  loss=categorical_crossentropy,

                  metrics=['accuracy'])

    

    return model



from keras.utils import plot_model

model=build_model()

plot_model(model, to_file='model.png')
train_generator, validation_generator = create_train_data_sets()



steps_per_epoch_train = train_generator.n//train_generator.batch_size

steps_per_epoch_validation = validation_generator.n//validation_generator.batch_size





def training():

    model = build_model()



    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,

                                  verbose=1, mode='max', min_lr=0.000001)



    checkpoint = ModelCheckpoint("shokunin-july.h5", monitor='val_acc', verbose=1,

                                 save_best_only=True, mode='max')



    history = model.fit_generator(

        generator=train_generator,

        steps_per_epoch=steps_per_epoch_train,

        epochs=EPOCHS,

        validation_data=validation_generator,

        validation_steps=steps_per_epoch_validation,

        callbacks=[checkpoint, reduce_lr])



    return history, model
history, model = training()
def analyse_results(epochs):

    metrics = ['loss', "acc", 'val_loss','val_acc']

        

    plt.style.use("ggplot")

    (fig, ax) = plt.subplots(1, 4, figsize=(30, 5))

    fig.subplots_adjust(hspace=0.1, wspace=0.3)



    for (i, l) in enumerate(metrics):

        title = "Loss for {}".format(l) if l != "loss" else "Total loss"

        ax[i].set_title(title)

        ax[i].set_xlabel("Epoch #")

        ax[i].set_ylabel(l.split('_')[-1])

        ax[i].plot(np.arange(0, epochs), history.history[l], label=l)

        ax[i].legend() 



analyse_results(EPOCHS)
def analyse_more(history):

    metrics = ['loss', "acc", 'val_loss','val_acc']

    

    plt.subplot(211)

    plt.title('Loss')

    plt.plot(history.history['loss'], label='train')

    plt.plot(history.history['val_loss'], label='test')

    plt.legend()

    

    # plot accuracy during training

    plt.subplot(212)

    plt.title('Accuracy')

    plt.plot(history.history['acc'], label='train')

    plt.plot(history.history['val_acc'], label='test')

    plt.legend()

    

analyse_more(history)
df_test = pd.read_csv("../input/synimg/synimg/test/data_nostyle.csv")



test_datagen = ImageDataGenerator(rescale=1./255.)



test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory="../input/synimg/",

    x_col="filepath",

    target_size=(32, 64),

    color_mode="rgb",

    batch_size=BATCH_SIZE,

    class_mode=None,

    shuffle=False

)



test_generator.reset()



predications = model.predict_generator(test_generator, verbose=1, steps=312)
def style_of(predication):

    def possibility_of(item):

        return item.get('possibility')

    

    labeled = [{'label': styles_encoder.inverse_transform([index])[0], 'possibility': possibility} for (index, possibility) in

               enumerate(predication)]



    return max(labeled, key=possibility_of)



def summarize_prediction(predications):    

    labels = map(lambda x: style_of(x)['label'], predications)

    zipped = dict(zip(df_test.id, labels))

    

    return [{"id": k, "style_name": v} for k, v in zipped.items()]



submission = pd.DataFrame(summarize_prediction(predications))



submission.style_name.value_counts().plot.bar()

submission.to_csv("submission.csv",index=False)
submission.head()
from IPython.display import FileLink, FileLinks

FileLinks('.')