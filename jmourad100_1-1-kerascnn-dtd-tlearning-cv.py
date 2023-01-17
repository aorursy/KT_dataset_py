import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import os, time, random, cv2, tqdm

import imgaug as ia

from imgaug import augmenters as iaa



import tensorflow as tf

from keras import backend as K

from keras.models import Model

from keras.applications import *

from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, concatenate

from keras.layers import Activation, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adam, RMSprop

from keras.metrics import *

from keras.utils.np_utils import to_categorical

from keras.utils import Sequence

from keras import regularizers



from keras.utils.vis_utils import plot_model

from IPython.display import Image



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.utils import class_weight, shuffle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



! ls ../input/dtd-r1.0.1/dtd/images/
%matplotlib inline

pd.set_option('max_colwidth', 400)

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 16

t_start = time.time()



DATA_DIR = "../input/dtd-r1.0.1/dtd/images/"

CATEGORIES = os.listdir(DATA_DIR)

Ncategories = len(CATEGORIES)



# Fix random seed for reproducibility (then init all keras model seed)

seed=1994

np.random.seed(seed)

tf.set_random_seed(seed)

ia.seed(seed)



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
###########################

######## Hyperparameters

##########################

img_width, img_height = (300, 300)

Nepochs = 25 ##

Batch_size = 30 ##

Learning_rate = 1e-04

Patience = 3



if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)
labels_df = []

for category_id, category in enumerate(CATEGORIES):

    for file in os.listdir(os.path.join(DATA_DIR, category)):

        # Solve not an image pb: ../input/dtd-r1.0.1/dtd/images/waffled/.directory

        if file == ".directory": continue

        labels_df.append(['{}/{}'.format(category, file), category, category_id])

labels_df = pd.DataFrame(labels_df, columns=['Img_name', 'Category', 'Category_id'])



labels_df.sample(n=10, random_state=seed).head(n=6)
Temp = labels_df.groupby('Category_id').apply(lambda x: x.sample(1))

Temp.reset_index(drop=True, inplace=True)

plt.figure(figsize=(26, Ncategories//2))

for i in range(Ncategories):

    plt.subplot(6, 8, i+1)

    img = cv2.imread('{}{}'.format(DATA_DIR, Temp["Img_name"][i]))

    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title(Temp["Category"][i]);

    plt.axis('off')  

plt.show()
class data_generator(Sequence):

    

    def create_train(dataset, batch_size, shape, augment=True):

        assert shape[2] == 3

        while True:

            dataset = shuffle(dataset)

            for start in range(0, dataset.shape[0], batch_size):

                end = min(start + batch_size, dataset.shape[0])

                batch_images = []

                X_train_batch = dataset.iloc[start:end, :]

                batch_labels = np.zeros((X_train_batch.shape[0], Ncategories))

                for i in range(X_train_batch.shape[0]):

                    image = data_generator.load_image( X_train_batch.iloc[i, 0], shape)

                    if augment:

                        image = data_generator.augment(image)

                    batch_images.append(image/255.)

                    # OneHotEncode Category_id

                    batch_labels[i][X_train_batch.iloc[i, 2]] = 1

                    

                yield np.array(batch_images, np.float32), batch_labels

                

    def create_valid(dataset, batch_size, shape, augment=False):

        assert shape[2] == 3

        while True:

            # dataset = shuffle(dataset)

            for start in range(0, dataset.shape[0], batch_size):

                end = min(start + batch_size, dataset.shape[0])

                batch_images = []

                X_valid_batch = dataset.iloc[start:end, :]

                batch_labels = np.zeros((X_valid_batch.shape[0], Ncategories))

                for i in range(X_valid_batch.shape[0]):

                    image = data_generator.load_image( X_valid_batch.iloc[i, 0], shape)

                    if augment:

                        image = data_generator.augment(image)

                    batch_images.append(image/255.)

                    # OneHotEncode Category_id

                    batch_labels[i][X_valid_batch.iloc[i, 2]] = 1

                    

                yield np.array(batch_images, np.float32), batch_labels

                

    def load_image(path, shape):

        image = cv2.imread(str(DATA_DIR) + str(path))

        image = cv2.resize(image, (shape[0], shape[1]))

        return image

    

    def augment(image):

        seq = iaa.Sequential([

            iaa.OneOf([

                iaa.Noop(),

                iaa.Fliplr(0.9),

                iaa.Flipud(0.9),

                iaa.Crop(px=(0, 20))

            ]),

#         iaa.EdgeDetect(alpha=(0.0, 1.0)),

#         iaa.Dropout(p=(0, 0.2), per_channel=0.5),

#         iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),

#         iaa.OneOf([

#             iaa.Noop(),

#             iaa.GaussianBlur(sigma=(0.0, 1.0)),

#             iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),

#             iaa.PerspectiveTransform(scale=(0.04, 0.08)),

#             iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0))

#         ]),

#         iaa.SomeOf(2, [

#             iaa.Affine(rotate=45),

#             iaa.AdditiveGaussianNoise(scale=0.2*255),

#             iaa.Add(50, per_channel=True),

#             iaa.Sharpen(alpha=0.5),

#             iaa.WithChannels(0, iaa.Add((10, 100))),

#             iaa.Grayscale(alpha=(0.0, 1.0)),

#             iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),

#         ])

         ], random_order=True)

        image_aug = seq.augment_image(image)

        return image_aug

def build_model(ModelName, InputShape = input_shape, froze=0.8):

    

    inp = Input(shape=input_shape)

    

    if ModelName == "Xception":

        base_model = Xception(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "VGG16":

        base_model = VGG16(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "VGG19":

        base_model = VGG19(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "ResNet50":

        base_model = ResNet50(input_tensor = inp, include_top=False, weights='imagenet')

    # ResNet152, ResNeXt50, ResNeXt101

    elif ModelName == "InceptionV3":

        base_model = InceptionV3(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "InceptionResNetV2":

        base_model = InceptionResNetV2(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "MobileNet":

        base_model = MobileNet(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "MobileNetV2":

        base_model = MobileNetV2(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "DenseNet121":

        base_model = DenseNet121(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "DenseNet201":

        base_model = DenseNet201(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "NASNetMobile":

        base_model = NASNetMobile(input_tensor = inp, include_top=False, weights='imagenet')

    elif ModelName == "NASNetLarge":

        base_model = NASNetLarge(input_tensor = inp, include_top=False, weights='imagenet')

        

    # frozen the first .froze% layers

    NtrainableLayers = round(len(base_model.layers)*froze)

    for layer in base_model.layers[:NtrainableLayers]:

        layer.trainable = False

    for layer in base_model.layers[NtrainableLayers:]:

        layer.trainable = True



    x_model = base_model.output

    x_model = GlobalAveragePooling2D(name='globalaveragepooling2d')(x_model)

    ## OR ##

    # x_model = Flatten()(x_model)

    # x_model = BatchNormalization()(x_model)

    # x_model = Dropout(0.5)(x_model)



    predictions = Dense(Ncategories, activation='softmax',name='output_layer')(x_model)

    model = Model(inputs=base_model.input, outputs=predictions)

    

    return model
MName = "Xception"

trainig_history = []

AccS = []

Nsplits = 10



Oof_pred_classes = np.zeros((labels_df.shape[0]))

Oof_true_classes = np.zeros((labels_df.shape[0]))



for i in range(1,11):

    

    with open('../input/dtd-r1.0.1/dtd/labels/train{}.txt'.format(i)) as f:

        train_imgs = f.readlines()



    train_imgs = [x.strip() for x in train_imgs]

    train = labels_df[labels_df['Img_name'].isin(train_imgs)]

    

    with open('../input/dtd-r1.0.1/dtd/labels/val{}.txt'.format(i)) as f:

        val_imgs = f.readlines()



    val_imgs = [x.strip() for x in val_imgs]

    val = labels_df[labels_df['Img_name'].isin(val_imgs)]

    

    K.clear_session()

    

    # Callbacks for training: Make sure the training works well and doesnt run too long or overfit too much

    BestModelWeightsPath = "{}_weights.best.hdf5".format(MName)

    

    checkpoint = ModelCheckpoint(

        BestModelWeightsPath, 

        monitor='val_acc', 

        verbose=1, 

        save_best_only=True, 

        mode='max', 

        save_weights_only = True

    )

    reduceLROnPlat = ReduceLROnPlateau(

        monitor='val_acc', 

        factor=0.2, 

        patience=Patience, 

        verbose=1, 

        mode='max', 

        cooldown=2, 

        min_lr=1e-7

    )

    early = EarlyStopping(

        monitor="val_acc", 

        mode="max", 

        patience=Patience*2

    )

    callbacks_list = [checkpoint, early, reduceLROnPlat]

    

    

    model = build_model(MName, InputShape = input_shape)



    model.compile(optimizer=Adam(lr=Learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])



    train_gen = data_generator.create_train(train, Batch_size, (img_height, img_width, 3), augment=True)

    val_gen = data_generator.create_valid(val, Batch_size, (img_height, img_width, 3), augment=False)



    history = model.fit_generator(

        generator= train_gen,

        steps_per_epoch=train.shape[0]//Batch_size,

        validation_data=val_gen,

        validation_steps=val.shape[0]//Batch_size,

        epochs=Nepochs,

        callbacks = callbacks_list

    )



    print("\nFine tuning")

    for l in model.layers[:]:

        l.trainable = True

    

    model.compile(optimizer=SGD(lr=Learning_rate*1e-02, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    

    history = model.fit_generator(

        generator = train_gen,

        steps_per_epoch = train.shape[0]//Batch_size,

        validation_data = val_gen,

        validation_steps = val.shape[0]//Batch_size,

        epochs = Nepochs//2,

        callbacks = callbacks_list

    )

    trainig_history.append(history)

    

    print('Loading Best Model')

    model.load_weights(BestModelWeightsPath)

    

    with open('../input/dtd-r1.0.1/dtd/labels/test{}.txt'.format(i)) as f:

        test_imgs = f.readlines()



    test_imgs = [x.strip() for x in test_imgs]

    test = labels_df[labels_df['Img_name'].isin(test_imgs)]

    

    val_gen__ = data_generator.create_valid(test, Batch_size, (img_height, img_width, 3), augment=False)

    acc = model.evaluate_generator(val_gen__, steps=test.shape[0]//Batch_size)[1]

    AccS.append(acc)

    print("\nFold {}/{} Accuracy = {:.3f}".format(i, Nsplits, acc))
print("Fold Accuracies : ", AccS)

print("\nModel cv accuracy = {:.3f} +/- {:.3f}".format(np.mean(AccS), np.std(AccS)))
plt.style.use('seaborn')

def plot_trainig_history(history, fold):

    fig, ax = plt.subplots(1, 2, figsize=(20,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('acc')

    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")

    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")

    ax[0].legend()

    ax[1].legend();

    fig.suptitle("Loss ans Accuracy for Fold {}".format(fold+1), fontsize=16)

    plt.show()

    

for h, f in zip(trainig_history, range(Nsplits)):

    plot_trainig_history(h, f)
t_finish = time.time()

print(f"Kernel run time = {(t_finish-t_start)/3600} hours")