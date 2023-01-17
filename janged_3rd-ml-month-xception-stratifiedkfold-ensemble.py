# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

import seaborn as sns

import matplotlib.pylab as plt

import PIL

from sklearn.model_selection import StratifiedKFold, KFold



from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator



from keras.applications import Xception



from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras import layers, models, optimizers
warnings.filterwarnings('ignore')

K.image_data_format()
IMAGE_SIZE = 299

BATCH_SIZE = 32

EPOCHS = 100

k_folds=5
DATA_PATH = '../input'
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
df_train.head()
plt.figure(figsize=(15,6))

sns.countplot('class', data=df_train)

plt.show()
df_train['class'].value_counts()
df_train['class'].value_counts().mean()
df_train['class'].value_counts().describe()
def crop_boxing_img(img_name, margin=-4, size=(IMAGE_SIZE,IMAGE_SIZE)):

    if img_name.split('_')[0] == 'train':

        PATH = TRAIN_IMG_PATH

        data = df_train

    else:

        PATH = TEST_IMG_PATH

        data = df_test



    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)



    return img.crop((x1, y1, x2, y2)).resize(size)
TRAIN_CROPPED_PATH = '../cropped_train'

TEST_CROPPED_PATH = '../cropped_test'
if (os.path.isdir(TRAIN_CROPPED_PATH) == False):

    os.mkdir(TRAIN_CROPPED_PATH)



if (os.path.isdir(TEST_CROPPED_PATH) == False):

    os.mkdir(TEST_CROPPED_PATH)



for i, row in df_train.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(TRAIN_CROPPED_PATH, row['img_file']))



for i, row in df_test.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(TEST_CROPPED_PATH, row['img_file']))
df_train['class'] = df_train['class'].astype('str')

df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]
model_path = './'
def get_callback(model_name, patient):

    ES = EarlyStopping(

        monitor='val_loss', 

        patience=patient, 

        mode='min', 

        verbose=1)

    RR = ReduceLROnPlateau(

        monitor = 'val_loss', 

        factor = 0.5, 

        patience = patient / 2, 

        min_lr=0.000001, 

        verbose=1, 

        mode='min')

    MC = ModelCheckpoint(

        filepath=model_name, 

        monitor='val_loss', 

        verbose=1, 

        save_best_only=True, 

        mode='min')



    return [ES, RR, MC]
def get_model(model_name, iamge_size):

    base_model = model_name(weights='imagenet', input_shape=(iamge_size,iamge_size,3), include_top=False)

    #base_model.trainable = False

    model = models.Sequential()

    model.add(base_model)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.Dropout(0.25))

 

    model.add(layers.Dense(196, activation='softmax'))

    model.summary()



    optimizer = optimizers.RMSprop(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])



    return model
train_datagen = ImageDataGenerator(

    rescale=1./255,

    #featurewise_center= True,  # set input mean to 0 over the dataset

    #samplewise_center=True,  # set each sample mean to 0

    #featurewise_std_normalization= True,  # divide inputs by std of the dataset

    #samplewise_std_normalization=True,  # divide each input by its std

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=False,

    zoom_range=0.3,

    shear_range=0.3,

    #brightness_range=(1, 1.2),

    fill_mode='nearest'

    )



valid_datagen = ImageDataGenerator(

    rescale=1./255,

    #featurewise_center= True,  # set input mean to 0 over the dataset

    #samplewise_center=True,  # set each sample mean to 0

    #featurewise_std_normalization= True,  # divide inputs by std of the dataset

    #samplewise_std_normalization=True  # divide each input by its std

    )

test_datagen = ImageDataGenerator(

    rescale=1./255

    #featurewise_center= True,  # set input mean to 0 over the dataset

    #samplewise_center=True,  # set each sample mean to 0

    #featurewise_std_normalization= True,  # divide inputs by std of the dataset

    #samplewise_std_normalization=True,  # divide each input by its std

    )
skf = StratifiedKFold(n_splits=k_folds, random_state=2019)

#skf = KFold(n_splits=k_folds, random_state=2019)
j = 1

model_xception_names = []

for (train_index, valid_index) in skf.split(

    df_train['img_file'], 

    df_train['class']):



    traindf = df_train.iloc[train_index, :].reset_index()

    validdf = df_train.iloc[valid_index, :].reset_index()



    print("=========================================")

    print("====== K Fold Validation step => %d/%d =======" % (j,k_folds))

    print("=========================================")

    

    train_generator = train_datagen.flow_from_dataframe(

        dataframe=traindf,

        directory=TRAIN_CROPPED_PATH,

        x_col='img_file',

        y_col='class',

        target_size= (IMAGE_SIZE, IMAGE_SIZE),

        color_mode='rgb',

        class_mode='categorical',

        batch_size=BATCH_SIZE,

        seed=2019,

        shuffle=True

        )



    valid_generator = valid_datagen.flow_from_dataframe(

        dataframe=validdf,

        directory=TRAIN_CROPPED_PATH,

        x_col='img_file',

        y_col='class',

        target_size= (IMAGE_SIZE, IMAGE_SIZE),

        color_mode='rgb',

        class_mode='categorical',

        batch_size=BATCH_SIZE,

        seed=2019,

        shuffle=True

        )



    model_name = model_path + str(j) + '_xception.hdf5'

    model_xception_names.append(model_name)

    

    model_xception = get_model(Xception, IMAGE_SIZE)

    

    try:

        model_xception.load_weights(model_name)

    except:

        pass

        

    history = model_xception.fit_generator(

        train_generator,

        steps_per_epoch=len(traindf.index) / BATCH_SIZE,

        epochs=EPOCHS,

        validation_data=valid_generator,

        validation_steps=len(validdf.index) / BATCH_SIZE,

        verbose=1,

        shuffle=False,

        callbacks = get_callback(model_name, 6)

        )

        

    j+=1
test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=TEST_CROPPED_PATH,

    x_col='img_file',

    y_col=None,

    target_size= (IMAGE_SIZE, IMAGE_SIZE),

    color_mode='rgb',

    class_mode=None,

    batch_size=BATCH_SIZE,

    shuffle=False

)
xception_prediction = []

for i, name in enumerate(model_xception_names):

    model_xception = get_model(Xception, IMAGE_SIZE)

    model_xception.load_weights(name)

    test_generator.reset()

    pred = model_xception.predict_generator(

        generator=test_generator,

        steps = len(df_test)/BATCH_SIZE,

        verbose=1

    )

    xception_prediction.append(pred)



y_pred_xception = np.mean(xception_prediction, axis=0)
preds_class_indices=np.argmax(y_pred_xception, axis=1)
labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

final_pred = [labels[k] for k in preds_class_indices]
submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission["class"] = final_pred

submission.to_csv("submission.csv", index=False)

submission.head()