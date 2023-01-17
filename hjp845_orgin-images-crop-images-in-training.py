import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import warnings

import seaborn as sns

import matplotlib.pylab as plt

import PIL

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import f1_score

from keras import backend as K

from keras import layers, models, optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import *

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



warnings.filterwarnings('ignore')

K.image_data_format()
pd.set_option("display.max_rows", 20)
BATCH_SIZE = 32

EPOCHS = 150

k_folds = 2

PATIENCE = 3

SEED = 2019

BASE_MODEL = Xception

IMAGE_SIZE = 299
os.listdir('../input')
# 기존 train crop 이미지 파일의 개수, 기존 train 이미지 파일 개수와 동일. 나는 train crop + train origin을 원함.

print(len(next(os.walk('../input/3rd-ml-month-car-image-cropping-dataset/train_crop'))[2])) 
DATA_PATH = '../input/2019-3rd-ml-month-with-kakr'



TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')



df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))



model_path = './'

if not os.path.exists(model_path):

    os.mkdir(model_path)
# df_train에 crop하고 origin 데이터 클래스 다 적혀 있는 df 만듬

df_train_crop = df_train.copy()

df_train_crop['img_file'] = [x[:-4] + '_crop.jpg' for x in df_train['img_file']] 
# 주의 합성이라 한번만 실행되어야 한다 *****

# axis = 0 은 행이 더 늘어나는 방향으로 붙여 주는 것이다. <-> axis = 1



df_train = pd.concat([df_train_crop, df_train], axis=0)
df_train
def crop_boxing_img(img_name, margin=0, size=(IMAGE_SIZE,IMAGE_SIZE)):

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
import os

os.listdir('../input/2019-3rd-ml-month-with-kakr/test')
%%time

# 크롭된 이미지 경로

#TRAIN_CROPPED_PATH = '../input/3rd-ml-month-car-image-cropping-dataset/train_crop'

# 기본과 크롭 섞은거!

TRAIN_CROPPED_PATH = '../input/train-origincrop/train_all/train_all'



# 크롭된 이미지 경로

TEST_CROPPED_PATH = '../input/3rd-ml-month-car-image-cropping-dataset/test_crop'

# 기본 사진으로 테스트!

#TEST_CROPPED_PATH = '../input/2019-3rd-ml-month-with-kakr/test'



# if (os.path.isdir(TRAIN_CROPPED_PATH) == False):

#     os.mkdir(TRAIN_CROPPED_PATH)



# if (os.path.isdir(TEST_CROPPED_PATH) == False):

#     os.mkdir(TEST_CROPPED_PATH)



# for i, row in df_train.iterrows():

#     cropped = crop_boxing_img(row['img_file'])

#     cropped.save(os.path.join(TRAIN_CROPPED_PATH, row['img_file']))



# for i, row in df_test.iterrows():

#     cropped = crop_boxing_img(row['img_file'])

#     cropped.save(os.path.join(TEST_CROPPED_PATH, row['img_file']))
df_train['class'] = df_train['class'].astype('str')

df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]
def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
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

    model.add(layers.Dense(2048, activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(0.15))

 

    model.add(layers.Dense(196, activation='softmax', kernel_initializer='lecun_normal'))

    model.summary()



    optimizer = optimizers.Nadam(lr=0.0002)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1_m, precision_m, recall_m])



    return model
#ref: https://github.com/yu4u/cutout-random-erasing/blob/master/cifar10_resnet.py

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    def eraser(input_img):

        img_h, img_w, img_c = input_img.shape

        p_1 = np.random.rand()



        if p_1 > p:

            return input_img



        while True:

            s = np.random.uniform(s_l, s_h) * img_h * img_w

            r = np.random.uniform(r_1, r_2)

            w = int(np.sqrt(s / r))

            h = int(np.sqrt(s * r))

            left = np.random.randint(0, img_w)

            top = np.random.randint(0, img_h)



            if left + w <= img_w and top + h <= img_h:

                break



        if pixel_level:

            c = np.random.uniform(v_l, v_h, (h, w, img_c))

        else:

            c = np.random.uniform(v_l, v_h)



        input_img[top:top + h, left:left + w, :] = c



        return input_img



    return eraser
train_datagen = ImageDataGenerator(

    rescale=1./255,

    #featurewise_center= True,  # set input mean to 0 over the dataset

    #samplewise_center=True,  # set each sample mean to 0

    #featurewise_std_normalization= True,  # divide inputs by std of the dataset

    #samplewise_std_normalization=True,  # divide each input by its std

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=False,

    zoom_range=0.2,

    #shear_range=0.2,

    #brightness_range=(1, 1.2),

    fill_mode='nearest',

    preprocessing_function = get_random_eraser(v_l=0, v_h=255),

    )



valid_datagen = ImageDataGenerator(

    rescale=1./255,

    #featurewise_center= True,  # set input mean to 0 over the dataset

    #samplewise_center=True,  # set each sample mean to 0

    #featurewise_std_normalization= True,  # divide inputs by std of the dataset

    #samplewise_std_normalization=True  # divide each input by its std

    )

test_datagen = ImageDataGenerator(

    rescale=1./255,

    #featurewise_center= True,  # set input mean to 0 over the dataset

    #samplewise_center=True,  # set each sample mean to 0

    #featurewise_std_normalization= True,  # divide inputs by std of the dataset

    #samplewise_std_normalization=True,  # divide each input by its std

    )
skf = StratifiedKFold(n_splits=k_folds, random_state=SEED)

#skf = KFold(n_splits=k_folds, random_state=SEED)
j = 1

model_names = []

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

        seed=SEED,

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

        seed=SEED,

        shuffle=True

        )

    

    model_name = model_path + str(j) + '_' + 'Xception' + '.hdf5'

    model_names.append(model_name)

    

    model = get_model(BASE_MODEL, IMAGE_SIZE)

    

    try:

        model.load_weights(model_name)

    except:

        pass

        

    history = model.fit_generator(

        train_generator,

        steps_per_epoch=len(traindf.index) / BATCH_SIZE,

        epochs=150, #########################################################

        validation_data=valid_generator,

        validation_steps=len(validdf.index) / BATCH_SIZE,

        verbose=1,

        shuffle=False,

        callbacks = get_callback(model_name, PATIENCE)

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
prediction = []

for i, name in enumerate(model_names):

    model = get_model(BASE_MODEL, IMAGE_SIZE)

    model.load_weights(name)

    

    test_generator.reset()

    pred = model.predict_generator(

        generator=test_generator,

        steps = len(df_test)/BATCH_SIZE,

        verbose=1

    )

    prediction.append(pred)



y_pred = np.mean(prediction, axis=0)
preds_class_indices=np.argmax(y_pred, axis=1)
labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

final_pred = [labels[k] for k in preds_class_indices]
submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

# if(JUST_FOR_TESTING):

#     submission=submission[:10]

submission["class"] = final_pred

submission.to_csv("submission.csv", index=False)

submission.head()