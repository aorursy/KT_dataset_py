# import libraries

import gc

import os

import sys

import PIL

import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras.backend as K

from tqdm import tqdm



warnings.filterwarnings('ignore')



from keras_preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit



from keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPool2D, Dropout, MaxPooling2D

from keras.models import Model

!pip install efficientnet==0.0.4

from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3

from keras.applications.xception import Xception

from efficientnet import EfficientNetB3
def build_model(model_name=None, include_top=False, input_shape=(256,256,3), fine_tuning=True, layer_to_freeze=None, load_pretrained : str = None, summary=False):

    pmodel_name = model_name.strip().lower()

    print(pmodel_name)



    if pmodel_name == 'resnet50': base_model = ResNet50(include_top=include_top, input_shape=input_shape)

    elif pmodel_name == 'inception_v3' : base_model = InceptionV3(include_top=include_top, input_shape=input_shape)

    elif pmodel_name == 'xception' : base_model = Xception(include_top=include_top, input_shape=input_shape)

    elif pmodel_name == 'efficient_net' : base_model = EfficientNetB3(include_top=include_top, input_shape=input_shape)

    else : raise ValueError



    if fine_tuning:

        # Freese layers

        assert layer_to_freeze != None, 'You must define layer\'s name to freese.'

        fr_layer_name = layer_to_freeze

        set_trainable = False



        for layer in base_model.layers:

            if not layer.name == fr_layer_name:

                set_trainable = True



            layer.trainable = set_trainable



    # change last layers

    last_1dconv_1 = Conv2D(1024, 1, activation='relu')(base_model.output)

    last_pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(last_1dconv_1)

    global_avg_pool = GlobalAveragePooling2D()(last_pool_1)

#     last_Dense_1 = Dense(512, activation='relu')(global_avg_pool)

    last_Dense_2 = Dense(196, activation='softmax')(global_avg_pool)



    # compile

    model = Model(base_model.input, last_Dense_2)



    # summary

    if summary:

        model.summary()



    # load pretrained weights

    if load_pretrained:

        model.load_weights(load_pretrained)



    return model



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

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



# Define steps per epoch

def get_steps(num_samples, batch_size):

    if (num_samples // batch_size) > 0:

        return (num_samples // batch_size) + 1

    else:

        return num_samples // batch_size



# https://www.kaggle.com/seriousran/cutout-augmentation-on-keras-efficientnet

def get_random_erazer(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    def erazer(input_img):

        img_h, img_w, img_c = input_img.shape

        p_1 = np.random.rand()



        if p_1 > p :

            return input_img



        while True:

            s = np.random.uniform(s_l, s_h) * img_h * img_w

            r = np.random.uniform(r_1, r_2)

            w = int(np.sqrt(s/r))

            h = int(np.sqrt(s*r))

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



    return erazer





# For prediction

def predict_class(model, num_samples, batch_size, train_generator, test_generator, output_name, DATA_PATH = '/kaggle/input/2019-3rd-ml-month-with-kakr', OUTPUT_PATH = './output'):



    # Prediction

    prediction = model.predict_generator(

        generator=test_generator,

        steps=get_steps(num_samples, batch_size),

        verbose=1

    )



    predicted_indices = np.argmax(prediction, axis=1)

    labels = (train_generator.class_indices)

    labels = dict((v, k) for k, v in labels.items())

    predictions = [labels[k] for k in predicted_indices]

    

    # Load submission form

    submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

    submission['class'] = predictions

    submission.to_csv(os.path.join(OUTPUT_PATH, '{}.csv'.format(output_name)), index=False)



def predict_class_ensemble(models, weights, num_samples, batch_size, train_generator, test_generator, DATA_PATH = '/kaggle/input/2019-3rd-ml-month-with-kakr', OUTPUT_PATH = './output'):

    predictions = []





    if not models == None:

        num_models = len(models)



        if not weights == None:

            num_weights = len(weights)

            for model_name in models:

                for i, weight_name in enumerate(weights):

                    print('=== predict {0} model - {1}\'s split ==='.format(model_name, i))

                    model = build_model(model_name=model_name, input_shape=(299,299,3), fine_tuning=False, summary=False)

                    model.load_weights(os.path.join('/kaggle/input/3rd-ml-month-efficeintnet-5-folds', weight_name))



                    test_generator.reset()



                    prediction = model.predict_generator(

                        generator=test_generator,

                        steps=get_steps(num_samples, batch_size),

                        verbose=1

                    )



                    predictions.append(prediction)



        else:

            for model_name in models:

                print('=== predict {} model ==='.format(model_name))

                model = build_model(model_name=model_name, input_shape=(299, 299, 3), fine_tuning=False, summary=False)

                model.load_weights(os.path.join('/kaggle/input/3rd-ml-month-efficeintnet-5-folds', '{}_model_50_epochs.h5'.format(model_name)))



                # 한번 예측 후에는 반드시 리셋 필수!

                test_generator.reset()



                prediction = model.predict_generator(

                    generator=test_generator,

                    steps=get_steps(num_samples, batch_size),

                    verbose=1

                )



                print(np.argmax(prediction, axis=-1)[:10])



                predictions.append(prediction)



        print('Complete!')

        predictions = np.array(predictions)

        print(predictions.shape)



        predictions = np.mean(predictions, axis=0)



        # 제출 형식으로 변환

        predict_indices = np.argmax(predictions, axis=-1)

        labels = (train_generator.class_indices)

        labels = dict((v,k) for k,v in labels.items())

        prediction_ensemble = [labels[k] for k in predict_indices]



        # 제출

        output_name = 'ensemble_' + ','.join(models) + '_{}_splits.csv'.format(num_weights)

        submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

        submission['class'] = prediction_ensemble

        submission.to_csv(os.path.join(OUTPUT_PATH, output_name), index=False)

        

def crop_boxing_img(img, pos, margin=16):

    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(width, pos[2] + margin)

    y2 = min(height, pos[3] + margin)



    cropped_img = img.crop((x1,y1,x2,y2))

    # plt.imshow(cropped_img)

    # plt.show()

    return cropped_img
# set random seed

RANDOM_SEED = 40



# 데이터 경로 설정

DATA_PATH = '/kaggle/input/2019-3rd-ml-month-with-kakr'

OUTPUT_PATH = './cropped_images'

os.listdir(DATA_PATH)



# 이미지 경로 설정

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')
# generate cropped images

if not os.path.exists('./output'):

    os.mkdir('./output')

if not os.path.exists('./cropped_images'):

    os.mkdir('./cropped_images')

    

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))



if not os.path.exists('./cropped_images/train_crop'):

    os.mkdir('./cropped_images/train_crop')



if not os.path.exists('./cropped_images/test_crop'):

    os.mkdir('./cropped_images/test_crop')



# 훈련 이미지 자르기

for i, img_name in tqdm(enumerate(df_train['img_file'])):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, img_name))

    pos = df_train.iloc[i][['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

    cropped_img = crop_boxing_img(img, pos)

    cropped_img.save(os.path.join(OUTPUT_PATH, 'train_crop/'+img_name))



# 시험 이미지 자르기

for i, img_name in tqdm(enumerate(df_test['img_file'])):

    img = PIL.Image.open(os.path.join(TEST_IMG_PATH, img_name))

    pos = df_test.iloc[i][['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

    cropped_img = crop_boxing_img(img, pos)

    cropped_img.save(os.path.join(OUTPUT_PATH, 'test_crop/' + img_name))
# 데이터 경로 설정

DATA_PATH = '/kaggle/input/2019-3rd-ml-month-with-kakr'

OUTPUT_PATH = './output'

os.listdir(DATA_PATH)



# 이미지 경로 설정

TRAIN_IMG_PATH = os.path.join('./cropped_images', 'train_crop')

TEST_IMG_PATH = os.path.join('./cropped_images', 'test_crop')



# 데이터 읽어오기

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))



# 모델링을 위한 데이터 준비

df_train['class'] = df_train['class'].astype('str')



df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]
# Parameters

img_size = (299, 299)

epochs = 50

batch_size = 16

learning_rate = 0.0002

base_model = 'efficient_net'

load_pretrained = None

patience = 5

n_splits=1
# Define Generator config

train_datagen = ImageDataGenerator(

    horizontal_flip = True,

    vertical_flip = False,

    zoom_range = 0.10,

    rotation_range = 20,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range=0.5,

    brightness_range=[0.5, 1.5],

    fill_mode='nearest',

    rescale=1./255,

    preprocessing_function=get_random_erazer(v_l=0, v_h=255)

)



val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)
# Train, Test spllit

splitter = StratifiedShuffleSplit(n_splits=n_splits, train_size=0.9, test_size=0.1, random_state=RANDOM_SEED)



scores = []
# 학습

for i, (trn_idx, val_idx) in enumerate(splitter.split(df_train['img_file'], df_train['class'])):



    print('============', '{}\'th Split'.format(i), '============\n')



    X_train = df_train.iloc[trn_idx].copy()

    X_val = df_train.iloc[val_idx].copy()

    print('Train : {0} / Test : {1}'.format(X_train.shape, X_val.shape))



    train_size = len(X_train)

    val_size = len(X_val)



    train_generator = train_datagen.flow_from_dataframe(

        dataframe = X_train,

        directory = TRAIN_IMG_PATH,

        x_col = 'img_file',

        y_col = 'class',

        target_size = img_size,

        color_mode = 'rgb',

        class_mode = 'categorical',

        batch_size = batch_size,

        seed=RANDOM_SEED,

        shuffle=True,

        interploation = 'bicubic'

    )



    val_generator = val_datagen.flow_from_dataframe(

        dataframe = X_val,

        directory = TRAIN_IMG_PATH,

        x_col = 'img_file',

        y_col = 'class',

        target_size = img_size,

        color_mode = 'rgb',

        class_mode = 'categorical',

        batch_size = batch_size,

        seed=RANDOM_SEED,

        shuffle=True,

        interploation = 'bicubic'

    )



    test_generator = test_datagen.flow_from_dataframe(

        dataframe = df_test,

        directory = TEST_IMG_PATH,

        x_col = 'img_file',

        y_col = None,

        target_size = img_size,

        color_mode = 'rgb',

        class_mode = None,

        batch_size = batch_size,

        shuffle=False

    )



    # 훈련 루틴

    """

    model = build_model(model_name=base_model, input_shape=(299, 299, 3), fine_tuning=False, summary=False)



    # Load pretrained

    if load_pretrained is not None:

        print('Loading Pretrained networks {}'.format(load_pretrained))

        model.load_weights(os.path.join(os.getcwd(), './weights/' + load_pretrained))





    optimizer = optimizers.adam(lr=learning_rate)



    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', f1_m])



    if not os.path.exists('./weights'):

        os.mkdir('./weights')



    # checkpoint_filename = 'inception_v3_model_{0}_epochs.h5'.format(epochs)

    if n_splits > 1:

        checkpoint_filename = '{0}_model_{1}_epochs_split_{2}.h5'.format(base_model,epochs, i)

    else:

        checkpoint_filename = '{0}_model_{1}_epochs.h5'.format(base_model, epochs)





    # Define callbacks

    reduce_lr = ReduceLROnPlateau(

        monitor='val_loss',

        factor=0.5,

        patience=int(patience / 2),

        verbose=1,

        mode='min',

        min_lr=0.0000001

    )



    early_stopping = EarlyStopping(

        monitor='val_loss',

        min_delta=0.000001,

        patience=patience,

        verbose=1,

        mode='min'

    )



    model_checkpoint = ModelCheckpoint(

        filepath=os.path.join('./weights', checkpoint_filename),

        monitor='val_loss',

        verbose=1,

        save_best_only=True,

        save_weights_only=True,

        mode='min',

        period=5

    )



    callbacks = [reduce_lr, early_stopping, model_checkpoint]



    # 훈련

    history = model.fit_generator(

        generator = train_generator,

        steps_per_epoch = get_steps(train_size, batch_size),

        epochs = epochs,

        callbacks = callbacks,

        validation_data = val_generator,

        validation_steps = get_steps(val_size, batch_size)

    )

    """

    ## 단일모델 예측

    # Predict class

    # predict_class(model, df_test.shape[0], batch_size, train_generator, test_generator, checkpoint_filename[:-3])



    # Predict ensemble

    # models = ['inception_v3', 'resnet50', 'xception', 'efficient_net']

    weights = ['efficient_net_model_50_epochs_split_0.h5',

               'efficient_net_model_50_epochs_split_1.h5',

               'efficient_net_model_50_epochs_split_2.h5',

               'efficient_net_model_50_epochs_split_3.h5',

               'efficient_net_model_50_epochs_split_4.h5']

    predict_class_ensemble(['efficient_net'], weights, df_test.shape[0], batch_size, train_generator, test_generator)



    gc.collect()
# output file이 지나지게 많아지기때문에 전처리한 사진은 삭제해주겠습니다.

!rm -r cropped_images