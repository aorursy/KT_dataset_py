import os

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from keras import backend as K

warnings.filterwarnings(action='ignore')



K.image_data_format()
# path 목록

MODEL_PATH = '../input/kakr3rdxception/kakr-3rd-xception'

FOLD_DATA_PATH = '../input/3rd-ml-df-folds/3rd_ml_df_folds/'

DATA_PATH = '../input/carimagesegcrop/car-image-segcropping'



# CSV 파일 경로

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
# Xception : (299, 299)

IMAGE_WIDTH, IMAGE_HEIGHT = (299, 299)

classes = df_class['id'].values.astype('str').tolist()



BATCH_SIZE = 32

EPOCHS = 30

K_FOLDS = 5

PATIENCE = 6
epochs = EPOCHS

batch_size = BATCH_SIZE



def get_total_batch(num_samples, batch_size):

    if (num_samples % batch_size) > 0:

        return (num_samples // batch_size) + 1

    else:

        return num_samples // batch_size
def recall(y_target, y_pred):

    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다

    # round : 반올림한다

    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다



    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다

    count_true_positive = K.sum(y_target_yn * y_pred_yn) 



    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체

    count_true_positive_false_negative = K.sum(y_target_yn)



    # Recall =  (True Positive) / (True Positive + False Negative)

    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다

    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())



    # return a single tensor value

    return recall





def precision(y_target, y_pred):

    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다

    # round : 반올림한다

    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다



    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다

    count_true_positive = K.sum(y_target_yn * y_pred_yn) 



    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체

    count_true_positive_false_positive = K.sum(y_pred_yn)



    # Precision = (True Positive) / (True Positive + False Positive)

    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다

    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())



    # return a single tensor value

    return precision





def f1score(y_target, y_pred):

    _recall = recall(y_target, y_pred)

    _precision = precision(y_target, y_pred)

    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다

    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())

    

    # return a single tensor value

    return _f1score
from sklearn.model_selection import StratifiedKFold, KFold

skfold = StratifiedKFold(n_splits=K_FOLDS, random_state=1993)
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing.image import ImageDataGenerator



datagen_train = ImageDataGenerator(

    rescale = 1./255,               

    featurewise_center = False,              # set input mean to 0 over the dataset

    samplewise_center = False,               # set each sample mean to 0

    featurewise_std_normalization = False,   # divide inputs by std of the dataset

    samplewise_std_normalization = False,    # divide each input by its std

    zca_whitening = False,                   # apply ZCA whitening

    rotation_range = 20,                     # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.1,                        # randomly zoom range

    width_shift_range = 0.1,                 # randomly zoom images horizontally (fraction of total width)

    height_shift_range = 0.1,                # randomly shift images vertically (fraction of total height)

    horizontal_flip = True,                  # randomly flip images

    vertical_flip = False,                   # randomly flip images

    preprocessing_function = preprocess_input

)



# validation, test셋 이미지는 augmentation을 적용하지 않습니다.

datagen_val = ImageDataGenerator(

    rescale = 1./255     

#     featurewise_center = False,              # set input mean to 0 over the dataset

#     samplewise_center = False,               # set each sample mean to 0

#     featurewise_std_normalization = False,   # divide inputs by std of the dataset

#     samplewise_std_normalization = False,    # divide each input by its std

#     zca_whitening = False,                   # apply ZCA whitening

#     rotation_range = 20,                     # randomly rotate images in the range (degrees, 0 to 180)

#     zoom_range = 0.1,                        # randomly zoom range

#     width_shift_range = 0.1,                 # randomly zoom images horizontally (fraction of total width)

#     height_shift_range = 0.1,                # randomly shift images vertically (fraction of total height)

#     horizontal_flip = True,                  # randomly flip images

#     vertical_flip = False,                   # randomly flip images

#     preprocessing_function = preprocess_input

)



df_train['class'] = df_train['class'].astype('str')   # 안해주면 Error 발생.
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.applications.xception import Xception

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D



models = {}



def get_model(base_model):

    base_model = base_model(weights='imagenet', include_top = False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    

    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(196, activation='softmax', kernel_initializer='he_normal'))          # 196개의 class를 분류해야하므로 활성화 함수로는 softmax, 

    model.summary()

    

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc', f1score])

    

    return model



__models = {"Xception" : Xception}
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback



def get_callbacks(model_save_filename, patience):

    # 더 이상 개선의 여지가 없을 때 학습을 종료시키는 역할.

    es = EarlyStopping(monitor = 'val_f1score', 

                       min_delta = 0,                   # 개선되고 있다고 판단하기 위한 최소 변화량.

                       patience = patience, 

                       verbose = 1,                     # 진행사항 출력여부 표시.

                       mode = 'max'                     # 관찰하고 있는 항목이 증가되는 것을 멈출 때 종료합니다.

                       )  

    

    # 모델의 정확도가 향상되지 않는 경우, learning rate를 줄여주는 역할.

    rr = ReduceLROnPlateau(monitor = 'val_f1score', 

                           factor = 0.5,                # 콜백이 호출되면 학습률을 줄이는 정도.

                           patience = patience / 2,

                           min_lr = 0.000001,

                           verbose = 1,

                           mode = 'max'

                           )

    

    # Keras에서 모델을 학습할 때마다 중간중간에 콜백 형태로 알려주는 역할.

    mc = ModelCheckpoint(filepath = model_save_filename, 

                         monitor = 'val_f1score', 

                         verbose = 1, 

                         save_best_only = True,         # 모델의 정확도가 최고값을 갱신했을 때만 저장하도록 하는 옵션.

                         mode = 'max'

                         )

    

    return [es, rr, mc]
import ssl

from keras.models import model_from_json



# ssl 에러를 해결하기 위함.

ssl._create_default_https_context = ssl._create_unverified_context

history_list = {}

for _m in __models:

    print("Model : ", _m)

    

    # 미리 fold를 나누어 생성해 둔 dataframe 파일을 사용합니다.

    for fold_index in range(K_FOLDS):

        os.system("training : model %s fold %d" % (_m, fold_index))



        # Model 생성.

        model = get_model(__models[_m])

        

        # 마찬가지로 미리 생성해둔 weight 파일을 불러와서 MODEL_PATH에 저장합니다.

        model_save_filename = ("%s_%d.h5" % (_m , fold_index))

        model_save_filepath = os.path.join(MODEL_PATH, model_save_filename)

        

        # 나눠진 dataframe을 load.

        df_train_filename = ("fold_%d_train.csv" % fold_index)

        df_val_filename = ("fold_%d_val.csv" % fold_index)



        dataframe_train = pd.read_csv(os.path.join(FOLD_DATA_PATH, df_train_filename))

        dataframe_val = pd.read_csv(os.path.join(FOLD_DATA_PATH, df_val_filename))

        

        # 아래 안해주면 에러남. categorical이어서 기준 col이 숫자값이면 안되는 것인듯.

        dataframe_train['class'] = dataframe_train['class'].astype('str')

        dataframe_val['class'] = dataframe_val['class'].astype('str')



        # ImageDataGenerator 생성(train/val)

        datagen_train_flow = datagen_train.flow_from_dataframe(dataframe=dataframe_train,

                                                   directory=os.path.join(DATA_PATH, "train_segcrop"),

                                                   x_col='img_file',

                                                   y_col="class",

                                                   classes = classes,

                                                   target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

                                                   color_mode='rgb',

                                                   class_mode='categorical',

                                                   batch_size=batch_size,

                                                   seed=1993)



        datagen_val_flow = datagen_val.flow_from_dataframe(dataframe=dataframe_val,

                                                   directory=os.path.join(DATA_PATH, "train_segcrop"),

                                                   x_col='img_file',

                                                   y_col="class",

                                                   classes = classes,

                                                   target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

                                                   color_mode='rgb',

                                                   class_mode='categorical',

                                                   batch_size=batch_size,

                                                   seed=1993)

        

        # 동일 이름의 weight 파일이 있으면 넘어간다는 의미.

        if os.path.exists(model_save_filepath) == True:

            print(">>>", model_save_filepath, " already trained... skip!")

            continue

        

        train_steps = get_total_batch(dataframe_train.shape[0], batch_size)

        val_steps = get_total_batch(dataframe_val.shape[0], batch_size)

            

        history = model.fit_generator(datagen_train_flow,

            epochs=epochs,

            steps_per_epoch = train_steps,

            validation_data = datagen_val_flow,

            validation_steps = val_steps,

            callbacks = get_callbacks(model_save_filepath, PATIENCE),

            verbose=1)

        

        history_list[model_save_filename] = history
fig, ax = plt.subplots(1, 1, figsize=(12,8 * len(__models)))

from cycler import cycler



# set color cycle : 그래프 색깔을 알아서 cycling해준다.

x = np.linspace(0, 1, 10)

number = 5

cmap = plt.get_cmap('gnuplot')

colors = [cmap(i) for i in np.linspace(0, 1, number)]

ax.set_prop_cycle(cycler('color', colors))



for hname in history_list:

    history = history_list[hname]

    plot_label = "val_score : " + hname

    ax.plot(history.history['val_f1score'], label=plot_label)        

ax.legend()        

plt.show()
datagen_submit = ImageDataGenerator(preprocessing_function=preprocess_input)



# 앞서 저장한 5개의 sub-model을 loading하는 함수.

def load_sub_models():

    sub_models = []

    for _m in __models:

        print("Model ", _m, " : ")

        for _, _, filenames in os.walk(MODEL_PATH):

            for fname in filenames:

                if fname.find(_m) >= 0 and fname.find(".h5") >= 0:        

                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> loading ", fname)

                    model = get_model(__models[_m])



                    # model에 데이터파일을 올린다.

                    # model 명이 있는 것들 중에서만 올린다.

                    fpath = os.path.join(MODEL_PATH, fname)

                    print("model weight fpath:", fpath)

                    model.load_weights(fpath)



                    sub_models.append(model)

    return sub_models



submodels = load_sub_models()
from numpy import dstack     # 각 sub-model의 예측 결과를 dataset으로 생성하기 위한 dstack.



def make_meta_learner_dataset(submodels, df, imgdirname):

    datagen_submit = ImageDataGenerator(preprocessing_function=preprocess_input)

    stackX = None

    for model in submodels:



        # make prediction

        datagen_metalearner_flow = datagen_submit.flow_from_dataframe(

            dataframe=df,

            directory=os.path.join(DATA_PATH, imgdirname),

            x_col='img_file',

            y_col=None,

            target_size= (IMAGE_WIDTH, IMAGE_HEIGHT),

            color_mode='rgb',

            class_mode=None,

            batch_size=batch_size,

            shuffle=False)



        datagen_metalearner_flow.reset()

        pred = model.predict_generator(generator = datagen_metalearner_flow,

                                       steps = get_total_batch(df.shape[0], batch_size),

                                       verbose=1)

        

        if stackX is None:

            stackX = pred

        else:

            stackX = dstack((stackX, pred))

   

    print("stackX.shape = ", stackX.shape)

        

    # flatten predictions to [rows, members x probabilities]

    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

    

    return stackX

    # 5개의 sub-model에 대해 각 예측 결과를 dstack하면 (9960, 196, 5)의 shape를 가지게 된다.

    # 이를 meta learner의 training dataset으로 사용하기 위해 (9960, 196*5) 모양으로 reshape한다.
import keras

from keras import layers, models



def make_meta_learner_model(input_shape, output_class_count, dropout):

    print(input_shape)

    print(output_class_count)

    print(dropout)

    print(input_shape[1] * 2)

    

    model = Sequential()

    model.add(layers.Dense(units=input_shape[1] * 2, activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(dropout))

    

    print("01")

    

    model.add(layers.Dense(units=int(input_shape[1] / 2), activation='relu', kernel_initializer='he_normal'))

    model.add(layers.Dropout(dropout))

    

    print("02")

    

    model.add(layers.Dense(units=output_class_count, activation='softmax', kernel_initializer='he_normal'))

    

    print("03")

    

    #print(model.summary())



    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy', f1score])

    

    return model
from keras.utils import to_categorical



print("Build dataset for meta-learner...")

meta_train_X = make_meta_learner_dataset(submodels, df_train, "train_segcrop")

print("meta_train_X.shape=", meta_train_X.shape)
# 모델이 훈련된 label값에 맞게 Y값을 만들어야 한다.



labels = (datagen_train_flow.class_indices)

meta_train_Y = df_train['class'].values

meta_train_Y = [labels[x] for x in meta_train_Y]

meta_train_Y = to_categorical(meta_train_Y)

print("meta_train_Y.shape=", meta_train_Y.shape)
def train_meta_learner(X, Y):

    

    print("Training meta-learner model...")

    meta_learner_model = make_meta_learner_model(X.shape, len(classes), dropout=0.2)

    meta_learner_model.fit(X, Y, epochs=5, verbose=1, batch_size=128)



    return meta_learner_model



meta_learner_model = train_meta_learner(meta_train_X, meta_train_Y)
meta_submit_X = make_meta_learner_dataset(submodels, df_test, "test_segcrop")

pred = meta_learner_model.predict(meta_submit_X, batch_size=32)



submit_Y = np.argmax(pred, axis=1)

labels = (datagen_train_flow.class_indices)

labels = dict((v,k) for k, v in labels.items())



predictions = [labels[k] for k in submit_Y]



submission = pd.DataFrame()

submission['img_file'] = df_test['img_file']

submission["class"] = predictions

submission.to_csv("submission.csv", index=False)
from IPython.display import FileLinks

FileLinks('.') # input argument is specified folder