import gc

import os

import warnings

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D

from keras import layers

from keras.optimizers import SGD, RMSprop





import os

print(os.listdir("../input"))
#efficientnet download

!pip install -U efficientnet==0.0.4

from efficientnet import EfficientNetB3
#crop data directory

DATA_PATH = '../input/car-crop'

os.listdir(DATA_PATH)
#original data directory

DATA_PATH2 = '../input/2019-3rd-ml-month-with-kakr'

os.listdir(DATA_PATH2)
#semi_data directory

DATA_PATH3 = '../input/semi-detaset'

os.listdir(DATA_PATH3)
#crop merge directory

DATA_PATH4 = '../input/car-crop2'

os.listdir(DATA_PATH4)
# 이미지 폴더 경로

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')



# CSV 파일 경로

df_train = pd.read_csv(os.path.join(DATA_PATH2, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH2, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH2, 'class.csv'))



# 버전 1의 submission load

df_semi = pd.read_csv(os.path.join(DATA_PATH3, 'Pseudo Labelsing.csv'))



#버전 1에서 test를 tset로 저장하여 변경해줌 

name = list(map(lambda x:  x.replace("tset", "test"),df_semi['img_file']))

df_semi['img_file']=name

df_semi['img_file'] = df_semi['img_file']+'.jpg'

df_semi.head(5)
df_train["class"] = df_train["class"].astype('str')

df_semi["class"] = df_semi["class"].astype('str')

df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]



# train과 semi 데이터 병합

df_train2 = pd.concat([df_train, df_semi],axis=0)





its = np.arange(df_train2.shape[0])

train_idx, val_idx = train_test_split(its, train_size = 0.8, random_state=42)



X_train = df_train2.iloc[train_idx, :]

X_val = df_train2.iloc[val_idx, :]



print(X_train.shape)

print(X_val.shape)

print(df_test.shape)

df_train2.head(5)
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
import keras.backend as K

from keras.legacy import interfaces

from keras.optimizers import Optimizer





class AdamAccumulate(Optimizer):



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):

        if accum_iters < 1:

            raise ValueError('accum_iters must be >= 1')

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))

        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr



        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())



        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * completed_updates))



        t = completed_updates + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))



        # self.iterations incremented after processing a batch

        # batch:              1 2 3 4 5 6 7 8 9

        # self.iterations:    0 1 2 3 4 5 6 7 8

        # update_switch = 1:        x       x    (if accum_iters=4)  

        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)

        update_switch = K.cast(update_switch, K.floatx())



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]



        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):



            sum_grad = tg + g

            avg_grad = sum_grad / self.accum_iters_float



            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))

            else:

                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))

            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))

            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
# Parameter

img_size = (300, 300)

image_size = 300

nb_train_samples = len(X_train)

nb_validation_samples = len(X_val)

nb_test_samples = len(df_test)

epochs = 30

batch_size = 32



# Define Generator config

train_datagen =ImageDataGenerator(

    rescale=1./255,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    vertical_flip=False,

    fill_mode='nearest',

    preprocessing_function = get_random_eraser(v_l=0, v_h=1),

    )



val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)
#generator

train_generator = train_datagen.flow_from_dataframe(

    dataframe=X_train, 

    directory='../input/car-crop2/train2_crop',

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    seed=42

)



validation_generator = val_datagen.flow_from_dataframe(

    dataframe=X_val, 

    directory='../input/car-crop2/train2_crop',

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size

)



test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory='../input/car-crop/test_crop',

    x_col='img_file',

    y_col=None,

    target_size= img_size,

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False

)
#model

opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=5)

EfficientNet_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))





model = Sequential()

model.add(EfficientNet_model)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(2048, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(196, activation='softmax'))

model.summary()



#compile

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
def get_steps(num_samples, batch_size):

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
%%time

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



#model path

MODEL_SAVE_FOLDER_PATH = './model/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):

    os.mkdir(MODEL_SAVE_FOLDER_PATH)



model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'



patient = 3

callbacks_list = [

     EarlyStopping(

        # 모델의 검증 정확도 모니터링

        monitor='val_loss',

        # patient(정수)보다 정확도가 향상되지 않으면 훈련 종료

        patience=patient, 

        # 검증에 대해 판단하기 위한 기준, val_loss경우 감소되는 것이므로 min

        mode='min', 

        #얼마나 자세하게 정보를 나타낼것인가.

        verbose=1

                          

    ),

    ReduceLROnPlateau(

        monitor = 'val_loss', 

        #콜백 호출시 학습률(lr)을 절반으로 줄임

        factor = 0.5, 

        #위와 동일

        patience = patient / 2, 

        #최소학습률

        min_lr=0.00001,

        verbose=1,

        mode='min'

    ),

    ModelCheckpoint(

        filepath=model_path,

        monitor ='val_loss',

        # val_loss가 좋지 않으면 모델파일을 덮어쓰지 않는다

        save_best_only = True,

        verbose=1,

        mode='min') ]



    



history = model.fit_generator(

    train_generator,

    steps_per_epoch = get_steps(nb_train_samples, batch_size),

    epochs=epochs,

    validation_data = validation_generator,

    validation_steps = get_steps(nb_validation_samples, batch_size),

    callbacks = callbacks_list

)

gc.collect()
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, label='Training acc')

plt.plot(epochs, val_acc, label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.ylim(0.9,1)

plt.show()
plt.plot(epochs, loss, label='Training loss')

plt.plot(epochs, val_loss, label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.ylim(0,0.5)

plt.show()
%%time

test_generator.reset()

prediction = model.predict_generator(

    generator = test_generator,

    steps = get_steps(nb_test_samples, batch_size),

    verbose=1

)
submission = pd.read_csv(os.path.join(DATA_PATH2, 'sample_submission.csv'))

predicted_class_indices=np.argmax(prediction, axis=1)



# Generator class dictionary mapping

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission["class"] = predictions

submission.to_csv("submission_all.csv", index=False)

submission.head()