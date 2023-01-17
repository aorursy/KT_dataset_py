import time

t_start = time.time()
import warnings

warnings.filterwarnings("ignore")



import pandas as pd 

import numpy as np

import os, gc, sys

import matplotlib.pyplot as plt

import seaborn as sns



import keras

from keras import backend as K

from keras import layers, models, optimizers, applications

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img



from keras.models import Model, load_model

from keras.applications.resnet50 import preprocess_input

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint





sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))

from efficientnet import EfficientNetB5
# Model parameters



IMG_WIDTH       = 512

IMG_HEIGHT      = 512

CHANNEL         = 3



BATCH_SIZE      = 4



OLD_DATA_EPOCHS = 6

NEW_DATA_EPOCHS = 6

WARMUP_EPOCHS   = 3



NUM_CLASSES     = 5

SEED            = 2

n_folds         = 1



ES_PATIENCE     = 5

RLROP_PATIENCE  = 2

DECAY_DROP      = 0.2
def append_file_ext_png(file_name):

    return file_name + ".png"



def append_file_ext_jpeg(file_name):

    return file_name + ".jpeg"
BASE_DIR  = '/kaggle/input/aptos2019-blindness-detection/'



TRAIN_DIR = '/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train'

TEST_DIR  = '/kaggle/input/aptos2019-blindness-detection/test_images'
TEST_DF          = pd.read_csv(BASE_DIR + "test.csv",dtype='object')

TRAIN_DF         = pd.read_csv("/kaggle/input/diabetic-retinopathy-resized/trainLabels.csv",dtype='object')

TRAIN_DF.columns = ['id_code', 'diagnosis'] 



X_COL='id_code'

Y_COL='diagnosis'



TRAIN_DF[X_COL] = TRAIN_DF[X_COL].apply(append_file_ext_jpeg)

TEST_DF[X_COL]  = TEST_DF[X_COL].apply(append_file_ext_png)
df0 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '0']

df1 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '1']

df2 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '2']

df3 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '3']

df4 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '4']



df0 = df0.head(2000)

df1 = df1.head(2000)

df2 = df2.head(2000)



TRAIN_DF = df0.append([df1, df2, df3, df4],ignore_index = True)

from sklearn.utils import shuffle

TRAIN_DF = shuffle(TRAIN_DF)
print(TRAIN_DF.head())

print('************************')

print(TEST_DF.head())

print('************************')

print(len(TRAIN_DF))

print('************************')

print(len(TEST_DF))
# def crop_image_from_gray(img, tol=7):

#     # If for some reason we only have two channels

#     if img.ndim == 2:

#         mask = img > tol

#         return img[np.ix_(mask.any(1),mask.any(0))]

#     # If we have a normal RGB images

#     elif img.ndim == 3:

#         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#         mask = gray_img > tol

        

#         check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

#         if (check_shape == 0): # image is too dark so that we crop out everything,

#             return img # return original image

#         else:

#             img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

#             img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

#             img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

#             img = np.stack([img1,img2,img3],axis=-1)

#         return img

    

# def preprocess_image(path, sigmaX=10):

#     image = cv2.imread(path)

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     image = crop_image_from_gray(image)

#     image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))

#     image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)

#     return image
def crop_image_from_gray(image):

    IMAGE_SIZE = (IMG_WIDTH, IMG_WIDTH)

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)

    

    height, width, _ = image.shape

    center_x = int(width / 2)

    center_y = int(height / 2)

    radius = min(center_x, center_y)

    

    circle_mask = np.zeros((height, width), np.uint8)

    cv2.circle(circle_mask, (center_x, center_y), radius, color=1, thickness=-1)

    image = cv2.resize(cv2.bitwise_and(image, image, mask=circle_mask)[center_y - radius:center_y + radius, center_x - radius:center_x + radius], IMAGE_SIZE)

    

    return image



def preprocess_image(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    return image
# Code Source: https://github.com/CyberZHG/keras-radam/blob/master/keras_radam/optimizers.py

class RAdam(keras.optimizers.Optimizer):

    """RAdam optimizer.

    # Arguments

        lr: float >= 0. Learning rate.

        beta_1: float, 0 < beta < 1. Generally close to 1.

        beta_2: float, 0 < beta < 1. Generally close to 1.

        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

        decay: float >= 0. Learning rate decay over each update.

        weight_decay: float >= 0. Weight decay for each param.

        amsgrad: boolean. Whether to apply the AMSGrad variant of this

            algorithm from the paper "On the Convergence of Adam and

            Beyond".

        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.

        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.

        min_lr: float >= 0. Minimum learning rate after warmup.

    # References

        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)

        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)

    """



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,

                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):

        super(RAdam, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

            self.weight_decay = K.variable(weight_decay, name='weight_decay')

            self.total_steps = K.variable(total_steps, name='total_steps')

            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')

            self.min_lr = K.variable(lr, name='min_lr')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.initial_weight_decay = weight_decay

        self.initial_total_steps = total_steps

        self.amsgrad = amsgrad



    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr



        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))



        t = K.cast(self.iterations, K.floatx()) + 1



        if self.initial_total_steps > 0:

            warmup_steps = self.total_steps * self.warmup_proportion

            decay_steps = self.total_steps - warmup_steps

            lr = K.switch(

                t <= warmup_steps,

                lr * (t / warmup_steps),

                lr * (1.0 - K.minimum(t, decay_steps) / decay_steps),

            )



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]

        else:

            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]



        self.weights = [self.iterations] + ms + vs + vhats



        beta_1_t = K.pow(self.beta_1, t)

        beta_2_t = K.pow(self.beta_2, t)



        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0

        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)



        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)



            m_corr_t = m_t / (1.0 - beta_1_t)

            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t) + self.epsilon)

                self.updates.append(K.update(vhat, vhat_t))

            else:

                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t) + self.epsilon)



            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *

                         (sma_t - 2.0) / (sma_inf - 2.0) *

                         sma_inf / sma_t)



            p_t = K.switch(sma_t > 5, r_t * m_corr_t / v_corr_t, m_corr_t)



            if self.initial_weight_decay > 0:

                p_t += self.weight_decay * p



            p_t = p - lr * p_t



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {

            'lr': float(K.get_value(self.lr)),

            'beta_1': float(K.get_value(self.beta_1)),

            'beta_2': float(K.get_value(self.beta_2)),

            'decay': float(K.get_value(self.decay)),

            'weight_decay': float(K.get_value(self.weight_decay)),

            'epsilon': self.epsilon,

            'amsgrad': self.amsgrad,

            'total_steps': float(K.get_value(self.total_steps)),

            'warmup_proportion': float(K.get_value(self.warmup_proportion)),

            'min_lr': float(K.get_value(self.min_lr)),

        }

        base_config = super(RAdam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    

    base_model = EfficientNetB5(weights=None, 

                                       include_top=False,

                                       input_tensor=input_tensor)

    base_model.load_weights('/kaggle/input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5')

        



    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)

    x = Dense(2048, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    

    final_output = Dense(n_out, activation='softmax', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
es = EarlyStopping(monitor='val_loss',

                   mode='min', 

                   patience=ES_PATIENCE, 

                   restore_best_weights=True, 

                   verbose=1)



rlrop = ReduceLROnPlateau(monitor='val_loss', 

                          mode='min', 

                          patience=RLROP_PATIENCE, 

                          factor=DECAY_DROP, 

                          verbose=1)



model_checkpoint = ModelCheckpoint('EfficientNetB5.h5',

                                   verbose=1, 

                                   save_best_only=True)



callback_list = [es, rlrop, model_checkpoint]
datagen         = ImageDataGenerator(

                    rescale=1./255.,

                    validation_split=0.25)



train_generator = datagen.flow_from_dataframe(

                    dataframe=TRAIN_DF,

                    directory=TRAIN_DIR,

                    x_col=X_COL,

                    y_col=Y_COL,

                    subset="training",

                    batch_size=BATCH_SIZE,

                    seed=SEED,

                    zoom_range=0.2,

                    horizontal_flip=True,

                    vertical_flip=True,

                    class_mode="categorical",

                    preprocessing_function=preprocess_image,

                    target_size=(IMG_WIDTH,IMG_HEIGHT))



valid_generator=datagen.flow_from_dataframe(

                    dataframe=TRAIN_DF,

                    directory=TRAIN_DIR,

                    x_col=X_COL,

                    y_col=Y_COL,

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=SEED,

                    zoom_range=0.2,

                    horizontal_flip=True,

                    vertical_flip=True,

                    class_mode="categorical",

                    preprocessing_function=preprocess_image,    

                    target_size=(IMG_WIDTH,IMG_HEIGHT))



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model = create_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL), n_out=NUM_CLASSES)

        

for layer in model.layers:

    layer.trainable = False



for i in range(-10, 0):

    model.layers[i].trainable = True



metric_list = ["accuracy"]

optimizer = RAdam(lr=0.00005)

model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
history_warmup = model.fit_generator(generator=train_generator,

                          steps_per_epoch=STEP_SIZE_TRAIN,

                          validation_data=valid_generator,

                          validation_steps=STEP_SIZE_VALID,

                          epochs=WARMUP_EPOCHS,

                          verbose=1).history
for layer in model.layers:

    layer.trainable = True



optimizer = RAdam(lr=0.00005)

model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=metric_list)



gc.collect()



history_finetunning = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              epochs=OLD_DATA_EPOCHS,

                              callbacks=callback_list,

                              verbose=1).history
NEW_TRAIN_DIR       = '/kaggle/input/aptos2019-blindness-detection/train_images'

NEW_TRAIN_DF        = pd.read_csv(BASE_DIR + "train.csv",dtype='object')

NEW_TRAIN_DF[X_COL] = NEW_TRAIN_DF[X_COL].apply(append_file_ext_png)

NEW_TRAIN_DF.head()
datagen         = ImageDataGenerator(

                    rescale=1./255.,

                    validation_split=0.25)



train_generator = datagen.flow_from_dataframe(

                    dataframe=NEW_TRAIN_DF,

                    directory=NEW_TRAIN_DIR,

                    x_col=X_COL,

                    y_col=Y_COL,

                    subset="training",

                    batch_size=BATCH_SIZE,

                    seed=SEED,

                    zoom_range=0.2,

                    horizontal_flip=True,

                    vertical_flip=True,

                    class_mode="categorical",

                    preprocessing_function=preprocess_image,

                    target_size=(IMG_WIDTH,IMG_HEIGHT))



valid_generator=datagen.flow_from_dataframe(

                    dataframe=NEW_TRAIN_DF,

                    directory=NEW_TRAIN_DIR,

                    x_col=X_COL,

                    y_col=Y_COL,

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=SEED,

                    zoom_range=0.2,

                    horizontal_flip=True,

                    vertical_flip=True,

                    class_mode="categorical",

                    preprocessing_function=preprocess_image,    

                    target_size=(IMG_WIDTH,IMG_HEIGHT))



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# for layer in model.layers:

#     layer.trainable = True



# optimizer = RAdam(lr=0.00005)

# model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=metric_list)



gc.collect()



history_finetunning = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              epochs=NEW_DATA_EPOCHS,

                              callbacks=callback_list,

                              verbose=1).history
# history = {'loss': history_warmup['loss'] + history_finetunning['loss'], 

#            'val_loss': history_warmup['val_loss'] + history_finetunning['val_loss'], 

#            'acc': history_warmup['acc'] + history_finetunning['acc'], 

#            'val_acc': history_warmup['val_acc'] + history_finetunning['val_acc']}



# sns.set_style("whitegrid")

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



# ax1.plot(history['loss'], label='Train loss')

# ax1.plot(history['val_loss'], label='Validation loss')

# ax1.legend(loc='best')

# ax1.set_title('Loss')



# ax2.plot(history['acc'], label='Train Accuracy')

# ax2.plot(history['val_acc'], label='Validation accuracy')

# ax2.legend(loc='best')

# ax2.set_title('Accuracy')



# plt.xlabel('Epochs')

# sns.despine()

# plt.show()
optimizer = RAdam(lr=0.00005)

metric_list = ["accuracy"]

model = load_model('EfficientNetB5.h5', compile=False)

model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=metric_list)
test_datagen=ImageDataGenerator(rescale=1./255.)



test_generator=test_datagen.flow_from_dataframe(

                    dataframe=TEST_DF,

                    directory=TEST_DIR,

                    x_col=X_COL,

                    y_col=None,

                    batch_size=BATCH_SIZE,

                    seed=SEED,

                    shuffle=False,

                    class_mode=None,

                    preprocessing_function=preprocess_image,    

                    target_size=(IMG_WIDTH,IMG_HEIGHT))
if test_generator.n%BATCH_SIZE > 0:

    PREDICTION_STEPS = (test_generator.n//BATCH_SIZE) + 1

else:

    PREDICTION_STEPS = (test_generator.n//BATCH_SIZE)



print(PREDICTION_STEPS)
test_generator.reset()

preds = model.predict_generator(test_generator,

                                steps=PREDICTION_STEPS, 

                                verbose=1) 

predictions = [np.argmax(pred) for pred in preds]
filenames = test_generator.filenames

results = pd.DataFrame({'id_code':filenames, 'diagnosis':predictions})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])

results.astype({'diagnosis': 'int64'})

results.to_csv('submission.csv',index=False)

print(results.head(10))
t_finish = time.time()

total_time = round((t_finish-t_start) / 3600, 4)

print('Kernel runtime = {} hours ({} minutes)'.format(total_time, int(total_time*60)))