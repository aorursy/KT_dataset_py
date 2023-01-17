import os

import math

import cv2 as cv2

from PIL import Image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import tensorflow as tf



#import keras

from keras import models

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D



from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import LeakyReLU

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.utils import Sequence, to_categorical

#from keras import utils as np_utils



from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard, Callback

from keras.optimizers import Adam, Optimizer

import keras.backend as K



from sklearn.metrics import cohen_kappa_score, accuracy_score

from sklearn.utils import class_weight



import imgaug as ia

import imgaug.augmenters as iaa







from sklearn.model_selection import KFold, StratifiedKFold



import random



%matplotlib inline



def seed_everything(seed=999):

    

    print('Random seeds initialized')

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.set_random_seed(seed)

    ia.seed(seed)



seed_everything()





def normalize(x):

    return (x-dataset_mean)/dataset_std



def denormalize(x):

    return x*dataset_std+dataset_mean
TRAIN_PATH = './input/prep456_cropOnly_origAspect/'

VAL_PATH = './input/prep456_cropOnly_origAspect/'

WIDTH = 380

HEIGHT = WIDTH

target_size=(WIDTH, HEIGHT)


## Displaying some Sample Images

def display_samples_from_df(df, columns=4, rows=3, path=TRAIN_PATH):

    fig=plt.figure(figsize=(5*columns, 4*rows))



    idx=1

    

    for i in range(columns*rows):

        image_path = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        

        img = Image.open(path+image_path)

        height, width = img.size

        fig.add_subplot(rows, columns, idx)

        plt.title(image_id)

        plt.imshow(img)

        idx+=1

    

    plt.tight_layout()

    

def display_samples_from_list(x, y, columns=4, rows=3):

    fig=plt.figure(figsize=(5*columns, 4*rows))



    idx=1

    

    for i in range(columns*rows):

        image_path = x[i]

        image_id = y[i]



        img = Image.fromarray(image_path.astype("uint8"))

        fig.add_subplot(rows, columns, idx)

        plt.title(image_id)

        plt.imshow(img)

        idx+=1

    

    plt.tight_layout() 



#Separate validation data

def getValSet(validation_method, train_df, foldIdx=0, fold=3):

    

    if (validation_method == 'same'):

        print("Calculate Validation set with same distribution")

        target=0.2

        row_list_train = []

        row_list_val = []

        for index, row in train_df.iterrows():



            if random.random()<target: 

                row_list_val.append({"id_code": row["id_code"], "diagnosis":row["diagnosis"]})

            else:

                row_list_train.append({"id_code": row["id_code"], "diagnosis":row["diagnosis"]})



        train_df = pd.DataFrame(row_list_train)

        val_df = pd.DataFrame(row_list_val)



    if (validation_method == 'kFold'):

        print("Get kFold Validation")

        

        source = train_df.copy()

        

        kf = StratifiedKFold(n_splits = fold, shuffle = False, random_state = 2019)

        split = kf.split(source['id_code'], source['diagnosis'])

        for _ in range(foldIdx+1): 

            result=next(split) 



        val_df =  pd.DataFrame()

        train_df = pd.DataFrame()

        val_df = pd.concat([val_df, source.iloc[result[1]]])

        train_df = pd.concat([train_df, source.iloc[result[0]]])

        val_df.reset_index(inplace=True)

        train_df.reset_index(inplace=True)

        

    if (validation_method == 'new'):

        print("Get Validation set fron the Train.CSV")

        val_df = pd.read_csv('./input/train.csv')

        val_df['id_code'] = val_df['id_code']+".png"

        val_df['diagnosis'] = val_df['diagnosis'].astype("int")



    train_df['diagnosis'] = train_df['diagnosis'].astype('str')

    val_df['diagnosis'] = val_df['diagnosis'].astype('str')



    print(train_df['diagnosis'].value_counts())

    print(val_df['diagnosis'].value_counts())

    return (train_df, val_df)



#train_df = pd.read_csv('./input/oldtrain.csv')

#train, val = getValSet('kFold', train_df, foldIdx=2)
def color(images, random_state, parents, hooks): # without cutting out a circle

    

    blank=np.zeros( images[0].shape, np.uint8 )

    cropfactor=0.95

    

    for i, img in enumerate(images):

        

        img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0), WIDTH/60) ,-4 ,128)

        cv2.circle(blank , ( img.shape[1]//2 , img.shape[0]//2) , int( img.shape[1]//2*cropfactor ) , ( 1 , 1 , 1 ) , -1 , 8 , 0 )

        images[i]=img*blank+0*(1-blank)



    return images



def colorCrop(images, random_state, parents, hooks): # cutout of a circle

    

    blank=np.zeros( images[0].shape, np.uint8 )

    cropfactor=0.95

    

    for i, img in enumerate(images):

        

        img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0), WIDTH/60) ,-4 ,128)      

        cv2.circle(blank , ( img.shape[1]//2 , img.shape[0]//2) , int( img.shape[1]//2*cropfactor ) , ( 1 , 1 , 1 ) , -1 , 8 , 0 )

        images[i]=img*blank+0*(1-blank)



    return images



augment_orig_val = iaa.Sequential()







augment_strong = iaa.Sequential(

    [

        iaa.Fliplr(0.5), # horizontally flip 50% of all images

        iaa.Flipud(0.5), # vertically flip 20% of all images



        iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1  ))),

        

        iaa.Sometimes(0.5, iaa.Affine(

            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},

            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},

            rotate=(-180, 180),

        )),



        iaa.SomeOf((0, 3),

            [

                iaa.Sometimes(0.1, 

                    iaa.Superpixels(

                        p_replace=(0, 0.25),

                        n_segments=(20, 200)

                    )

                ),

                iaa.OneOf([

                    iaa.GaussianBlur((0, 3.0)),

                    iaa.AverageBlur(k=(2, 7)),

                    iaa.MedianBlur(k=(3, 5)),

                ]),

                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                iaa.Emboss(alpha=(0, 1.0), strength=(1, 2.0)),

                iaa.Sometimes(0.2, iaa.OneOf([

                    iaa.EdgeDetect(alpha=(0, 0.7)),

                    iaa.DirectedEdgeDetect(

                        alpha=(0, 0.7), direction=(0.0, 1.0)

                    ),

                ])),

                iaa.AdditiveGaussianNoise(

                    scale=(0.1, 0.03*255), per_channel=0.5

                ),

                iaa.OneOf([

                    iaa.Dropout((0.01, 0.15), per_channel=0.5),

                    iaa.CoarseDropout(

                        (0.03, 0.15), size_percent=(0.03, 0.05)),

                ]),

                iaa.Add((-10, 10)),

                iaa.Multiply((0.8, 1.5)),

                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                iaa.Grayscale(alpha=(0.0, 1.0)),

                iaa.Sometimes(0.1, 

                    iaa.ElasticTransformation(alpha=(0.0, 2), sigma=0.25)

                ),



                iaa.Sometimes(0.1, iaa.PiecewiseAffine(scale=(0.01, 0.05)))

            ],

            # do all of the above augmentations in random order

            random_order=True

        )

    ]) 





#augment = augment_origcolor_v3

#x, y = validation_generator.__getitem__(0)

#img = denormalize(x)

#img = img.astype('uint8')

#img = augment(images=img)

#display_samples_from_list(img, y)
def decoder(pred):

    if pred < 0.5: return 0

    elif pred < 1.5: return 1

    elif pred < 2.5: return 2

    elif pred < 3.5: return 3

    else: return 4



def decode(predictions):

    return np.array([decoder(pred) for pred in predictions])



def tightCrop(img):    #cuts out the middle of a circle

    

    r=img.shape[1]//2

    a=int(math.sqrt(2)*r*1.05)



    img = img[img.shape[0]//2-a//2:img.shape[0]//2+a//2, r-a//2:r+a//2, :]    

    img = cv2.resize(img, (WIDTH, HEIGHT))

    return img



def tightCrop43(img):    #cuts out a rectangle from the middle of a circle

    

    r=img.shape[1]//2

    x=int(math.sqrt(4/25*r**2)*1.05)

    sidea=4*x

    sideb=3*x

    

    startx = (img.shape[1]-sidea)//2

    starty = (img.shape[0]-sideb)//2



    img = img[starty:starty+sideb, startx:startx+sidea, :]    

    return img



def middleCrop(img): # crops out the middle of a rectangle

    

    h, w, _ = img.shape

    startx = int(w/2-h/2)

    return img[:, startx:startx+h, :]



def load_image(p_path, crop=False):

    

    try:

        image = cv2.imread(p_path) #read

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #color transform

    except:

        print(f"Can't read: {p_path}")

        return None

        

    if crop==True: image = crop_image_from_gray(image) #crop black border

        

    imgType = getImageType(image)

    if (imgType == 0): image = tightCrop43(image)

    if (imgType == 1): image = tightCrop43(image)

    

    image = cv2.resize(image, (WIDTH, HEIGHT)) #resize

    return image



# load X,Y from a dataframe

def load_df(df, max_load=1000000000, path=TRAIN_PATH, crop=False):



    x = np.zeros((min(max_load, df.shape[0]), WIDTH, HEIGHT, 3), dtype='uint8')

    y = np.zeros((min(max_load, df.shape[0])), dtype='uint8')

    

    for idx,row in df.iterrows():

        x[idx, :, :, :] = load_image(row['path'], crop)

        y[idx] = row['diagnosis']

        

        if idx>=max_load-1: break

    

    print(f'Return shape: {x.shape}')

    return x,y



# crops images

def crop_image_from_gray(img,tol=7):

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = gray_img>tol

        

    check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

    if (check_shape == 0): # image is too dark so that we crop out everything,

        return img # return original image

    else:

        img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

        img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

        img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

        img = np.stack([img1,img2,img3],axis=-1)

    return img

    

def getImageType(img, limit=20):

    

    #print(np.max(img[ img.shape[0]//4, 0,  : ]))

    #print(np.max(img[ 0, img.shape[0]//4,  : ]))

    #print(np.max(img[ img.shape[1]//2,  image.shape[0]//2,  : ]))

    

    if (img.shape[1]-img.shape[0])<img.shape[0]/50: return 0 # circle

    if np.max(img[ img.shape[0]//3, 0,  : ])<limit: return 1 # sides touch sides

    return 2



#image = './input/preptest456_cropOnly_origAspect/0e9d6c70eaf7.png' #circle

#image = './input/prep456_cropOnly_origAspect/0097f532ac9f.png'

#image = './input/preptest456_cropOnly_origAspect/0eb6e41a8658.png'

#image = cv2.imread(image)

#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



#print("Image type: ",getImageType(image))

#image = tightCrop43(image)

#image = Image.fromarray(image)



#plt.figure(figsize = (8,8))

#plt.imshow(image) 
class DataSequence(Sequence):



    def __init__(self, df, batch_size, x_col, y_col, augment=True, shuffle=True, augmenter=None, label_smoothing=0.1):

        self.df = df

        self.batch_size = batch_size

        self.x_col = x_col

        self.y_col = y_col

        self.augment = augment

        self.shuffle = shuffle

        self.n = df['diagnosis'].count()

        self.num_classes = len(np.unique(df[y_col]))

        self.myAugmenter = augmenter

        self.label_smoothing = label_smoothing



    def __len__(self):

        return int(math.ceil(len(self.df) / float(self.batch_size)))



    def on_epoch_end(self):

        # shuffle when needed

        if self.shuffle: self.df = self.df.sample(frac=1).reset_index(drop=True)

        

    def get_batch_labels(self, idx):

        # Fetch a batch of labels

        y = np.array(self.df[self.y_col][idx * self.batch_size: (idx + 1) * self.batch_size])

        y = to_categorical(y, 5)

        

        y = y * (1-self.label_smoothing) + self.label_smoothing/5

        

        return y



    def get_batch_features(self, idx):

        # Fetch a batch of inputs

        images = np.array([load_image(im) for im in self.df[self.x_col][idx * self.batch_size: (1 + idx) * self.batch_size]])



        

        if self.augment==True: trans_images = normalize(np.array(self.myAugmenter(images=images.astype('uint8'))))

        else: trans_images=normalize(np.array(images))

        

        return trans_images



    def __getitem__(self, idx):

        batch_x = self.get_batch_features(idx)

        batch_y = self.get_batch_labels(idx)

        return batch_x, batch_y



def fitData(df, path = TRAIN_PATH): 

    fit_datagen =  ImageDataGenerator(

        featurewise_center=True,

        featurewise_std_normalization=True)

        

    x_fit, y_fit = load_df(df, 300, path)

    fit_datagen.fit(x_fit)

        

    global dataset_mean

    global dataset_std

    dataset_mean = fit_datagen.mean

    dataset_std = fit_datagen.std

            

    print(fit_datagen.std)

    print(fit_datagen.mean)          



def getGenerators(train_df, val_df, batch_size, augmenter, label_smoothing=0.1):



    global dataset_mean

    global dataset_std



    fitData(train_df, TRAIN_PATH)

 

    train_generator = DataSequence(

            train_df, 

            x_col="path",

            y_col="diagnosis",

            batch_size=batch_size,

            augment=True,

            shuffle=True,

            augmenter=augmenter[0],

            label_smoothing=label_smoothing

            )



    validation_generator = DataSequence(

            val_df, 

            x_col="path",

            y_col="diagnosis",

            batch_size=batch_size,

            augment=True,

            shuffle=False,

            augmenter=augmenter[1],

            label_smoothing=label_smoothing

    )



    return train_generator, validation_generator
train_df = pd.read_csv('./input/oldtrain.csv') # 2015 training dataset

train_df['id_code'] = train_df['id_code']+".png"

train_df['diagnosis'] = train_df['diagnosis'].astype("int")



train_df, val_df = getValSet("new", train_df) 

train_generator, validation_generator = getGenerators(train_df, val_df, TRAIN_PATH, VAL_PATH, 32, (augment_origcolor, augment_orig_val))



print('Done')
import efficientnet.keras as efn 



def build_model():

    

    seed_everything()

    base = efn.EfficientNetB5(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, 3), pooling='avg')

    base.trainable=True



    #dropout_dense_layer = 0.2 # for B0

    #dropout_dense_layer = 0.3 # for B3

    dropout_dense_layer = 0.4  # for B5    

     

    model = Sequential()

    model.add(base)

    model.add(Dropout(dropout_dense_layer))

    model.add(Dense(5, activation='softmax'))

    

    model.compile(

        loss='categorical_crossentropy',

        optimizer=Adam(lr=0.001),

        metrics=['accuracy']

    )

    model.summary()

    return model



model = build_model()
x, y = train_generator.__getitem__(1)

x, y = validation_generator.__getitem__(0)



display_samples_from_list(denormalize(x), y)
model_name = "model_efficientb5_cat_base.h5"

checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

reduce_lr_on_plateau = ReduceLROnPlateau(monitor = 'loss',patience = 5, factor=0.3, verbose=1)
model.layers[0].trainable = False

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'] )

model.summary()
history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=train_generator.n/train_generator.batch_size,

                              validation_data=validation_generator,

                              validation_steps=validation_generator.n/validation_generator.batch_size,

                              epochs=5,

                              workers=6,

                              callbacks=[checkpoint, reduce_lr_on_plateau])





history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
model.layers[0].trainable = True

limit = 300

for i in range(limit): model.layers[0].layers[i].trainable = False

for i in range(limit, 570): model.layers[0].layers[i].trainable = True

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'] )

model.summary()
train_generator, validation_generator = getGenerators(train_df, val_df, TRAIN_PATH, VAL_PATH, 16, (augment_origcolor, augment_orig_val))

history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=train_generator.n/train_generator.batch_size,

                              validation_data=validation_generator,

                              validation_steps=validation_generator.n/validation_generator.batch_size,

                              epochs=15,

                              workers=6,

                              callbacks=[checkpoint, reduce_lr_on_plateau])





history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
model.save_weights(model_name)
from keras.callbacks import Callback

import keras.backend as K

import numpy as np



class SGDRScheduler(Callback):

    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage

        ```python

            schedule = SGDRScheduler(min_lr=1e-5,

                                     max_lr=1e-2,

                                     steps_per_epoch=np.ceil(epoch_size/batch_size),

                                     lr_decay=0.9,

                                     cycle_length=5,

                                     mult_factor=1.5)

            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])

        ```

    # Arguments

        min_lr: The lower bound of the learning rate range for the experiment.

        max_lr: The upper bound of the learning rate range for the experiment.

        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 

        lr_decay: Reduce the max_lr after the completion of each cycle.

                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.

        cycle_length: Initial number of epochs in a cycle.

        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References

        Blog post: jeremyjordan.me/nn-learning-rate

        Original paper: http://arxiv.org/abs/1608.03983

    '''

    def __init__(self,

                 min_lr,

                 max_lr,

                 steps_per_epoch,

                 lr_decay=1,

                 cycle_length=10,

                 mult_factor=1.1):



        self.min_lr = min_lr

        self.max_lr = max_lr

        self.lr_decay = lr_decay



        self.batch_since_restart = 0

        self.next_restart = cycle_length



        self.steps_per_epoch = steps_per_epoch



        self.cycle_length = cycle_length

        self.mult_factor = mult_factor





    def clr(self):

        '''Calculate the learning rate.'''

        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)

        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))

        #print(lr)

        return lr



    def on_train_begin(self, logs={}):

        '''Initialize the learning rate to the minimum value at the start of training.'''

        K.set_value(self.model.optimizer.lr, self.max_lr)



    def on_batch_end(self, batch, logs={}):

        '''Record previous batch statistics and update the learning rate.'''



        self.batch_since_restart += 1

        K.set_value(self.model.optimizer.lr, self.clr())



    def on_epoch_end(self, epoch, logs={}):

        '''Check for end of current cycle, apply restarts when necessary.'''

        if epoch + 1 == self.next_restart:

            self.batch_since_restart = 0

            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)

            self.next_restart += self.cycle_length

            self.max_lr *= self.lr_decay

def train(load_model='model_efficientB5_cat_base.h5', model_name = "model_efficientB5_cat_ph2.h5", epochs=50):

    

    schedule = SGDRScheduler(min_lr=1e-5,

                                     max_lr=3e-2,

                                     steps_per_epoch=np.ceil(train_generator.n/train_generator.batch_size),

                                     cycle_length=5)    

    

    #model = build_model()

    seed_everything()

    

    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    reduce_lr_on_plateau = ReduceLROnPlateau(monitor = 'loss',patience = 4, factor=0.5, verbose=1)



    if load_model != None: 

        print("Loading model: ", load_model)

        model.load_weights(load_model)

        

    history = model.fit_generator(generator=train_generator,

                                  steps_per_epoch=train_generator.n/train_generator.batch_size,

                                  validation_data=validation_generator,

                                  validation_steps=validation_generator.n/validation_generator.batch_size,

                                  epochs=epochs,

                                  verbose=1,

                                  workers=6,

                                  class_weight=class_weights,

                                  callbacks=[checkpoint, reduce_lr_on_plateau])





    history_df = pd.DataFrame(history.history)

    #history_df[['loss', 'val_loss']].plot()

    #history_df[['acc', 'val_acc']].plot()

    print('Minimum validation loss: ', history_df['val_loss'].min())

    return history_df['val_loss'].min()
origtrain_df = pd.read_csv('./input/train.csv')

origtrain_df['path'] = TRAIN_PATH+origtrain_df.id_code+'.png'

#origtrain_df.id_code += '.png'

Kfold = 3

TEST_PATH = './input/preptest456_cropOnly_origAspect/'

pseudo = pd.read_csv('./input/linear-stacking-blended-with-best-lb-as-reg-v2-submission.csv')

pseudo['path'] = TEST_PATH+pseudo.id_code+'.png'

pseudo.head()

newtrain = pd.concat([origtrain_df, pseudo], sort=True)

#newtrain.id_code += '.png'

newtrain.head()
val_df = pd.read_csv('./input/cleanvalidation.csv')

val_df['path'] = TRAIN_PATH+val_df.id_code+'.png'

val_df.count()
train_df = pd.merge(newtrain, val_df, on='id_code', indicator=True, how='outer').query('_merge=="left_only"')

train_df.drop(columns=['diagnosis_y', 'path_y', '_merge'], inplace=True)

train_df.rename(columns={'diagnosis_x':'diagnosis', 'path_x':'path'}, inplace=True)

train_df.head()
#train_df, val_df = getValSet( 'kFold',newtrain, foldIdx=0, fold=5)

train_generator, validation_generator = getGenerators(

        train_df, 

        val_df, 

        batch_size=16, 

        augmenter=(augment_strong, augment_orig_val)

)

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(train_df.diagnosis),

                                                 train_df.diagnosis)

class_weights
x, y = train_generator.__getitem__(0)

#x, y = validation_generator.__getitem__(2)



display_samples_from_list(denormalize(x), y)
import tensorflow as tf



def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=train_generator.batch_size, name='kappa'):

    """A continuous differentiable approximation of discrete kappa loss.

        Args:

            y_pred: 2D tensor or array, [batch_size, num_classes]

            y_true: 2D tensor or array,[batch_size, num_classes]

            y_pow: int,  e.g. y_pow=2

            N: typically num_classes of the model

            bsize: batch_size of the training or validation ops

            eps: a float, prevents divide by zero

            name: Optional scope/name for op_scope.

        Returns:

            A tensor with the kappa loss."""

    

    with tf.name_scope(name):

        y_true = tf.to_float(y_true)

        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))

        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))

        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

    

        pred_ = y_pred ** y_pow

        try:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))

        except Exception:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

    

        hist_rater_a = tf.reduce_sum(pred_norm, 0)

        hist_rater_b = tf.reduce_sum(y_true, 0)

    

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

    

        nom = tf.reduce_sum(weights * conf_mat)

        denom = tf.reduce_sum(weights * tf.matmul(

            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /

                              tf.to_float(bsize))

    

        return nom / (denom + eps)
model.layers[0].trainable = True

limit = 300

for i in range(limit): model.layers[0].layers[i].trainable = False

for i in range(limit, 570): model.layers[0].layers[i].trainable = True

model.compile(loss=kappa_loss, optimizer=Adam(0.001), metrics=['accuracy'] )

model.summary()
train(load_model='model_efficientB5_cat_base.h5', model_name = "model_efficientB5_cat_ph2.h5", epochs=40) 
x_val, y_val = load_df(val_df)

x_val = normalize(x_val)



x_eval = x_val

y_eval = y_val

display_samples_from_list(denormalize(x_eval), y_val)
y_true = np.array(y_eval).astype('int')



model.load_weights('./model_efficientB5_cat_ph2.h5')

predictions = model.predict(x_eval)

predictions = np.array(predictions).squeeze()



y_pred = np.argmax(predictions, axis=1)



print(y_pred[:30])

print(y_true[:30])



print("Validation accuracy: ", np.mean(np.equal(y_true, y_pred)))

print("Cohen Kappa score: %.3f" % cohen_kappa_score(y_pred, y_true, weights='quadratic'))
from sklearn.metrics import classification_report, confusion_matrix



print('Confusion Matrix')

print(confusion_matrix(y_true, y_pred))

print('Classification Report')

target_names = ['0', '1', '2', '3', '4']

print(classification_report(y_true, y_pred, target_names=target_names))

eval_augment = iaa.Sequential(

    [



        iaa.Fliplr(0.3), # horizontally flip 50% of all images

        iaa.Sometimes(0.1, iaa.Crop(percent=(0, 0.1))),



        iaa.Sometimes(0.5, iaa.Affine(

            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},

            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},

            rotate=(-5, 5))),



        iaa.SomeOf((0, 2),

            [

                iaa.Sharpen(alpha=(0, 0.5), lightness=(0.85, 1.15)),

                iaa.Emboss(alpha=(0, 0.5), strength=(0.8, 1.0)),

                iaa.Sometimes(0.3, iaa.EdgeDetect(alpha=(0, 0.3))),

                iaa.AdditiveGaussianNoise(),

                iaa.Multiply((0.9, 1.3)),

                iaa.ContrastNormalization((0.7, 1.2)),



            ], random_order=True

        )

    ]) 
predictions = []

seed_everything()

tta_steps = 3

predictions = np.zeros((tta_steps, y_eval.shape[0]))

for tta in range(tta_steps):

    preds = np.zeros((x_eval.shape[0]))

       

    x_trans = eval_augment(images=denormalize(x_eval).astype('uint8'))

    preds = model.predict(normalize(x_trans))

    predictions[tta, :] = np.argmax(preds, axis=1)

    print(f'TTA round {tta+1} done')

  

    

predictions = np.mean(predictions, axis=0)

y_pred = (predictions).astype("int")

y_true = np.array(y_eval).astype('int')



print(y_pred[:30])

print(y_true[:30])



print("Validation accuracy with TTT: ", np.mean(np.equal(y_true, y_pred)))

print("Cohen Kappa score: ", cohen_kappa_score(y_pred, y_true, weights='quadratic'))
submit_df = pd.read_csv('./input/sample_submission.csv')

TEST_PATH = './input/test_images/'

submit_df['path'] = TEST_PATH+submit_df.id_code+'.png'

show_df = submit_df.copy()

show_df['id_code'] += '.png'



x_test, y_test = load_df(show_df, path=TEST_PATH, crop=True)

display_samples_from_list(x_test, y_test)
#show_df.sample(frac=1).reset_index(drop=True)

fitData(train_df, path = TRAIN_PATH)

#dataset_mean = [152.86581, 78.63777, 20.671019]

#dataset_std = [43.88956  25.214554 18.860243]

seed_everything()



tta_steps = 1

predictions = []

predictions = np.zeros((tta_steps, x_test.shape[0]))

for tta in range(tta_steps):

    preds = np.zeros((x_test.shape[0]))

       

    if tta_steps>1: x_trans = eval_augment(images=x_test.astype('uint8'))

    else: x_trans = x_test.astype('uint8')

    

    preds = model.predict(normalize(x_trans))

    predictions[tta, :] = np.argmax(preds, axis=1)

    print(f'TTA round {tta+1} done')



predictions = np.mean(predictions, axis=0)

y_pred = predictions.astype('int')

#y_pred = predictions #raw
submit_df['diagnosis'] = y_pred.squeeze()

submit_df = submit_df.drop(columns=['path'])

submit_df.to_csv('submission.csv',index=False)



submit_df.hist()
submit_df.diagnosis.value_counts()
ref = pd.read_csv('./submission829.csv')

#ref = pd.read_csv('./submission_classifier_0798.csv')



ref.hist()
ref.diagnosis.value_counts()