import time

kernel_start = time.time()

LIMIT = 8.8



DO_TRAIN = True

DO_TEST = True

USE_TTA = True



RAND = 12345



!pip install tensorflow-gpu==1.14.0 --quiet

!pip install keras==2.2.4 --quiet

!pip install segmentation-models --quiet



import albumentations as albu

import cv2, gc, os

import keras

from keras import backend as K

from keras.models import Model

from keras.layers import Input

from keras import layers

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.losses import binary_crossentropy

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from skimage.exposure import adjust_gamma

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from keras.models import load_model

import segmentation_models as sm

from sklearn.metrics import roc_auc_score, accuracy_score



import keras.backend as K

from keras.legacy import interfaces

from keras.optimizers import Optimizer
sub = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')

sub['Image'] = sub['Image_Label'].map(lambda x: x.split('.')[0])

sub['Label'] = sub['Image_Label'].map(lambda x: x.split('_')[1])

# LOAD TEST CLASSIFIER PREDICTIONS

sub['p'] = pd.read_csv('../input/cloud-classifiers/pred_cls.csv').p.values

sub['p'] += np.load('../input/cloud-classifiers/pred_cls0.npy').reshape((-1)) * 0.5

sub['p'] += pd.read_csv('../input/cloud-classifiers/pred_cls3.csv').p.values * 3.0

sub['p'] += np.load('/kaggle/input/cloud-classifiers/pred_cls4b.npy') * 0.6

sub['p'] /= 5.1



train = pd.read_csv('../input/cloud-images-resized/train_384x576.csv')

train['Image'] = train['Image_Label'].map(lambda x: x.split('.')[0])

train['Label'] = train['Image_Label'].map(lambda x: x.split('_')[1])

train2 = pd.DataFrame({'Image':train['Image'][::4]})

train2['e1'] = train['EncodedPixels'][::4].values

train2['e2'] = train['EncodedPixels'][1::4].values

train2['e3'] = train['EncodedPixels'][2::4].values

train2['e4'] = train['EncodedPixels'][3::4].values

train2.set_index('Image',inplace=True,drop=True)

train2.fillna('',inplace=True); train2.head()

train2[['d1','d2','d3','d4']] = (train2[['e1','e2','e3','e4']]!='').astype('int8')

for k in range(1,5): train2['o'+str(k)] = 0

# LOAD TRAIN CLASSIFIER PREDICTIONS

train2[['o1','o2','o3','o4']] = np.load('../input/cloud-classifiers/oof_cls.npy')

train2[['o1','o2','o3','o4']] += np.load('../input/cloud-classifiers/oof_cls0.npy') * 0.5

train2[['o1','o2','o3','o4']] += np.load('../input/cloud-classifiers/oof_cls3.npy') * 3.0

train2[['o1','o2','o3','o4']] += np.load('../input/cloud-classifiers/oof_cls4b.npy') * 0.6

train2[['o1','o2','o3','o4']] /= 5.1

train2.head()
def mask2rleXXX(img0, shape=(576,384), grow=(525,350)):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    a = (shape[1]-img0.shape[0])//2

    b = (shape[0]-img0.shape[1])//2

    img = np.zeros((shape[1],shape[0]),dtype=np.uint8)

    img[a:a+img0.shape[0],b:b+img0.shape[1]] = img0

    img = cv2.resize(img,grow)

    

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2maskX(mask_rle, shape=(2100,1400), shrink=1):

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T[::shrink,::shrink]



def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
class DataGenerator2(keras.utils.Sequence):

    # USES GLOBAL VARIABLE TRAIN2 COLUMNS E1, E2, E3, E4

    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=24, shuffle=False, width=544, height=352, scale=1/128., sub=1., mode='train_seg',

                 path='../input/cloud-images-resized/train_images_384x576/', flips=False, augment=False, shrink1=1,

                 shrink2=1, dim=(576,384), clean=False):

        'Initialization'

        self.list_IDs = list_IDs

        self.shuffle = shuffle

        self.batch_size = batch_size

        self.path = path

        self.scale = scale

        self.sub = sub

        self.path = path

        self.width = width

        self.height = height

        self.mode = mode

        self.flips = flips

        self.augment = augment

        self.shrink1 = shrink1

        self.shrink2 = shrink2

        self.dim = dim

        self.clean = clean

        self.on_epoch_end()

        

    def __len__(self):

        'Denotes the number of batches per epoch'

        ct = int(np.floor( len(self.list_IDs) / self.batch_size))

        if len(self.list_IDs)>ct*self.batch_size: ct += 1

        return int(ct)



    def __getitem__(self, index):

        'Generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X, msk = self.__data_generation(indexes)

        if self.augment: X, msk = self.__augment_batch(X, msk)

        if (self.mode=='train_seg')|(self.mode=='validate_seg'): return X, msk

        else: return X



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(int( len(self.list_IDs) ))

        if self.shuffle: np.random.shuffle(self.indexes)



    def __data_generation(self, indexes):

        'Generates data containing batch_size samples' 

        # Initialization

        lnn = len(indexes); ex = self.shrink1; ax = self.shrink2

        X = np.empty((lnn,self.height,self.width,3),dtype=np.float32)

        msk = np.empty((lnn,self.height//ax,self.width//ax,4),dtype=np.int8)

        

        # Generate data

        for k in range(lnn):

            img = cv2.imread(self.path + self.list_IDs[indexes[k]]+'.jpg')

            #img = cv2.resize(img,(self.dim[0]//ex,self.dim[1]//ex),interpolation = cv2.INTER_AREA)

            img = img[::ex,::ex,:]

            # AUGMENTATION FLIPS

            hflip = False; vflip = False

            if (self.flips):

                if np.random.uniform(0,1)>0.5: hflip=True

                if np.random.uniform(0,1)>0.5: vflip=True

            if vflip: img = cv2.flip(img,0) # vertical

            if hflip: img = cv2.flip(img,1) # horizontal

            # AUGMENTATION SHAKE

            a = np.random.randint(0,self.dim[0]//ex//ax-self.width//ax+1)

            b = np.random.randint(0,self.dim[1]//ex//ax-self.height//ax+1)

            if (self.mode=='predict'):

                a = (self.dim[0]//ex//ax-self.width//ax)//2

                b = (self.dim[1]//ex//ax-self.height//ax)//2

            img = img[b*ax:self.height+b*ax,a*ax:self.width+a*ax]

            # NORMALIZE IMAGES

            X[k,] = img*self.scale - self.sub      

            # LABELS

            if (self.mode!='predict'):

                for j in range(1,5):

                    rle = train2.loc[self.list_IDs[indexes[k]],'e'+str(j)]

                    if self.clean:

                        if  train2.loc[self.list_IDs[indexes[k]],'o'+str(j)]<0.4:

                            rle = ''

                    mask = rle2maskX(rle,shrink=ex*ax,shape=self.dim)

                    if vflip: mask = np.flip(mask,axis=0)

                    if hflip: mask = np.flip(mask,axis=1)

                    msk[k,:,:,j-1] = mask[b:self.height//ax+b,a:self.width//ax+a]



        return X, msk

    

    def __random_transform(self, img, masks):

        composition = albu.Compose([

            #albu.HorizontalFlip(p=0.5),

            #albu.VerticalFlip(p=0.5),

            albu.ShiftScaleRotate(rotate_limit=30, scale_limit=0.1, p=0.5)

        ])

        

        composed = composition(image=img, mask=masks)

        aug_img = composed['image']

        aug_masks = composed['mask']

        

        return aug_img, aug_masks

    

    def __augment_batch(self, img_batch, masks_batch):

        for i in range(img_batch.shape[0]):

            img_batch[i, ], masks_batch[i, ] = self.__random_transform(

                img_batch[i, ], masks_batch[i, ])

        

        return img_batch, masks_batch
class AdamAccumulate(Optimizer):



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=8, **kwargs):

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
def kaggle_acc(y_true, y_pred0, pix=0.5, area=24000, dim=(352,544)):

 

    # PIXEL THRESHOLD

    y_pred = K.cast( K.greater(y_pred0,pix), K.floatx() )

    

    # MIN AREA THRESHOLD

    s = K.sum(y_pred, axis=(1,2))

    s = K.cast( K.greater(s, area), K.floatx() )



    # REMOVE MIN AREA

    s = K.reshape(s,(-1,1))

    s = K.repeat(s,dim[0]*dim[1])

    s = K.reshape(s,(-1,1))

    y_pred = K.permute_dimensions(y_pred,(0,3,1,2))

    y_pred = K.reshape(y_pred,shape=(-1,1))

    y_pred = s*y_pred

    y_pred = K.reshape(y_pred,(-1,y_pred0.shape[3],dim[0],dim[1]))

    y_pred = K.permute_dimensions(y_pred,(0,2,3,1))



    # COMPUTE KAGGLE ACC

    total_y_true = K.sum(y_true, axis=(1,2))

    total_y_true = K.cast( K.greater(total_y_true, 0), K.floatx() )



    total_y_pred = K.sum(y_pred, axis=(1,2))

    total_y_pred = K.cast( K.greater(total_y_pred, 0), K.floatx() )



    return 1 - K.mean( K.abs( total_y_pred - total_y_true ) )





def kaggle_dice(y_true, y_pred0, pix=0.5, area=24000, dim=(352,544)):

 

    # PIXEL THRESHOLD

    y_pred = K.cast( K.greater(y_pred0,pix), K.floatx() )

    

    # MIN AREA THRESHOLD

    s = K.sum(y_pred, axis=(1,2))

    s = K.cast( K.greater(s, area), K.floatx() )



    # REMOVE MIN AREA

    s = K.reshape(s,(-1,1))

    s = K.repeat(s,dim[0]*dim[1])

    s = K.reshape(s,(-1,1))

    y_pred = K.permute_dimensions(y_pred,(0,3,1,2))

    y_pred = K.reshape(y_pred,shape=(-1,1))

    y_pred = s*y_pred

    y_pred = K.reshape(y_pred,(-1,y_pred0.shape[3],dim[0],dim[1]))

    y_pred = K.permute_dimensions(y_pred,(0,2,3,1))



    # COMPUTE KAGGLE DICE

    intersection = K.sum(y_true * y_pred, axis=(1,2))

    total_y_true = K.sum(y_true, axis=(1,2))

    total_y_pred = K.sum(y_pred, axis=(1,2))

    return K.mean( (2*intersection+1e-9) / (total_y_true+total_y_pred+1e-9) )
filters = [256, 128, 64, 32, 16]

REDUCTION = 0; RED = 2**REDUCTION

filters = filters[:5-REDUCTION]



BATCH_SIZE = 16

jaccard_loss = sm.losses.JaccardLoss() 



skf = KFold(n_splits=3, shuffle=True, random_state=RAND)

for k, (idxT0, idxV0) in enumerate( skf.split(train2) ):



    train_idx = train2.index[idxT0]

    val_idx = train2.index[idxV0]



    if k==0: idx_oof_0 = val_idx.copy()

    elif k==1: idx_oof_1 = val_idx.copy()

    elif k==2: idx_oof_2 = val_idx.copy()



    print('#'*20)

    print('### Fold',k,'###')

    print('#'*20)



    if not DO_TRAIN: continue



    train_generator = DataGenerator2(

        train_idx, flips=True, augment=True, shuffle=True, shrink2=RED, batch_size=BATCH_SIZE,

    )



    val_generator = DataGenerator2(

        val_idx, shrink2=RED, batch_size=BATCH_SIZE

    )



    opt = AdamAccumulate(lr=0.001, accum_iters=8)

    model = sm.Unet(

        'efficientnetb2', 

        classes=4,

        encoder_weights='imagenet',

        decoder_filters = filters,

        input_shape=(None, None, 3),

        activation='sigmoid'

    )

    model.compile(optimizer=opt, loss=jaccard_loss, metrics=[dice_coef,kaggle_dice,kaggle_acc])



    checkpoint = ModelCheckpoint('model_'+str(k)+'.h5', save_best_only=True)

    es = EarlyStopping(monitor='val_dice_coef', min_delta=0.001, patience=5, verbose=1, mode='max')

    rlr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=2, verbose=1, mode='max', min_delta=0.001)

    

    history = model.fit_generator(

         train_generator,

         validation_data=val_generator,

         callbacks=[rlr, es, checkpoint],

         epochs=30,

         verbose=2, workers=2

    )

    history_df = pd.DataFrame(history.history)

    history_df.to_csv('history_'+str(k)+'.csv', index=False)



    del train_idx, val_idx, train_generator, val_generator, opt, model, checkpoint, es, rlr, history, history_df

    K.clear_session(); x=gc.collect()
!pip install tta-wrapper --quiet

from tta_wrapper import tta_segmentation



if DO_TEST:

    model1 = load_model('model_0.h5',custom_objects={'dice_coef':dice_coef,

            'jaccard_loss':jaccard_loss,'AdamAccumulate':AdamAccumulate,

            'kaggle_dice':kaggle_dice,'kaggle_acc':kaggle_acc})

    if USE_TTA:

        model1 = tta_segmentation(model1, h_flip=True, h_shift=(-10, 10), v_flip=True, v_shift=(-10, 10), merge='mean')

    model2 = load_model('model_1.h5',custom_objects={'dice_coef':dice_coef,

            'jaccard_loss':jaccard_loss,'AdamAccumulate':AdamAccumulate,

            'kaggle_dice':kaggle_dice,'kaggle_acc':kaggle_acc})

    if USE_TTA:

        model2 = tta_segmentation(model2, h_flip=True, h_shift=(-10, 10), v_flip=True, v_shift=(-10, 10), merge='mean')

    model3 = load_model('model_2.h5',custom_objects={'dice_coef':dice_coef,

            'jaccard_loss':jaccard_loss,'AdamAccumulate':AdamAccumulate,        

            'kaggle_dice':kaggle_dice,'kaggle_acc':kaggle_acc})

    if USE_TTA:

        model3 = tta_segmentation(model3, h_flip=True, h_shift=(-10, 10), v_flip=True, v_shift=(-10, 10), merge='mean')
print('Computing masks for',len(sub)//4,'test images with 3 models'); sub.EncodedPixels = ''

PTH = '../input/cloud-images-resized/test_images_384x576/'; bs = 4

if USE_TTA: bs=1

test_gen = DataGenerator2(sub.Image[::4].values, width=576, height=384, batch_size=bs, mode='predict',path=PTH)



sz = 20000.*(576/525)*(384/350)/RED/RED



pixt = [0.5,0.5,0.5,0.35] #; pixt = [0.4,0.4,0.4,0.4]

szt = [25000., 20000., 22500., 15000.] #; szt = [20000., 20000., 20000., 20000.]

for k in range(len(szt)): szt[k] = szt[k]*(576./525.)*(384./350.)/RED/RED



if DO_TEST:

    for b,batch in enumerate(test_gen):

        btc = model1.predict_on_batch(batch)

        btc += model2.predict_on_batch(batch)

        btc += model3.predict_on_batch(batch)

        btc /= 3.0



        for j in range(btc.shape[0]):

            for i in range(btc.shape[-1]):

                mask = (btc[j,:,:,i]>pixt[i]).astype(int); rle = ''

                if np.sum(mask)>szt[i]: rle = mask2rleXXX( mask ,shape=(576//RED,384//RED))

                sub.iloc[4*(bs*b+j)+i,1] = rle

        if b%(100//bs)==0: print(b*bs,', ',end='')

        t = np.round( (time.time() - kernel_start)/60,1 )

        if t > LIMIT*60:

            print('#### EXCEEDED TIME LIMIT. STOPPING NOW ####')

            break



    sub[['Image_Label','EncodedPixels']].to_csv('sub_seg.csv',index=False)

    sub.loc[(sub.p<0.5)&(sub.Label=='Fish'),'EncodedPixels'] = ''

    sub.loc[(sub.p<0.3)&(sub.Label=='Flower'),'EncodedPixels'] = ''

    sub.loc[(sub.p<0.5)&(sub.Label=='Gravel'),'EncodedPixels'] = ''

    sub.loc[(sub.p<0.5)&(sub.Label=='Sugar'),'EncodedPixels'] = ''

    sub[['Image_Label','EncodedPixels']].to_csv('submission.csv',index=False)



sub.head(10)