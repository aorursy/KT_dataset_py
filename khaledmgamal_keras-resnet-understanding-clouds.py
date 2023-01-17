# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow==1.13.1

#!pip install tensorflow-gpu==1.13.1
!pip install -U --pre segmentation-models --user
!pip install -U albumentations --user 

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import keras

import albumentations as A

import segmentation_models as sm

path="/kaggle/input/understanding_cloud_organization"

train = pd.read_csv(f'{path}/train.csv')

sub = pd.read_csv(f'{path}/sample_submission.csv')
train.head()
n_train = len(os.listdir(f'{path}/train_images'))

n_test = len(os.listdir(f'{path}/test_images'))

print(f'There are {n_train} images in train dataset')

print(f'There are {n_test} images in test dataset')
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])

train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])



sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])

sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])



image_id_list = train['im_id'].unique()


train.head()
def visualize(**images):

    """PLot images in one row."""

    n = len(images)

    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):

        plt.subplot(1, n, i + 1)

        plt.xticks([])

        plt.yticks([])

        plt.title(' '.join(name.split('_')).title())

        plt.imshow(image)

    plt.show()



def rle_decode(mask_rle: str = '', shape: tuple =(1400, 2100)):

    '''

    Decode rle encoded mask.

    

    :param mask_rle: run-length as string formatted (start length)

    :param shape: (height, width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape, order='F')



def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def bulid_mask(train_df,image_id,image_shape):

    masks=np.zeros((*image_shape,4))

    for i, (idx, row) in enumerate(train.loc[train['im_id'] == image_id].iterrows()):

        

        mask_rle = row['EncodedPixels']

        

        try: # label might not be there!

            mask = rle_decode(mask_rle)

        except:

            mask = np.zeros(image_shape)

        masks[:,:,i]=mask

   

    return masks 



def show_image_mask(image_id):

    image = Image.open(f"{path}/train_images/{image_id}")

    print("actual mask")

    mask=bulid_mask(train,image_id ,(1400, 2100))

    #mask=bulid_mask(train,image_id ,(350, 525))

    

    visualize(

       image=image, 

       Fish_mask=mask[..., 0].squeeze(),

       Flower_mask=mask[..., 1].squeeze(),

       Gravel_mask=mask[..., 2].squeeze(),

       Suger_mask=mask[..., 3].squeeze(),    

    )
fig = plt.figure(figsize=(25, 16))

for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), 4)):

    for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):

        ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])

        im = Image.open(f"{path}/train_images/{row['Image_Label'].split('_')[0]}")

        plt.imshow(im)

        mask_rle = row['EncodedPixels']

        try: # label might not be there!

            mask = rle_decode(mask_rle)

        except:

            mask = np.zeros((1400, 2100))

        plt.imshow(mask, alpha=0.5, cmap='gray')

        ax.set_title(f"Image: {row['Image_Label'].split('_')[0]}. Label: {row['label']}")

np.random.seed(0)

for i in np.random.randint(0,len(image_id_list),size=5):

    image_id=image_id_list[i]

    show_image_mask(image_id)



    

class Dataset:

    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    

    Args:

        images_dir (str): path to images folder

        masks_dir (str): path to segmentation masks folder

        class_values (list): values of classes to extract from segmentation mask

        augmentation (albumentations.Compose): data transfromation pipeline 

            (e.g. flip, scale, etc.)

        preprocessing (albumentations.Compose): data preprocessing 

            (e.g. noralization, shape manipulation, etc.)

    

    """

    

    CLASSES = ['Fish', 'Flower','Gravel','Suger']

    

    def __init__(

            self,

            tain_df,

            images_dir, 

            image_id_list, 

            classes=None, 

            augmentation=None, 

            preprocessing=None,

    ):

        self.ids = image_id_list

        self.images_fps = [f"{images_dir}/train_images/{image_id}" for image_id in self.ids]



        self.train_df= tain_df 

        self.augmentation = augmentation

        self.preprocessing = preprocessing

    

    def __getitem__(self, i):

        

        # read data

        image = cv2.imread(self.images_fps[i])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        

        # apply augmentations

        mask=bulid_mask(self.train_df,self.ids[i] ,(1400, 2100))

        if self.augmentation:

            sample = self.augmentation(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

        

        # apply preprocessing

        if self.preprocessing:

            sample = self.preprocessing(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

            

        return image, mask

        

    def __len__(self):

        return len(self.ids)

    

    

    

    

class Dataloder(keras.utils.Sequence):

    """Load data from dataset and form batches

    

    Args:

        dataset: instance of Dataset class for image loading and preprocessing.

        batch_size: Integet number of images in batch.

        shuffle: Boolean, if `True` shuffle image indexes each epoch.

    """

    

    def __init__(self, dataset, batch_size=1, shuffle=False):

        self.dataset = dataset

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indexes = np.arange(len(dataset))



        self.on_epoch_end()



    def __getitem__(self, i):

        

        # collect batch data

        start = i * self.batch_size

        stop = (i + 1) * self.batch_size

        data = []

        for j in range(start, stop):

            data.append(self.dataset[j])

        

        # transpose list of lists

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        

        return batch

    

    def __len__(self):

        """Denotes the number of batches per epoch"""

        return len(self.indexes) // self.batch_size

    

    def on_epoch_end(self):

        """Callback function to shuffle indexes each epoch"""

        if self.shuffle:

            self.indexes = np.random.permutation(self.indexes)







def round_clip_0_1(x, **kwargs):

    return x.round().clip(0, 1)



# define heavy augmentations

def get_training_augmentation():

    train_transform = [



        A.HorizontalFlip(p=0.5),



        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),



        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),

        A.RandomCrop(height=320, width=320, always_apply=True),



        A.IAAAdditiveGaussianNoise(p=0.2),

        A.IAAPerspective(p=0.5),



        A.OneOf(

            [

                A.CLAHE(p=1),

                A.RandomBrightness(p=1),

                A.RandomGamma(p=1),

            ],

            p=0.9,

        ),



        A.OneOf(

            [

                A.IAASharpen(p=1),

                A.Blur(blur_limit=3, p=1),

                A.MotionBlur(blur_limit=3, p=1),

            ],

            p=0.9,

        ),



        A.OneOf(

            [

                A.RandomContrast(p=1),

                A.HueSaturationValue(p=1),

            ],

            p=0.9,

        ),

        A.Lambda(mask=round_clip_0_1)

    ]

    return A.Compose(train_transform)





def get_validation_augmentation():

    """Add paddings to make image shape divisible by 32"""

    test_transform = [

        #A.PadIfNeeded(384, 480),

        #A.PadIfNeeded(320, 320),

        #A.PadIfNeeded(min_height=384, min_width=480, always_apply=True, border_mode=0)

        A.Resize(320, 320, interpolation=1, always_apply=False, p=1)

        

    ]

    

    

    return A.Compose(test_transform)



def get_preprocessing(preprocessing_fn):

    """Construct preprocessing transform

    

    Args:

        preprocessing_fn (callbale): data normalization function 

            (can be specific for each pretrained neural network)

    Return:

        transform: albumentations.Compose

    

    """

    

    _transform = [

        A.Lambda(image=preprocessing_fn),

    ]

    return A.Compose(_transform)    
'''

#BACKBONE = 'efficientnetb3'

BACKBONE='resnet18'

BATCH_SIZE = 8

CLASSES = ['Fish', 'Flower','Gravel','Suger']

LR = 0.001

EPOCHS = 10



preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES))  # case for binary and multiclass segmentation

activation = 'sigmoid' #if n_classes == 1 else 'softmax'



#create model

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)



# define optomizer

optim = keras.optimizers.Adam(LR)



# Segmentation models losses can be combined together by '+' and scaled by integer or float factor

# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1,1])) 

focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()

total_loss = dice_loss + (1 * focal_loss)



# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses

# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 



metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]



# compile keras model with defined optimozer, loss and metrics

model.compile(optim, total_loss, metrics)

'''
import keras.backend as K

from keras.legacy import interfaces

from keras.optimizers import Optimizer

from keras.losses import binary_crossentropy

import tensorflow as tf

class AdamAccumulate(Optimizer):



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):

        if accum_iters < 1:

            raise ValueError('accum_iters must be >= 1')

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lrr = K.variable(lr,name='lr')

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



        lr = self.lrr



        completed_updates = K.cast(tf.math.floordiv(self.iterations, self.accum_iters), K.floatx())



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

        config = {'lr': float(K.get_value(self.lrr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))



def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    

preprocess_input = sm.get_preprocessing('resnet18')    

EPOCHS = 15

BATCH_SIZE = 8

CLASSES = ['Fish', 'Flower','Gravel','Suger']

n_classes = len(CLASSES)

opt = AdamAccumulate(lr=0.002, accum_iters=8)



model = sm.Unet(

    'resnet18', 

    classes=4,

    input_shape=(320, 320, 3),

    activation='sigmoid'

)

model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice_coef])
from sklearn.model_selection import train_test_split

train_id_list, val_id_list = train_test_split(

    image_id_list, random_state=2019, test_size=0.1

)

train_dataset = Dataset(

    train,

    path, 

    train_id_list, 

    classes=['Fish', 'Flower','Gravel','Suger'],

    augmentation=get_validation_augmentation(),#get_training_augmentation(),

    preprocessing=get_preprocessing(preprocess_input),

)



# Dataset for validation images

valid_dataset = Dataset(

    train,

    path, 

    val_id_list, 

    classes=['Fish', 'Flower','Gravel','Suger'], 

    augmentation=get_validation_augmentation(),

    preprocessing=get_preprocessing(preprocess_input),

)



train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)



# check shapes for errors

assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)

assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)


'''

# define callbacks for learning rate scheduling and best checkpoints saving

callbacks = [

    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),

    #keras.callbacks.ReduceLROnPlateau(),

]

# train model

history = model.fit_generator(

    train_dataloader, 

    steps_per_epoch=len(train_dataloader), 

    epochs=EPOCHS, 

    callbacks=callbacks, 

    validation_data=valid_dataloader, 

    validation_steps=len(valid_dataloader),

)





# Plot training & validation iou_score values

#plt.figure(figsize=(30, 5))

#plt.subplot(121)

#plt.plot(history.history['iou_score'])

#plt.plot(history.history['val_iou_score'])

#plt.title('Model iou_score')

#plt.ylabel('iou_score')

#plt.xlabel('Epoch')

#plt.legend(['Train', 'Test'], loc='upper left')



# Plot training & validation loss values

#plt.subplot(122)

#plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

#plt.title('Model loss')

#plt.ylabel('Loss')

#plt.xlabel('Epoch')

#plt.legend(['Train', 'Test'], loc='upper left')

#plt.show()



history_df = pd.DataFrame(history.history)

history_df.to_csv('history.csv', index=False)



history_df[['loss', 'val_loss']].plot()

history_df[['dice_coef', 'val_dice_coef']].plot()

history_df[['lr']].plot()

'''
model.summary()
model.load_weights('/kaggle/input/resnet18-trained-model/best_model.h5')
def denormalize(x):

    """Scale image to range 0..1 for correct plot"""

    x_max = np.percentile(x, 98)

    x_min = np.percentile(x, 2)    

    x = (x - x_min) / (x_max - x_min)

    x = x.clip(0, 1)

    return x

def predict_mask(image_file,threshold):



    image = cv2.imread(image_file)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    width = 320#384 

    height = 320#480

    dim = (width, height)

    #if image.shape[0]==width and image.shape[0]==width:

        #augment=get_validation_augmentation()

        #image=augment(image=image) 

    if image.shape[0]!=width and image.shape[0]!=width:

        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        #augment=get_validation_augmentation()

        #image=augment(image=image) 



    preprocss=get_preprocessing(preprocess_input)

    image=preprocss(image=image)

    image = np.expand_dims(image['image'], axis=0)

    p_mask = model.predict(image)

    p_mask=(p_mask > threshold).astype(np.float_)

    image = cv2.resize(image.squeeze(axis=0),(2100,1400), interpolation = cv2.INTER_AREA)

    p_mask =cv2.resize(p_mask.squeeze(axis=0), (2100,1400), interpolation = cv2.INTER_AREA)

    #image = cv2.resize(image.squeeze(axis=0),(2100,1400), interpolation = cv2.INTER_AREA)

    #p_mask =cv2.resize(p_mask.squeeze(axis=0), (350,525), interpolation = cv2.INTER_AREA)

    

    visualize(

        image=denormalize(image),

        Fish_mask=p_mask[..., 0],

        Flower_mask=p_mask[..., 1],

        Gravel_mask=p_mask[..., 2],

        Suger_mask=p_mask[..., 3],  

    )

    

    return image,p_mask





def post_process(mask, min_size):

    """

    Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored

    """

    

    #mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    #predictions = np.zeros((1400,2100), np.float32)

    predictions = np.zeros((350,525), np.float32)

    

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num

np.random.seed(0)

for i in np.random.randint(0,len(image_id_list),size=5):

    image_id=image_id_list[i]

    show_image_mask(image_id)

    print("predicted mask")

    image,p_mask=predict_mask(f"{path}/train_images/{image_id}",0.45)

sub.head()

print("Number of test images ",len(sub['im_id'].unique()))
'''

test_path='/kaggle/input/understanding_cloud_organization/test_images/'

mini_number=15000

encoded_pixels = []

test_images_list=sub['im_id'].unique()

for i in range(len(test_images_list)):

    image,p_mask=predict_mask(f"{test_path}{test_images_list[i]}",0.45)

    #p_mask =cv2.resize(p_mask, (350, 525), interpolation = cv2.INTER_AREA)

    for m in range(p_mask.shape[-1]):

        pred_mask= p_mask[...,m].astype('float32') 

        print(i,len(test_images_list))

        num_predict=pred_mask.sum()            

        pred_mask, num_predict = post_process(pred_mask, mini_number)

            

        if num_predict == 0:

            encoded_pixels.append('')

        else:

            r = mask2rle(pred_mask)

            encoded_pixels.append(r)

sub['EncodedPixels'] = encoded_pixels

'''
#sub.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
#from IPython.display import FileLink

#FileLink(r'submission.csv')
#from IPython.display import FileLink

#FileLink(r'best_model.h5')
#os.listdir(path)