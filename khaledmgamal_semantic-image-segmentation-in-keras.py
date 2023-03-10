# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# Any results you write to the current directory are saved as output.
!pip install tensorflow==1.13.1
#!pip install tensorflow-gpu==1.13.1
!pip install -U --pre segmentation-models --user
!pip install -U albumentations --user 

#!pip install -U albumentations>=0.3.0 --user 



import tensorflow as tf

#print (tf.__version__)

tf.__version__

#tf.get_session()

!python3 -c 'import tensorflow as tf; print(tf.__version__)'



import os



import cv2

import keras

import numpy as np

import matplotlib.pyplot as plt

import albumentations as A

import segmentation_models as sm



DATA_DIR = '/kaggle/input/camvid-tiramisu/repository/alexgkendall-SegNet-Tutorial-bb68b64/CamVid'



x_train_dir = os.path.join(DATA_DIR, 'train')

y_train_dir = os.path.join(DATA_DIR, 'trainannot')



x_valid_dir = os.path.join(DATA_DIR, 'val')

y_valid_dir = os.path.join(DATA_DIR, 'valannot')



x_test_dir = os.path.join(DATA_DIR, 'test')

y_test_dir = os.path.join(DATA_DIR, 'testannot')





# helper function for data visualization

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

    

# helper function for data visualization    

def denormalize(x):

    """Scale image to range 0..1 for correct plot"""

    x_max = np.percentile(x, 98)

    x_min = np.percentile(x, 2)    

    x = (x - x_min) / (x_max - x_min)

    x = x.clip(0, 1)

    return x

    



# classes for data loading and preprocessing

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

    

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 

               'tree', 'signsymbol', 'fence', 'car', 

               'pedestrian', 'bicyclist', 'unlabelled']

    

    def __init__(

            self, 

            images_dir, 

            masks_dir, 

            classes=None, 

            augmentation=None, 

            preprocessing=None,

    ):

        self.ids = os.listdir(images_dir)

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        

        # convert str names to class values on masks

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        

        self.augmentation = augmentation

        self.preprocessing = preprocessing

    

    def __getitem__(self, i):

        

        # read data

        image = cv2.imread(self.images_fps[i])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)

        #print(mask.shape)

        #print(np.unique(mask))

        # extract certain classes from mask (e.g. cars)

        masks = [(mask == v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')

        #print(len(masks),masks[0].shape,np.unique(masks[0]))

        #print(mask.shape)

        #print(np.unique(mask))

        #print(np.unique(masks))        # add background if mask is not binary

        if mask.shape[-1] != 1:

            background = 1 - mask.sum(axis=-1, keepdims=True)

            mask = np.concatenate((mask, background), axis=-1)

        #print(mask.shape)

        #print(np.unique(mask))



        # apply augmentations

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

        A.PadIfNeeded(384, 480)

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





BACKBONE = 'efficientnetb3'

BATCH_SIZE = 8

CLASSES = ['car', 'pedestrian']

LR = 0.0001

EPOCHS = 40



preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation

activation = 'sigmoid' if n_classes == 1 else 'softmax'



#create model

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)



# define optomizer

optim = keras.optimizers.Adam(LR)



# Segmentation models losses can be combined together by '+' and scaled by integer or float factor

# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5])) 

focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()

total_loss = dice_loss + (1 * focal_loss)



# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses

# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 



metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]



# compile keras model with defined optimozer, loss and metrics

model.compile(optim, total_loss, metrics)

'''

# Lets look at data we have

# Dataset for train images

train_dataset = Dataset(

    x_train_dir, 

    y_train_dir, 

    classes=CLASSES, 

    augmentation=get_training_augmentation(),

    preprocessing=get_preprocessing(preprocess_input),

)



# Dataset for validation images

valid_dataset = Dataset(

    x_valid_dir, 

    y_valid_dir, 

    classes=CLASSES, 

    augmentation=get_validation_augmentation(),

    preprocessing=get_preprocessing(preprocess_input),

)



train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)



# check shapes for errors

assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)

assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)



# define callbacks for learning rate scheduling and best checkpoints saving

callbacks = [

    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),

    keras.callbacks.ReduceLROnPlateau(),

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

'''
'''

# Plot training & validation iou_score values

plt.figure(figsize=(30, 5))

plt.subplot(121)

plt.plot(history.history['iou_score'])

plt.plot(history.history['val_iou_score'])

plt.title('Model iou_score')

plt.ylabel('iou_score')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')



# Plot training & validation loss values

plt.subplot(122)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

'''


test_dataset = Dataset(

    x_test_dir, 

    y_test_dir, 

    classes=CLASSES, 

    augmentation=get_validation_augmentation(),

    preprocessing=get_preprocessing(preprocess_input),

)



test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

#model.load_weights('best_model.h5')



model.load_weights('/kaggle/input/trained-model-semantic-image-segmentation/best_model.h5')

scores = model.evaluate_generator(test_dataloader)



print("Loss: {:.5}".format(scores[0]))

for metric, value in zip(metrics, scores[1:]):

    print("mean {}: {:.5}".format(metric.__name__, value))



n = 10

ids = np.random.choice(np.arange(len(test_dataset)), size=n)



for i in ids:

    

    image, gt_mask = test_dataset[i]

    image = np.expand_dims(image, axis=0)

    pr_mask = model.predict(image)

    

    visualize(

        image=denormalize(image.squeeze()),

        gt_mask=gt_mask.squeeze(),

        pr_mask=pr_mask.squeeze(),

    )



def predict_mask(image_file):



    image = cv2.imread(image_file)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    width = 384 

    height = 480

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

    visualize(

        image=denormalize(image.squeeze(axis=0)),

        predicted_mask=p_mask.squeeze(axis=0),

        

    )

    return image,p_mask



image,p_mask=predict_mask('/kaggle/input/street-photo/The-world-before-your-feet-New-York-City-Photo-credit-garin-chadwick-1280692-unsplash.jpg')



image,p_mask=predict_mask('/kaggle/input/camvid-tiramisu/repository/alexgkendall-SegNet-Tutorial-bb68b64/CamVid/test/Seq05VD_f02910.png')

#from IPython.display import FileLink

#FileLink(r'best_model.h5')