import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
import cv2

import matplotlib.pyplot as plt

from os.path import isfile

import torch.nn.init as init

import torch

import torch.nn as nn

from PIL import Image, ImageFilter

from sklearn.model_selection import train_test_split, StratifiedKFold

from torch.utils.data import Dataset

from torchvision import transforms

from torch.optim import Adam, SGD, RMSprop

import time

from torch.autograd import Variable

import torch.functional as F

from tqdm import tqdm

from sklearn import metrics

import urllib

import pickle

import cv2

import torch.nn.functional as F

from torchvision import models

import seaborn as sns

import random

import sys
package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'

sys.path.append(package_path)

from efficientnet_pytorch import EfficientNet

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything(1234)

TTA         = 5

num_classes = 1

IMG_SIZE    = 300

test = '../input/aptos2019-blindness-detection/test_images/'

def expand_path(p):

    p = str(p)

    if isfile(test + p + ".png"):

        return test + (p + ".png")

    return p



def p_show(imgs, label_name=None, per_row=3):

    n = len(imgs)

    rows = (n + per_row - 1)//per_row

    cols = min(per_row, n)

    fig, axes = plt.subplots(rows,cols, figsize=(15,15))

    for ax in axes.flatten(): ax.axis('off')

    for i,(p, ax) in enumerate(zip(imgs, axes.flatten())): 

        img = Image.open(expand_path(p))

        ax.imshow(img)

        ax.set_title(train_df[train_df.id_code == p].diagnosis.values)

def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img

class MyDataset(Dataset):

    

    def __init__(self, dataframe, transform=None):

        self.df = dataframe

        self.transform = transform

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        label = self.df.diagnosis.values[idx]

        label = np.expand_dims(label, -1)

        

        p = self.df.id_code.values[idx]

        p_path = expand_path(p)

        image = cv2.imread(p_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = crop_image_from_gray(image)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 30) ,-4 ,128)

        image = transforms.ToPILImage()(image)

        

        if self.transform:

            image = self.transform(image)

        

        return image, label

test_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation((-120, 120)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



testset        = MyDataset(pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv'), 

                 transform=test_transform)

test_loader    = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

model = EfficientNet.from_name('efficientnet-b0')

in_features = model._fc.in_features

model._fc = nn.Linear(in_features, num_classes)

model.load_state_dict(torch.load('../input/enet-test/weight_best(3).pt'))

model.cuda()

for param in model.parameters():

    param.requires_grad = False

sample = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

test_pred = np.zeros((len(sample), 1))

model.eval()



for _ in range(TTA):

    with torch.no_grad():

        for i, data in tqdm(enumerate(test_loader)):

            images, _ = data

            images = images.cuda()

            pred = model(images)

            test_pred[i * 16:(i + 1) * 16] += pred.detach().cpu().squeeze().numpy().reshape(-1, 1)



output = test_pred / TTA

preds1 = output.copy()

coef = [0.57, 1.37, 2.57, 3.57]

for i, pred in enumerate(output):

    if pred < coef[0]:

        output[i] = 0

    elif pred >= coef[0] and pred < coef[1]:

        output[i] = 1

    elif pred >= coef[1] and pred < coef[2]:

        output[i] = 2

    elif pred >= coef[2] and pred < coef[3]:

        output[i] = 3

    else:

        output[i] = 4

submission1 = pd.DataFrame({'id_code':pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv').id_code.values,

                          'diagnosis':np.squeeze(output).astype(int)})
package_dir = "../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/"

sys.path.insert(0, package_dir)

import torchvision

import torch.nn as nn

from tqdm import tqdm

from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch

from torchvision import transforms

import pretrainedmodels



device = torch.device("cuda:0")

ImageFile.LOAD_TRUNCATED_IMAGES = True
class RetinopathyDatasetTest(Dataset):

    def __init__(self, csv_file, transform):

        self.data = pd.read_csv(csv_file)

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')

        image = Image.open(img_name)

        image = self.transform(image)

        return {'image': image}

model = pretrainedmodels.__dict__['resnet101'](pretrained=None)



model.avg_pool = nn.AdaptiveAvgPool2d(1)

model.last_linear = nn.Sequential(

                          nn.BatchNorm1d(2048, eps=1e-3, momentum=0.9, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.2),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.ReLU(),

                          nn.BatchNorm1d(2048, eps=1e-3, momentum=0.9, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.2),

                          nn.Linear(in_features=2048, out_features=1, bias=True),

                         )

model.load_state_dict(torch.load("../input/mmmodel/model.bin"))

model = model.to(device)

for param in model.parameters():

    param.requires_grad = False



model.eval()



test_transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    transforms.RandomErasing()

])

test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/sample_submission.csv',

                                      transform=test_transform)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds1 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds1[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds2 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds2[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds3 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds3[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds4 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds4[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds5 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds5[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds6 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds6[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds7 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds7[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds8 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds8[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds9 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds9[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds10 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model(x_batch.to(device))

    test_preds10[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

test_preds = (test_preds1 + test_preds2 + test_preds3 + test_preds4 + test_preds5

             + test_preds6 + test_preds7 + test_preds8 + test_preds9 + test_preds10) / 10.0

coef = [0.5, 1.5, 2.5, 3.5]



preds2 = test_preds.copy()

for i, pred in enumerate(test_preds):

    if pred < coef[0]:

        test_preds[i] = 0

    elif pred >= coef[0] and pred < coef[1]:

        test_preds[i] = 1

    elif pred >= coef[1] and pred < coef[2]:

        test_preds[i] = 2

    elif pred >= coef[2] and pred < coef[3]:

        test_preds[i] = 3

    else:

        test_preds[i] = 4

submission2 = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

submission2.diagnosis = test_preds.astype(int)

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate)

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model

import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy, categorical_crossentropy

#from keras.applications.resnet50 import preprocess_input

from keras.applications.densenet import DenseNet121,DenseNet169

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import imgaug as ia



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 300

NUM_CLASSES = 5
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

x = df_train['id_code']

y = df_train['diagnosis']



x, y = shuffle(x, y, random_state=42)

y = to_categorical(y, num_classes=NUM_CLASSES)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4,

                                                      stratify=y, random_state=42)

sometimes = lambda aug: iaa.Sometimes(0.8, aug)

seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.5), # horizontally flip 50% of all images

            iaa.Flipud(0.2), # vertically flip 20% of all images

            sometimes(iaa.CropAndPad(

            percent=(-0.05, 0.1),

            pad_mode=ia.ALL,

            pad_cval=(0, 255)

             )),

            sometimes(iaa.Affine(

                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)

                rotate=(-45, 45), # rotate by -45 to +45 degrees

                shear=(-16, 16), # shear by -16 to +16 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 5),

                [

                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                    iaa.OneOf([

                        iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0

                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7

                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7

                    ]),

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.5)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    # search either for all edges or for directed edges,

                    # blend the result with the original image using a blobby mask

                    iaa.SimplexNoiseAlpha(iaa.OneOf([

                        iaa.EdgeDetect(alpha=(0.5, 1.0)),

                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                    ])),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images

                    iaa.OneOf([

                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels

                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),

                    ]),

                    iaa.Invert(0.05, per_channel=True), # invert color channels

                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation

                    # either change the brightness of the whole image (sometimes

                    # per channel) or change the brightness of subareas

                    iaa.OneOf([

                        iaa.Multiply((0.5, 1.5), per_channel=0.5),

                        iaa.FrequencyNoiseAlpha(

                            exponent=(-4, 0),

                            first=iaa.Multiply((0.5, 1.5), per_channel=True),

                            second=iaa.ContrastNormalization((0.5, 2.0))

                        )

                    ]),

                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast

                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around

                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

                ],

                random_order=True

            )

        ],

        random_order=True)

class My_Generator(Sequence):



    def __init__(self, image_filenames, labels,

                 batch_size, is_train=True,

                 mix=False, augment=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.is_augment = augment

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

        else:

            pass

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')

            img = cv2.resize(img, (SIZE, SIZE))

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        if(self.is_mix):

            batch_images, batch_y = self.mix_up(batch_images, batch_y)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y

def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = DenseNet121(include_top=False,

                   weights=None,

                   input_tensor=input_tensor)

    base_model.load_weights("../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.2)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.2)(x)

    final_output = Dense(n_out, activation='softmax', name='final_output')(x)

    model = Model(input_tensor, final_output) 

    return model

# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 20; batch_size = 32

checkpoint = ModelCheckpoint('../working/densenet_.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)



train_generator = My_Generator(train_x, train_y, 128, is_train=True)

train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)

valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)



model = create_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=NUM_CLASSES)

# reference link: https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow

def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-12, N=5, bsize=32, name='kappa'):

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

    

        return nom*0.5 / (denom + eps) + categorical_crossentropy(y_true, y_pred)*0.5

from keras.callbacks import Callback

class QWKEvaluation(Callback):

    def __init__(self, validation_data=(), batch_size=32, interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.batch_size = batch_size

        self.valid_generator, self.y_val = validation_data

        self.history = []



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict_generator(generator=self.valid_generator,

                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),

                                                  workers=1, use_multiprocessing=False,

                                                  verbose=1)

            def flatten(y):

                return np.argmax(y, axis=1).reshape(-1)

            

            score = cohen_kappa_score(flatten(self.y_val),

                                      flatten(y_pred),

                                      labels=[0,1,2,3,4],

                                      weights='quadratic')

            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))

            self.history.append(score)

            if score >= max(self.history):

                print('saving checkpoint: ', score)

                self.model.save('../working/densenet_bestqwk.h5')



qwk = QWKEvaluation(validation_data=(valid_generator, valid_y),

                    batch_size=batch_size, interval=1)

# warm up model

for layer in model.layers:

    layer.trainable = False



for i in range(-3,0):

    model.layers[i].trainable = True



model.compile(

    loss='categorical_crossentropy',

    optimizer=Adam(1e-5))



model.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),

    epochs=2,

    workers=WORKERS, use_multiprocessing=True,

    verbose=1,

    callbacks=[qwk])

for layer in model.layers:

    layer.trainable = True

callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early, qwk]

model.compile(loss='categorical_crossentropy',

            # loss=kappa_loss,

            optimizer=Adam(lr=1e-5), metrics=['categorical_accuracy'])

model.fit_generator(

    train_mixup,

    steps_per_epoch=np.ceil(float(len(train_x)) / float(batch_size)),

    validation_data=valid_generator,

    validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),

    epochs=epochs,

    verbose=1,

    workers=1, use_multiprocessing=False,

    callbacks=callbacks_list)

submission3 = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

model.load_weights('../working/densenet_bestqwk.h5')

predicted = []

# reference:https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb 

for i, name in tqdm(enumerate(submission3['id_code'])):

    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    X = np.array((image[np.newaxis])/255)

    score_predict=((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

    label_predict = np.argmax(score_predict)

    predicted.append(label_predict)

submission3['diagnosis'] = predicted

import torch

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



from sklearn.model_selection import StratifiedKFold

from joblib import load, dump

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *

from fastai.callbacks import *

from torchvision import models as md

from torch import nn

from torch.nn import functional as F

import re

import math

import collections

from functools import partial

from torch.utils import model_zoo

from sklearn import metrics

from collections import Counter

import json
GlobalParams = collections.namedtuple('GlobalParams', [

    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',

    'num_classes', 'width_coefficient', 'depth_coefficient',

    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])





# Parameters for an individual model block

BlockArgs = collections.namedtuple('BlockArgs', [

    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',

    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])





# Change namedtuple defaults

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)





def relu_fn(x):

    """ Swish activation function """

    return x * torch.sigmoid(x)





def round_filters(filters, global_params):

    """ Calculate and round number of filters based on depth multiplier. """

    multiplier = global_params.width_coefficient

    if not multiplier:

        return filters

    divisor = global_params.depth_divisor

    min_depth = global_params.min_depth

    filters *= multiplier

    min_depth = min_depth or divisor

    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)

    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%

        new_filters += divisor

    return int(new_filters)





def round_repeats(repeats, global_params):

    """ Round number of filters based on depth multiplier. """

    multiplier = global_params.depth_coefficient

    if not multiplier:

        return repeats

    return int(math.ceil(multiplier * repeats))





def drop_connect(inputs, p, training):

    """ Drop connect. """

    if not training: return inputs

    batch_size = inputs.shape[0]

    keep_prob = 1 - p

    random_tensor = keep_prob

    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)

    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor

    return output





def get_same_padding_conv2d(image_size=None):

    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.

        Static padding is necessary for ONNX exporting of models. """

    if image_size is None:

        return Conv2dDynamicSamePadding

    else:

        return partial(Conv2dStaticSamePadding, image_size=image_size)



class Conv2dDynamicSamePadding(nn.Conv2d):

    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):

        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2



    def forward(self, x):

        ih, iw = x.size()[-2:]

        kh, kw = self.weight.size()[-2:]

        sh, sw = self.stride

        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)

        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:

            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)





class Conv2dStaticSamePadding(nn.Conv2d):

    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2



        # Calculate padding based on image size and save it

        assert image_size is not None

        ih, iw = image_size if type(image_size) == list else [image_size, image_size]

        kh, kw = self.weight.size()[-2:]

        sh, sw = self.stride

        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)

        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)

        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:

            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        else:

            self.static_padding = Identity()



    def forward(self, x):

        x = self.static_padding(x)

        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return x





class Identity(nn.Module):

    def __init__(self,):

        super(Identity, self).__init__()



    def forward(self, input):

        return input





########################################################################

############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############

########################################################################





def efficientnet_params(model_name):

    """ Map EfficientNet model name to parameter coefficients. """

    params_dict = {

        # Coefficients:   width,depth,res,dropout

        'efficientnet-b0': (1.0, 1.0, 224, 0.2),

        'efficientnet-b1': (1.0, 1.1, 240, 0.2),

        'efficientnet-b2': (1.1, 1.2, 260, 0.2),

        'efficientnet-b3': (1.2, 1.4, 300, 0.2),

        'efficientnet-b4': (1.4, 1.8, 380, 0.2),

        'efficientnet-b5': (1.6, 2.2, 456, 0.2),

        'efficientnet-b6': (1.8, 2.6, 528, 0.2),

        'efficientnet-b7': (2.0, 3.1, 600, 0.2),

    }

    return params_dict[model_name]





class BlockDecoder(object):

    """ Block Decoder for readability, straight from the official TensorFlow repository """



    @staticmethod

    def _decode_block_string(block_string):

        """ Gets a block through a string notation of arguments. """

        assert isinstance(block_string, str)



        ops = block_string.split('_')

        options = {}

        for op in ops:

            splits = re.split(r'(\d.*)', op)

            if len(splits) >= 2:

                key, value = splits[:2]

                options[key] = value



        # Check stride

        assert (('s' in options and len(options['s']) == 1) or

                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))



        return BlockArgs(

            kernel_size=int(options['k']),

            num_repeat=int(options['r']),

            input_filters=int(options['i']),

            output_filters=int(options['o']),

            expand_ratio=int(options['e']),

            id_skip=('noskip' not in block_string),

            se_ratio=float(options['se']) if 'se' in options else None,

            stride=[int(options['s'][0])])



    @staticmethod

    def _encode_block_string(block):

        """Encodes a block to a string."""

        args = [

            'r%d' % block.num_repeat,

            'k%d' % block.kernel_size,

            's%d%d' % (block.strides[0], block.strides[1]),

            'e%s' % block.expand_ratio,

            'i%d' % block.input_filters,

            'o%d' % block.output_filters

        ]

        if 0 < block.se_ratio <= 1:

            args.append('se%s' % block.se_ratio)

        if block.id_skip is False:

            args.append('noskip')

        return '_'.join(args)



    @staticmethod

    def decode(string_list):

        """

        Decodes a list of string notations to specify blocks inside the network.



        :param string_list: a list of strings, each string is a notation of block

        :return: a list of BlockArgs namedtuples of block args

        """

        assert isinstance(string_list, list)

        blocks_args = []

        for block_string in string_list:

            blocks_args.append(BlockDecoder._decode_block_string(block_string))

        return blocks_args



    @staticmethod

    def encode(blocks_args):

        """

        Encodes a list of BlockArgs to a list of strings.



        :param blocks_args: a list of BlockArgs namedtuples of block args

        :return: a list of strings, each string is a notation of block

        """

        block_strings = []

        for block in blocks_args:

            block_strings.append(BlockDecoder._encode_block_string(block))

        return block_strings





def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,

                 drop_connect_rate=0.2, image_size=None, num_classes=1000):

    """ Creates a efficientnet model. """



    blocks_args = [

        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',

        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',

        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',

        'r1_k3_s11_e6_i192_o320_se0.25',

    ]

    blocks_args = BlockDecoder.decode(blocks_args)



    global_params = GlobalParams(

        batch_norm_momentum=0.99,

        batch_norm_epsilon=1e-4,

        dropout_rate=dropout_rate,

        drop_connect_rate=drop_connect_rate,

        # data_format='channels_last',  # removed, this is always true in PyTorch

        num_classes=num_classes,

        width_coefficient=width_coefficient,

        depth_coefficient=depth_coefficient,

        depth_divisor=8,

        min_depth=None,

        image_size=image_size,

    )



    return blocks_args, global_params





def get_model_params(model_name, override_params):

    """ Get the block args and global params for a given model """

    if model_name.startswith('efficientnet'):

        w, d, s, p = efficientnet_params(model_name)

        # note: all models have drop connect rate = 0.2

        blocks_args, global_params = efficientnet(

            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)

    else:

        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:

        # ValueError will be raised here if override_params has fields not included in global_params.

        global_params = global_params._replace(**override_params)

    return blocks_args, global_params





url_map = {

    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',

    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',

    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',

    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',

    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth',

    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth',

}



def load_pretrained_weights(model, model_name, load_fc=True):

    """ Loads pretrained weights, and downloads if loading for the first time. """

    state_dict = model_zoo.load_url(url_map[model_name])

    if load_fc:

        model.load_state_dict(state_dict)

    else:

        state_dict.pop('_fc.weight')

        state_dict.pop('_fc.bias')

        res = model.load_state_dict(state_dict, strict=False)

        assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'

    print('Loaded pretrained weights for {}'.format(model_name))

    

    

class MBConvBlock(nn.Module):

    """

    Mobile Inverted Residual Bottleneck Block



    Args:

        block_args (namedtuple): BlockArgs, see above

        global_params (namedtuple): GlobalParam, see above



    Attributes:

        has_se (bool): Whether the block contains a Squeeze and Excitation layer.

    """



    def __init__(self, block_args, global_params):

        super().__init__()

        self._block_args = block_args

        self._bn_mom = 1 - global_params.batch_norm_momentum

        self._bn_eps = global_params.batch_norm_epsilon

        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)

        self.id_skip = block_args.id_skip  # skip connection and drop connect



        # Get static or dynamic convolution depending on image size

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)



        # Expansion phase

        inp = self._block_args.input_filters  # number of input channels

        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels

        if self._block_args.expand_ratio != 1:

            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)

            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)



        # Depthwise convolution phase

        k = self._block_args.kernel_size

        s = self._block_args.stride

        self._depthwise_conv = Conv2d(

            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise

            kernel_size=k, stride=s, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)



        # Squeeze and Excitation layer, if desired

        if self.has_se:

            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))

            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)

            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)



        # Output phase

        final_oup = self._block_args.output_filters

        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)

        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)



    def forward(self, inputs, drop_connect_rate=None):

        """

        :param inputs: input tensor

        :param drop_connect_rate: drop connect rate (float, between 0 and 1)

        :return: output of block

        """



        # Expansion and Depthwise Convolution

        x = inputs

        if self._block_args.expand_ratio != 1:

            x = relu_fn(self._bn0(self._expand_conv(inputs)))

        x = relu_fn(self._bn1(self._depthwise_conv(x)))



        # Squeeze and Excitation

        if self.has_se:

            x_squeezed = F.adaptive_avg_pool2d(x, 1)

            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))

            x = torch.sigmoid(x_squeezed) * x



        x = self._bn2(self._project_conv(x))



        # Skip connection and drop connect

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters

        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:

            if drop_connect_rate:

                x = drop_connect(x, p=drop_connect_rate, training=self.training)

            x = x + inputs  # skip connection

        return x





class EfficientNet(nn.Module):

    """

    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods



    Args:

        blocks_args (list): A list of BlockArgs to construct blocks

        global_params (namedtuple): A set of GlobalParams shared between blocks



    Example:

        model = EfficientNet.from_pretrained('efficientnet-b0')



    """



    def __init__(self, blocks_args=None, global_params=None):

        super().__init__()

        assert isinstance(blocks_args, list), 'blocks_args should be a list'

        assert len(blocks_args) > 0, 'block args must be greater than 0'

        self._global_params = global_params

        self._blocks_args = blocks_args



        # Get static or dynamic convolution depending on image size

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)



        # Batch norm parameters

        bn_mom = 1 - self._global_params.batch_norm_momentum

        bn_eps = self._global_params.batch_norm_epsilon



        # Stem

        in_channels = 3  # rgb

        out_channels = round_filters(32, self._global_params)  # number of output channels

        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)



        # Build blocks

        self._blocks = nn.ModuleList([])

        for block_args in self._blocks_args:



            # Update block input and output filters based on depth multiplier.

            block_args = block_args._replace(

                input_filters=round_filters(block_args.input_filters, self._global_params),

                output_filters=round_filters(block_args.output_filters, self._global_params),

                num_repeat=round_repeats(block_args.num_repeat, self._global_params)

            )



            # The first block needs to take care of stride and filter size increase.

            self._blocks.append(MBConvBlock(block_args, self._global_params))

            if block_args.num_repeat > 1:

                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            for _ in range(block_args.num_repeat - 1):

                self._blocks.append(MBConvBlock(block_args, self._global_params))



        # Head

        in_channels = block_args.output_filters  # output of final block

        out_channels = round_filters(1280, self._global_params)

        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)



        # Final linear layer

        self._dropout = self._global_params.dropout_rate

        self._fc = nn.Linear(out_channels, self._global_params.num_classes)



    def extract_features(self, inputs):

        """ Returns output of the final convolution layer """



        # Stem

        x = relu_fn(self._bn0(self._conv_stem(inputs)))



        # Blocks

        for idx, block in enumerate(self._blocks):

            drop_connect_rate = self._global_params.drop_connect_rate

            if drop_connect_rate:

                drop_connect_rate *= float(idx) / len(self._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)



        # Head

        x = relu_fn(self._bn1(self._conv_head(x)))



        return x



    def forward(self, inputs):

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """



        # Convolution layers

        x = self.extract_features(inputs)



        # Pooling and final linear layer

        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        if self._dropout:

            x = F.dropout(x, p=self._dropout, training=self.training)

        x = self._fc(x)

        return x



    @classmethod

    def from_name(cls, model_name, override_params=None):

        cls._check_model_name_is_valid(model_name)

        blocks_args, global_params = get_model_params(model_name, override_params)

        return EfficientNet(blocks_args, global_params)



    @classmethod

    def from_pretrained(cls, model_name, num_classes=1000):

        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})

        return model



    @classmethod

    def get_image_size(cls, model_name):

        cls._check_model_name_is_valid(model_name)

        _, _, res, _ = efficientnet_params(model_name)

        return res



    @classmethod

    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):

        """ Validates model name. None that pretrained weights are only available for

        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """

        num_models = 4 if also_need_pretrained_weights else 8

        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]

        if model_name.replace('-','_') not in valid_models:

            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

#making model

md_ef = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)

#copying weighst to the local directory 

!mkdir models

!cp '../input/kaggle-public-copy/abcdef.pth' 'models'

def get_df():

    base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

    train_dir = os.path.join(base_image_dir,'train_images/')

    df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

    df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

    df = df.drop(columns=['id_code'])

    df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

    test_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

    return df, test_df



df, test_df = get_df()

#you can play around with tfms and image sizes

bs = 64

sz = 224

tfms = get_transforms(do_flip=True,flip_vert=True)

data = (ImageList.from_df(df=df,path='./',cols='path') 

        .split_by_rand_pct(0.2) 

        .label_from_df(cols='diagnosis',label_cls=FloatList) 

        .transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 

        .databunch(bs=bs,num_workers=4) 

        .normalize(imagenet_stats)  

       )

def qk(y_pred, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_pred), y, weights='quadratic'), device='cuda:0')

learn = Learner(data, 

                md_ef, 

                metrics = [qk], 

                model_dir="models").to_fp16()



learn.data.add_test(ImageList.from_df(test_df,

                                      '../input/aptos2019-blindness-detection',

                                      folder='test_images',

                                      suffix='.png'))

learn.load('abcdef');

#https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa

class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

        print(-loss_partial(self.coef_['x']))



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']



coefficients=[0.5, 1.5, 2.5, 3.5]

opt = OptimizedRounder()

preds,y = learn.get_preds(DatasetType.Test)

preds4 = np.copy(preds)

tst_pred = opt.predict(preds, coefficients)

test_df.diagnosis = tst_pred.astype(int)

submission4 = test_df.copy()

print ('done')
score = [0.777, 0.758, 0.783]

weight = [0.35, 0.2, 0.45]

predsData = weight[0]*preds1 + weight[1]*preds2 + weight[2]*preds4
coef = [0.5, 1.5, 2.5, 3.5]

for i, pred in enumerate(predsData):

    if pred < coef[0]:

        predsData[i] = 0

    elif pred >= coef[0] and pred < coef[1]:

        predsData[i] = 1

    elif pred >= coef[1] and pred < coef[2]:

        predsData[i] = 2

    elif pred >= coef[2] and pred < coef[3]:

        predsData[i] = 3

    else:

        predsData[i] = 4

submission = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

submission.diagnosis = predsData.astype(int)

submission.to_csv('submission.csv', index=False)