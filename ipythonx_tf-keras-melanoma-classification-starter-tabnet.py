!pip install pytorch-tabnet

!pip install tensorflow-addons==0.9.1    

!pip install git+https://github.com/aleju/imgaug.git

!pip install -U git+https://github.com/qubvel/efficientnet

!pip install -U git+https://github.com/albumentations-team/albumentations

!pip install git+https://github.com/qubvel/classification_models.git
# basic imports

import torch

from pytorch_tabnet.tab_model import TabNetClassifier



import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm

from glob import glob

from pylab import rcParams

import matplotlib.pyplot as plt

from scipy.stats import rankdata

import os, gc, cv2, random, warnings, math, sys

from collections import Counter, defaultdict



# sklearn

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV, train_test_split



# tf 

import tensorflow as tf

import tensorflow_addons as tfa

import efficientnet.tfkeras as efn 

import tensorflow.keras.layers as L

from tensorflow.keras.models import Model

from tensorflow.keras import backend as K

from tensorflow.keras.utils import Sequence

from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                                        EarlyStopping, ReduceLROnPlateau, CSVLogger)



# augmentation libs [albumentation, img_aug]

import albumentations

import imgaug.augmenters as iaa

from PIL import Image, ImageOps, ImageEnhance

from albumentations.augmentations import functional as F

from albumentations.core.transforms_interface import ImageOnlyTransform



warnings.simplefilter('ignore')

sys.path.insert(0, "/kaggle/input/keras-tta-wrapper")
def display(df, path):

    fig = plt.figure(figsize=(20, 16))

    for class_id in [0, 1]:

        for i, (idx, row) in enumerate(df.loc[df['target'] == class_id].sample(4, random_state=101).iterrows()):

            ax = fig.add_subplot(4, 4, class_id * 4 + i + 1, xticks=[], yticks=[])



            image = cv2.imread(os.path.join(path + '{}.jpg'.format(row['image_name'])))

            ax.set_title('Label: {}'.format(class_id) )

            plt.imshow(image)

    

    

# helper function to plot sample from dataloader/generator 

def plot_imgs(dataset_show, is_train=True):

    rcParams['figure.figsize'] = 30,20

    for i in range(2):

        f, ax = plt.subplots(1,5)

        for p in range(5):

            idx = np.random.randint(0, len(dataset_show))

            if is_train:

                img, label = dataset_show[idx]

            else:

                img = dataset_show[idx]

            ax[p].grid(False)

            ax[p].imshow(img[0])

            ax[p].set_title(idx)

    plt.show()
batch_size = 48

dim = 256, 256



root = '../input/melanoma-merged-external-data-512x512-jpeg/'

df = pd.read_csv(os.path.join(root, 'marking.csv'))



train_images = os.path.join(root, '512x512-dataset-melanoma/512x512-dataset-melanoma/')

test_images  = os.path.join(root, '512x512-test/512x512-test/')
# for reproducibiity

def seed_all(s):

    random.seed(s)

    np.random.seed(s)

    tf.random.set_seed(s)

    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    os.environ['PYTHONHASHSEED'] = str(s) 

    

# seed all

seed_all(101)
print(df.info())

df.head()
# each patient has more than one samples

df.rename({"image_id": "image_name"},axis='columns',inplace =True) 

print('Unique Image Id  : ', len(df.image_name.unique()))

print('Unique Patient Id: ',len(df.patient_id.unique()))
# significant imbalance

sns.set_style('darkgrid')

sns.countplot(df.target)

plt.show()
display(df, train_images)
def stratified_group_k_fold(X, y, groups, k, seed=None):

    """ https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation """

    labels_num = np.max(y) + 1

    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))

    y_distr = Counter()

    for label, g in zip(y, groups):

        y_counts_per_group[g][label] += 1

        y_distr[label] += 1



    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))

    groups_per_fold = defaultdict(set)



    def eval_y_counts_per_fold(y_counts, fold):

        y_counts_per_fold[fold] += y_counts

        std_per_label = []

        for label in range(labels_num):

            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])

            std_per_label.append(label_std)

        y_counts_per_fold[fold] -= y_counts

        return np.mean(std_per_label)

    

    groups_and_y_counts = list(y_counts_per_group.items())

    random.Random(seed).shuffle(groups_and_y_counts)



    for g, y_counts in tqdm(sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])), total=len(groups_and_y_counts)):

        best_fold = None

        min_eval = None

        for i in range(k):

            fold_eval = eval_y_counts_per_fold(y_counts, i)

            if min_eval is None or fold_eval < min_eval:

                min_eval = fold_eval

                best_fold = i

        y_counts_per_fold[best_fold] += y_counts

        groups_per_fold[best_fold].add(g)



    all_groups = set(groups)

    for i in range(k):

        train_groups = all_groups - groups_per_fold[i]

        test_groups = groups_per_fold[i]



        train_indices = [i for i, g in enumerate(groups) if g in train_groups]

        test_indices = [i for i, g in enumerate(groups) if g in test_groups]



        yield train_indices, test_indices
df['patient_id'] = df['patient_id'].fillna(df['image_name'])

df['sex'] = df['sex'].fillna('unknown')

df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('unknown')

df['age_approx'] = df['age_approx'].fillna(round(df['age_approx'].mean()))



patient_id_2_count = df[['patient_id', 

                         'image_name']].groupby('patient_id').count()['image_name'].to_dict()

df = df.set_index('image_name')



def get_stratify_group(row):

    stratify_group = row['sex']

    stratify_group += f'_{row["source"]}'

    stratify_group += f'_{row["target"]}'

    patient_id_count = patient_id_2_count[row["patient_id"]]

    if patient_id_count > 80:   stratify_group += f'_80'

    elif patient_id_count > 60: stratify_group += f'_60'

    elif patient_id_count > 50: stratify_group += f'_50'

    elif patient_id_count > 30: stratify_group += f'_30'

    elif patient_id_count > 20: stratify_group += f'_20'

    elif patient_id_count > 10: stratify_group += f'_10'

    else: stratify_group += f'_0'

    return stratify_group



df['stratify_group'] = df.apply(get_stratify_group, axis=1)

df['stratify_group'] = df['stratify_group'].astype('category').cat.codes

df.loc[:, 'fold'] = 0
%%time

skf = stratified_group_k_fold(X=df.index, 

                              y=df['stratify_group'], 

                              groups=df['patient_id'], 

                              k=3, seed=101)



for fold_number, (train_index, val_index) in enumerate(skf):

    df.loc[df.iloc[val_index].index, 'fold'] = fold_number
df.reset_index(inplace=True)

df.to_csv('innat_df.csv', index=False)

df.head()
def int_parameter(level, maxval):

    """Helper function to scale `val` between 0 and maxval .

    Args:

    level: Level of the operation that will be between [0, `PARAMETER_MAX`].

    maxval: Maximum value that the operation can have. This will be scaled to

      level/PARAMETER_MAX.

    Returns:

    An int that results from scaling `maxval` according to `level`.

    """

    return int(level * maxval / 10)



def float_parameter(level, maxval):

    """Helper function to scale `val` between 0 and maxval.

    Args:

    level: Level of the operation that will be between [0, `PARAMETER_MAX`].

    maxval: Maximum value that the operation can have. This will be scaled to

      level/PARAMETER_MAX.

    Returns:

    A float that results from scaling `maxval` according to `level`.

    """

    return float(level) * maxval / 10.





def sample_level(n):

    return np.random.uniform(low=0.1, high=n)





def autocontrast(pil_img, _):

    return ImageOps.autocontrast(pil_img)





def equalize(pil_img, _):

    return ImageOps.equalize(pil_img)





def posterize(pil_img, level):

    level = int_parameter(sample_level(level), 4)

    return ImageOps.posterize(pil_img, 4 - level)



def rotate(pil_img, level):

    degrees = int_parameter(sample_level(level), 30)

    if np.random.uniform() > 0.5:

        degrees = -degrees

    return pil_img.rotate(degrees, resample=Image.BILINEAR)



def solarize(pil_img, level):

    level = int_parameter(sample_level(level), 256)

    return ImageOps.solarize(pil_img, 256 - level)



def shear_x(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, level, 0, 0, 1, 0),

                           resample=Image.BILINEAR)



def shear_y(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, 0, level, 1, 0),

                           resample=Image.BILINEAR)



def translate_x(pil_img, level):

    level = int_parameter(sample_level(level), pil_img.size[0] / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, level, 0, 1, 0),

                           resample=Image.BILINEAR)



def translate_y(pil_img, level):

    level = int_parameter(sample_level(level), pil_img.size[0] / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, 0, 0, 1, level),

                           resample=Image.BILINEAR)



# operation that overlaps with ImageNet-C's test set

def color(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Color(pil_img).enhance(level)



# operation that overlaps with ImageNet-C's test set

def contrast(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Contrast(pil_img).enhance(level)



# operation that overlaps with ImageNet-C's test set

def brightness(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Brightness(pil_img).enhance(level)



# operation that overlaps with ImageNet-C's test set

def sharpness(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Sharpness(pil_img).enhance(level)





augmentations = [

    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,

    translate_x, translate_y

]



def normalize(image):

    """Normalize input image channel-wise to zero mean and unit variance."""

    return image - 127



def apply_op(image, op, severity):

    #   image = np.clip(image, 0, 255)

    pil_img = Image.fromarray(image)  # Convert to PIL.Image

    pil_img = op(pil_img, severity)

    return np.asarray(pil_img)



def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):

    """Perform AugMix augmentations and compute mixture.

    Args:

    image: Raw input image as float32 np.ndarray of shape (h, w, c)

    severity: Severity of underlying augmentation operators (between 1 to 10).

    width: Width of augmentation chain

    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly

      from [1, 3]

    alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:

    mixed: Augmented and mixed image.

    """

    ws = np.float32(

      np.random.dirichlet([alpha] * width))

    m = np.float32(np.random.beta(alpha, alpha))



    mix = np.zeros_like(image).astype(np.float32)

    for i in range(width):

        image_aug = image.copy()

        depth = depth if depth > 0 else np.random.randint(1, 4)

        for _ in range(depth):

            op = np.random.choice(augmentations)

            image_aug = apply_op(image_aug, op, severity)

            

        # Preprocessing commutes since all coefficients are convex

        mix += ws[i] * image_aug

    mixed = (1 - m) * image + m * mix

    return mixed
class AugMix(ImageOnlyTransform):



    def __init__(self, severity=3, width=3, depth=-1, 

                 alpha=1., always_apply=False, p=0.5):

        super().__init__(always_apply, p)

        self.severity = severity

        self.width = width

        self.depth = depth

        self.alpha = alpha



    def apply(self, image, **params):

        image = augment_and_mix(image,

            self.severity,

            self.width,

            self.depth,

            self.alpha)

        return image

    

    

# augmix augmentation using albumentation

albu_transforms_train = albumentations.Compose([

    AugMix(severity=3, width=3, alpha=1., p=1.),

])
# general augmentation methods using img_aug library

iaa_train_transform = iaa.Sequential([

    iaa.OneOf([ ## rotate

        iaa.Affine(rotate=0),

        iaa.Affine(rotate=90),

        iaa.Affine(rotate=180),

        iaa.Affine(rotate=270),

    ]),

    iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),

    iaa.Fliplr(0.5),

    iaa.Flipud(0.5),

])
class SIMelanomaGenerate(Sequence):

    def __init__(self, data, batch_size, 

                 dim, mixcutup=False, shuffle=False, 

                 is_train=False, transform=False):

        '''initiate params

        data      : dataframe

        batch_size: batch size for training

        dim       : image resolution

        mixcutup  : True for "mixup" augmentation 

        shuffle   : shuffling the data set

        is_train  : false for test set. 

        transform : Augmentaiton on Train set (AugMix, img_aug*)

        '''

        self.dim        = dim

        self.data       = data

        self.shuffle    = shuffle

        self.is_train   = is_train

        self.mix        = mixcutup

        self.transform  = transform

        self.batch_size = batch_size

        self.label      = self.data['target'] if self.is_train else np.nan

        self.list_idx   = data.index.values

        self.indices    = np.arange(len(self.list_idx))

        self.on_epoch_end()



    def __len__(self):

        return int(np.ceil(len(self.data) / float(self.batch_size)))



    def __getitem__(self, index):

        batch_idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        idx = [self.list_idx[k] for k in batch_idx]

        

        # placeholder

        Data   = np.empty((self.batch_size, *self.dim, 3))

        Target = np.empty((self.batch_size), dtype = np.float32)

        

        for i, k in enumerate(idx):

            # load the image file using cv2

            if self.is_train:

                image = cv2.imread(train_images + self.data['image_name'][k] + '.jpg',

                                  cv2.IMREAD_COLOR) 

            else:

                image = cv2.imread(test_images + self.data['image_name'][k] + '.jpg',

                                  cv2.IMREAD_COLOR)

                

            # resize and scaling 

            image = cv2.resize(image, self.dim)

            

            # all about transformation 

            if self.transform:

                if np.random.rand() < 0.9:

                    # image augmentation using "img_aug"

                    image = iaa_train_transform.augment_image(image)

                else:

                    # image augmentation using "albumentation"

                    res   = albu_transforms_train(image=image)

                    image = res['image'].astype(np.float32)

                

            # image scaling

            image = image.astype(np.float32)/255.0 

        

            # pass training set or simply test samples 

            if self.is_train:

                Data[i,:, :, :] =  image

                Target[i] = self.label.loc[k]

            else:

                Data[i,:, :, :] =  image

                

            # mix_up augmenation

            if self.mix:

                Data, Target = self.mix_up(Data, Target)

                

        return Data, Target if self.is_train else Data

    

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(self.indices)

    

    @staticmethod

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y
def fold_generator(fold):

    # for way one - data generator

    train_labels = df[df.fold != fold].reset_index(drop=True)

    val_labels = df[df.fold == fold].reset_index(drop=True)



    # training generator

    train_generator = SIMelanomaGenerate(train_labels, batch_size, 

                                         dim, mixcutup=False, shuffle = True, 

                                         is_train = True, transform = True)



    # validation generator: no shuffle , not augmentation

    val_generator = SIMelanomaGenerate(val_labels, batch_size, dim, 

                                       mixcutup=False, shuffle = False, 

                                       is_train = True, transform = None)



    return train_generator, val_generator, train_labels, val_labels
gc.collect()
# Generalized mean pool - GeM

gm_exp = tf.Variable(3.0, dtype = tf.float32)

def GeM2d(X):

    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 

                        axis = [1, 2], 

                        keepdims = False) + 1.e-7)**(1./gm_exp)

    return pool



def Net(input_dim):

    input = L.Input(input_dim)

    efnet = efn.EfficientNetB3(weights='noisy-student',

                               include_top = False, 

                               input_tensor = input)

    

    # GeM

    lambda_layer = L.Lambda(GeM2d) 

    lambda_layer.trainable_weights.extend([gm_exp])

    features     = lambda_layer(efnet.output)

    

    # tails

    features     = L.Dense(512, activation='relu',name='relu_act') (features)

    features     = L.Dropout(0.5)(features)

    classifier   = L.Dense(1, activation='sigmoid',name='predictions') (features)

    

    model        = Model(efnet.input, classifier)

    return model
# Optimizer

radam  = tfa.optimizers.RectifiedAdam(lr=0.001)

ranger = tfa.optimizers.Lookahead(radam, 

                                  sync_period=6, 

                                  slow_step_size=0.5)



# Loss

def focal_loss(alpha=0.25,gamma=2.0):

    def focal_crossentropy(y_true, y_pred):

        bce    = K.binary_crossentropy(y_true, y_pred)

        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

        p_t    = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

        alpha_factor      = y_true*alpha + ((1-alpha)*(1-y_true))

        modulating_factor = K.pow((1-p_t), gamma)

        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)

    return focal_crossentropy
def Call_Back(each_fold):

    # model check point

    checkpoint = ModelCheckpoint('../working/fold_{}.h5'.format(each_fold), 

                                 monitor='val_loss', 

                                 verbose= 0,save_best_only=True, 

                                 mode= 'min',save_weights_only=True)

    

    csv_logger = CSVLogger('../working/history_{}.csv'.format(each_fold))

    

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',

                                   factor=0.3, patience=2,

                                   verbose=1, mode='auto',

                                   epsilon=0.0001, cooldown=1, min_lr=0.00001)

    

    return [checkpoint, csv_logger,reduceLROnPlat]
def folds_training(each_fold):

    # clean space

    tf.keras.backend.clear_session()

    gc.collect()



    # call each fold set

    print('\nFold No. {}'.format(each_fold))

    train_generator, val_generator, train_labels, val_labels = fold_generator(each_fold)

     

    # Train set fold

    print('Train Generator: \n', train_labels.target.value_counts())

    plot_imgs(train_generator)

    

    # Valid set fold

    print('Valid Generator: \n', val_labels.target.value_counts())

    plot_imgs(val_generator) 

    

    # building the complete model and compile

    model = Net(input_dim=(*dim,3))

    model.compile(

        optimizer = ranger,

        loss      = [tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.05)],

        metrics   = [tf.keras.metrics.AUC()]

    )

    

    # print out the model params

    if each_fold == 0: 

        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])

        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))

        print('Trainable params: {:,}'.format(trainable_count))

        print('Non-trainable params: {:,}'.format(non_trainable_count))

        

    # invoke callbacks functions

    callbacks = Call_Back(each_fold)

    steps_per_epoch = np.ceil(float(len(train_labels)) / float(batch_size))

    validation_steps = np.ceil(float(len(val_labels)) / float(batch_size))



    # fit generator

    train_history = model.fit_generator(

        train_generator,

        steps_per_epoch=steps_per_epoch,

        validation_data=val_generator,

        validation_steps=validation_steps,

        epochs=5, verbose=1,

        callbacks=callbacks

    )

    

    del model



# calling method to run on all folds

[folds_training(each_fold) for each_fold in range(len(df.fold.value_counts()))] 
df_test = pd.DataFrame({

    'image_name': os.listdir(test_images)

})



df_test['image_name'] = df_test['image_name'].str.split('.').str[0]

print(df_test.shape)

df_test.head()
from tta_wrapper import tta_classification 



# calling test generator

batch_size = 1

model_check_points = sorted(glob('../working/*.h5'))



test_generator = SIMelanomaGenerate(df_test, batch_size, dim, 

                                    shuffle   = False, 

                                    is_train  = False, 

                                    transform = None)



for each_check_points in model_check_points:

    # define and load weights

    model  = Net(input_dim=(*dim,3))

    model.load_weights(each_check_points)

    

    # test time augmentation: horizontal flip, vertical flip, rotate etc

    tta_model = tta_classification(model, h_flip=True, v_flip=True,

                                   rotation=(90,270), h_shift=(-5, 5), 

                                   merge='mean')

    

    # predict and take mean

    df_test[each_check_points.split('/')[-1]] = tta_model.predict(test_generator,

                                                              steps=np.ceil(float(len(df_test)) / float(batch_size)),

                                                              verbose=1)
# rank the predicted data and average ensemble 

df_test['target'] = (rankdata(df_test["fold_0.h5"].astype(float).values) + 

                     rankdata(df_test["fold_1.h5"].astype(float).values) +

                     rankdata(df_test["fold_2.h5"].astype(float).values))



df_test = df_test[['image_name', 'target']]
df_test.to_csv('img_submission.csv', index=False)

df_test.head()
root = '../input/siim-isic-melanoma-classification/'



train = df.copy()

test  = pd.read_csv(os.path.join(root , 'test.csv'))
print('train set:')

train.head()
print('test set')

test.head()
# pre-processing, same as train set (while making group-stratify-kfold)

test['sex'] = test['sex'].fillna('unknown')

test['age_approx'] = test['age_approx'].fillna(round(test['age_approx'].mean()))

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('unknown')
print('Columns Items\n')

print('train set cols\n', train.columns.tolist())

print('\ntest set cols\n',test.columns.tolist())
le = LabelEncoder()



# feat: sex

train.sex = le.fit_transform(train.sex.astype('str'))

test.sex  = le.transform(test.sex.astype('str'))



# feat: anatom site general challenge

train.anatom_site_general_challenge = le.fit_transform(train.anatom_site_general_challenge.astype('str'))

test.anatom_site_general_challenge  = le.transform(test.anatom_site_general_challenge.astype('str'))



# apporx age

train.age_approx = le.fit_transform(train['age_approx'].astype('str'))

test.age_approx  = le.transform(test['age_approx'].astype('str'))
print('train set:')

train = train[['sex', 'age_approx',

               'anatom_site_general_challenge',

               'target', 'fold']]



train.head()
print('test set')



test = test[['sex','age_approx','anatom_site_general_challenge']]

test.head()
if "Set" not in train.columns:

    train["Set"] = np.random.choice(

        ["train", "valid"], p=[0.8, 0.2], size=(train.shape[0],)

    )

    

train_indices = train[train.Set == "train"].index

valid_indices = train[train.Set == "valid"].index
class TabNetTuner(TabNetClassifier):

    def fit(self, X, y, *args, **kwargs):

        

        self.n_d = self.n_a

        

        X_train, X_valid, y_train, y_valid = train_test_split(

            X, y, test_size=0.2, 

            random_state=0, 

            shuffle=True, 

            stratify=y

        )

        

        return super().fit(

            X_train,y_train,

            patience=3,

            X_valid=X_valid,y_valid=y_valid,

            num_workers=os.cpu_count(),max_epochs=10, 

            batch_size=2048, virtual_batch_size=512

        )
# define tuner

tb = TabNetTuner()



# define param

# list(tb.get_params().keys())

grid = {

    "n_a": [16, 32],

    "n_independent": [3, 4, 5],

    "n_shared": [1, 2], 

    "n_steps": [3, 5],

    "clip_value": [1.],

    "gamma": [0.5, 2.],

    "momentum": [0.1, 0.005],

    "lambda_sparse": [0.1, 0.01],

    "verbose": [1],

    'seed':[42]

}



# define searching object

rand_search = RandomizedSearchCV(

    tb, grid,n_iter=5,

    scoring="roc_auc",n_jobs=1,

    iid=False,refit=False,

    cv=[(train_indices, valid_indices)],

    verbose=1,pre_dispatch=0,

    random_state=42,

    return_train_score=False,

)
# get relevant features

features = list(set(train.columns.tolist()) - set(['target']) -

                set(["Set"]) - set(['fold']) - set(['stratify_group']))



label = ['target']



X = train[features].values

y = train[label].values.squeeze(1)



print(features, X.shape)

print(label, y.shape)
rand_search.fit(X, y)

rand_search.best_params_
tab_net = TabNetClassifier(**rand_search.best_params_)



def fold_generator(fold):

    print('Fold Number: ', fold)

    

    train_labels = train[train.fold != fold].reset_index(drop=True)

    val_labels   = train[train.fold == fold].reset_index(drop=True)

    

    X_train = train_labels[features].values

    y_train = train_labels[label].values.squeeze(1)

    

    X_val   = val_labels[features].values

    y_val   = val_labels[label].values.squeeze(1)

    

    print(X_train.shape)

    print(y_train.shape)



    tab_net.fit(X_train,y_train,

              X_val,y_val,weights=1,

              max_epochs=10,patience=7, 

              batch_size=2048, virtual_batch_size=512,

              num_workers=0,drop_last=False)

    

    print("Validation score: {:<8.5f}".format(roc_auc_score(y_val,

                                                            tab_net.predict_proba(X_val)[:,1])))

    

    test[fold] = tab_net.predict_proba(test[features].values)[:,1]
[fold_generator(i) for i in range(3)] 
print('test set with all folds cols [0:3)')

test.head()
sample = pd.read_csv(os.path.join(root , 'sample_submission.csv'))

sample.target = test.iloc[:, 3:].astype(float).mean(axis=1)

sample.to_csv('tab_submission.csv',index=False)