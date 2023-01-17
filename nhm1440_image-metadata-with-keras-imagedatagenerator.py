!pip install git+git://github.com/stared/livelossplot.git
!pip install -q efficientnet

import pandas as pd 
import numpy as np 
import cv2

import pydicom 
import random

import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import concatenate, Dense, Dropout, Input, Flatten, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential, Model, load_model
from keras.utils import plot_model

import tensorflow as tf
from tensorflow.python.keras import backend as K

import skimage.io
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import plotly.graph_objects as go

from livelossplot import PlotLossesKeras

import efficientnet.tfkeras as efn 


base_directory_path = '../input/siim-isic-melanoma-classification'

base_train_df = pd.read_csv(base_directory_path + '/train.csv')
test_df = pd.read_csv(base_directory_path + '/test.csv')

base_train_df = base_train_df.rename(columns = {'anatom_site_general_challenge': 'body_part'})
test_df = test_df.rename(columns = {'anatom_site_general_challenge': 'body_part'})

print('------------------------------------')
print('Train data size: ', len(base_train_df))
print('Test data size: ', len(test_df))
print('------------------------------------\n')


print('------------------------------------')
print('# of categories in each column \n')
print(base_train_df.nunique())
print('------------------------------------\n')


missing = base_train_df.isnull().sum().sort_values(ascending = False) # taking values list(as an array) of series
percent = missing * 100 / len(base_train_df)
missing_percent_df  = pd.concat([missing, percent], axis = 1, keys = ['missing', 'percent'])
print('------------------------------------')
print('% of missing values in columns \n')
print(missing_percent_df)
print('------------------------------------\n')

base_train_df.head()

test_df

print('------------------------------------')
print('% of categories in \'body parts\' \n')
print(base_train_df['body_part'].value_counts(ascending = True, normalize = True))
print('------------------------------------\n')

print('------------------------------------')
print('% of categories in \'gender\' \n')
print(base_train_df['sex'].value_counts(ascending = True, normalize = True))
print('------------------------------------\n')

print('------------------------------------')
print('% of categories in \'diagnosis\' \n')
print(base_train_df['diagnosis'].value_counts(ascending = True, normalize = True))
print('------------------------------------\n')

print('------------------------------------')
print('% of categories in \'benign_malignant\' \n')
print(base_train_df['benign_malignant'].value_counts(ascending = True, normalize = True))
print('------------------------------------\n')

pd.crosstab(base_train_df['patient_id'], base_train_df['target']).head()

# taking only malignant cases
malignant_df = base_train_df[base_train_df['benign_malignant'] == 'malignant']
# excluding NAN sex values 
malignant_df = base_train_df[base_train_df['sex'].isna() == False]

pd.crosstab(malignant_df['benign_malignant'], malignant_df['sex'])

# taking only NAN sex values
missing_sex_df = base_train_df[base_train_df['sex'].isna() == True]

print('patients whose sex values are not known')
print(missing_sex_df['patient_id'].value_counts())
print('\n')

print('# of benign vs malignant tumors in these patients')
print(missing_sex_df['benign_malignant'].value_counts())
print('\n')

print('# of missing age values in these patients')
print(missing_sex_df['age_approx'].isnull().sum())

base_train_df.loc[base_train_df.patient_id == 'IP_5205991', 'sex'] = 'female'
base_train_df.loc[base_train_df.patient_id == 'IP_9835712', 'sex'] = 'female'
base_train_df['sex'].isnull().sum()
# taking only NAN age values
missing_age_df = base_train_df[base_train_df['age_approx'].isna() == True]

# patients whose age values are not known
print(missing_age_df['patient_id'].value_counts())
base_train_df[base_train_df.patient_id == 'IP_0550106']

# age median of the females 
female_age_median = base_train_df[base_train_df['sex'] == 'female'].age_approx.median()
# age median of the males 
male_age_median = base_train_df[base_train_df['sex'] == 'male'].age_approx.median()

for patient in missing_age_df.patient_id.unique():
    l = base_train_df.loc[base_train_df.patient_id == patient, 'sex'].unique()
    if l[0] == 'female':
        base_train_df.loc[base_train_df.patient_id == patient, 'age_approx'] = female_age_median
    elif l[0] == 'male':
        base_train_df.loc[base_train_df.patient_id == patient, 'age_approx'] = male_age_median

base_train_df.age_approx.isnull().sum()
base_train_df = base_train_df[~ base_train_df['age_approx'].isin([0])]
#print(base_train_df[~ base_train_df['age_approx'].isin([0]))
base_train_df.age_approx.min()
parts_train = base_train_df.copy()

parts_train['flag'] = np.where(base_train_df['body_part'].isna() == True, 'missing', 'non_missing')

# Figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5))

sns.countplot(parts_train['flag'], hue = parts_train['sex'], ax = ax1)

sns.distplot(parts_train[parts_train['flag'] == 'missing']['age_approx'], 
             hist = False, rug = True, label = 'missing', ax = ax2, kde_kws = dict(linewidth=4))

sns.distplot(parts_train[parts_train['flag'] == 'non_missing']['age_approx'], 
             hist = False, rug = True, label = 'non_missing', ax = ax2, kde_kws = dict(linewidth=4))


ax1.set_title('sex for missing and non_missing body parts', fontsize = 13)
ax2.set_title('age for missing and non_missing body parts', fontsize = 13)


base_train_df['body_part'].fillna('torso', inplace = True)
base_train_df['body_part'].isnull().sum()

base_train_df = base_train_df[~base_train_df['diagnosis'].isin(['atypical melanocytic proliferation', 'cafe-au-lait macule'])]

print(base_train_df['diagnosis'].value_counts())

BENIGN_SAMPLE = 10000

malignant_df = base_train_df[base_train_df['target'] == 1]
benign_df = base_train_df[base_train_df['target'] == 0]

base_train_df = pd.concat([benign_df.sample(BENIGN_SAMPLE, replace = False, random_state = 1234), malignant_df])
base_train_df = base_train_df.reset_index(drop = True)

base_train_df.head()
base_train_df['image_name'] = '../input/siim-isic-melanoma-classification/jpeg/train/' + base_train_df['image_name'] + '.jpg'

base_train_df

sample_image_name = os.path.basename(base_train_df['image_name'][1111])
sample_image_name

sample_image = cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/' + 'ISIC_0431547.jpg')
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

sample_dicom_path = '../input/siim-isic-melanoma-classification/train/' + 'ISIC_0431547' + '.dcm'

dicom_file = pydicom.read_file(sample_dicom_path)
dicom_file_to_imgArray = dicom_file.pixel_array # automatically extracts image

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5))


ax1.imshow(dicom_file_to_imgArray)
ax2.imshow(sample_image)

print(dicom_file)


test_df = pd.read_csv(base_directory_path + '/test.csv')
test_df = test_df.rename(columns = {'anatom_site_general_challenge': 'body_part'})

test_df.isnull().sum().sort_values(ascending = False)

parts_test = test_df.copy()

parts_test['flag'] = np.where(parts_test['body_part'].isna() == True, 'missing', 'non_missing')
#print(parts_test)
#print(parts_test.flag.value_counts())

median = parts_test[parts_test['flag'] == 'missing']['age_approx'].median()

parts_test[(parts_test['flag'] == 'non_missing') & (parts_test['age_approx'] == median)]['body_part'].value_counts()

test_df['body_part'].fillna('torso', inplace = True)
print(test_df['body_part'].isnull().sum())
print(test_df.isnull().sum())
test_df.head()
external_directory_path = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/'

external_df = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/marking.csv')
external_df = external_df.rename(columns = {'anatom_site_general_challenge': 'body_part'})

external_df
external_df.dropna(inplace = True)

base_train_df_copy = pd.read_csv(base_directory_path + '/train.csv')
patient_id_list = base_train_df_copy['patient_id'].tolist()


same_patients_df = external_df.loc[external_df['patient_id'].isin(patient_id_list)]
not_same_patients = external_df.loc[~external_df['patient_id'].isin(patient_id_list)]

malignant_cases_df = not_same_patients[not_same_patients['target'] == 1]



external_train_df = pd.concat([same_patients_df, malignant_cases_df])

external_train_df = external_train_df.reset_index(drop = True)

external_train_df = external_train_df.rename(columns = {'image_id': 'image_name'})
external_train_df['image_name'] = external_directory_path + external_train_df['image_name'] + '.jpg'


external_train_df
# checking if there are missing values
base_train_df.isnull().sum()
base_train_df = pd.get_dummies(base_train_df, columns = ['body_part'], prefix = [''])
base_train_df = pd.get_dummies(base_train_df, columns = ['sex'], prefix = [''])
base_train_df['age_approx_norm'] = base_train_df['age_approx'] / base_train_df['age_approx'].max()

del base_train_df['patient_id'], base_train_df['benign_malignant'], base_train_df['age_approx'], base_train_df['diagnosis']
base_train_df = base_train_df[['image_name', '_female', '_male', '_head/neck', '_lower extremity', '_oral/genital', '_palms/soles', '_torso', '_upper extremity', 'age_approx_norm', 'target']]

base_train_df.head()
external_train_df['age_approx_norm'] = external_train_df['age_approx'] / external_train_df['age_approx'].max()
external_train_df = pd.get_dummies(external_train_df, columns = ['body_part'], prefix = [''])
external_train_df = pd.get_dummies(external_train_df, columns = ['sex'], prefix = [''])
del external_train_df['patient_id'], external_train_df['age_approx'], external_train_df['source'] #test_df['age_category_freq'], test_df['age_category'], 


external_train_df = external_train_df[['image_name', '_female', '_male', '_head/neck', '_lower extremity', '_oral/genital', '_palms/soles', '_torso', '_upper extremity', 'age_approx_norm', 'target']]
external_train_df.head()

test_df = pd.get_dummies(test_df, columns=['sex'], prefix = [''])
test_df = pd.get_dummies(test_df, columns=['body_part'], prefix = [''])
test_df['age_approx_norm'] = test_df['age_approx'] / test_df['age_approx'].max()

del test_df['patient_id'], test_df['age_approx'] 

test_df.head()

#test_df['image_name'] = '../input/siim-isic-melanoma-classification/jpeg/test' + test_df['image_name'] + '.jpg'
test_df['image_name'] = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-test/512x512-test/' + test_df['image_name'] + '.jpg'
test_df.head()

IMAGE_SHAPE = (224, 224, 3)
META_DIM = 9

def cnn_net(name):
    
    if name == "EfficientNet":
        
        model = efn.EfficientNetB0(weights = 'imagenet', # noisy-student
                                   include_top = False,
                                   input_shape = IMAGE_SHAPE)

    #model = VGG16(weights = 'imagenet', 
    #              include_top = False, 
    #              input_shape = IMAGE_SHAPE) # Note if 'top=False' then we can add 'pooling='avg'' 
                                               # it will also automatically flatten the layer after convolution
        
        for layer in model.layers: 
            layer.trainable = False
            
    if name == "VGG16":
        model = VGG16(include_top = False, 
                      weights = 'imagenet', 
                      input_shape = IMAGE_SHAPE) # pooling = 'avg'

    x = Flatten()(model.output)
    #x = model.output
    #x = GlobalAveragePooling2D()(x)
    
    #x = Dense(512, activation = 'relu')(x)
    #x = Dropout(0.2)(x, training = True)
    
    #x = Dense(256, activation = 'relu')(x)
    #x = Dropout(0.2)(x, training = True)
 
    #x = Dense(128, activation = 'relu')(x)
    #x = Dropout(0.2)(x, training = True)
    
    x = Dense(8, activation = 'relu')(x)
    #x = Dropout(0.1)(x, training = True)

    
    #x = Dense(8, activation = 'relu')(x)
        
    model = Model(model.input, x)
            
    
    return model

CNN_NET = cnn_net('VGG16')
plot_model(CNN_NET, to_file = 'model_architecture.png', show_shapes = True, show_layer_names = False)

def mlp_net():
    
    model = Sequential()
    model.add(Dense(8, input_dim = META_DIM, activation = "relu"))
    model.add(Dense(4, input_dim = META_DIM, activation = "relu"))

    #model.add(Dense(8, activation = "relu"))
        
    return model

MLP_NET = mlp_net()
plot_model(MLP_NET, to_file = 'model_architecture.png', show_shapes = True, show_layer_names = False)

def concatenated_net(cnn, mlp):
    
    combinedInput = concatenate([cnn.output, mlp.output])
    
    #x = Dense(128, activation="relu")(combinedInput)
    #x = Dropout(0.2)(x, training = True)
    
    #x = Dense(64, activation="relu")(combinedInput)
    #x = Dropout(0.2)(x, training = True)

    x = Dense(1, activation="sigmoid")(combinedInput) # because our metric is AUC, i.e. 
                                                      # softmax with two neurons will not work
    
    model = Model(inputs = [cnn.input, mlp.input], outputs = x)
    return model

concatenated_model = concatenated_net(CNN_NET, MLP_NET)
plot_model(concatenated_model, to_file = 'model_architecture.png', show_shapes = True, show_layer_names = False)

#SVG(model_to_dot(model, dpi=48, rankdir="LR").create(prog='dot', format='svg'))

def focal_loss(alpha = 0.25, gamma = 2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true * y_pred) + ((1-y_true) * (1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true * alpha + ((1-alpha) * (1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor * modulating_factor * bce, axis = -1)
    
    return focal_crossentropy

opt = Adam(lr = 1e-05)
concatenated_model.compile(loss = focal_loss(), metrics = [tf.keras.metrics.AUC(name = 'auc')], optimizer = opt)
#concatenated_model.compile(loss = 'binary_crossentropy', metrics = [tf.keras.metrics.AUC(name = 'auc')], optimizer = opt)

def custom_prep(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hairs = 10
    thickness = 1
    color = (0, 0, 0) # Black hair 

        
    height, width, _ = image.shape
    max_hair_number = random.randint(0, hairs)
    
    
    for _ in range(max_hair_number):
        # The start_point of the line is on upper left part [(0, w/2), (0, h/2)] of an image
        start_point = (random.randint(0, width // 2), random.randint(0, height // 2))
        # The end_point of the line 
        end_point = (random.randint(0, width), random.randint(0, height))
        cv2.line(image, start_point, end_point, color, thickness)
             
        center_coordinates = (width // 2, height // 2) 
        axesLength = (random.randint(0, width // 2), random.randint(0, height // 2)) 
        angle = random.randint(0, 360)
        start_angle = 0
        end_angle = 180
        cv2.ellipse(image, center_coordinates, axesLength, angle, start_angle, end_angle, color, thickness)

    return image   


sample_image = cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/ISIC_8233560.jpg')
sample_image_hair = custom_prep(sample_image)

sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)


f, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5))

ax1.imshow(sample_image)
ax2.imshow(sample_image_hair)
sample_image_hair = cv2.cvtColor(sample_image_hair, cv2.COLOR_BGR2RGB)

cv2.imwrite('filename.jpeg', sample_image_hair) 
train, validation = train_test_split(external_train_df, # base_train_df
                                     test_size = 0.15, 
                                     stratify = external_train_df['target']) # base_train_df

train_datagen = ImageDataGenerator(
    rescale = 1. / 255.,
    rotation_range = 180,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [0.3,1.3],
    fill_mode = 'reflect', # nearest
    preprocessing_function = custom_prep  
)

val_datagen = ImageDataGenerator(rescale = 1./255)


BATCH_SIZE = 8

train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col = 'image_name',
    y_col = train.columns[1:],
    target_size = (IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
    batch_size = BATCH_SIZE,
    shuffle = True,
    class_mode = 'raw')

validation_generator = val_datagen.flow_from_dataframe(
    validation,
    x_col = 'image_name',
    y_col = validation.columns[1:],
    target_size = (IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
    shuffle = False,
    batch_size = BATCH_SIZE,
    class_mode='raw')

def own_train_generator_func():
    count = 0
    while True:
        if count == len(train.index):
            train_generator.reset()
            #break
        count += 1
        data = train_generator.next()
        
        imgs = data[0]
        meta = data[1][:,:-1]
        targets = data[1][:,-1:]
        
        yield [imgs, meta], targets

def own_validation_generator_func():
    count = 0
    while True:
        if count == len(validation.index):
            validation_generator.reset()
            #break
        count += 1
        data = validation_generator.next()
                
        imgs = data[0]
        meta = data[1][:,:-1]
        targets = data[1][:,-1:]
        
        yield [imgs, meta], targets
        
EPOCHS = 3

UPDATES_PER_EPOCH = train.shape[0] // BATCH_SIZE # + 1
VALIDATION_STEPS = validation.shape[0] // BATCH_SIZE

print("Number of training and validation steps: {} and {}".format(UPDATES_PER_EPOCH, VALIDATION_STEPS))

def lrfn(epoch):
    return 1e-4 * (0.7 ** np.floor(epoch / 3))

lr_sched = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)


rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

checkpoint = ModelCheckpoint(
   'best_model.h5', 
    monitor = 'val_auc',
    mode = 'max', # because val_acc is monitored
    verbose = 0, 
    save_best_only = True, 
    save_weights_only = True)

early_stopper = EarlyStopping(
    monitor = 'val_auc', 
    mode = 'max', 
    patience = 5, 
    restore_best_weights = True, 
    verbose = 1)

reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    mode = 'min',
    factor = 0.5, 
    patience = 3, 
    min_lr = 1e-6,
    min_delta = 1e-4,
    verbose = 1)

callbacks_list = [PlotLossesKeras(), checkpoint,  early_stopper, lr_sched]


concatenated_model.fit(
    own_train_generator_func(),
    steps_per_epoch = UPDATES_PER_EPOCH,
    epochs = EPOCHS,
    validation_data = own_validation_generator_func(),
    validation_steps = VALIDATION_STEPS,
    callbacks = [checkpoint, PlotLossesKeras()])

submission = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submission.head()

test_df['image_name'][0]
target = []

for index in range(len(test_df)):
    img = cv2.imread(test_df['image_name'][index])
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

    meta = test_df.iloc[index, 1:]
    meta = meta.to_numpy()
    meta = np.reshape(meta, (1, META_DIM))
    meta = meta.astype(np.float32)

    combined_input = [img, meta]
    prediction = concatenated_model.predict(combined_input)
    target.append(prediction[0][0])


submission['target'] = target


submission.to_csv('combined_submission.csv', index = False)


stratified_kf = StratifiedKFold(
    n_splits = 5, 
    shuffle = True, 
    random_state = 1234)
    
fold_indices = list(stratified_kf.split(external_train_df, external_train_df['target']))

train_datagen = ImageDataGenerator(
    rescale = 1. / 255.,
    rotation_range = 30,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [0.1,1.0],
    fill_mode = 'reflect', # nearest
    #preprocessing_function = custom_prep  
)

val_datagen = ImageDataGenerator(rescale = 1. / 255.)

def lrfn(epoch):
    return 1e-4 * (0.7 ** np.floor(epoch / 3))

def get_callbacks(weights_name):
    
    patience_es = 5
    patience_lr = 3
    
    checkpoint = ModelCheckpoint(
        filepath = weights_name, 
        monitor = 'val_auc',
        mode = 'max',
        verbose = 0,
        save_best_only = True,
        save_weights_only = True)
    
    early_stopper = EarlyStopping(
        monitor = 'val_auc', 
        mode = 'max', 
        patience = patience_es, 
        restore_best_weights = True, 
        verbose = 1)
    
    reduce_lr = ReduceLROnPlateau(
        monitor = 'val_loss', 
        mode = 'min',
        factor = 0.5, 
        patience = patience_lr, 
        min_lr = 1e-6,
        min_delta = 1e-4,
        verbose = 1)
    
    lr_sched = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)


    return [checkpoint, early_stopper, lr_sched] # PlotLossesKeras(), reduce_lr


def own_train_generator_func():
    count = 0
    while True:
        if count == len(train.index):
            train_generator.reset()
            #break
        count += 1
        data = train_generator.next()
        
        imgs = data[0]
        meta = data[1][:,:-1]
        targets = data[1][:,-1:]
        
        yield [imgs, meta], targets

        
def own_validation_generator_func():
    count = 0
    while True:
        if count == len(validation.index):
            validation_generator.reset()
            #break
        count += 1
        data = validation_generator.next()
                
        imgs = data[0]
        meta = data[1][:,:-1]
        targets = data[1][:,-1:]
        
        yield [imgs, meta], targets
        
BATCH_SIZE = 8
EPOCHS = 3


for j, (train_indices, validation_indices) in enumerate(fold_indices):
    
    print('\nFold ',j)

    train = external_train_df.iloc[train_indices]
    validation = external_train_df.iloc[validation_indices]
    
    UPDATES_PER_EPOCH = train.shape[0] // BATCH_SIZE # + 1
    VALIDATION_STEPS = validation.shape[0] // BATCH_SIZE


    train_generator = train_datagen.flow_from_dataframe(
        train,
        x_col = 'image_name',
        y_col = train.columns[1:],
        target_size = (IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        batch_size = BATCH_SIZE,
        shuffle = True,
        class_mode = 'raw')

    validation_generator = val_datagen.flow_from_dataframe(
        validation,
        x_col = 'image_name',
        y_col = validation.columns[1:],
        target_size = (IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        shuffle = False,
        batch_size = BATCH_SIZE,
        class_mode = 'raw')
    
    
    weights_name = 'fold_' + str(j) + '_weights.h5'
    callbacks_list = get_callbacks(weights_name)

    
    concatenated_model.fit(
    own_train_generator_func(),
    steps_per_epoch = UPDATES_PER_EPOCH,
    epochs = EPOCHS,
    validation_data = own_validation_generator_func(),
    validation_steps = VALIDATION_STEPS,
    callbacks = callbacks_list)


concatenated_model.load_weights('../input/fold-weights/fold_0_weights.h5')

target_1 = []

for index in range(len(test_df)):
    img = cv2.imread(test_df['image_name'][index])
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

    meta = test_df.iloc[index, 1:]
    meta = meta.to_numpy()
    meta = np.reshape(meta, (1, META_DIM))
    meta = meta.astype(np.float32)

    combined_input = [img, meta]
    prediction = concatenated_model.predict(combined_input)
    target_1.append(prediction[0][0])

concatenated_model.load_weights('../input/fold-weights/fold_0_weights.h5')

target_1 = []

for index in range(len(test_df)):
    img = cv2.imread(test_df['image_name'][index])
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

    meta = test_df.iloc[index, 1:]
    meta = meta.to_numpy()
    meta = np.reshape(meta, (1, META_DIM))
    meta = meta.astype(np.float32)

    combined_input = [img, meta]
    prediction = concatenated_model.predict(combined_input)
    target_1.append(prediction[0][0])
    
    
submission1 = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submission1['target'] = target_1
submission1.to_csv('meta_and_images_fold_1.csv', index = False)
submission1.head()
concatenated_model.load_weights('../input/fold-weights/fold_1_weights.h5')

target_2 = []

for index in range(len(test_df)):
    img = cv2.imread(test_df['image_name'][index])
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

    meta = test_df.iloc[index, 1:]
    meta = meta.to_numpy()
    meta = np.reshape(meta, (1, META_DIM))
    meta = meta.astype(np.float32)

    combined_input = [img, meta]
    prediction = concatenated_model.predict(combined_input)
    target_2.append(prediction[0][0])

submission2 = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submission2['target'] = target_2
submission2.to_csv('meta_and_images_fold_2.csv', index = False)
submission2.head()


concatenated_model.load_weights('../input/fold-weights/fold_2_weights.h5')

target_3 = []

for index in range(len(test_df)):
    img = cv2.imread(test_df['image_name'][index])
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

    meta = test_df.iloc[index, 1:]
    meta = meta.to_numpy()
    meta = np.reshape(meta, (1, META_DIM))
    meta = meta.astype(np.float32)

    combined_input = [img, meta]
    prediction = concatenated_model.predict(combined_input)
    target_3.append(prediction[0][0])

submission3 = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submission3['target'] = target_3
submission3.to_csv('meta_and_images_fold_3.csv', index = False)
submission3.head()


concatenated_model.load_weights('../input/fold-weights/fold_3_weights.h5')

target_4 = []

for index in range(len(test_df)):
    img = cv2.imread(test_df['image_name'][index])
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

    meta = test_df.iloc[index, 1:]
    meta = meta.to_numpy()
    meta = np.reshape(meta, (1, META_DIM))
    meta = meta.astype(np.float32)

    combined_input = [img, meta]
    prediction = concatenated_model.predict(combined_input)
    target_4.append(prediction[0][0])

submission4 = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submission4['target'] = target_4
submission4.to_csv('meta_and_images_fold_4.csv', index = False)
submission4.head()

concatenated_model.load_weights('../input/fold-weights/fold_4_weights.h5')

target_5 = []

for index in range(len(test_df)):
    img = cv2.imread(test_df['image_name'][index])
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

    meta = test_df.iloc[index, 1:]
    meta = meta.to_numpy()
    meta = np.reshape(meta, (1, META_DIM))
    meta = meta.astype(np.float32)

    combined_input = [img, meta]
    prediction = concatenated_model.predict(combined_input)
    target_5.append(prediction[0][0])

submission5 = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submission5['target'] = target_5
submission5.to_csv('meta_and_images_fold_5.csv', index = False)
submission5.head()

submission_concat = pd.concat([submission1, submission2, submission3, submission4, submission5], axis = 1)
del submission_concat['image_name']

submission_concat.head()
mean = submission_concat.mean(axis = 1)
median = submission_concat.median(axis = 1)
minimum = submission_concat.min(axis = 1)
maximum = submission_concat.max(axis = 1)

submission = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submission['target'] = median
submission.to_csv('median.csv', index = False)
