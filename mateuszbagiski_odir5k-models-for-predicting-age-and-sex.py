# Import everything we need

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, image as mpimg
from tqdm import tqdm
from time import time
from collections import Counter
import random

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics, utils, callbacks
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from skimage.transform import resize

IMG_SIZE = 512
BATCH_SIZE = 32



# Set up all the paths and load the data

train_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images'
test_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Testing Images'
main_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K'
history_dir = '/kaggle/working/history'
models_dir = '/kaggle/working/models'
try:
    os.mkdir(history_dir)
    os.mkdir(models_dir)
except Exception as e:
    pass


data = pd.read_excel(os.path.join(main_dir,'data.xlsx'), sheet_name=None)
data = pd.DataFrame(data[list(data.keys())[0]])

conditions = list(data.columns[-8:])


# copy-pasted from previous notebook's output:

diagnostic_keyphrases = {'N': ['normal fundus'], 
 'D': ['nonproliferative retinopathy',
  'non proliferative retinopathy',
  'proliferative retinopathy'],
 'G': ['glaucoma'],
 'C': ['cataract'],
 'A': ['age-related macular degeneration'],
 'H': ['hypertensive'],
 'M': ['myopi'],
 'O': ['macular epiretinal membrane',
  'epiretinal membrane',
  'drusen',
  'lens dust',
  'myelinated nerve fibers',
  'laser spot',
  'vitreous degeneration',
  'refractive media opacity',
  'spotted membranous change',
  'tessellated fundus',
  'maculopathy',
  'chorioretinal atrophy',
  'branch retinal vein occlusion',
  'retinal pigmentation',
  'white vessel',
  'post retinal laser surgery',
  'epiretinal membrane over the macula',
  'retinitis pigmentosa',
  'central retinal vein occlusion',
  'optic disc edema',
  'post laser photocoagulation',
  'retinochoroidal coloboma',
  'atrophic change',
  'optic nerve atrophy',
  'old branch retinal vein occlusion',
  'depigmentation of the retinal pigment epithelium',
  'chorioretinal atrophy with pigmentation proliferation',
  'central retinal artery occlusion',
  'old chorioretinopathy',
  'pigment epithelium proliferation',
  'retina fold',
  'abnormal pigment ',
  'idiopathic choroidal neovascularization',
  'branch retinal artery occlusion',
  'vessel tortuosity',
  'pigmentation disorder',
  'rhegmatogenous retinal detachment',
  'macular hole',
  'morning glory syndrome',
  'atrophy',
  'low image quality',
  'arteriosclerosis',
  'asteroid hyalosis',
  'congenital choroidal coloboma',
  'macular coloboma',
  'optic discitis',
  'oval yellow-white atrophy',
  'wedge-shaped change',
  'wedge white line change',
  'retinal artery macroaneurysm',
  'retinal vascular sheathing',
  'suspected abnormal color of  optic disc',
  'suspected retinal vascular sheathing',
  'suspected retinitis pigmentosa',
  'silicone oil eye']}

con2img = {condition:[] for condition in conditions}
for i, row in data.iterrows():
    image_L = row['Left-Fundus']
    image_R = row['Right-Fundus']
    if row['N']==1:
        con2img['N'] += [image_L, image_R]
        continue
        
    keyphrases_L = row['Left-Diagnostic Keywords']
    keyphrases_R = row['Right-Diagnostic Keywords']
    diagnosed_conditions = []
    for condition in conditions[1:]:
        if row[condition]==1:
            diagnosed_conditions.append(condition)
            
    if 'normal fundus' in keyphrases_L:
        con2img['N'].append(image_L)
        for condition in diagnosed_conditions:
            con2img[condition].append(image_R)
        continue
    if 'normal fundus' in keyphrases_R:
        con2img['N'].append(image_R)
        for condition in diagnosed_conditions:
            con2img[condition].append(image_L)
        continue
    
    for condition in diagnosed_conditions:
        if any(keyphrase in keyphrases_L for keyphrase in diagnostic_keyphrases[condition]):
            con2img[condition].append(image_L)
        if any(keyphrase in keyphrases_R for keyphrase in diagnostic_keyphrases[condition]):
            con2img[condition].append(image_R)

            
img2con = {}
for condition in conditions:
    for im in con2img[condition]:
        if im not in img2con:
            img2con[im] = [condition]
        else:
            img2con[im] = sorted(img2con[im]+[condition])

# Sanity check: are the numbers the same as in the previous notebook?     
for condition in conditions:
    print(condition, len(con2img[condition]))
print(len(img2con))
imgdata_columns = ['Image', 'Patient Age', 'Patient Sex', *conditions]
imgdata = []

for i, row in data.iterrows():
    image_L = row['Left-Fundus']
    image_R = row['Right-Fundus']
    if image_L in img2con:
        image_conditions = [int(condition in img2con[image_L]) for condition in conditions]
        imgdata.append([image_L, row['Patient Age'], row['Patient Sex'], *image_conditions])
    if image_R in img2con:
        image_conditions = [int(condition in img2con[image_R]) for condition in conditions]
        imgdata.append([image_R, row['Patient Age'], row['Patient Sex'], *image_conditions])

imgdata = pd.DataFrame(imgdata, columns=imgdata_columns)
#imgdata['Patient Sex'] = imgdata['Patient Sex'].apply(lambda x:0  if x=='Female' else 1) # encode sex: 'Female'=>0, 'Male'=>1

imgdata_labels = []
for i, row in imgdata.iterrows():
    row_labels = []
    for condition in conditions:
        if row[condition]==1:
            row_labels.append(condition)
    imgdata_labels.append(row_labels)
imgdata['Labels'] = imgdata_labels
imgdata
con2ind = {condition: [] for condition in conditions}
for i, row in imgdata.iterrows():
    for condition in conditions:
        if row[condition]==1:
            con2ind[condition].append(i)
for condition in conditions:
    con2ind[condition] = np.array(con2ind[condition])
            
print('con2img sizes:')
for condition in conditions:
    print(condition, len(con2img[condition]))

print('\ncon2ind sizes:')
for condition in conditions:
    print(condition, len(con2img[condition]))
def image_prep(image, target_shape=(IMG_SIZE,IMG_SIZE,3)):
    non_0_rows = np.array([row_idx for row_idx, row in enumerate(image) if np.count_nonzero(row)!=0])
    non_0_cols = np.array([col_idx for col_idx, col in enumerate(image.transpose(1,0,2)) if np.count_nonzero(col)!=0])
    image = image[non_0_rows.min():non_0_rows.max()+1, non_0_cols.min():non_0_cols.max()+1, :] # clip
    image = resize(image, target_shape) # resize
    return image

# TEST:
image = mpimg.imread(os.path.join(train_dir, os.listdir(train_dir)[0]))
print('original:\t',image.shape)
plt.imshow(image)
plt.show()

image_prepped = image_prep(image)
print('preprocessed:\t', image_prepped.shape)
plt.imshow(image_prepped)
plt.show()
N_indices = con2ind['N'].copy() # get indices of 'N' images in the imgdata DataFrame
np.random.shuffle(N_indices) # randomly shuffle these indices
N_indices_train, N_indices_val, N_indices_test = N_indices[:2480], N_indices[2480:2790], N_indices[2790:] # split these indices into training, validation, and test set
N_indices_train.shape, N_indices_val.shape, N_indices_test.shape # sanity check
# idg template for training and validation data
train_idg = IDG(
    horizontal_flip=True, vertical_flip=True, rotation_range=180, # a modest data augmentation
    rescale=1./255.,
    preprocessing_function=image_prep # preprocessing function defined earlier
)
# idg template for testing data 
test_idg = IDG(
    rescale=1./255.,
    preprocessing_function=image_prep
)

# training data generator for age prediction
age_train_generator = train_idg.flow_from_dataframe(
    dataframe = imgdata.iloc[N_indices_train, :],
    directory = train_dir,
    x_col='Image',
    y_col='Patient Age',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='raw',
    target_size=(IMG_SIZE, IMG_SIZE)
)
# validation data generator for age prediction
age_val_generator = train_idg.flow_from_dataframe(
    dataframe = imgdata.iloc[N_indices_val, :],
    directory = train_dir,
    x_col='Image',
    y_col='Patient Age',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='raw',
    target_size=(IMG_SIZE, IMG_SIZE)
)
# testing data generator for age prediction
age_test_generator = test_idg.flow_from_dataframe(
    dataframe = imgdata.iloc[N_indices_test, :],
    directory = train_dir,
    x_col='Image',
    y_col='Patient Age',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='raw',
    target_size=(IMG_SIZE, IMG_SIZE)
)
resnet152 = ResNet152(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
resnet152.trainable = False # We freeze the weights of the convolutional base

age_model = models.Sequential(name='age_model')
age_model.add(resnet152)
age_model.add(layers.Flatten(name='flatten'))
age_model.add(layers.Dense(1, activation=None, name='dense_output'))

age_model.compile(
    optimizer=optimizers.Adam(lr=1e-5),
    loss='huber',
    metrics=['mae']
)

age_model.summary()

age_history = age_model.fit(
    age_train_generator, 
    validation_data=age_val_generator, 
    epochs=8, steps_per_epoch=len(N_indices_train)//BATCH_SIZE, validation_steps=len(N_indices_val)//BATCH_SIZE,
    verbose=1,
    shuffle=True
)
pd.DataFrame(age_history.history).to_csv('history/age_history_0.csv') # Save the training history
age_model.save('models/age_model_0.h5') # Save the model with its current weights
train_loss = age_history.history['loss']
val_loss = age_history.history['val_loss']
train_mae = age_history.history['mae']
val_mae = age_history.history['val_mae']

plt.plot(range(1,9), train_loss, 'r-', label='training loss')
plt.plot(range(1,9), val_loss, 'b--', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Dense classifier training: Loss')
plt.legend()
plt.show()

plt.plot(range(1,9), train_mae, 'r-', label='training MAE')
plt.plot(range(1,9), train_mae, 'b--', label='validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Dense classifier training: MAE')
plt.legend()
plt.show()
for layer in age_model.layers[0].layers:
    if 'conv5_block3' in layer.name:
        layer.trainable = True
        
age_model.compile(
    optimizer=optimizers.Adam(1e-7),
    loss='huber',
    metrics=['mae']
)
age_history = age_model.fit(
        age_train_generator, 
        validation_data=age_val_generator, 
        epochs=16, steps_per_epoch=len(N_indices_train)//BATCH_SIZE, validation_steps=len(N_indices_val)//BATCH_SIZE,
        verbose=1,
        shuffle=True
)
pd.DataFrame(age_history.history).to_csv('history/age_history_1.csv')
age_model.save('models/age_model_1.h5')
train_loss = age_history.history['loss']
val_loss =  age_history.history['val_loss']
train_mae = age_history.history['mae']
val_mae = age_history.history['val_mae']


plt.plot(range(1,17), train_loss, 'r-', label='training loss')
plt.plot(range(1,17), val_loss, 'b--', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Fine-tuning: Loss')
plt.legend()
plt.show()

plt.plot(range(1,17), train_mae, 'r-', label='training MAE')
plt.plot(range(1,17), train_mae, 'b--', label='validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Fine-tuning: MAE')
plt.legend()
plt.show()
age_model_before_ft = models.load_model('models/age_model_0.h5')
age_model_after_ft = models.load_model('models/age_model_1.h5')

score_before_ft = age_model_before_ft.evaluate(age_test_generator)
score_after_ft = age_model_after_ft.evaluate(age_test_generator)
print('Evaluation:')
print('\tBefore fine-tuning:\tLoss: %.3f\tMAE: %.3f' % (score_before_ft[0], score_before_ft[1]))
print('\tAfter fine-tuning: \tLoss: %.3f\tMAE: %.3f' % (score_after_ft[0], score_after_ft[1]))

age_model = models.load_model('models/age_model_0.h5')

age_test_generators = {'N': age_test_generator}
for condition in conditions[1:]:
    age_test_generators[condition] = test_idg.flow_from_dataframe(
        dataframe = imgdata.iloc[con2ind[condition], :],
        directory = train_dir,
        x_col='Image',
        y_col='Patient Age',
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode='raw',
        target_size=(IMG_SIZE, IMG_SIZE)
    )

age_model_scores = {}
for condition in conditions:
    age_model_scores[condition] = age_model.evaluate(age_test_generators[condition], verbose=1)
for condition in conditions:
    print('%s:\tLoss:\t%.3f\t\tMAE:\t%.3f' % (condition, age_model_scores[condition][0], age_model_scores[condition][1]) )
age_groups_indices = [
    np.asarray(imgdata.query('`Patient Age` < 30').index),
    np.asarray(imgdata.query('30 <= `Patient Age` < 60').index),
    np.asarray(imgdata.query('60 <= `Patient Age`').index)
]

print('%i images from people below 30'%len(age_groups_indices[0]))
print('%i images from people between 30 and 60'%len(age_groups_indices[1]))
print('%i images from people older than 60'%len(age_groups_indices[2]))
age_groups_test_generators = []
for age_group_indices in age_groups_indices:
    age_groups_test_generators.append(
        test_idg.flow_from_dataframe(
            dataframe = imgdata.iloc[age_group_indices, :],
            directory = train_dir,
            x_col='Image',
            y_col='Patient Age',
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode='raw',
            target_size=(IMG_SIZE, IMG_SIZE)
        )
    )
age_groups_test_scores = []
for age_group_test_generator in age_groups_test_generators:
    age_groups_test_scores.append(age_model.evaluate(age_group_test_generator, verbose=1))
print('1-29:\tLOSS:\t%.3f\tMAE:\t%.3f'%(age_groups_test_scores[0][0],age_groups_test_scores[0][1]))
print('30-59:\tLOSS:\t%.3f\tMAE:\t%.3f'%(age_groups_test_scores[1][0],age_groups_test_scores[1][1]))
print('60+:\tLOSS:\t%.3f\tMAE:\t%.3f'%(age_groups_test_scores[2][0],age_groups_test_scores[2][1]))
N_age_groups_indices = [
    np.asarray(imgdata.query('`Patient Age` < 30 & N==1').index),
    np.asarray(imgdata.query('30 <= `Patient Age` < 60 & N==1 ').index),
    np.asarray(imgdata.query('60 <= `Patient Age` & N==1').index)
]

print('%i images from healthy people below 30'%len(N_age_groups_indices[0]))
print('%i images from healthy people between 30 and 60'%len(N_age_groups_indices[1]))
print('%i images from healthy people older than 60'%len(N_age_groups_indices[2]))

N_age_groups_test_generators = []
for N_age_group_indices in N_age_groups_indices:
    N_age_groups_test_generators.append(
        test_idg.flow_from_dataframe(
            dataframe = imgdata.iloc[N_age_group_indices, :],
            directory = train_dir,
            x_col='Image',
            y_col='Patient Age',
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode='raw',
            target_size=(IMG_SIZE, IMG_SIZE)
        )
    )
    
N_age_groups_test_scores = []
for N_age_group_test_generator in N_age_groups_test_generators:
    N_age_groups_test_scores.append(age_model.evaluate(N_age_group_test_generator, verbose=1))
print('\tHealthy patients only:\n')
print('1-29:\tLOSS:\t%.3f\tMAE:\t%.3f'%(N_age_groups_test_scores[0][0],N_age_groups_test_scores[0][1]))
print('30-59:\tLOSS:\t%.3f\tMAE:\t%.3f'%(N_age_groups_test_scores[1][0],N_age_groups_test_scores[1][1]))
print('60+:\tLOSS:\t%.3f\tMAE:\t%.3f'%(N_age_groups_test_scores[2][0],N_age_groups_test_scores[2][1]))
female_N_indices = []
male_N_indices = []
for i, row in imgdata.query('N == 1').iterrows():
    if row['Patient Sex']=='Female':
        female_N_indices.append(i)
    elif row['Patient Sex']=='Male':
        male_N_indices.append(i)

N_per_sex = len(female_N_indices)
np.random.shuffle(male_N_indices)
female_N_indices = np.array(female_N_indices)
male_N_indices = np.array(male_N_indices[:N_per_sex]) # cutting out a randomly sampled male overrepresantation

female_N_indices_train, female_N_indices_val, female_N_indices_test = female_N_indices[:int(.8*N_per_sex)], female_N_indices[int(.8*N_per_sex):int(.9*N_per_sex)], female_N_indices[int(.9*N_per_sex):]
male_N_indices_train, male_N_indices_val, male_N_indices_test = male_N_indices[:int(.8*N_per_sex)], male_N_indices[int(.8*N_per_sex):int(.9*N_per_sex)], male_N_indices[int(.9*N_per_sex):]

sex_N_indices_train = np.concatenate([female_N_indices_train, male_N_indices_train])
sex_N_indices_val = np.concatenate([female_N_indices_val, male_N_indices_val])
sex_N_indices_test = np.concatenate([female_N_indices_test, male_N_indices_test])
print(np.bincount(imgdata['Patient Sex'].apply(lambda x:0 if x=='Female' else 1)[sex_N_indices_train]))
print(np.bincount(imgdata['Patient Sex'].apply(lambda x:0 if x=='Female' else 1)[sex_N_indices_val]))
print(np.bincount(imgdata['Patient Sex'].apply(lambda x:0 if x=='Female' else 1)[sex_N_indices_test]))
sex_train_generator = train_idg.flow_from_dataframe(
    dataframe = imgdata.iloc[sex_N_indices_train, :],
    directory = train_dir,
    x_col='Image',
    y_col='Patient Sex',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='categorical',
    target_size=(IMG_SIZE, IMG_SIZE)
)
sex_val_generator = train_idg.flow_from_dataframe(
    dataframe = imgdata.iloc[sex_N_indices_val, :],
    directory = train_dir,
    x_col='Image',
    y_col='Patient Sex',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='categorical',
    target_size=(IMG_SIZE, IMG_SIZE)
)
sex_test_generator = test_idg.flow_from_dataframe(
    dataframe = imgdata.iloc[sex_N_indices_test, :],
    directory = train_dir,
    x_col='Image',
    y_col='Patient Sex',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='categorical',
    target_size=(IMG_SIZE, IMG_SIZE)
)
resnet152.trainable = False

sex_model = models.Sequential(name='sex_model')
sex_model.add(resnet152)
sex_model.add(layers.Flatten(name='flatten'))
sex_model.add(layers.Dense(1, activation='sigmoid', name='dense_output'))

sex_model.compile(
    optimizer=optimizers.Adam(lr=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

sex_model.summary()
sex_history = sex_model.fit(
    sex_train_generator,
    validation_data=sex_val_generator,
    epochs=4, steps_per_epoch=len(sex_N_indices_train)//BATCH_SIZE, validation_steps=len(sex_N_indices_val)//BATCH_SIZE,
    shuffle=True
)
pd.DataFrame(sex_history.history).to_csv('history/sex_history_0.csv')
sex_model.save('models/sex_model_0.h5')
train_loss = sex_history.history['loss']
val_loss = sex_history.history['val_loss']
train_acc = sex_history.history['accuracy']
val_acc = sex_history.history['val_accuracy']

plt.plot(range(1,5), train_loss, 'r-', label='training loss')
plt.plot(range(1,5), val_loss, 'b--', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Sex prediction: Loss')
plt.legend()
plt.show()

plt.plot(range(1,5), train_acc, 'r-', label='training accuracy')
plt.plot(range(1,5), val_acc, 'b--', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Sex prediction: Accuracy')
plt.legend()
plt.show()