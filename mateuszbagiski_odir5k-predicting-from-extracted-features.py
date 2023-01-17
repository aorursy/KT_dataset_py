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
from tensorflow.keras import models, layers, optimizers, losses, metrics, utils, callbacks, applications
from sklearn.model_selection import train_test_split as tts
import cv2 as cv

# Image size: 1024px x 1024px (x 3 color (RGB) channels)
IMG_SIZE = 1024

# Batch size: 32 images
BATCH_SIZE = 32

# Paths to directories

train_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images'
test_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Testing Images'
main_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K'

history_dir = '/kaggle/working/history'
if not os.path.isdir(history_dir):
    os.mkdir(history_dir)

    models_dir = '/kaggle/working/models'
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)



# A function to load and preprocess image (clip out the black background and resize it properly)

def load_prep_img(image_path, target_shape=(IMG_SIZE, IMG_SIZE)):
    image = cv.imread(image_path, cv.IMREAD_COLOR) # load from the directory
    non_0_rows = np.array([row_idx for row_idx, row in enumerate(image) if np.count_nonzero(row)!=0])
    non_0_cols = np.array([col_idx for col_idx, col in enumerate(image.transpose(1,0,2)) if np.count_nonzero(col)!=0])
    image = image[non_0_rows.min():non_0_rows.max()+1, non_0_cols.min():non_0_cols.max()+1, :] # clip
    image = cv.resize(image, target_shape)
    return image
    

# Function test:
image_path = os.path.join(train_dir, os.listdir(train_dir)[0])

image = cv.imread(image_path, cv.IMREAD_COLOR)
print('Original (raw) image:\t',image.shape)
plt.imshow(image)
plt.show()

image_prepped = load_prep_img(image_path)
print('Preprocessed image:\t', image_prepped.shape)
plt.imshow(image_prepped)
plt.show()
data = pd.read_excel(os.path.join(main_dir, 'data.xlsx'))
data.head()
classes = list(data.columns[7:15])
data['Patient Labels'] = data.apply(lambda x:[class_ for class_ in classes if x[class_]==1], axis=1)
data.head()
# Names of columns in the DataFrame
col_names = ['Image', 'Keywords', 'Age', 'Sex', 'Patient Labels']

# DataFrame for left-eye images
eyes_L = data[['Left-Fundus', 'Left-Diagnostic Keywords', 'Patient Age', 'Patient Sex', 'Patient Labels']]
eyes_L.columns = col_names

# DataFrame for right-eye images
eyes_R = data[['Right-Fundus', 'Right-Diagnostic Keywords', 'Patient Age', 'Patient Sex', 'Patient Labels']]
eyes_R.columns = col_names

# DataFrame for left-eye and right-eye images combined
eyes_df = pd.concat([eyes_L, eyes_R], axis=0)

eyes_df
# Keywords characteristic for 'O' class:

O_keywords = [
    'macular epiretinal membrane',
    'epiretinal membrane',
    'drusen',
    #'lens dust',
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
    #'low image quality',
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
    'silicone oil eye',
    'fundus laser photocoagulation spots',
    'glial remnants anterior to the optic disc',
    'intraretinal microvascular abnormality'
    

]
def generate_eye_labels(keywords, patient_labels):
    eye_labels = []
    
    if 'normal fundus' in keywords:
        eye_labels.append('N')
        if list(set(keywords.replace('，', ',').split(',')))==['normal fundus']: # there were two images, for which 'normal fundus' keyphrase was duplicated
            eye_labels.append('N+') # healthy fundus without any caveats (like lens dust or low image quality)
            return eye_labels # in this case we know that there are no other keywords, so we can already quit the function and return the list
        else:
            eye_labels.append('N-') # healthy fundus but with some caveats
    if 'lens dust' in keywords:
        eye_labels.append('LD') # lens dust
    if 'low image quality' in keywords:
        eye_labels.append('LIQ') # low image quality
    if 'D' in patient_labels and ('proliferative retinopathy' in keywords or 'diabetic' in keywords):
        eye_labels.append('D') # diabetes
    if 'suspected glaucoma' in keywords:
        eye_labels.append('SG') # suspected glaucoma (it may be real glaucoma or may not)
    elif 'glaucoma' in keywords:
        eye_labels.append('G') # glaucoma
    if 'cataract' in keywords:
        eye_labels.append('C') # cataract
    if 'age-related' in keywords:
        eye_labels.append('A') # AMD
    if 'hypertensi' in keywords:
        eye_labels.append('H') # hypertension
    if 'myopi' in keywords:
        eye_labels.append('M') # myopia
    if 'O' in patient_labels and (any(O_keyword in keywords for O_keyword in O_keywords)):
        eye_labels.append('O') # other (anything else)
    return eye_labels




eyes_df['Eye Labels'] = eyes_df.apply(lambda x: generate_eye_labels(x['Keywords'], x['Patient Labels']), axis=1)
eyes_df
c = Counter()
for eye_label in eyes_df['Eye Labels']:
    c[tuple(eye_label)] +=1 
c
def extract_dataframe(criteria=['C'], n_max=0, shuffle=True, df=eyes_df):
    if type(criteria)!=type(list()):
        criteria = [criteria]
    disease_criteria = [criterion for criterion in criteria if criterion!='Male' and criterion!='Female']
    
    if disease_criteria==[]:
        df['extract'] = 1
    else:
        df['extract'] = df['Eye Labels'].apply(lambda x: 1 if all(criterion in x for criterion in disease_criteria) else 0)
    
    if 'Male' in criteria:
        df['extract'] = df['extract'] * df['Sex'].apply(lambda x: 1 if x=='Male' else 0 )
    elif 'Female' in criteria:
        df['extract'] = df['extract'] * df['Sex'].apply(lambda x: 1 if x=='Female' else 0 )
        
    extract_df = df.query(' `extract` == 1 ')
    extract_df.drop('extract', axis=1, inplace=True)
    df.drop('extract', axis=1, inplace=True)
    if shuffle:
        extract_df = extract_df.sample(frac=1)
    extract_df.reset_index(drop=True, inplace=True)
    if n_max!=0:
        extract_df = extract_df.iloc[:n_max, :]
    
    return extract_df



males_with_cataract_df = extract_dataframe(['C', 'Male'])
print(males_with_cataract_df['Eye Labels'].value_counts()) # Distribution of eye labels in males with cataract
print(males_with_cataract_df['Patient Labels'].value_counts()) # Distribution of patient labels in males with cataract
print(males_with_cataract_df['Sex'].value_counts()) # Distribution of sexes in males with cataract - unsurprisingly boring
males_with_cataract_df
all_eye_labels = [*classes, 'N-', 'N+','LD','LIQ','SG']
for criterion in [*all_eye_labels, 'Male', 'Female']:
    print(criterion, extract_dataframe(criterion).shape[0])
print("There are %i images of perfectly healthy male eyes in the dataset" % extract_dataframe(['Male','N+']).shape[0])
print("There are %i images of perfectly healthy female eyes in the dataset" % extract_dataframe(['Female', 'N+']).shape[0])
print("There are %i images of perfectly healthy eyes in total in the dataset" % extract_dataframe(['N+']).shape[0])
# Load the extractor - pre-trained ResNet152
extractor = applications.ResNet152(include_top=False, weights='imagenet', pooling='max', input_shape=(IMG_SIZE, IMG_SIZE, 3))

def extract_features(extract_df, directory=train_dir, verbose=True):
    if verbose:
        print(extract_df.shape[0], "images are being processed...")
    extracts = []
    for i, row in tqdm(extract_df.iterrows()):
        image_path = os.path.join(directory, row['Image'])
        image = load_prep_img(image_path, (IMG_SIZE, IMG_SIZE)).reshape(1, IMG_SIZE, IMG_SIZE, 3)
        extract = extractor.predict(image)
        extracts.append(extract)
    
    return np.array(extracts).reshape(-1,2048)
n_max = 1024

N_extracts = extract_features(extract_df=extract_dataframe(criteria='N+', n_max=n_max))
N_labels = np.zeros(shape=(n_max,))

print(N_extracts.shape, N_labels.shape)
# A function for splitting data into training, validation, and testing sets of given relative sizes
def tvt_split(X, y, split_sizes=[8,1,1], stratify=True):
    split_sizes = np.array(split_sizes)
    if stratify:
        train_X, test_X, train_y, test_y = tts(X, y, test_size=split_sizes[2]/split_sizes.sum(), stratify=y)
        train_X, val_X, train_y, val_y = tts(train_X, train_y, test_size=split_sizes[1]/(split_sizes[0]+split_sizes[1]), stratify=train_y)
    else:
        train_X, test_X, train_y, test_y = tts(X, y, test_size=split_sizes[2]/split_sizes.sum())
        train_X, val_X, train_y, val_y = tts(train_X, train_y, test_size=split_sizes[1]/(split_sizes[0]+split_sizes[1]))
    return train_X, val_X, test_X, train_y, val_y, test_y

# A function generating training, validation, and test sets, given the criteria and maximum number of examples per (positive or negative) class
def generate_datasets(criteria=['C'], n_max=1024):
    X_extracts = extract_features(extract_dataframe(criteria, n_max=n_max))
    X_n = X_extracts.shape[0]
    X_labels = np.ones(shape=(X_n))
    
    extracts = np.concatenate([X_extracts, N_extracts[:X_n, :]], axis=0) # Previously generated extracts of healthy eyes are the negative class
    labels = np.concatenate([X_labels, N_labels[:X_n]], axis=0)
    
    return  tvt_split(extracts, labels) #train_X, val_X, test_X, train_y, val_y, test_y
train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets(criteria=['D'], n_max=100)
for X, y in [[train_X, train_y], [val_X, val_y], [test_X, test_y]]:
    print(X.shape, y.shape) # Sanity-check for the shape of each set
    print(np.bincount(y.astype(np.int32))) # Sanity check for equal distribution of classes in each set
# Build a model template, whose copies will be trained to classify all the clinical conditions

model_template = models.Sequential(name='model_template', layers=[
    layers.Input(shape=(2048,)),
    layers.BatchNormalization(),
    layers.Dense(256, kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(.5),
    layers.Dense(32, kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(.5),
    layers.Dense(2, activation='softmax')
])

# Callbacks for training models
def generate_callbacks(filepath, monitor='val_acc', mode='max'):
    return [
        callbacks.EarlyStopping(patience=50), # Stop training after 50 epochs of no improvement
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=20, verbose=0), # Reduce learning rate by a factor of 10, if performance hasn't been improving for 20 epochs
        callbacks.ModelCheckpoint(filepath=filepath, monitor=monitor, mode=mode, save_best_only=True, save_freq='epoch', save_weights_only=True)
    ]

# Function to evalute the given model on the training, validation, and testing set
def evaluate_model(model):
    print("Training set:\tLoss: %f\tMetric: %f"% tuple(model.evaluate(train_X, train_y, verbose=0)))
    print("Validation set:\tLoss: %f\tMetric: %f"% tuple(model.evaluate(val_X, val_y, verbose=0)))
    print("Testing set:\tLoss: %f\tMetric: %f"% tuple(model.evaluate(test_X, test_y, verbose=0)))


model_template.summary()
def plot_history(history):
    epochs = np.arange(1, len(history.history['loss'])+1)
    print("epochs:", len(epochs))
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, train_loss, 'r-', label='train_loss')
    plt.plot(epochs, val_loss, 'g--', label='val_loss')
    plt.legend()
    print("Training and validation loss:")
    plt.show()
    
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, train_acc, 'r-', label='train_acc')
    plt.plot(epochs, val_acc, 'g--', label='val_acc')
    plt.legend()
    print("Training and validation accuracy:")
    plt.show()
    
    lr = history.history['lr']
    plt.plot(epochs, lr, 'b--', label='lr')
    plt.legend()
    print("Learning rate:")
    plt.show()
# Generate datasets for diabetes

train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets('D')
D_model = models.clone_model(model_template)

D_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'D_model.h5')
print(model_filepath)

D_history = D_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=2
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(D_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
D_model.load_weights(model_filepath)# = models.load_model(model_filepath)
evaluate_model(D_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(D_history.history['val_acc']))
plot_history(D_history)
# Generate datasets for glaucoma

train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets('G')
G_model = models.clone_model(model_template)

G_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'G_model.h5')

G_history = G_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(G_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
G_model.load_weights(model_filepath)# = models.load_model(model_filepath)
evaluate_model(G_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(G_history.history['val_acc']))

plot_history(G_history)
train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets('C')
C_model = models.clone_model(model_template)

C_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'C_model.h5')

C_history = C_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(C_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
C_model.load_weights(model_filepath)# = models.load_model(model_filepath)
evaluate_model(C_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(C_history.history['val_acc']))

plot_history(C_history)
train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets('A')
A_model = models.clone_model(model_template)

A_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'A_model.h5')

A_history = A_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(A_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
A_model.load_weights(model_filepath)
evaluate_model(A_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(A_history.history['val_acc']))

plot_history(A_history)
train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets('H')
H_model = models.clone_model(model_template)

H_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'H_model.h5')

H_history = H_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(H_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
H_model.load_weights(model_filepath)
evaluate_model(H_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(H_history.history['val_acc']))

plot_history(H_history)
train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets('M')
M_model = models.clone_model(model_template)

M_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'M_model.h5')

M_history = M_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(M_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
H_model.load_weights(model_filepath)
evaluate_model(M_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(M_history.history['val_acc']))

plot_history(M_history)
train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets('O')
O_model = models.clone_model(model_template)

O_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'O_model.h5')

O_history = O_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(O_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
O_model.load_weights(model_filepath)
evaluate_model(O_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(O_history.history['val_acc']))

plot_history(O_history)
n_max = 1024

# Generate extracts from images of female retinas and assign labels
Female_extracts = extract_features(extract_dataframe(['Female', 'N+'], n_max=n_max))
Female_labels = np.zeros(shape=(n_max,))

# Generate extracts from images of male retinas and assign labels
Male_extracts = extract_features(extract_dataframe(['Male', 'N+'], n_max=n_max))
Male_labels = np.ones(shape=(n_max,))
extracts = np.concatenate([Female_extracts, Male_extracts], axis=0)
labels = np.concatenate([Male_labels, Female_labels], axis=0)

#train_X, test_X, train_y, test_y = tts(extracts, labels, test_size=1/10, stratify=labels)
#train_X, val_X, train_y, val_y = tts(train_X, train_y, test_size=1/9, stratify=train_y)
train_X, val_X, test_X, train_y, val_y, test_y = tvt_split(extracts, labels, stratify=True)
for X, y in [[train_X, train_y], [val_X, val_y], [test_X, test_y]]:
    print(X.shape, y.shape) # Sanity-check for the shape of each set
    print(np.bincount(y.astype(np.int32))) # Sanity check for equal distribution of classes in each set
sex_model = models.clone_model(model_template)
'''
sex_model = models.Sequential(name='sex_model', layers=[
    layers.Input(shape=(2048,)),
    layers.BatchNormalization(),
    layers.Dense(512, kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(.5),
    layers.Dense(128, kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(.5),
    layers.Dense(32, kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(.5),
    layers.Dense(2, activation='softmax')
])
'''

sex_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'sex_model.h5')

sex_history = sex_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(sex_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
sex_model.load_weights(model_filepath)
evaluate_model(sex_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(sex_history.history['val_acc']))

plot_history(sex_history)
ages_df = eyes_df.query(' `Keywords` == "normal fundus" ')
ages_extracts = extract_features(ages_df)
ages_labels = ages_df['Age']

train_X, val_X, test_X, train_y, val_y, test_y = tvt_split(X=ages_extracts, y=ages_labels, stratify=False)
for X, y in [[train_X, train_y], [val_X, val_y], [test_X, test_y]]:
    print(X.shape, y.shape) # Sanity-check for the shape of each set
    #print(np.bincount(y.astype(np.int32))) # Sanity check for equal distribution of classes in each set
# Since this is a regression task, we need to use a different final layer - with one output node and no (or identity) activation function
age_model = models.Sequential(name='age_model', layers=[
    layers.Input(shape=(2048,)),
    layers.BatchNormalization(),
    layers.Dense(256, kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(.5),
    layers.Dense(32, kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(.5),
    layers.Dense(1)
])

# For the same reason we specify loss function and metric appropriate for regression: Mean Squared Error (MSE) and Mean Absolute Error (MAE), respectively 
age_model.compile(
    optimizer = optimizers.RMSprop(3e-3),
    loss='mse',
    metrics=['mae']
)

model_filepath = os.path.join(models_dir, 'age_model.h5')

age_history = age_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 128, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath, monitor='val_mae', mode='min'),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(age_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
age_model.load_weights(model_filepath)
evaluate_model(age_model)

print("\nThe lowest validation MAE achieved by this model was", np.min(age_history.history['val_mae']))


epochs = np.arange(1, len(age_history.history['loss'])+1)
print("epochs:", len(epochs))

train_mse = age_history.history['loss']
val_mse = age_history.history['val_loss']
plt.plot(epochs, train_mse, 'r-', label='train_mse')
plt.plot(epochs, val_mse, 'g--', label='val_mse')
plt.legend()
print("Training and validation Mean Squared Error:")
plt.show()

train_mae = age_history.history['mae']
val_mae = age_history.history['val_mae']
plt.plot(epochs, train_mae, 'r-', label='train_mae')
plt.plot(epochs, val_mae, 'g--', label='val_mae')
plt.legend()
print("Training and validation Mean Absolute Error:")
plt.show()

lr = age_history.history['lr']
plt.plot(epochs, lr, 'b--', label='lr')
plt.legend()
print("Learning rate:")
plt.show()
train_X, val_X, test_X, train_y, val_y, test_y = generate_datasets(['LD', 'N-']) # Only images of healthy fundi (other than having lens dust)
LD_model = models.clone_model(model_template)

LD_model.compile(
    optimizer = optimizers.RMSprop(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model_filepath = os.path.join(models_dir, 'LD_model.h5')

LD_history = LD_model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 256, batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = generate_callbacks(filepath=model_filepath),
    verbose=0
)
print("\tEvaluation of the model at the end of training\n")
evaluate_model(LD_model)

print("\n\tEvaluation of the model instance at the best point in the training (the highest validation accuracy)\n")
LD_model.load_weights(model_filepath)
evaluate_model(LD_model)

print("\nThe highest validation accuracy achieved by this model was", np.max(LD_history.history['val_acc']))

plot_history(LD_history)
