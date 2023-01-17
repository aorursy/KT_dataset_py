%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import pickle
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()
# from tqdm import tqdm

# pd.options.display.max_rows = 999
# pd.options.display.max_columns = 999
import glob
def get_path(str, first=True, parent_dir='../input/**/'):
    res_li = glob.glob(parent_dir+str)
    return res_li[0] if first else res_li
DATA_DIR = '../input/dogs-vs-cats-redux-kernels-edition/'
evals = pd.read_csv('../input/dvc-prepare-evalset/evals.csv')
evals.head()
H, W, C = 224, 224, 3 #at least 197
batch_size = 32
eval_batch_size = batch_size * 4
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

train_gen = ImageDataGenerator(
    rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #channel_shift_range=0.2,
    #vertical_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    #rescale=1./255,#!!!!!
    preprocessing_function=preprocess_input
)
test_gen = ImageDataGenerator(
    #rescale=1./255,#!!!!!
    preprocessing_function=preprocess_input
)
train_flow = train_gen.flow_from_directory(
    './', # Empty dir
    class_mode=None, 
    target_size=(H, W),
    batch_size=batch_size,
    shuffle=True,
)
valid_flow = test_gen.flow_from_directory(
    './', # Empty dir
    class_mode=None, 
    target_size=(H, W),
    batch_size=eval_batch_size,
    shuffle=False,
)
test_flow = test_gen.flow_from_directory(
    './', # Empty dir
    class_mode=None, 
    target_size=(H, W),
    batch_size=eval_batch_size,
    shuffle=False,
)
def set_data_flow(flow, eval_mode, shuffle=True, valid_fold=0, n_valid=128*8, evals=evals):
    flow.class_indices = {'dog': 0, 'cat': 1}
    if eval_mode=='train':
        flow.directory = DATA_DIR+'train'
        mask = (evals['is_test']==0) & (evals['eval_set']!=valid_fold)
    elif eval_mode=='valid':
        shuffle = False
        flow.directory = DATA_DIR+'train'
        mask = (evals['is_test']==0) & (evals['eval_set']==valid_fold)
    elif eval_mode=='test':
        shuffle = False
        flow.directory = DATA_DIR+'test'
        mask = (evals['is_test']==1)
    flow.samples = len(evals.loc[mask, 'target'].values) if eval_mode!='valid' else n_valid
    flow.n = len(evals.loc[mask, 'target'].values) if eval_mode!='valid' else n_valid
    filenames_arr = evals.loc[mask, 'img_id'].apply(lambda x: x+'.jpg').values
    target_arr = evals.loc[mask, 'target'].values
    if eval_mode=='valid':
        filenames_arr = filenames_arr[:n_valid]
        target_arr = target_arr[:n_valid]
    if shuffle:
        indexes = np.arnage(flow.samples)
        np.random.permutatione(indexes)
        filenames_arr = filenames_arr[indexes]
        target_arr = target_arr[indexes]
    flow.filenames = filenames_arr.tolist()
    flow.classes = target_arr
    flow.class_mode = 'binary'
    flow.num_classes = len(np.unique(target_arr))
    print(f'Found {flow.n} images belonging to {flow.num_classes} classes.')
    return flow
train_flow = set_data_flow(train_flow, 'valid', valid_fold=0)
valid_flow = set_data_flow(valid_flow, 'valid', valid_fold=1)
test_flow = set_data_flow(test_flow, 'test', valid_fold=None)
MODEL_NAME = f'resnet50_weights_tf_dim_ordering_tf_kernels_notop'
MODEL_PATH = f'../input/keras-pretrained-models/{MODEL_NAME}.h5'
from keras.applications.resnet50 import ResNet50
def get_pretrained_model(weight_path=MODEL_PATH, trainable=False):
    input_shape = (H, W, C)
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    base_model.load_weights(weight_path)
    for l in base_model.layers:
        l.trainable = trainable
    return base_model

encoder = get_pretrained_model(weight_path=MODEL_PATH, trainable=False)
import keras.backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import optimizers, losses, activations, models
from keras.layers import Conv2D, Dense, Input, Flatten, Concatenate, Dropout, Activation
from keras.layers import BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras import applications
n_final_state = 32

def get_model(encoder, n_final_state, lr=1e-3, decay=1e-8):
    input_shape = (H, W, C)
    
    input_x = encoder.inputs
    
    x = encoder.output # (None, 1, 1, 2048)
    x = Flatten()(x)
    
    d1 = Dense(
        64, activation='relu'
    )(x)
    #d1 = Dropout(0.5)(d1)
    d1 = BatchNormalization()(d1)
    
    final_state = Dense(
        n_final_state, activation='relu', name='final_state'
    )(d1)
    #x = Dropout(0.5)(final_state)
    final_state = BatchNormalization()(final_state)
    
    outputs = Dense(1, activation='sigmoid')(final_state)
    model = Model(inputs=input_x, outputs=outputs)
    optimizer=optimizers.Adam(lr=lr, decay=decay)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

model = get_model(encoder, n_final_state=n_final_state)
model.summary()
for l in model.layers:
    if l.trainable:
        print(l.name)
train_steps = int(np.ceil(train_flow.n / batch_size))
valid_steps = int(np.ceil(valid_flow.n / eval_batch_size))
test_steps = int(np.ceil(test_flow.n / eval_batch_size))
epochs = 10

print('BATCH_SIZE: {} EPOCHS: {}'.format(batch_size, epochs))
print(f'train {train_steps} steps')
print(f'valid {valid_steps} steps')
print(f'test {test_steps} steps')

file_path='model.h5'
checkpoint = ModelCheckpoint(
    file_path, monitor='val_loss', verbose=1, 
    save_best_only=True, 
    save_weights_only=True,
    mode='min'
)
early = EarlyStopping(monitor='val_loss', mode='min', patience=30)
callbacks_list = [checkpoint, early]

K.set_value(model.optimizer.lr, 0.0001)

gc.collect();
history = model.fit_generator(
    generator=train_flow, 
    steps_per_epoch=train_steps,
    validation_data=valid_flow,
    validation_steps=valid_steps,
    epochs=epochs, 
    shuffle=False,
    verbose=1,
    callbacks=callbacks_list
)
eval_res = pd.DataFrame(history.history)
eval_res.to_csv('eval_res_init.csv', index=False)
for c in ['acc', 'loss']:
    eval_res[[c, f'val_{c}']].plot(figsize=[18, 4]);
    plt.xlabel('Epoch'); plt.ylabel(c);
    plt.title(c); plt.grid();
model.load_weights('model.h5')
print(f'encoder totally {len(encoder.layers)} layers')
print(f'model totally {len(model.layers)} layers')
last_5_layer_names = [_.name for _ in encoder.layers[::-1][:5]]
print('encoder last 5 layers:', last_5_layer_names)
for l in model.layers[::-1][6:12]:
    print('Fine-tune', l.name) #, l.trainable
    l.trainable = True
epochs = 15

print('BATCH_SIZE: {} EPOCHS: {}'.format(batch_size, epochs))
print(f'train {train_steps} steps')
print(f'valid {valid_steps} steps')
print(f'test {test_steps} steps')

file_path='model.h5'
checkpoint = ModelCheckpoint(
    file_path, monitor='val_loss', verbose=1, 
    save_best_only=True, 
    save_weights_only=True,
    mode='min'
)
early = EarlyStopping(monitor='val_loss', mode='min', patience=30)
callbacks_list = [checkpoint, early]

K.set_value(model.optimizer.lr, 0.00001)
K.set_value(model.optimizer.decay, 1e-9)

gc.collect();
history = model.fit_generator(
    train_flow, 
    steps_per_epoch=train_steps,
    validation_data=valid_flow,
    validation_steps=valid_steps,
    epochs=epochs, 
    verbose=1,
    callbacks=callbacks_list
)
eval_res = pd.DataFrame(history.history)
eval_res.to_csv('eval_res_finetune.csv', index=False)
for c in ['acc', 'loss']:
    eval_res[[c, f'val_{c}']].plot(figsize=[18, 4]);
    plt.xlabel('Epoch'); plt.ylabel(c);
    plt.title(c); plt.grid();
model.load_weights('model.h5')
# final_state_model = Model(model.inputs, model.get_layer('final_state').output)
# valid_state = final_state_model.predict_generator(valid_flow, steps=valid_steps, verbose=1)
pred_val = model.predict_generator(valid_flow, steps=valid_steps, verbose=1)
pred_val.shape, valid_flow.classes.shape
pred_val = pred_val.ravel()
y_valid =  valid_flow.classes.copy()
from sklearn.metrics import log_loss, accuracy_score
val_loss = log_loss(y_valid, pred_val)
val_acc = accuracy_score(y_valid, np.round(pred_val))
print(f'valid loss: {val_loss}\t valid accuracy: {val_acc}')
pred_test = model.predict_generator(test_flow, steps=test_steps, verbose=1)
pred_test = pred_test.ravel()
np.save('valid_pred.npy', pred_val)
np.save('test_pred.npy', pred_test)
evals.loc[evals['is_test']==1, 'img_id'].shape
mask = evals['is_test']==1
sub = {
    'id': evals.loc[mask, 'img_id'].values.astype('int'),
    'label': pred_test,
}
sub = pd.DataFrame(sub).sort_values(by='id').reset_index(drop=True)
sub['label'] = 1 - sub['label']
sub.head()
subname = f'resnet50ft_{val_loss:.6f}.csv'
sub.to_csv(subname, index=False)
print(subname, 'saved')
