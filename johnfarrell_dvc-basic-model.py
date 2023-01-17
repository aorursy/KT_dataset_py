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
H, W, C = 150, 150, 3
batch_size = 32
eval_batch_size = batch_size * 4
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    #channel_shift_range=0.2,
    horizontal_flip=True,
    #vertical_flip=True,
    #rescale=1./255,#!!!!NO!
    preprocessing_function=lambda x:(x-x.mean())/x.std()
)
test_gen = ImageDataGenerator(
    #rescale=1./255,
    preprocessing_function=lambda x:(x-x.mean())/x.std()
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
def set_data_flow(flow, eval_mode, shuffle=True, valid_fold=0, n_valid=128*16, evals=evals):
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
        indexes = np.arange(flow.samples)
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
import keras.backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import optimizers, losses, activations, models
from keras.layers import Conv2D, Dense, Input, Flatten, Concatenate, Dropout, Activation
from keras.layers import BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras import applications
n_final_state = 32

def get_model(n_final_state, lr=1e-3, decay=1e-8):
    input_shape = (H, W, C)
    
    input_x = Input(shape=input_shape)
    
    c1 = Conv2D(32, (3, 3))(input_x)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3))(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3))(c2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = MaxPooling2D((2, 2))(c3)
    
    flat = Flatten()(c3)
    
    d1 = Dense(
        64, activation='relu'
    )(flat)
    #d1 = Dropout(0.5)(d1)
    d1 = BatchNormalization()(d1)
    
    final_state = Dense(
        n_final_state, activation='relu', name='final_state'
    )(d1)
    
    x = Dropout(0.5)(final_state)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_x, outputs=outputs)
    optimizer=optimizers.Adam(lr=lr, decay=decay)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

model = get_model(n_final_state=n_final_state)
model.summary()
train_steps = int(np.ceil(train_flow.n / batch_size))
valid_steps = int(np.ceil(valid_flow.n / eval_batch_size))
test_steps = int(np.ceil(test_flow.n / eval_batch_size))
epochs = 50

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

K.set_value(model.optimizer.lr, 0.0005)

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
eval_res.to_csv('eval_res.csv', index=False)
for c in ['acc', 'loss']:
    eval_res[[c, f'val_{c}']].plot(figsize=[18, 6]);
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
subname = f'basic_{val_loss:.6f}.csv'
sub.to_csv(subname, index=False)
print(subname, 'saved')
