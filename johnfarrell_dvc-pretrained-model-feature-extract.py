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
H, W, C = 224, 224, 3 #pretrained model requires at least 197
batch_size = 32
eval_batch_size = batch_size * 4
batch_size = eval_batch_size
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

train_gen = ImageDataGenerator(
    #rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #channel_shift_range=0.2,
    #vertical_flip=True,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
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
# MODEL_NAME = f'resnet50_weights_tf_dim_ordering_tf_kernels_notop'
# MODEL_PATH = f'../input/keras-pretrained-models/{MODEL_NAME}.h5'
MODEL_NAME = f'vgg16_weights_tf_dim_ordering_tf_kernels_notop'
MODEL_PATH = f'../input/keras-pretrained-models/{MODEL_NAME}.h5'
from keras.applications.vgg16 import VGG16
def get_pretrained_model(weight_path=MODEL_PATH, trainable=False):
    input_shape = (H, W, C)
    #base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    base_model.load_weights(weight_path)
    for l in base_model.layers:
        l.trainable = trainable
    return base_model

encoder = get_pretrained_model(weight_path=MODEL_PATH, trainable=False)
for bx, by in valid_flow: break
tmp = encoder.predict(bx)
tmp.shape
train_steps = int(np.ceil(train_flow.n / batch_size))
valid_steps = int(np.ceil(valid_flow.n / eval_batch_size))
test_steps = int(np.ceil(test_flow.n / eval_batch_size))
print(f'train {train_steps} steps')
print(f'valid {valid_steps} steps')
print(f'test {test_steps} steps')
%%time
X_train = []
y_train = []
for i in tqdm(range(train_steps)):
    bx,by = next(train_flow)
    X_train.extend(encoder.predict(bx))
    y_train.extend(by)
np.stack(X_train, 0).shape, np.stack(y_train).shape
%%time
X_valid = []
y_valid = []
for i in tqdm(range(valid_steps)):
    bx,by = next(valid_flow)
    X_valid.extend(encoder.predict(bx))
    y_valid.extend(by)
# %%time
# X_test = []
# y_test = []
# for i in tqdm(range(test_steps)):
#     bx,by = next(test_flow)
#     X_test.extend(encoder.predict(bx))
#     y_test.extend(by)
def get_stacked_data(X, y):
    X, y = np.stack(X, 0), np.stack(y)
    print(X.shape, y.shape)
    return X, y
X_train, y_train = get_stacked_data(X_train, y_train)
X_valid, y_valid = get_stacked_data(X_valid, y_valid)
# X_test, y_test = get_stacked_data(X_test, y_test)
def get_reshaped_data(X, y):
    X, y = X.reshape(X.shape[0], -1), y
    print(X.shape, y.shape)
    return X, y
X_train, y_train = get_reshaped_data(X_train, y_train)
X_valid, y_valid = get_reshaped_data(X_valid, y_valid)
# X_test, y_test = get_reshaped_data(X_test, y_test)
from scipy import sparse
def save_feature(savename, X, y):
    #np.save(f'X_{savename}.npy', X) #big
    sparse.save_npz(f'X_{savename}.npz', sparse.csr_matrix(X), compressed=True)
    np.save(f'y_{savename}.npy', y)
    
save_feature('train', X_train, y_train)
save_feature('valid', X_valid, y_valid)
# save_feature('test', X_test, y_test)
import keras.backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import optimizers, losses, activations, models
from keras.layers import Conv2D, Dense, Input, Flatten, Concatenate, Dropout, Activation
from keras.layers import BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras import applications
n_feature = X_train.shape[1]
n_feature
n_final_state = 32

def get_model(n_final_state, lr=1e-3, decay=1e-8):
    #input_shape = (H, W, C)
    input_shape = (n_feature,)
    
    input_x = Input(shape=input_shape)
    
    d1 = Dense(
        64, activation='relu'
    )(input_x)
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
epochs = 10

print('BATCH_SIZE: {} EPOCHS: {}'.format(batch_size, epochs))

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
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    validation_data=(X_valid, y_valid),
    epochs=epochs, 
    verbose=1,
    shuffle=False,
    callbacks=callbacks_list
)
model.load_weights(file_path)
pred_val = model.predict(X_valid)
pred_val = pred_val.ravel()
from sklearn.metrics import log_loss, accuracy_score
val_loss = log_loss(y_valid, pred_val)
val_acc = accuracy_score(y_valid, np.round(pred_val))
print(f'valid loss: {val_loss}\t valid accuracy: {val_acc}')
np.save('valid_pred.npy', pred_val)
# np.save('test_pred.npy', pred_test)
# mask = evals['is_test']==1
# sub = {
#     'id': evals.loc[mask, 'img_id'].values.astype('int'),
#     'label': pred_test,
# }
# sub = pd.DataFrame(sub).sort_values(by='id').reset_index(drop=True)
# sub['label'] = 1 - sub['label']

# subname = f'resnet50ft_{val_loss:.6f}.csv'
# sub.to_csv(subname, index=False)
# print(subname, 'saved')