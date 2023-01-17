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
evals['path'] = evals['path'].apply(lambda x: x.replace('../input/', DATA_DIR))
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
def get_filenames_targets(eval_mode, valid_fold, n_valid, evals=evals):
    if eval_mode=='train':
        mask = (evals['is_test']==0) & (evals['eval_set']!=valid_fold)
    elif eval_mode=='valid':
        mask = (evals['is_test']==0) & (evals['eval_set']==valid_fold)
    elif eval_mode=='test':
        mask = (evals['is_test']==1)
    else:
        raise NotImplementedError
    filenames_arr = evals.loc[mask, 'path'].values
    target_arr = evals.loc[mask, 'target'].values
    return filenames_arr, target_arr
import keras.backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import load_img, img_to_array
import threading

class ImageIterator(Iterator):
    def __init__(
        self, img_generator, 
        eval_mode, valid_fold, n_valid=128*16, 
        evals=evals, 
        target_size=(H, W),
        num_class=1,
        batch_size=batch_size,
        shuffle=False,
        use_tta='fliplr',
        seed=42
    ):
        shuffle = True if eval_mode=='train' else False
        filenames_arr, target_arr = get_filenames_targets(
            eval_mode, valid_fold, n_valid, evals=evals
        )
        if eval_mode=='valid' and n_valid is not None:
            filenames_arr = filenames_arr[:n_valid]
            target_arr = target_arr[:n_valid]
        if shuffle:
            indexes = np.arange(flow.samples)
            np.random.permutatione(indexes)
            filenames_arr = filenames_arr[indexes]
            target_arr = target_arr[indexes]
        assert len(filenames_arr)==len(target_arr)
        n = len(filenames_arr)
        
        self.img_generator = img_generator
        self.class_indices = {'dog': 0, 'cat': 1}
        self.eval_mode = eval_mode
        self.n = n
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = filenames_arr
        self.classes = target_arr
        self.num_class = num_class
        self.seed = seed
        self.use_tta = use_tta
        self.lock = threading.Lock()
        
        super(ImageIterator, self).__init__(
            n=n, batch_size=batch_size, shuffle=shuffle, seed=seed
        )
    
    def _get_batches_of_transformed_samples(self, index_array):
        X = np.zeros((len(index_array),) + (H, W, C), dtype=K.floatx())
        Y = np.zeros((len(index_array), self.num_class), dtype=K.floatx())
        
        for i, idx in enumerate(index_array):
            with self.lock:
                x = load_img(
                    path=self.filenames[idx],
                    target_size=self.target_size
                )
                x = img_to_array(x)
                
            if self.use_tta=='fliplr':
                x = x[:, ::-1, :].copy()
                
            X[i] = x.astype(K.floatx())
            Y[i] = self.classes[idx].astype(K.floatx())
        ### latest keras version supports ImageDataGenerator.apply_transform
        return next(self.img_generator.flow(
           X, Y, batch_size=len(index_array), shuffle=False            
        ))
    
    def next(self):
        with self.lock: 
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)       
train_flow = ImageIterator(train_gen, 'valid', 1, batch_size=batch_size, use_tta=None)
valid_flow = ImageIterator(test_gen, 'valid', 0, batch_size=eval_batch_size, use_tta=None)
valid_tta_flow = ImageIterator(test_gen, 'valid', 0, batch_size=eval_batch_size, use_tta='fliplr')
test_flow = ImageIterator(test_gen, 'test', None, batch_size=eval_batch_size, use_tta=None)
test_tta_flow = ImageIterator(test_gen, 'test', None, batch_size=eval_batch_size, use_tta='fliplr')
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
print(f'train {train_steps} steps')
print(f'valid {valid_steps} steps')
print(f'test {test_steps} steps')
## https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/snapshot.py
## https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L146
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler

class SnapshotModelCheckpoint(Callback):
    """Callback that saves the snapshot weights of the model.
    Saves the model weights on certain epochs (which can be considered the
    snapshot of the model at that epoch).
    Should be used with the cosine annealing learning rate schedule to save
    the weight just before learning rate is sharply increased.
    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model', verbose=1):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            if self.verbose>0:
                print("Saved snapshot at weights/%s_%d.h5" % (self.fn_prefix, epoch))
                
class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.
    From the paper "Snapshot Ensembles: Train 1, Get M For Free" (https://openreview.net/pdf?id=BJYwwY9ll)
    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1, verbose=1):
        """
        Initialize a snapshot callback builder.
        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.verbose = verbose

    def get_callbacks(self, model_prefix='Model'):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.
        Args:
            model_prefix: prefix for the filename of the weights.
        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        if not os.path.exists('weights/'):
            os.makedirs('weights/')

        callback_list = [
            ModelCheckpoint('weights/%s-Best.h5' % model_prefix, monitor='val_acc',
                            save_best_only=True, save_weights_only=True),
            LearningRateScheduler(schedule=self._cosine_anneal_schedule),
            SnapshotModelCheckpoint(
                self.T, self.M, fn_prefix='weights/%s' % model_prefix, verbose=self.verbose
            )
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)
epochs = 50

print('BATCH_SIZE: {} EPOCHS: {}'.format(batch_size, epochs))

file_path='model.h5'
checkpoint = ModelCheckpoint(
    file_path, monitor='val_loss', verbose=1, 
    save_best_only=True, 
    save_weights_only=True,
    mode='min'
)
early = EarlyStopping(monitor='val_loss', mode='min', patience=30)
#callbacks_list = [checkpoint, early]
#K.set_value(model.optimizer.lr, 0.0005)
snapshot_cb = SnapshotCallbackBuilder(epochs, 10, 0.0005*2)
callbacks_list = snapshot_cb.get_callbacks(model_prefix='Model')

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
def predict(model, modelpath, data_flow, steps, workers=4, verbose=1):
    model.load_weights(modelpath)
    pred = model.predict_generator(
        generator=data_flow,
        steps=steps, 
        use_multiprocessing=True  if workers>1 else False, 
        workers=workers, 
        verbose=verbose
    )
    return pred
glob.glob('./weights/Model*.h5')
pred_val_best = predict(
    model, './weights/Model-Best.h5', 
    valid_flow, valid_steps, workers=4
)
pred_val_best_tta = predict(
    model, './weights/Model-Best.h5', 
    valid_tta_flow, valid_steps, workers=4
)
print(pred_val_best.shape, pred_val_best_tta.shape)
sns.distplot(pred_val_best)
sns.distplot(pred_val_best_tta)
plt.legend(['normal', 'fliplr']);
plt.grid();
from sklearn.metrics import log_loss, accuracy_score
pred_val_best = pred_val_best.ravel()
pred_val_best_tta = pred_val_best_tta.ravel()
y_valid =  valid_flow.classes.copy()
val_loss = log_loss(y_valid, pred_val_best)
val_acc = accuracy_score(y_valid, np.round(pred_val_best))
print(f'normal \nvalid loss: {val_loss:.6f}\t valid accuracy: {val_acc:.4%}')
val_loss = log_loss(y_valid, pred_val_best_tta)
val_acc = accuracy_score(y_valid, np.round(pred_val_best_tta))
print(f'tta \nvalid loss: {val_loss:.6f}\t valid accuracy: {val_acc:.4%}')

pred_val_best_avg = (pred_val_best+pred_val_best_tta)/2.

val_loss = log_loss(y_valid, pred_val_best_avg)
val_acc = accuracy_score(y_valid, np.round(pred_val_best_avg))
print(f'avg(normal tta) \nvalid loss: {val_loss:.6f}\t valid accuracy: {val_acc:.4%}')
print(sorted(glob.glob('./weights/Model*'))[2:-1])
pred_val_li = [predict(
    model, p, valid_flow, valid_steps, workers=4, verbose=1
) for p in sorted(glob.glob('./weights/Model*'))[2:-1]]
pred_val_tta_li = [predict(
    model, p, valid_tta_flow, valid_steps, workers=4, verbose=1
) for p in sorted(glob.glob('./weights/Model*'))[2:-1]]
for i,(p, p_tta) in enumerate(zip(pred_val_li, pred_val_tta_li)):
    print(i+2, 'th snapshot normal loss: {:.6f} acc: {:.6f}'.format(
        log_loss(y_valid, p),
        accuracy_score(y_valid, np.round(p))
    ))
    print(i+2, 'th snapshot tta    loss: {:.6f} acc: {:.6f}'.format(
        log_loss(y_valid, p_tta),
        accuracy_score(y_valid, np.round(p_tta))
    ))
X_meta = pred_val_li + pred_val_tta_li
X_meta = np.hstack(X_meta)
X_meta.shape
from sklearn.linear_model import LogisticRegressionCV
meta_model = LogisticRegressionCV(scoring='neg_log_loss')
meta_model.fit(X_meta, y_valid)
print(meta_model.coef_, meta_model.intercept_)
pred_val_ens_meta = meta_model.predict_proba(X_meta)[:, 1]
print('snapshot-ens meta loss: {:.6f} acc: {:.6f}'.format(
    log_loss(y_valid, pred_val_ens_meta),
    accuracy_score(y_valid, np.round(pred_val_ens_meta))
))
pred_val_ens_avg = X_meta[:, 3:].mean(1)
print('snapshot-ens avg loss: {:.6f} acc: {:.6f}'.format(
    log_loss(y_valid, pred_val_ens_avg),
    accuracy_score(y_valid, np.round(pred_val_ens_avg))
))
pred_test_best = predict(
    model, './weights/Model-Best.h5', 
    test_flow, test_steps, workers=4
)
pred_test_best_tta = predict(
    model, './weights/Model-Best.h5', 
    test_tta_flow, test_steps, workers=4
)
%%time
pred_test_li = [predict(
    model, p, test_flow, test_steps, workers=4, verbose=1
) for p in sorted(glob.glob('./weights/Model*'))[2:-1]]
pred_test_tta_li = [predict(
    model, p, test_tta_flow, test_steps, workers=4, verbose=1
) for p in sorted(glob.glob('./weights/Model*'))[2:-1]]
pred_test_best = pred_test_best.ravel()
pred_test_best_tta = pred_test_best_tta.ravel()
X_meta_test = pred_test_li + pred_test_tta_li
X_meta_test = np.hstack(X_meta_test)
pred_test_best_avg = (pred_test_best+pred_test_best_tta)/2.
pred_test_ens_meta = meta_model.predict_proba(X_meta_test)[:, 1]
pred_test_ens_avg = X_meta_test[:, 3:].mean(1)
def make_sub(pred_test, name, val_loss, evals=evals):
    mask = evals['is_test']==1
    sub = {
        'id': evals.loc[mask, 'img_id'].values.astype('int'),
        'label': pred_test,
    }
    sub = pd.DataFrame(sub).sort_values(by='id').reset_index(drop=True)
    sub['label'] = 1 - sub['label']
    subname = f'{name}_{val_loss:.6f}.csv'
    sub.to_csv(subname, index=False)
    print(subname, 'saved')
for pair in [
    ('best',     pred_test_best,     pred_val_best),
    ('best_tta', pred_test_best_tta, pred_val_best_tta),
    ('best_avg', pred_test_best_avg, pred_val_best_avg), 
    ('ens_meta', pred_test_ens_meta, pred_val_ens_meta),
    ('ens_avg',  pred_test_ens_avg,  pred_val_ens_avg)
]:
    name, p_t, p_v = pair
    val_loss = log_loss(y_valid, p_v)
    make_sub(p_t, name, val_loss, evals=evals)