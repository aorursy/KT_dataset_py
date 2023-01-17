import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
import gc
import matplotlib.pyplot as plt
from keras.utils import Sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Dropout
print('Reading datasets')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('Merging test and train')
test['target'] = np.nan
train = train.append(test).reset_index() # merge train and test
del test
print('Done, shape=',np.shape(train))
def rank_gauss(x, title=None):
    #Trying to implement rankGauss in python, here are my steps
    # 1) Get the index of the series
    # 2) sort the series
    # 3) standardize the series between -1 and 1
    # 4) apply erfinv to the standardized series
    # 5) create a new series using the index
    # Am i missing something ??
    # I subtract mean afterwards. And do not touch 1/0 (binary columns). 
    # The basic idea of this "RankGauss" was to apply rank trafo and them shape them like gaussians. 
    # Thats the basic idea. You can try your own variation of this.
    
    if(title!=None):
        fig, axs = plt.subplots(3, 3)
        fig.suptitle(title)
        axs[0][0].hist(x)

    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    if(title!=None):
        print('1)', max(temp), min(temp))
        axs[0][1].hist(temp)
    rank_x = temp.argsort() / N
    if(title!=None):
        print('2)', max(rank_x), min(rank_x))
        axs[0][2].hist(rank_x)
    rank_x -= rank_x.mean()
    if(title!=None): 
        print('3)', max(rank_x), min(rank_x))
        axs[1][0].hist(rank_x)
    rank_x *= 2
    if(title!=None):
        print('4)', max(rank_x), min(rank_x))
        axs[1][1].hist(rank_x)
    efi_x = erfinv(rank_x)
    if(title!=None): 
        print('5)', max(efi_x), min(efi_x))
        axs[1][2].hist(efi_x)
    efi_x -= efi_x.mean()
    if(title!=None):
        print('6)', max(efi_x), min(efi_x))
        axs[2][0].hist(efi_x)
        plt.show()

    return efi_x

for i in train.columns:
    if i.endswith('cat'): # could be train[i].dtype == 'object' + labelencode, or maybe one hot encode...
        print('Categorical: ',i)
        train[i] = rank_gauss(train[i].values, i) # display rank gauss tranformation
    elif i.endswith('bin'):
        print('Binary: ',i)
    else:
        print('Numeric: ',i)
# i'm doing this cause i don't know if some keras backend have threading problems...
import threading
class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
    def acquire_read(self):
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()
    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()
    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()
    def release_write(self):
        self._read_ready.release()
from math import ceil
class DAESequence(Sequence):
    def __init__(self, df, batch_size=128, random_cols=.15, random_rows=1, use_cache=False, use_lock=False, verbose=True):
        self.df = df.values.copy()     # ndarray baby
        self.batch_size = int(batch_size)
        self.len_data = df.shape[0]
        self.len_input_columns = df.shape[1]
        if(random_cols <= 0):
            self.random_cols = 0
        elif(random_cols >= 1):
            self.random_cols = self.len_input_columns
        else:
            self.random_cols = int(random_cols*self.len_input_columns)
        if(self.random_cols > self.len_input_columns):
            self.random_cols = self.len_input_columns
        self.random_rows = random_rows
        self.cache = None
        self.use_cache = use_cache
        self.use_lock = use_lock
        self.verbose = verbose
        
        self.lock = ReadWriteLock()
        self.on_epoch_end()

    def on_epoch_end(self):
        if(not self.use_cache):
            return
        if(self.use_lock):
            self.lock.acquire_write()
        if(self.verbose):
            print("Doing Cache")
        self.cache = {}
        for i in range(0, self.__len__()):
            self.cache[i] = self.__getitem__(i, True)
        if(self.use_lock):
            self.lock.release_write()
        gc.collect()
        if(self.verbose):
            print("Done")

    def __len__(self):
        return int(ceil(self.len_data / float(self.batch_size)))

    def __getitem__(self, idx, doing_cache=False):
        if(not doing_cache and self.cache is not None and not (self.random_cols <=0 or self.random_rows<=0)):
            if(idx in self.cache.keys()):
                if(self.use_lock):
                    self.lock.acquire_read()
                ret0, ret1 = self.cache[idx][0], self.cache[idx][1]
                if(self.use_lock):
                    self.lock.release_read()
                if (not doing_cache and self.verbose):
                    print('DAESequence Cache ', idx)
                return ret0, ret1
        idx_end = min(idx + self.batch_size, self.len_data)
        cur_len = idx_end - idx
        rows_to_sample = int(self.random_rows * cur_len)
        input_x = self.df[idx: idx_end]
        if (self.random_cols <= 0 or self.random_rows <= 0 or rows_to_sample<=0):
            return input_x, input_x # not dae
        # here start the magic
        random_rows = np.random.randint(low=0, high=self.len_data-rows_to_sample, size=rows_to_sample)
        random_rows[random_rows>idx] += cur_len # just to don't select twice the current rows
        cols_to_shuffle = np.random.randint(low=0, high=self.len_input_columns, size=self.random_cols)
        noise_x = input_x.copy()
        noise_x[0:rows_to_sample, cols_to_shuffle] = self.df[random_rows[:,None], cols_to_shuffle]
        if(not doing_cache and self.verbose):
            print('DAESequence ', idx)
        return noise_x, input_x

print("Create Model")
dae_data = train[train.columns.drop(['id','target'])] # only get "X" vector

# reduce data size, we are in kaggle =)
dae_data = dae_data[0:1000]

len_input_columns, len_data = dae_data.shape[1], dae_data.shape[0]
NUM_GPUS=1
#kernel_initializer='Orthogonal'  # this one give non NaN more often than others 

# from https://kaggle2.blob.core.windows.net/forum-message-attachments/250927/8325/nn.cfg.log
#L0: 221(in)-1500 'r'ReLU  lRate:0.003 lRateDecay:0.995 regL2:0 regL1:0 dropout:0  w:222x1500  out(x3):1501x128 (0.00210051 GB) init..(uni:1 sp:1)[min|max|mean|std:-0.0672672|0.0672671|-4.74564e-05|0.0388202]
#L1: 1500(in)-1500 'r'ReLU  lRate:0.003 lRateDecay:0.995 regL2:0 regL1:0 dropout:0  w:1501x1500  out(x3):1501x128 (0.00977451 GB) init..(uni:1 sp:1)[min|max|mean|std:-0.0258199|0.0258199|8.51905e-06|0.0148989]
#L2: 1500(in)-1500 'r'ReLU  lRate:0.003 lRateDecay:0.995 regL2:0 regL1:0 dropout:0  w:1501x1500  out(x3):1501x128 (0.00977451 GB) init..(uni:1 sp:1)[min|max|mean|std:-0.0258199|0.0258199|8.51905e-06|0.0148989]
#L3: 1500(in)-221 'l'linear  lRate:0.003 lRateDecay:0.995 regL2:0 regL1:0 dropout:0  w:1501x221  out(x3):222x128 (0.00144055 GB) init..(uni:1 sp:1)[min|max|mean|std:-0.0258199|0.0258198|-1.80977e-05|0.0149005]

# is uni:1 = uniform? what about sp:1 ?
# std ~= sqrt(2 / (input+output))   (first layer)
# std ~= sqrt(1 / (input+output))   (others layer)
kernel_initializer_0=keras.initializers.RandomNormal(mean=-4.74564e-05, stddev=0.0388202, seed=None)   # sqrt(2/(221+1500)) = 0.0341 vs 0,0388
kernel_initializer_1=keras.initializers.RandomNormal(mean=8.51905e-06, stddev=0.0148989, seed=None)    # sqrt(1/(1500+1500)) = 0.018 vs 0,014
kernel_initializer_2=keras.initializers.RandomNormal(mean=8.51905e-06, stddev=0.0148989, seed=None)    # sqrt(1/(1500+1500)) = 0.018 vs 0,014
kernel_initializer_3=keras.initializers.RandomNormal(mean=-1.80977e-05, stddev=0.0149005, seed=None)   # sqrt(1/(1500+221)) = 0.024 vs 0,014

print("Input len=", len_input_columns, len_data)
model_dae = Sequential()
model_dae.add(Dense(units=len_input_columns*10, activation='relu', dtype='float32', name='Hidden1', input_shape=(len_input_columns,), kernel_initializer=kernel_initializer_0))
model_dae.add(Dense(units=len_input_columns*10, activation='relu', dtype='float32', name='Hidden2', kernel_initializer=kernel_initializer_1))
model_dae.add(Dense(units=len_input_columns*10, activation='relu', dtype='float32', name='Hidden3', kernel_initializer=kernel_initializer_2))
model_dae.add(Dense(units=len_input_columns, activation='linear', dtype='float32', name='Output', kernel_initializer=kernel_initializer_3))
model_opt = keras.optimizers.SGD(lr=0.003, decay=1-0.995, momentum=0, nesterov=False) # decay -> Oscar Takeshita comment

try:
    print('Loading model from file')
    model_dae = keras.models.load_model('DAE.keras.model.h5')
except Exception as e:
    print("Can't load previous fitting parameters and model", repr(e))
if(NUM_GPUS>1):
    try:
        multi_gpu_model = keras.utils.multi_gpu_model(model_dae, gpus=NUM_GPUS)
        multi_gpu_model.compile(loss='mean_squared_error', optimizer=model_opt)
        print("MULTI GPU MODEL")
        print(multi_gpu_model.summary())
    except Exception as e:
        print("Can't run multi gpu, error=", repr(e))
        model_dae.compile(loss='mean_squared_error', optimizer=model_opt)
        NUM_GPUS=0
else:
    model_dae.compile(loss='mean_squared_error', optimizer=model_opt)

print("BASE MODEL")
print(model_dae.summary())

from math import ceil
batch_size = 128
multi_process_workers = 2
if (NUM_GPUS > 1):
    multi_gpu_model.fit_generator(
        DAESequence(dae_data, batch_size=batch_size*NUM_GPUS, verbose=False),
        steps_per_epoch=int(ceil(dae_data.shape[0]/(batch_size*NUM_GPUS))),
        workers=multi_process_workers, use_multiprocessing=True if multi_process_workers>1 else False,
        epochs=1000,
        verbose=1,
        callbacks=[
            # keras.callbacks.LambdaCallback(on_epoch_end=lambda x,y: model_dae.save('DAE.keras.model.h5')) # save weights 
        ])
else: # single CPU/GPU
    model_dae.fit_generator(
        DAESequence(dae_data, batch_size=batch_size, verbose=False),
        steps_per_epoch=int(ceil(dae_data.shape[0]/batch_size)),
        epochs=1000,
        workers=multi_process_workers, use_multiprocessing=True if multi_process_workers>1 else False,
        verbose=1, callbacks=[
            # keras.callbacks.LambdaCallback(on_epoch_end=lambda x,y: model_dae.save('DAE.keras.model.h5')) # save weights
        ])
    
#model_dae.save('DAE.keras.model.h5') # save weights
plt.hist(model_dae.get_weights(), bins = 100)
plt.show()
# here we can measure the error from input -> output, it's not what we want, right?
from sklearn.metrics import mean_squared_error
if (NUM_GPUS > 1):
    dae_denoised_data = multi_gpu_model.predict(dae_data)
else:
    dae_denoised_data = model_dae.predict(dae_data)

print("DAE MSE from train data: ", mean_squared_error(dae_data, dae_denoised_data))
#after you have DAE fitted...
your_new_df=train[['id','target']].copy()
# let's cut it again...
# reduce data size, we are in kaggle =)
your_new_df = your_new_df[0:1000]

for i in ['1','2','3']:
    print('Hidden layer',i)
    columns_names = ['Hidden_'+str(i)+'_'+str(l) for l in range(0, len_input_columns*10)]
    for l in columns_names:
        your_new_df[l] = 0 # create columns (maybe it's not optimized)
    intermediate_layer_model = Model(inputs=model_dae.input, outputs=model_dae.get_layer('Hidden' + i).output)
    your_new_df[columns_names] = intermediate_layer_model.predict(dae_data)

print('DONE!')
print(your_new_df)
#your_new_df.to_pickle("DAE-hidden-features.pickle")

# now =) use it and win :P
