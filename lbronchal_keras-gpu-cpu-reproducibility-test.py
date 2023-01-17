import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import os
import multiprocessing
import numpy as np 
import pandas as pd 
import tensorflow as tf
import random as rn
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from keras.models import Model, Sequential
data_train = pd.read_csv("../input/train.csv")
y = data_train['label'].astype('int32')
X = data_train.drop('label', axis=1).astype('float32')
X = X.values.reshape(-1, 28, 28, 1)
y = to_categorical(y)
SEED = 1
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train /= 255
X_val /= 255
def create_model():
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(3, 3),
                     strides=2,
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Dropout(0.5))
    model.add(Conv2D(30, kernel_size=(3, 3), strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def execute_test(mode='gpu', n_repeat=5, seed=1):
    n_epochs = 2
    batch_size = 128    
    num_cores=1  
    
    if type(seed)==int:
        seed_list = [seed]*n_repeat
    else:
        if (type(seed) in [list, tuple]) and (len(seed) >= n_repeat): 
            seed_list = seed
        else:
            raise ValueError('seed must be an integer or a list/tuple the lenght n_repeat')
        
    if mode=='gpu':
        num_GPU = 1
        num_CPU = 1
        gpu_name = tf.test.gpu_device_name()
        if (gpu_name != ''):
            gpu_message = gpu_name  
            print("Testing with GPU: {}".format(gpu_message))
        else:
            gpu_message = "ERROR <GPU NO AVAILABLE>"
            print("Testing with GPU: {}".format(gpu_message))
            return  
    else:    
        num_CPU = 1
        num_GPU = 0
        max_cores = multiprocessing.cpu_count()
        print("Testing with CPU: using {} core ({} availables)".format(num_cores, max_cores))

    results = []    
    for i in range(n_repeat):
        os.environ['PYTHONHASHSEED'] = '0'                      
        np.random.seed(seed_list[i])
        rn.seed(seed_list[i])
        tf.set_random_seed(seed_list[i])

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                      inter_op_parallelism_threads=num_cores, 
                                      allow_soft_placement=True,
                                      device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        model = create_model()

        model.fit(X_train, y_train, batch_size = batch_size, epochs=n_epochs, verbose=0)
        eval_acc = model.evaluate(x=X_val, y=y_val, batch_size=batch_size, verbose=0)[1]
        results.append(eval_acc)
        print("Accuracy Test {}: {}".format(i, eval_acc))
    K.clear_session()
    return results
res_cpu_same_seed = execute_test(mode='cpu', n_repeat=5, seed=SEED)
print("mean: {}".format(np.mean(res_cpu_same_seed)))
print("std: {}".format(np.std(res_cpu_same_seed)))    
_ = execute_test(mode='cpu', n_repeat=1, seed=SEED*2)
_ = execute_test(mode='cpu', n_repeat=1, seed=SEED*10)
res_gpu_same_seed = execute_test(mode='gpu', n_repeat=10, seed=SEED)
print("mean: {}".format(np.mean(res_gpu_same_seed)))
print("std: {}".format(np.std(res_gpu_same_seed)))   
res_gpu_diff_seed = execute_test(mode='gpu', n_repeat=10, seed=[i*10 for i in range(10)])
print("mean: {}".format(np.mean(res_gpu_diff_seed)))
print("std: {}".format(np.std(res_gpu_diff_seed)))   
model = create_model()
model.fit(X_train, y_train, batch_size = 128, epochs=2, verbose=0)
for i in range(5):
    eval_acc = model.evaluate(x=X_val, y=y_val, batch_size=128, verbose=0)[1]
    print("Accuracy Test: {}".format(eval_acc))