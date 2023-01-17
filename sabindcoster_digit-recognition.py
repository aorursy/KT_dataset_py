# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on Tpu" , tpu.master())
except ValueError as e:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:" , strategy.num_replicas_in_sync)

BATCH_SIZE = 16 * strategy.num_replicas_in_sync 
(X_train , Y_train) ,(X_test , Y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
X_train = X_train / 255
X_train = X_train.reshape((-1,28,28))
print("Training Set Shape",X_train.shape)
plt.imshow(X_train[0])
print("Shape of image before padding",X_train[0].shape)

X_test = X_test / 255
X_test = X_test.reshape((-1,28,28))
print("Shape Of test data",X_test.shape)
plt.imshow(X_test[0])

# reshaping the lables to the no of classes i.e 10
def reshape_labels(Y):
    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I

Y_train = reshape_labels(Y_train)
print(Y_train.shape)
Y_test = reshape_labels(Y_test)
print(Y_test.shape)


# reshaping
X_train = np.array([np.pad(img,(2)) for img in X_train])
X_test = np.array([np.pad(img,(2)) for img in X_test])
print("Training Image Shape",X_train.shape)
plt.imshow(X_train[0])
print("Training Image Shape",X_test.shape)
plt.imshow(X_test[0])
print("Images Shape after Padding",X_train[0].shape)
# repeating channel 1 three times 
X_train = np.repeat(np.expand_dims(X_train,axis=3),3,axis=3)
X_test = np.repeat(np.expand_dims(X_test,axis=3),3,axis=3)
print("Shape After Adding Channel",X_train.shape)
print("Shape After Adding Channel",X_test.shape)
!pip install efficientnet

import efficientnet.tfkeras as efn

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss' , patience = 3)
learning_rate_start =  0.00001

learning_rate_max = 0.00005 * strategy.num_replicas_in_sync

learning_rate_min = 0.0001

learning_rate_boost_epochs = 3

learning_rate_sustain_epochs = 0 

learning_rate_decay = 0.9

def learning_rate_schedule(epoch):
    if epoch < learning_rate_boost_epochs:
        
        lr = (learning_rate_max - learning_rate_start) / learning_rate_boost_epochs * epoch + learning_rate_start
        
    elif epoch < learning_rate_boost_epochs + learning_rate_sustain_epochs:
        
        lr = learning_rate_max
        
    else:
        
        lr = (learning_rate_max - learning_rate_min) * learning_rate_decay **(epoch - learning_rate_boost_epochs - learning_rate_sustain_epochs) + learning_rate_min
        
    return lr


learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule , verbose = True)
with strategy.scope():
    transfer = efn.EfficientNetB7(include_top = False , input_shape=(32,32,3) , weights='noisy-student' , pooling='avg')
    for layers in transfer.layers:
        layers.trainable = True
    # adding flatten layer
    X = tf.keras.layers.Flatten()(transfer.layers[-1].output)
    X = tf.keras.layers.Dropout(0.2)(X)
    # adding softmax layer
    X = tf.keras.layers.Dense(10 , activation = 'softmax')(X)

    model = tf.keras.models.Model(inputs = transfer.input , outputs = X)

    model.compile(loss='categorical_crossentropy' , optimizer = tf.keras.optimizers.Adam() , metrics=['accuracy'])

model.summary()
model.fit(x = X_train , y= Y_train , epochs = 50 , batch_size = BATCH_SIZE , validation_data = (X_test , Y_test) ,callbacks=[learning_rate_callback])
test  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv") 

X_test = test.values
del(test)
X_test = X_test / 255
X_test = np.reshape(X_test , (-1,28,28))
X_test = np.array([np.pad(img,(2)) for img in X_test])
print("Test Image Shape",X_test.shape)
plt.imshow(X_test[0])
print("Images Shape after Padding",X_test[0].shape)
# repeating channel 1 three times 
X_test = np.repeat(np.expand_dims(X_test,axis=3),3,axis=3)
print("Shape After Adding Channel",X_test.shape)

predict = model.predict(X_test)
predict[3]
max_predict = np.amax(predict)
print(max_predict)

for idx,res in enumerate(predict[3]):
    if round(res) > 0:
        print(idx)
plt.imshow(X_test[3])
final_preds = []
for pred_list in predict:
    idx = np.where(pred_list == pred_list.max())
    final_preds.append(idx[0][0])
pred_id = [num for num in range(1,len(final_preds) + 1)]
data = pd.DataFrame({'Imageid':pred_id , 'Label' : final_preds})
data.to_csv('submission.csv', index=False)
data
