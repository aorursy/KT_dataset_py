!conda install -c conda-forge gdcm -y
!pip install pylibjpeg pylibjpeg-libjpeg

import pydicom
import matplotlib.pyplot as plt

img = pydicom.read_file('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/8bc04c1a5852/b3f09354bb04/ecdfcd5104b6.dcm')
plt.imshow(img.pixel_array)
plt.show()
from PIL import Image
import numpy as np

img = pydicom.read_file('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/8bc04c1a5852/b3f09354bb04/ecdfcd5104b6.dcm')

arr = img.pixel_array.astype(np.float32)

arr -= arr.min()
arr *= 1./arr.max()
arr *= 255
arr = arr.astype(np.uint8)

image = Image.merge("RGB", (Image.fromarray(arr[0::2,0::2]), Image.fromarray(arr[0::2,1::2]), Image.fromarray(arr[1::2,1::2]))).resize((224, 224))

arr = np.array(image)

plt.imshow(arr)
plt.show()

np.save("arr.npy", arr)
!ls -l arr.npy
# https://qiita.com/sasayabaku/items/9e376ba8e38efe3bcf79

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard
model = Sequential()

model.add(Conv2D(512,1,input_shape=(512,512,1)))
model.add(Activation('relu'))

model.add(Conv2D(512,1))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,1))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024//300))
model.add(Activation('relu'))
model.add(Dropout(0.99))

model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=["accuracy"])

model.summary()

plot_model(model, to_file='./model.png')
LABEL=4
from keras.models import model_from_json
model_arc_str = open('./cnn_model.'+str(LABEL)+'.json').read()
model = model_from_json(model_arc_str)
model.load_weights('./cnn_model_weight.'+str(LABEL)+'.hdf5')
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=["accuracy"])
import csv
import pydicom
# import matplotlib.pyplot as plt
import numpy as np

def storeFileNames(label=4, interval=100, maxSize=2000000, offset=0):
    PRINT_INT=10000
    global xf_train, y_train
    with open('/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv') as f:
        reader = csv.reader(f)

        count = 0
        for index, row in enumerate(reader):
            #print(row[label])
            #if count%PRINT_INT==0 and (index==0 or index>interval):
            #    print(index, ":", count, "/", maxSize)
            if index==0:
                continue
            if (index+offset)%interval!=0:
                continue
            if count>=maxSize:
                return
            count += 1
            y_train = np.append(y_train, float(row[label]))
            #xf_train.append('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/'+row[0]+'/'+row[1]+'/'+row[2]+'.dcm')
            xf_train.append(row[0]+'/'+row[1]+'/'+row[2])

def getPixels(fileName):
    #print("AAA"+fileName)
    img = pydicom.read_file('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/'+fileName+'.dcm')
    img = img.pixel_array.astype(np.float32)
    img -= img.min()
    img /= img.max()
    #print(fileName, len(img))
    return img
                            
def getBatch(batch_size):
    # https://aotamasaki.hatenablog.com/entry/2018/08/27/124349

    global xf_train, y_train
    SIZE = len(xf_train)

    n_batches = SIZE//batch_size
    #print(SIZE, "SIZE")
    #print(batch_size, "batch_size")
    #print(n_batches, "n_batches")

    i = 0
    while (i < n_batches):
        #print("doing", i, "/", n_batches)
        y_batch = y_train[(i * batch_size):(i * batch_size + batch_size)]
        
        x_batch_name = xf_train[(i * batch_size):(i * batch_size + batch_size)]
        #print(i, x_batch_name, (i * batch_size), (i * batch_size + batch_size))

        x_batch = np.array([getPixels(fileName)
                            for fileName in x_batch_name]).reshape(batch_size, 512, 512, 1)
        i += 1
        yield x_batch, y_batch
LABEL = 4 # 4-16
INTERVAL = 17
OFFSET = LABEL%INTERVAL
SIZE = 100000
N_BATCH = 4
N_EPOCHS = 1

xf_train = []
y_train = np.array([])

storeFileNames(LABEL, INTERVAL, SIZE, OFFSET)

for epoch in range(N_EPOCHS):
    print("EPOCH", epoch, "/", N_EPOCHS)
    acc = []
    loss = []
    
    index = -1
    for X_batch, Y_batch in getBatch(N_BATCH):
        # print(len(X_batch))
        index += 1
        if index%100==0:
            print(" BATCH:", index, "/", SIZE//N_BATCH)
        model.train_on_batch(X_batch, Y_batch)
        score = model.evaluate(X_batch, Y_batch, verbose=0)
        #print("batch accuracy:", score[1], index)
        loss.append(score[0])
        acc.append(score[1])
    print("Train loss", np.mean(loss), "train accuracy", np.mean(acc))
#    score = model.evaluate(X_test, Y_test)
#    print("Test loss:", score[0])
#    print("Test accuracy:", score[1])

import os
json_string = model.to_json()
open(os.path.join('./', 'cnn_model.'+str(LABEL)+'.json'), 'w').write(json_string)
!ls -l ./cnn_model*.json

model.save_weights(os.path.join('./', 'cnn_model_weight.'+str(LABEL)+'.hdf5'))
!ls -l ./cnn_model_weight.*.hdf5