import numpy as np
import cv2
from tqdm import tqdm #Progress bar
import os
TRAIN_IMAGE_DIR = '../input/train/images/' #img_id is x(input)
TRAIN_MASK_DIR = '../input/train/masks/'   #rle_mask is y(output)
TEST_IMAGE_DIR = '../input/test/images/'

train_d = os.listdir(TRAIN_IMAGE_DIR) 
x = [np.array(cv2.imread(TRAIN_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(train_d)] #cv2.imread=openCV image read
x = np.array(x)/255

y = [np.array(cv2.imread(TRAIN_MASK_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(train_d)]
y = np.array(y)/255
print(x.shape,y.shape)
x=np.expand_dims(x,axis=3) #EXPAND DIM OF X AND INSERT NEW AXIS @ 3 
y=np.expand_dims(y,axis=3)
print(x.shape,y.shape)
from keras.layers import MaxPooling2D,Conv2D,Dense,Dropout,Input,Conv2DTranspose,Concatenate
from keras.models import Sequential,Model
from keras.optimizers import Adam
import keras
def conv_block(num_layers,inp,units,kernel_size):
    x = input
    for l in range(num_layers): #repeat 32-24-16 ----4 times
        x = Conv2D(units, kernel_size=kernel_size,padding='SAME',activation='relu')(x)
    return x
input = Input(shape=(101,101,1))
cnn1 = conv_block(5,input,32,3)
cnn2 = conv_block(5,input,24,5)
cnn3 = conv_block(5,input,16,7)
cnn4 = conv_block(5,input,8,9)
cnn5 = conv_block(5,input,4,11)
concat = Concatenate()([cnn1,cnn2,cnn3,cnn4,cnn5])

d1 = Conv2D(16,1,activation='relu')(concat)
out = Conv2D(1,1,activation='sigmoid')(d1) #filter_size = 1 ,so that 1x1 filter will scan over for more learning

model = Model(inputs=[input], outputs=[out])
adam=Adam(lr=0.001)
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary() # start_dim=(101,101,1) == #end_dim=(101,101,1)
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
model.fit(x,y,epochs=50,batch_size=128,validation_split=0.2,verbose=True)
#test_data
test_d=os.listdir(TEST_IMAGE_DIR)

x_test = [np.array(cv2.imread(TEST_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(test_d)]
x_test = np.array(x_test)/255
print(x_test.shape)
x_test = np.expand_dims(x_test,axis=3)
print(x_test.shape)
predict=model.predict(x_test,verbose=True)
#copy-pasted for rendering
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1
            
    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

pred_dict = {fn[:-4]:RLenc(np.round(predict[i,:,:,0])) for i,fn in tqdm(enumerate(test_d))}
import pandas as pd
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')

