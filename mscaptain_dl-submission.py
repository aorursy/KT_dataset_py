import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

from tqdm import tqdm_notebook

import cv2



import keras

from keras.layers import UpSampling2D, Conv2D, Activation

from keras import Model
tr = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

print(len(tr))

tr.head()
# Only ClassId=4



df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)

df_train = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')].reset_index(drop=True)

print(len(df_train))

df_train.head()
def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )
img_size = 256
def keras_generator(batch_size):

    while True:

        x_batch = []

        y_batch = []

        

        for i in range(batch_size):            

            fn = df_train['ImageId_ClassId'].iloc[i].split('_')[0]

            img = cv2.imread( '../input/severstal-steel-defect-detection/train_images/'+fn )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            

            

            mask = rle2mask(df_train['EncodedPixels'].iloc[i], img.shape)

            

            img = cv2.resize(img, (img_size, img_size))

            mask = cv2.resize(mask, (img_size, img_size))

            

            x_batch += [img]

            y_batch += [mask]

                                    

        x_batch = np.array(x_batch)

        y_batch = np.array(y_batch)



        yield x_batch, np.expand_dims(y_batch, -1)
for x, y in keras_generator(4):

    break

    

print(x.shape, y.shape)
plt.imshow(x[3])
plt.imshow(np.squeeze(y[3]))
from keras.applications.vgg19 import VGG19

base_model = VGG19(weights=None, input_shape=(img_size,img_size,3), include_top=False)

base_model.load_weights('../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
base_model.trainable = False
base_out = base_model.output

up = UpSampling2D(32, interpolation='bilinear')(base_out)

conv = Conv2D(1, (1, 1))(up)

conv = Activation('sigmoid')(conv)



model = Model(input=base_model.input, output=conv)



model.compile(keras.optimizers.Adam(lr=0.0001), 'binary_crossentropy')
%%time

batch_size = 16

model.fit_generator(keras_generator(batch_size),

              steps_per_epoch=100,                    

              epochs=5,                    

              verbose=1,

              shuffle=True)
pred = model.predict(x)

plt.imshow(np.squeeze(pred[3]))
testfiles=os.listdir("../input/severstal-steel-defect-detection/test_images/")

len(testfiles)
%%time

test_img = []

for fn in tqdm_notebook(testfiles):

        img = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+fn )

        img = cv2.resize(img,(img_size,img_size))       

        test_img.append(img)
%%time

predict = model.predict(np.asarray(test_img))

print(len(predict))
def mask2rle(img):

    tmp = np.rot90( np.flipud( img ), k=3 )

    rle = []

    lastColor = 0;

    startpos = 0

    endpos = 0



    tmp = tmp.reshape(-1,1)   

    for i in range( len(tmp) ):

        if (lastColor==0) and tmp[i]>0:

            startpos = i

            lastColor = 1

        elif (lastColor==1)and(tmp[i]==0):

            endpos = i-1

            lastColor = 0

            rle.append( str(startpos)+' '+str(endpos-startpos+1) )

    return " ".join(rle)
%%time

pred_rle = []

for img in predict:      

    img = cv2.resize(img, (1600, 256))

    tmp = np.copy(img)

    tmp[tmp<np.mean(img)] = 0

    tmp[tmp>0] = 1

    pred_rle.append(mask2rle(tmp))
img_t = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ testfiles[4])

plt.imshow(img_t)
mask_t = rle2mask(pred_rle[4], img.shape)

plt.imshow(mask_t)
sub = pd.read_csv( '../input/severstal-steel-defect-detection/sample_submission.csv', converters={'EncodedPixels': lambda e: ' '} )

sub.head()
%%time

for fn, rle in zip(testfiles, pred_rle):

    sub['EncodedPixels'][(sub['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == fn) & \

                        (sub['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4'))] = rle
img_s = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ sub['ImageId_ClassId'][47].split('_')[0])

plt.imshow(img_s)
mask_s = rle2mask(sub['EncodedPixels'][47], (256, 1600))

plt.imshow(mask_s)
sub.to_csv('submission.csv', index=False)