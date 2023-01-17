import os
train_img_dir = '../input/train/images/'
train_mask_dir = '../input/train/masks/'
test_img_dir = '../input/test/images/'
train_img_names = [x.split('.')[0] for x in os.listdir(train_img_dir)]
train_img_names[:5]
train_num = len(train_img_names)
train_num
train_img_dict_i_to_names = dict()
train_img_dict_names_to_i = dict()
for i in range(train_num):
    train_img_dict_i_to_names[i] = train_img_names[i]
    train_img_dict_names_to_i[train_img_names[i]] = i
#train_img_dict_i_to_names
#train_img_dict_names_to_i
train_mask_names = [x.split('.')[0] for x in os.listdir(train_mask_dir)]
train_img_names == train_mask_names
from skimage.data import imread
train_img_shape = imread(train_img_dir + train_img_names[0]+'.png').shape
train_mask_shape = imread(train_mask_dir + train_img_names[0]+'.png').shape
train_img_shape,train_mask_shape
import numpy as np
train_img = np.zeros((train_num, train_img_shape[0], train_img_shape[1], train_img_shape[2]))
train_mask = np.zeros((train_num, train_mask_shape[0], train_mask_shape[1]))
for i in range(train_num):
    train_img[i] = i
    train_mask[i] = i
    train_img[i,:,:,:] = imread(train_img_dir + train_img_names[i]+'.png')
    train_mask[i,:,:] = imread(train_mask_dir + train_img_names[i]+'.png')
train_img.shape,train_mask.shape
train_img[100,:,:,0],train_mask[100,:,:]
chk = 0
for i in range(train_num):
    chk = chk + (train_img[i,:,:,0]*2-train_img[i,:,:,1]-train_img[i,:,:,2]).sum()
chk
train_img_mono = np.zeros((train_num, train_img_shape[0], train_img_shape[1]))
train_img_mono = train_img[:,:,:,0]
train_img_mono.shape
train_img.sum()/3 == train_img_mono.sum()
train_img_mono.max(),train_mask.max()
train_mask_8bit = np.zeros((train_mask.shape[0],train_mask.shape[1],train_mask.shape[1]))
for i in range(train_num):
    train_mask_8bit[i,:,:]= np.maximum(train_mask[i,:,:]/255-2,0)
train_mask_8bit[100,:,:]
train_mask.sum()/65535 == train_mask_8bit.sum()/255
train_img_mono.shape,train_img_mono.max(),train_mask_8bit.shape,train_mask_8bit.max()
import pandas as pd
train_dir = '../input/'
train = pd.read_csv(train_dir + 'train.csv')
train.head(7)
def rle_to_mask(rle_list, SHAPE):
    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask
for i in range(train_num):
    rle = train.loc[i,'rle_mask']
    rle_list = str(rle).split()
    mask = rle_to_mask(rle_list, train_mask_shape)
    img_name = train.loc[i,'id']
    num = train_img_dict_names_to_i[img_name]
    mask_ans = train_mask_8bit[num]
    mask_ans = mask_ans.reshape(train_mask_shape[0],train_mask_shape[1])
    if (mask-mask_ans).sum() != 0:
        print('{} is NG'.format(img_name))
train_img_mono = train_img_mono.reshape(train_img_mono.shape[0],train_img_mono.shape[1],train_img_mono.shape[2],1)
train_mask_8bit = train_mask_8bit.reshape(train_mask_8bit.shape[0],train_mask_8bit.shape[1],train_mask_8bit.shape[2],1)
train_img_mono.shape,train_mask_8bit.shape
train_img_mono_pad = np.zeros((train_img_mono.shape[0],128,128,train_img_mono.shape[3]))
train_mask_8bit_pad = np.zeros((train_img_mono.shape[0],128,128,train_img_mono.shape[3]))
top_pad = 13
bottom_pad = 14
left_pad = 13
right_pad = 14
for i in range(train_num):
    train_img_mono_pad[i,top_pad:-bottom_pad,left_pad:-right_pad,0] = train_img_mono[i,:,:,0]
    train_mask_8bit_pad[i,top_pad:-bottom_pad,left_pad:-right_pad,0] = train_mask_8bit[i,:,:,0]
train_img_mono_pad.shape,train_mask_8bit_pad.shape
train_img_mono.sum() == train_img_mono_pad.sum()
train_mask_8bit.sum() == train_mask_8bit_pad.sum()
train_img_mono_pad = train_img_mono_pad.astype('float32')/255
train_mask_8bit_pad = train_mask_8bit_pad.astype('float32')/255
def calc_IoU(A,B):
    AorB = np.logical_or(A,B).astype('int')
    AandB = np.logical_and(A,B).astype('int')
    IoU = AandB.sum() / AorB.sum()
    return IoU
import math
def calc_metric_oneimage(mask, mask_pred):
    score = 0.0
    if mask.sum() == 0:
        if mask_pred.sum() == 0:
            score = 1.0
        else:
            score = 0.0
    else:
        IoU = calc_IoU(mask,mask_pred)
        score = math.ceil((max(IoU-0.5,0))/0.05)/10
    return score
def calc_metric_allimage(masks, mask_preds_prod):
    num = masks.shape[0]
    tmp = mask_preds_prod > 0.5
    mask_preds = tmp.astype('int')
    scores = list()
    for i in range(num):
        score = calc_metric_oneimage(masks[i], mask_preds[i])
        scores.append(score)
    return np.mean(scores)
import tensorflow as tf
def my_metric(masks, mask_preds_prod):
    return tf.py_func(calc_metric_allimage, [masks, mask_preds_prod], tf.float64)
from keras.models import *
from keras.layers import *
inputs = Input(shape=(train_img_mono_pad.shape[1],train_img_mono_pad.shape[2],train_img_mono_pad.shape[3]))
conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
upcv6 = UpSampling2D(size=(2,2))(conv5)
upcv6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv6)
mrge6 = concatenate([conv4, upcv6], axis=3)
conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge6)
conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
upcv7 = UpSampling2D(size=(2,2))(conv6)
upcv7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv7)
mrge7 = concatenate([conv3, upcv7], axis=3)
conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge7)
conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
upcv8 = UpSampling2D(size=(2,2))(conv7)
upcv8 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv8)
mrge8 = concatenate([conv2, upcv8], axis=3)
conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge8)
conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
upcv9 = UpSampling2D(size=(2,2))(conv8)
upcv9 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv9)
mrge9 = concatenate([conv1, upcv9], axis=3)
conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge9)
conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
model = Model(inputs=inputs, outputs=conv10)
from keras.optimizers import *
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [my_metric])
model.summary()
model.fit(train_img_mono_pad, train_mask_8bit_pad, epochs=20, batch_size=32)
model.save('first_180924.h5')
def calc_metric_allimage_with_threshold(masks, mask_preds_prod, threshold):
    num = masks.shape[0]
    tmp = mask_preds_prod > threshold
    mask_preds = tmp.astype('int')
    scores = list()
    for i in range(num):
        score = calc_metric_oneimage(masks[i], mask_preds[i])
        scores.append(score)
    return np.mean(scores)
threshold_list = [x/100 for x in range(0,100,5)]
pred_train_img = model.predict(train_img_mono_pad)
metric_tmp = dict()
for threshold in threshold_list:
    tmp = calc_metric_allimage_with_threshold(train_mask_8bit_pad, pred_train_img, threshold)
    metric_tmp[threshold] = tmp
best_threshold = max(metric_tmp, key=metric_tmp.get)
best_threshold, metric_tmp[best_threshold]
test_img_names = [x.split('.')[0] for x in os.listdir(test_img_dir)]
test_num = len(test_img_names)
test_num
test_img_dict_i_to_names = dict()
test_img_dict_names_to_i = dict()
for i in range(test_num):
    test_img_dict_i_to_names[i] = test_img_names[i]
    test_img_dict_names_to_i[test_img_names[i]] = i
test_img_shape = imread(test_img_dir + test_img_names[0]+'.png').shape
test_img_shape
test_img = np.zeros((test_num, test_img_shape[0], test_img_shape[1], test_img_shape[2]))
for i in range(test_num):
    test_img[i] = i
    test_img[i,:,:,:] = imread(test_img_dir + test_img_names[i]+'.png')
test_img.shape
chk = 0
for i in range(test_num):
    chk = chk + (test_img[i,:,:,0]*2-test_img[i,:,:,1]-test_img[i,:,:,2]).sum()
chk
test_img_mono = np.zeros((test_num, test_img_shape[0], test_img_shape[1]))
test_img_mono = test_img[:,:,:,0]
test_img_mono.shape
test_img.sum()/3 == test_img_mono.sum()
test_img_mono.max()
test_img_mono = test_img_mono.reshape(test_img_mono.shape[0],test_img_mono.shape[1],test_img_mono.shape[2],1)
test_img_mono.shape
test_img_mono_pad = np.zeros((test_img_mono.shape[0],128,128,test_img_mono.shape[3]))
for i in range(test_num):
    test_img_mono_pad[i,top_pad:-bottom_pad,left_pad:-right_pad,0] = test_img_mono[i,:,:,0]
test_img_mono_pad.shape
test_img_mono.sum() == test_img_mono_pad.sum()
test_img_mono_pad = test_img_mono_pad.astype('float32')/255
predict = model.predict(test_img_mono_pad)
predict.shape
predict_mod = np.zeros((predict.shape[0],101,101))
for i in range(test_num):
    predict_mod[i,:,:] = predict[i,top_pad:-bottom_pad,left_pad:-right_pad,0]
predict_mod.shape
predict_mask = np.zeros((predict_mod.shape[0],predict_mod.shape[1],predict_mod.shape[2]))
predict_mask = predict_mod > best_threshold
def mask_to_rle(mask):
    mask_flat = mask.flatten('F')
    flag = 0
    rle_list = list()
    for i in range(mask_flat.shape[0]):
        if flag == 0:
            if mask_flat[i] == 1:
                flag = 1
                starts = i+1
                rle_list.append(starts)
        else:
            if mask_flat[i] == 0:
                flag = 0
                ends = i
                rle_list.append(ends-starts+1)
    if flag == 1:
        ends = mask_flat.shape[0]
        rle_list.append(ends-starts+1)
    #sanity check
    if len(rle_list) % 2 != 0:
        print('NG')
    if len(rle_list) == 0:
        rle = np.nan
    else:
        rle = ' '.join(map(str,rle_list))
    return rle
import math

for i,name in train_img_dict_i_to_names.items():
    mask = train_mask_8bit[i]/255
    rle = mask_to_rle(mask)
    rle_ans = train['rle_mask'][train['id']==name].values[0]
    if rle == rle_ans:
        continue
    elif math.isnan(rle) == True and math.isnan(rle_ans) == True:
        continue
    else:
        print('NG')
submit_name = list()
submit_rle = list()
for i in range(test_num):
    name = test_img_dict_i_to_names[i]
    rle = mask_to_rle(predict_mask[i])
    submit_name.append(name)
    submit_rle.append(rle)
submit_df = pd.DataFrame({'id':submit_name, 'rle_mask':submit_rle})
submit_df.head()
submit_df.shape
submit_df.to_csv('first_180929.csv', index=False)
