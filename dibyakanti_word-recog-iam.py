import scipy.io as sio
from keras import backend as K
import numpy as np
import os
mat=sio.loadmat('../input/imgs1_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=imgs
y_train=labels
mat=sio.loadmat('../input/imgs2_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs3_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs4_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs5_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs6_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs7_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs8_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs9_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs10_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs11_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs12_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs13_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs14_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs15_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs16_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs17_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs18_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs19_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs20_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs21_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs22_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs23_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs24_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs25_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs26_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs27_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs28_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs29_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs30_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs31_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs32_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs33_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs34_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs35_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs36_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs37_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs38_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs39_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs40_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs41_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs42_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs43_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs44_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs45_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs46_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs47_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs48_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs49_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs50_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs51_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs52_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs53_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs54_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs55_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs56_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs57_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs58_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs59_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs60_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs61_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs62_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs63_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs64_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs65_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs66_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs67_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs68_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs69_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs70_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs71_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs72_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs73_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs74_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs75_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs76_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs77_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs78_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs79_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs80_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs81_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs82_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs83_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs84_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs85_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs86_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs87_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs88_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs89_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs90_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs91_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs92_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs93_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs94_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs95_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/imgs96_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test = imgs
y_test = labels
mat=sio.loadmat('../input/imgs97_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs98_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs99_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs100_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs101_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs102_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs103_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs104_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs105_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs106_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs107_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs108_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs109_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs110_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs111_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs112_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs113_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs114_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/imgs115_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)

x_train=1-x_train.astype('float32')/255
x_train=np.swapaxes(x_train,0,2)
x_train=np.expand_dims(x_train,axis=-1)
x_test=1-x_test.astype('float32')/255
x_test=np.swapaxes(x_test,0,2)
x_test=np.expand_dims(x_test,axis=-1)
u=np.unique(y_train)
y_train1=np.zeros(y_train.shape)
for i in range(0,u.shape[0]):
  y_train1[np.where(y_train==u[i])]=i

y_test1=np.zeros(y_test.shape)
for i in range(0,u.shape[0]):
  y_test1[np.where(y_test==u[i])]=i

Op=u.shape[0]
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Lambda, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

N=256

def custom_layer_fn(A,B):
  A1=tf.repeat(A,N,axis=-1)
  return A1*B

x1=Input((200,75,1))
#x=Conv2D(16,(3,3),activation='relu')(x1)
#x=Conv2D(16,(3,3),activation='relu')(x)
#x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(32,(3,3),activation='relu')(x1)
x=Conv2D(32,(3,3),activation='relu')(x)
x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
#x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(32,(3,3),activation='relu')(x)
x=Conv2D(32,(3,3),activation='relu')(x)
x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
#x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(64,(3,3),activation='relu')(x)
x=Conv2D(64,(3,3),activation='relu')(x)
x=Conv2D(64,(3,3),activation='relu',padding='same')(x)
#x=Conv2D(64,(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2DTranspose(64,(3,3),activation='relu')(x)
x=Conv2DTranspose(64,(3,3),activation='relu',padding='same')(x)
x=Conv2DTranspose(64,(3,3),activation='relu',padding='same')(x)
#x=Conv2DTranspose(128,(3,3),activation='relu',padding='same')(x)
x=UpSampling2D((2,2))(x)
x=Conv2DTranspose(128,(3,3),activation='relu')(x)
x=Conv2DTranspose(128,(3,3),activation='relu')(x)
x=Conv2DTranspose(128,(3,3),activation='relu',padding='same')(x)
#x=Conv2DTranspose(256,(3,3),activation='relu',padding='same')(x)
x=UpSampling2D((2,2))(x)
x=Conv2DTranspose(256,(3,3),activation='relu')(x)
x=Conv2DTranspose(256,(3,3),activation='relu')(x)
x=Conv2DTranspose(256,(3,3),activation='relu',padding='same')(x)
#x=Conv2DTranspose(512,(3,3),activation='relu',padding='same')(x)
x=UpSampling2D((2,2))(x)
#x=Conv2DTranspose(16,(3,3),activation='relu')(x)
#x=Conv2DTranspose(16,(3,3),activation='relu')(x)
#x=UpSampling2D((2,2))(x)
#x=Conv2D(N,(3,3),activation='relu')(x)
x=Conv2D(N,(5,4),activation='relu')(x)
x=Conv2D(N,(5,3),activation='relu')(x)
x=Lambda(lambda x: custom_layer_fn(x[0],x[1]))([x1,x])
x=Conv2D(N,(3,3),activation='relu')(x)
#x=Conv2D(N,(3,3),activation='relu',padding='same')(x)
#x=Conv2D(N,(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(int(N),(3,3),activation='relu')(x)
#x=Conv2D(int(N),(3,3),activation='relu',padding='same')(x)
#x=Conv2D(int(N/2),(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(int(N/2),(3,3),activation='relu')(x)
#x=Conv2D(int(N/2),(3,3),activation='relu',padding='same')(x)
#x=Conv2D(int(N),(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
#x=Conv2D(int(N/4),(3,3),activation='relu')(x)
#x=Conv2D(int(N/4),(3,3),activation='relu',padding='same')(x)
#x=Conv2D(int(N/2),(3,3),activation='relu',padding='same')(x)
#x=MaxPooling2D(pool_size=(2,2))(x)
x=Flatten()(x)
#x=Dense(8192,activation='relu')(x)
#x=Dropout(0.2)(x)
#x=Dense(4096,activation='relu')(x)
#x=Dense(2048,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(Op,activation='softmax')(x)
#x=Dense(1000,activation='softmax')(x)
#x=Dense()

model=Model(inputs=x1, outputs=x)
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,beta_1=0.1,beta_2=0.9,amsgrad=False),metrics=['accuracy'])
model.summary()

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, fill_mode='nearest')
datagen.fit(x_train)
history = model.fit(datagen.flow(x_train,np_utils.to_categorical(y_train1[0]),batch_size=32),validation_data=(x_test,np_utils.to_categorical(y_test1[0])),verbose=1,epochs=15)
np.max(x_test)

print(history.history.keys())
# summarize history for accuracy
import matplotlib.pyplot as plt 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()