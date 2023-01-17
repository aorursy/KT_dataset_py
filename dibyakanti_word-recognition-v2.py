# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import scipy.io as sio
from keras import backend as K
import numpy as np
import os
mat=sio.loadmat('../input/word-iam/imgs1_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=imgs
y_train=labels
mat=sio.loadmat('../input/word-iam/imgs2_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs3_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs4_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs5_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs6_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs7_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs8_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs9_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs10_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs11_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs12_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs13_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs14_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs15_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs16_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs17_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs18_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs19_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs20_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs21_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs22_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs23_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs24_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs25_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs26_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs27_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs28_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs29_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs30_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs31_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs32_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs33_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs34_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs35_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs36_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs37_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs38_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs39_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs40_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs41_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs42_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs43_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs44_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs45_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs46_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs47_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs48_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs49_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs50_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs51_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs52_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs53_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs54_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs55_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs56_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs57_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs58_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs59_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs60_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs61_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs62_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs63_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs64_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs65_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs66_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs67_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs68_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs69_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs70_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs71_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs72_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs73_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs74_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs75_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs76_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs77_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs78_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs79_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs80_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs81_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs82_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs83_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs84_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs85_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs86_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs87_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs88_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs89_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs90_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs91_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs92_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs93_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs94_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs95_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_train=np.concatenate((x_train,imgs),axis=-1)
y_train=np.concatenate((y_train,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs96_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test = imgs
y_test = labels
mat=sio.loadmat('../input/word-iam/imgs97_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs98_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs99_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs100_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs101_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs102_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs103_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs104_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs105_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs106_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs107_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs108_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs109_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs110_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs111_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs112_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs113_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs114_new.mat')
imgs = mat['imgs_new']
labels = mat['labels_new']
x_test=np.concatenate((x_test,imgs),axis=-1)
y_test=np.concatenate((y_test,labels),axis=-1)
mat=sio.loadmat('../input/word-iam/imgs115_new.mat')
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
x_test.shape
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
x=Conv2D(32,(3,3),activation='relu')(x1)
x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(64,(3,3),activation='relu')(x)
x=Conv2D(64,(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(128,(3,3),activation='relu')(x)
x=Conv2D(128,(3,3),activation='relu',padding='same')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2DTranspose(256,(3,3),activation='relu')(x)
x=Conv2DTranspose(256,(3,3),activation='relu',padding='same')(x)
x=UpSampling2D((2,2))(x)
x=Conv2DTranspose(512,(3,3),activation='relu')(x)
x=Conv2DTranspose(512,(3,3),activation='relu',padding='same')(x)
x=UpSampling2D((2,2))(x)
x=Conv2DTranspose(512,(3,3),activation='relu',padding='same')(x)
x=Conv2DTranspose(512,(3,3),activation='relu',padding='same')(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(N,(5,4),activation='relu')(x)
x=Conv2D(N,(5,3),activation='relu')(x)
x=Lambda(lambda x: custom_layer_fn(x[0],x[1]))([x1,x])
x=Conv2D(int(N),(3,3),activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Conv2D(int(N/2),(3,3),activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Flatten()(x)
x=Dense(4096,activation='relu')(x)
x=Dense(2048,activation='relu')(x)
x=Dense(Op,activation='softmax')(x)
model=Model(inputs=x1, outputs=x)
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,beta_1=0.1,beta_2=0.9,amsgrad=False),metrics=['accuracy'])
model.summary()
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, fill_mode='nearest')
datagen.fit(x_train)
history = model.fit(datagen.flow(x_train,np_utils.to_categorical(y_train1[0]),batch_size=16),validation_data=(x_test,np_utils.to_categorical(y_test1[0])),verbose=1,epochs=10)
