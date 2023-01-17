import numpy as np

import pandas as pd

import keras

import matplotlib.pyplot as plt

from PIL import Image
lib = pd.read_csv('/kaggle/input/testcanweuploaddatasetofourclass/FaceScore.csv')# 导入

print(lib.shape,"共计 5500 条数据")# 看数据形状（主要是长度）

lib.head()# 预览
lib['Rating'].hist()# 分数分布图
files = lib['Filename'].values.reshape(len(lib),1)# 利用 values 得到对应的ndarraay 然后转换一维向量到二维

Y = lib['Rating'].values.reshape(len(lib),1)# 同理

print(files.shape)
demo = Image.open('/kaggle/input/testcanweuploaddatasetofourclass/images/ftw1.jpg')# 加载图片

print(demo.size)# 打出一下像素大小

demo# 图片查看
SIZE = 128# 还是按照一般规律压缩到 128

N = len(files)# 样本量

X=np.zeros([N,SIZE,SIZE,3])# 先创建数据集大小

i = 0

for name in files:

    #print(name,type(name[0])) 用循环取得还是 ndarray

    Im=Image.open('/kaggle/input/testcanweuploaddatasetofourclass/images/'+name[0])# 批量导入

    Im=Im.resize([SIZE,SIZE])# 调整形状

    Im=np.array(Im)/255# 归一化到 0-1

    X[i,]=Im# 存进去

    i += 1
plt.imshow(X[0,:,:,:])# 验证转换有效
from sklearn.model_selection import train_test_split # 数据集分隔

X0,X1,Y0,Y1=train_test_split(X,Y,test_size=0.3,random_state=0)# 避免随机性
from keras.models import Sequential

from keras import layers

mymodel = Sequential()

mymodel.add(layers.InputLayer(input_shape=(128,128,3)))# 超级坑，如果没有这层，下面 summary 看起来一样，但是数据读不进去会是 nan

mymodel.add(layers.Flatten())# 拉平

mymodel.add(layers.Dense(1))# 单个线性层，不带激活

mymodel.summary()# 看效果
from keras import optimizers# 优化器

mymodel.compile(loss='mse',optimizer=optimizers.Adam(lr=0.001),metrics=['mse'])#用 MSE 准则 adam 优化
mymodel.fit(X0,Y0,batch_size=100,epochs=500)
MyPic=Image.open('/kaggle/input/testcanweuploaddatasetofourclass/images/ftw1.jpg')# 这个图自己弄一个，不想爆照就用猫了

MyPic=MyPic.resize((SIZE,SIZE)) # 调整结构

MyPic=np.array(MyPic)/255 # 调整到 0-1

plt.imshow(MyPic)# 看图

MyPic=MyPic.reshape((1,SIZE,SIZE,3))# 单一样本也是 N=1 的输入，不能是 SIZE,SIZE,3

mymodel.predict(MyPic)# 啊这分数有点惨烈
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