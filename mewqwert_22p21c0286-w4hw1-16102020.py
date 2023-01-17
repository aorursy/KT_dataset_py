import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import cv2
from skimage import feature
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.util import invert
import time

import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as  np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import vgg16
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

import pickle
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
tpu_strategy.num_replicas_in_sync
start_path = '/kaggle/input/thai-mnist-classification'
data = {'test' : start_path + '/test.rules.csv', 
        'train' : start_path + '/train.rules.csv', 
        'mnist' : start_path + '/mnist.train.map.csv', 
        'drop' : '/kaggle/input/drop-list/drop_lists.csv'}
ls_img = {'train' : os.listdir(start_path + '/train'), 
          'test' : os.listdir(start_path + '/test')}
drops = pd.read_csv(data['drop']).values
drops = drops.reshape(len(drops)).tolist()
mnist = pd.read_csv(data['mnist'], index_col='id')
train = pd.read_csv(data['train']).drop(['id'], axis = 1).drop_duplicates().reset_index(drop=True)
# train['predict'] don't have missing data
test = pd.read_csv(data['test']).drop(['predict'], axis=1).fillna(10)
mnist_drop = mnist.drop(drops, axis=0).reset_index()
mnist3 = mnist.copy()
mnist3.index.names = ['feature3']
train3 = train.set_index(['feature3'])
train = pd.merge(mnist3, train3, how='right', on='feature3').reset_index(drop=True).rename(columns={'category': 'feature3'})
mnist2 = mnist.copy()
mnist2.index.names = ['feature2']
train2 = train.set_index(['feature2'])
train = pd.merge(mnist2, train2, how='right', on='feature2',).reset_index(drop=True).rename(columns={'category': 'feature2'})
mnist1 = mnist.copy()
mnist1.index.names = ['feature1']
train1 = train.set_index(['feature1'])
train = pd.merge(mnist1, train1, how='right', on='feature1').reset_index(drop=True).rename(columns={'category': 'feature1'})
train.head()
print(test.shape)
test.head()
print(train.shape)
train.head()
print(mnist.shape)
mnist.reset_index().head()
print(mnist_drop.shape)
mnist_drop.head()
train[pd.isna(train['feature1'])][:5] # P = F2+F3
train[train['feature1'] == 0][:5] # P = F2*F3
train[train['feature1'] == 1][:5] # P = abs(F2-F3)
train[train['feature1'] == 2][:5] # P = (F2+F3) * abs(F2-F3)
train[train['feature1'] == 3][:5] # P = abs((F3(F3 +1) - F2(F2-1))/2)
train[train['feature1'] == 4][:5] # P = 50 + (F2-F3)
train[train['feature1'] == 5][:5] # P = data.iloc[:, 1:3].min(axis=1)
train[train['feature1'] == 6][:5] # P = data.iloc[:, 1:3].max(axis=1)
train[train['feature1'] == 7][:5] # P = ((F2*F3)%9) * 11
train[train['feature1'] == 8][:5] # P = ( ( (F2**2+1)*F2 + F3*(F3+1) - 1)%99 + 1 )%99
train[train['feature1'] == 9][:5] # P = 50 + F2
x_train = train.fillna(10).iloc[:, :3].values
le = LabelEncoder()
y_train = train.iloc[:, -1].values
x1 = le.fit_transform(x_train[:, 0]).reshape(len(x_train), 1)
x2 = x_train[:, 1:]
x_train = np.concatenate((x1,x2),axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
rf = RandomForestRegressor(n_estimators=1, random_state=0, bootstrap=False)
rf.fit(xtrain, ytrain)
rf_pred = np.array(rf.predict(xtest).round(),  dtype=int)
acc_rf = accuracy_score(ytest, rf_pred)
acc_rf
dt = DecisionTreeRegressor(random_state=9)
dt.fit(xtrain, ytrain)
dt_pred = np.array(dt.predict(xtest).round(),  dtype=int)
acc_dt = accuracy_score(ytest, dt_pred)
acc_dt
class getdata():
    def __init__(self, data_path, list_name_img, test=False):
        self.dataPath = data_path
        self.n_index = len(list_name_img)
        if test:
            self.name = list_name_img
        else:
            self.name = list_name_img.id.values
            self.category = list_name_img.category.values
    def get_img(self,img_index,mode='rgb'):
        img = cv2.imread( os.path.join(self.dataPath,self.name[img_index]) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img
data_num = getdata(start_path + '/train', mnist.reset_index())
samp_img = invert(data_num.get_img(12,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2)
ax1.imshow(samp_img)
cvh =  convex_hull_image(samp_img)
ax2.imshow(cvh)
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
samp_crop = convex_crop(samp_img,pad=10)
print(samp_crop.shape)
plt.imshow(samp_crop)
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=10)
    img = cv2.resize(img,(128,128))
    return img
def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(128,128))
    img = ((img > thes)*255).astype(np.uint8)
#    img = cv2.resize(img,(64,64))
#    img = ((img > thes)*255).astype(np.uint8)
    return img
samp_img = data_num.get_img(1,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(10,7))
ax1.imshow(convex_resize(samp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(samp_img))
ax2.set_title('Thresholding')
fig, ax = plt.subplots(5, 5, figsize=(15,15))
for i in range(5):
    for j in range(5):
        img_index = i**2 + j*2
        ax[i][j].imshow(thes_resize(data_num.get_img(img_index,'gray')))
        ax[i][j].set_axis_off()
x_mnist = []
switch = 0
for i in range(data_num.n_index):
    if switch == 0:
        time_start = time.time()
        switch = 1
    x_mnist.append(thes_resize(data_num.get_img(i,'gray')))
    if (data_num.n_index-i) % 200 == 0 or i == 0:
        time_end = time.time()
        switch = 0
        print(data_num.n_index-i, f'Loading {i/data_num.n_index*100:.1f}% - Time: {int(time_end-time_start)} s')
print('Complete')
x_mnist = np.array(x_mnist)
y_mnist = data_num.category
x_mnist = x_mnist.reshape((-1,128,128,1))
y_mnist = tf.keras.utils.to_categorical(y_mnist)
x_train_mnist, x_test_mnist, y_train_mnist, y_test_mnist = train_test_split(x_mnist, y_mnist, test_size=0.2, random_state=0)
x_train_mnist = x_train_mnist/255
x_train_mnist = tf.convert_to_tensor(x_train_mnist)
x_train_mnist = tf.image.grayscale_to_rgb(x_train_mnist)
x_test_mnist = x_test_mnist/255
x_test_mnist = tf.convert_to_tensor(x_test_mnist)
x_test_mnist = tf.image.grayscale_to_rgb(x_test_mnist)
with tpu_strategy.scope():
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    for l in vgg.layers:
        l.trainable = False
    x = vgg.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(vgg.input, x)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001*tpu_strategy.num_replicas_in_sync),
                  metrics=['accuracy'])
    model.summary()
history = model.fit(x_train_mnist,y_train_mnist,
                    validation_data=(x_test_mnist,y_test_mnist),
                    batch_size=64*tpu_strategy.num_replicas_in_sync,
                    epochs=100,
                    verbose=True)
model.save('model128.h5')
pred_num = getdata(start_path + '/test', np.array(ls_img['test']), test=True)
test_num = []
switch = 0
for i in range(pred_num.n_index):
    if switch == 0:
        time_start = time.time()
        switch = 1
    test_num.append(thes_resize(pred_num.get_img(i,'gray')))
    if (pred_num.n_index-i) % 200 == 0 or i == 0:
        time_end = time.time()
        switch = 0
        print(pred_num.n_index-i, f'Loading {i/pred_num.n_index*100:.1f}% - Time: {int(time_end-time_start)} s')
print('Complete')
test_num = np.array(test_num)
test_num = test_num.reshape((-1,128,128,1))
test_num = test_num/255
test_num = tf.convert_to_tensor(test_num)
test_num = tf.image.grayscale_to_rgb(test_num)
y_pred = model.predict(test_num)
test_pred = y_pred.argmax(axis=1)
df_test = pd.DataFrame({'id':pred_num.name, 'predict':test_pred})
df_test.head()
number_test = pd.read_csv('/kaggle/input/number-test/number_test.csv')
check_test = pd.merge(number_test.iloc[:, :1], df_test, how='left', on='id')
accuracy_score(number_test.predict.values, check_test.predict.values)
weight = model.get_weights()
pklfile= os.getcwd()+'/modelweights128.pkl'
fpkl= open(pklfile, 'wb')  
pickle.dump(weight, fpkl, protocol= pickle.HIGHEST_PROTOCOL)
fpkl.close()
test_img3 = df_test.copy().set_index(['id'])
test_img3.index.names = ['feature3']
test3 = test.set_index(['feature3'])
new_test = pd.merge(test_img3, test3, how='right', on='feature3').reset_index(drop=True).rename(columns={'predict': 'feature3'})
test_img2 = df_test.copy().set_index(['id'])
test_img2.index.names = ['feature2']
test2 = new_test.set_index(['feature2'])
new_test = pd.merge(test_img2, test2, how='right', on='feature2').reset_index(drop=True).rename(columns={'predict': 'feature2'})
test_img1 = df_test.copy().set_index(['id'])
test_img1.index.names = ['feature1']
test1 = new_test.set_index(['feature1'])
new_test = pd.merge(test_img1, test1, how='right', on='feature1').reset_index(drop=True).rename(columns={'predict': 'feature1'}).fillna(10)
new_test.head()
x_test = new_test.iloc[:, :3].values
x_test1 = le.transform(x_test[:, 0])
x_pred = np.concatenate((x_test1.reshape(len(x_test1), 1), x_test[:, 1:3]), axis=1)
submit = pd.read_csv(start_path+'/submit.csv').drop(['predict'], axis=1)
submit.head()
y_dt = np.array(dt.predict(x_pred).round(),  dtype=int)
y_dt = pd.DataFrame({'id':new_test.id.values, 'predict':y_dt})
df_dt = pd.merge(submit, y_dt, how='left', on='id')
df_dt.head()
y_rf = np.array(rf.predict(x_pred).round(),  dtype=int)
y_rf = pd.DataFrame({'id':new_test.id.values, 'predict':y_rf})
df_rf = pd.merge(submit, y_rf, how='left', on='id')
df_rf.head()
def func(data, n):
    dat = data.copy()
    data = dat[dat['feature1'] == n]
    f2 = data['feature2']
    f3 = data['feature3']
    id_dat = data.iloc[:,-1:]
    if n == 0: compute = f2*f3
    elif n == 1: compute = abs(f2-f3)
    elif n == 2: compute = (f2+f3)*abs(f2-f3)
    elif n == 3: compute = abs((f3*(f3+1) - f2*(f2-1))/2)
    elif n == 4: compute = 50+(f2-f3)
    elif n == 5: compute = data.iloc[:, 1:3].min(axis=1)
    elif n == 6: compute = data.iloc[:, 1:3].max(axis=1)
    elif n == 7: compute = ((f2*f3)%9)*11
    elif n == 8: compute = (((f2**2+1)*f2 + f3*(f3+1) -1)%99 + 1)%99
    elif n == 9: compute = 50+f2
    elif n == 10: compute = f2+f3
    compute = pd.DataFrame({'predict':compute})
    return pd.concat([id_dat,compute], axis=1, ignore_index=True)
for i in range(11):
    da = func(new_test, i)
    if i == 0:
        first = da
        continue
    first = pd.concat([first, da], axis=0, ignore_index=True)
first = first.rename(columns={0:'id', 1:'predict'})
operate = pd.merge(submit, first, how='left', on='id')
operate.head()
check_operate = pd.read_csv('/kaggle/input/submit/submit.csv')
acc_dt = accuracy_score(check_operate.predict.values, df_dt.predict.values)
acc_rf = accuracy_score(check_operate.predict.values, df_rf.predict.values)
acc_op = accuracy_score(check_operate.predict.values, operate.predict.values)
print('accuracy_score DecisionTree :', acc_dt)
print('accuracy_score RandomForest :', acc_rf)
print('accuracy_score Operation:', acc_op)
operate.to_csv('submit_pord.csv', index = False)