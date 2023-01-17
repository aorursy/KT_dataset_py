# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

from PIL import TiffImagePlugin

import openslide

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#경로가 제대로 된 게 맞는지 리스팅 해보자.
'''
참고자료 : https://itholic.github.io/python-listdir-glob/
'''
#os.listdir('/kaggle/input/prostate-cancer-grade-assessment/train_images/')
'''
openslide 라이브러리를 이용하면 tiff 파일을 열 수 있다.
'''

example = openslide.OpenSlide('/kaggle/input/prostate-cancer-grade-assessment/train_images/6d1a11077fe4183a4109d649cf319923.tiff')
patch = example.read_region((5000,6700), 0 , (256,256))

display(patch)

example.close()
example2 = openslide.OpenSlide('/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/002a4db09dad406c85505a00fb6f6144_mask.tiff')
patch = example2.read_region((5000,6700), 0 , (256,256))

display(patch)

example2.close()
data = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
data
'''
우선 마스크가 있는 데이터만 골라내고자 한다.
'''
data['mask'] = ''
data
'''
os.path.isfile('파일이름')으로 마스크가 존재하는 데이터인지 체크하자.
마스크가 있는 데이터는 'yes'로 표기해두고,
마스크가 없는 데이터는 'no'로 표기해두도록 하자.

참고자료 : 
https://stackoverflow.com/questions/38876816/change-value-of-a-dataframe-column-based-on-a-filter
'''


for im_id in data['image_id']:
    if os.path.isfile('../input/prostate-cancer-grade-assessment/train_label_masks/' + im_id + '_mask.tiff'):
        data.loc[data['image_id'] == im_id, 'mask'] = 'yes'
    else:
        data.loc[data['image_id'] == im_id, 'mask'] = 'no'

'''
mask가 없는 데이터들을 출력한다.
'''
data[data['mask'] =='no']
'''
마스크가 없는 데이터를 (row를) 모조리 삭제해주자.
'''
data = data.drop(data[data['mask'] == 'no'].index)
data
'''
이미지를 numpy array로 바꾸는 방법은 세 가지가 있다.
1) PIL의 Image 라이브러리를 사용하는 방법
2) cv2의 imread를 사용하는 방법
3) skimage.io의 imread를 사용하는 방법

세 가지를 모두 실행해보니, 3번 방법만 tiff파일을 np array로 바꿀 수 있었다.
그런데 확실히..파일 하나가 너무나도 크다.

그리고, 이미지마다 shape가 제각각이어서 학습을 어떻게 시켜야 할지도 감이 잘 안온다.
'''
import skimage.io as io

img = io.imread('/kaggle/input/prostate-cancer-grade-assessment/train_images/6d1a11077fe4183a4109d649cf319923.tiff')
img.shape
img = io.imread('/kaggle/input/prostate-cancer-grade-assessment/train_images/001c62abd11fa4b57bf7a6c603a11bb9.tiff')
img.shape
'''
데이터셋을 train set, validation set으로 나눠보자.
sklearn의 train_test_split을 쓰면 쉽게 진행할 수 있다.

참고자료: 
https://blog.naver.com/PostView.nhn?blogId=siniphia&logNo=221396370872&parentCategoryNo=&categoryNo=22&viewDate=&isShowPopularPosts=true&from=search
https://rfriend.tistory.com/519
'''
from sklearn.model_selection import train_test_split

data_train, data_val = train_test_split(data, test_size = 0.2, random_state = 1030)
data_val, data_test = train_test_split(data_val, test_size=0.2, random_state = 1030)
print(data_train.shape)
print(data_test.shape)
print(data_val.shape)
for data in data_train.head()['image_id']:
    print(data)
'''
train set의 각 이미지를 numpy array형태로 바꾸자.
바꾸고 나면 각 이미지를 256 * 256으로 resize 해보자.
'''
import cv2
from PIL import Image

fig, axs = plt.subplots(1,5, figsize = (100,100))
it = 0

for img_id in data_train.tail()['image_id']:
    img = io.imread('/kaggle/input/prostate-cancer-grade-assessment/train_images/'+ img_id +'.tiff')
    resized = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    print(resized.shape)
    
    
    
    axs[it].imshow(resized)
    it = it+1
    
    

'''
Q. 왜 마스크는 아무것도 뜨지 않는가?
A. 마스크는 WSI(Whole Slide Imaging)같은 이미지가 아니다.
배열에 0~255의 값을 담는 대신, 0~6의 값만 담는다.
0~6은 각각 다른 클래스 라벨을 의미한다. (이 데이터셋 description에 상세 정보가 나와 있다.)
그렇기 때문에 mask를 시각화 하는 시도를 하더라도 거의 0에 가까운 숫자들만 있기 때문에
아주 어둡게 나타나게 된다.
color map을 쓰면 이런 이슈를 해결할 수 있다.
0~6까지 각각 다른 color을 할당해주자.

★★ imshow 함수를 쓸 때 colormap이 먹히도록 만들려면
이미지를 nparray로 만들어줘야 한다.
np.asarray(이미지변수명)[:, :, 0]
그런데 맨 뒷 대괄호 안의 내용의 의미는 뭐지?
참고자료 : https://www.kaggle.com/tanulsingh077/prostate-cancer-in-depth-understanding-eda-model

'''
import cv2
from PIL import Image
import matplotlib

fig, axs = plt.subplots(1,5, figsize = (200,200))
it = 0

for img_id in data_train.tail()['image_id']:
    img = io.imread('/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'+ img_id +'_mask.tiff')
    resized = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    print(resized.shape)
    
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    
    axs[it].imshow(np.asarray(resized)[:,:,0], cmap = cmap, interpolation='nearest', vmin=0, vmax=5)
    it = it+1
os.makedirs('./train_img')
os.makedirs('./train_mask')
os.makedirs('./val_img')
os.makedirs('./val_mask')
os.makedirs('./test_img')
os.makedirs('./test_mask')
os.listdir('./')
from tqdm.notebook import tqdm

'''
지금까지 진행한 바는 다음과 같다.
1. 원본 이미지와 마스크 이미지를 numpy array화 시키기
2. 원본 이미지와 마스크 이미지를 256*256*3으로 resizing하기
3. 마스크 이미지에 colormap 입히기
4. 변환시킨 원본 이미지와 마스크 이미지를 출력하기

5. 데이터셋을 train_set과 validation_set으로 나누기

이제 mask가 존재하는 모든 데이터에 대하여 데이터셋을 numpy array화 해서 저장하도록 하자.
다음과 같은 규칙으로 데이터셋을 만들기로 한다.

x : resizing한 원본 데이터
y : resizing한 마스크 데이터

x_train, y_train, x_test, y_test를 만들도록 한다.


이미지 데이터셋을 numpy array화 하는 방법은 다음과 같다.
1. 일반 empty 배열을 만든다.
2. 각 이미지를 numpy array화 시킨다. (예를들어 cv2.imread나 cv2.resize를 사용하는 등)
3. 해당 이미지를 배열에 append한다.
4. np.array(배열 이름)이 곧 데이터셋의 numpy array이다.

그런데... 이렇게 해줬더니 메모리 초과되어버린다.
데이터 용량이 크다보니 한 번에 로드하면 메모리가 버티지 못하는 것이다.

그렇기 때문에 train 데이터 전체를 numpy 파일 하나로 묶어서 처리하기보다는
train 데이터 하나당 numpy 파일 하나로 만들어서 처리하는 것이 더 합리적이다.

[참고사항]
skimage.io.imread보다
skimage.io.MultiImage가 더 빠르다. (훨씬 훨씬)
'''
i = 0

for img_id in tqdm(data_train['image_id']):
    #img = io.imread('/kaggle/input/prostate-cancer-grade-assessment/train_images/'+ img_id +'.tiff')
    #resized = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img = io.MultiImage('/kaggle/input/prostate-cancer-grade-assessment/train_images/'+ img_id +'.tiff')
    resized = cv2.resize(img[-1], (256,256), interpolation=cv2.INTER_CUBIC)
    
    #mask = io.imread('/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'+ img_id +'_mask.tiff')
    #resized_mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    
    np.save('./train_img/' + str(i) + '.npy', resized)
    #np.save('./train_mask/' + str(i) + '.npy', resized_mask)
    
    i = i+1
print("done!")
    

'''
같은 방법으로 mask데이터도 저장해보자.
'''

i = 0

for img_id in tqdm(data_train['image_id']):
    mask = io.MultiImage('/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'+ img_id +'_mask.tiff')
    resized_mask = cv2.resize(mask[-1], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    
    np.save('./train_mask/' + str(i) + '.npy', resized_mask)
    
    i = i+1
print("done!")
i = 0

for img_id in tqdm(data_val['image_id']):
    img = io.MultiImage('/kaggle/input/prostate-cancer-grade-assessment/train_images/'+ img_id +'.tiff')
    resized= cv2.resize(img[-1], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    
    np.save('./val_img/' + str(i) + '.npy', resized)
    
    i = i+1
print("done!")
i = 0

for img_id in tqdm(data_val['image_id']):
    mask = io.MultiImage('/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'+ img_id +'_mask.tiff')
    resized_mask = cv2.resize(mask[-1], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    
    np.save('./val_mask/' + str(i) + '.npy', resized_mask)
    
    i = i+1
print("done!")
i = 0

for img_id in tqdm(data_test['image_id']):
    img = io.MultiImage('/kaggle/input/prostate-cancer-grade-assessment/train_images/'+ img_id +'.tiff')
    resized= cv2.resize(img[-1], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    
    np.save('./test_img/' + str(i) + '.npy', resized)
    
    i = i+1
print("done!")
i = 0

for img_id in tqdm(data_test['image_id']):
    mask = io.MultiImage('/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'+ img_id +'_mask.tiff')
    resized_mask = cv2.resize(mask[-1], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    
    
    np.save('./test_mask/' + str(i) + '.npy', resized_mask)
    
    i = i+1
print("done!")
'''
hoh = np.load('./train_img/799.npy')

fig, axs = plt.subplots(1,5, figsize = (200,200))

axs[0].imshow(hoh)
#hoh = cv2.imread('./train_mask/799.npy')

#print(hoh)
'''
os.listdir('./')

mask = ''
resized_mask = ''
img = ''
resized = ''
'''
numpy 형태로 저장한 데이터를 넣을 수 있게
datagenerator을 디자인해보자.

참고자료 : https://sunshower76.github.io/frameworks/2020/02/09/Keras-Batch%EC%83%9D%EC%84%B1%ED%95%98%EA%B8%B01-(Seuquence&fit_generator)/
'''

class DataGenerator(Sequence):
    def __init__(self, list_IDs, imgs_dir, masks_dir, batch_size = 10, img_size = 256, n_channels=3, n_classes=1, shuffle=True):
        self.list_IDs = list_IDs
        #인덱스는 __get_item__ 함수에서 쓰이므로 만들어놔야 한다.
        #np.arange(3)은 array([0,1,2])이다. 즉, 아래 코드는 array용 index를 만들어주는 것.
        self.indexes = np.arange(len(self.list_IDs))
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    def __getitem__(self, index):
        #batch용 index들을 만들어준다.
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.list_IDs[k] for k in indexes]
        
        imgs = list()
        masks = list()
        
        for id_name in batch_ids:
            img, mask = self.__data_generation(id_name)
            imgs.append(img)
            masks.append(mask)
            
        imgs = np.array(imgs)
        masks = np.array(masks)
        
        return imgs, masks
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, id_name):
        img_path = os.path.join(self.imgs_dir, id_name)
        mask_path = os.path.join(self.masks_dir, id_name)
        
        img = np.load(img_path)
        mask = np.load(mask_path)
        
        '''
        img = img / 255.0
        mask = mask / 255.0
        
        '''
        
        return img, mask
import glob
#x_train_list = sorted(glob.glob(os.path.join('./train_img', '*.npy')))
#y_train_list = sorted(glob.glob(os.path.join('./train_mask', '*.npy')))
'''
train_img 폴더에 포함된 npy파일과
train_mask에 포함된 npy파일 이름은 모두 같다.
'''
train_id = sorted(os.listdir('./train_img'))
val_id = sorted(os.listdir('./val_img'))
print(train_id[3])
print(len(train_id))
print(len(val_id))
train_gen = DataGenerator(list_IDs=train_id, imgs_dir = './train_img', masks_dir = './train_mask', batch_size = 10, img_size=256)
val_gen = DataGenerator(list_IDs=val_id, imgs_dir = './val_img', masks_dir = './val_mask', batch_size = 10, img_size=256)
'''
모델을 작성하도록 해보자.
'''

inputs = Input(shape=(256,256,3))

net = Conv2D(32, kernel_size=3, activation = 'relu', padding='same')(inputs)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation = 'relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation = 'relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Dense(128, activation='relu')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)

net = UpSampling2D(size=2)(net)
outputs = Conv2D(3, kernel_size=3, activation='softmax', padding='same')(net)


model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['acc', 'mse'])

model.summary()
history = model.fit_generator(train_gen, validation_data = val_gen, epochs=2, verbose=1)
test_list = sorted(os.listdir('./test_img'))

print(len(test_list))
print(test_list[45])
test_idx = 8
x1_test = np.load('./test_img/' + test_list[test_idx])
y1_test = np.load('./test_mask/' + test_list[test_idx])
y_pred = model.predict(x1_test.reshape(1,256,256,3))


y_pred = np.clip(y_pred.reshape((256,256,3)), 0, 5)



print(x1_test.shape, y1_test.shape, y_pred.shape)


#y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)
#y_pred = y_pred * 10

cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title('input')
plt.imshow(x1_test)
plt.subplot(1, 3, 2)
plt.title('output')
plt.imshow(np.asarray(y_pred)[:,:,0], cmap = cmap, interpolation='nearest', vmin=0, vmax=5)
#plt.imshow(y_pred)
plt.subplot(1, 3, 3)
plt.title('groundtruth')
plt.imshow(np.asarray(y1_test)[:,:,0], cmap = cmap, interpolation='nearest', vmin=0, vmax=5)
print(y_pred)
print(y1_test)
