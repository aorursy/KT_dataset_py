import os, shutil



import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook



import warnings

warnings.filterwarnings(action='ignore')



print(os.listdir("../input"))
DATA_PATH ="../input/2019-3rd-ml-month-with-kakr"

DATA_PATH2 ="../input/car-crop"
# 이미지 폴더 경로

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')



TRAIN_IMG_PATH2 = os.path.join(DATA_PATH2, 'train_crop')

TEST_IMG_PATH2 = os.path.join(DATA_PATH2, 'test_crop')



# CSV 파일 경로

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
df_class2 = df_class.copy()

df_class2.rename(columns={'id':"class"},inplace=True)
comb = pd.merge(df_train, df_class2)

comb.head()
import PIL

from PIL import ImageDraw



#랜던으로 10개의 file 선택하기

tmp_imgs = np.random.choice(comb['img_file'],10)

plt.figure(figsize=(12,15))



for num, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, f_name))

    plt.subplot(5, 2, num + 1)

    plt.title(comb[comb['img_file']==f_name].reset_index()['name'][0])

    plt.imshow(img)

    plt.axis('off')
import PIL

from PIL import ImageDraw



# merge를 통해 같은 종류의 차들끼리 묶여짐

tmp_imgs = comb[comb['name']=='Audi S4 Sedan 2012'].iloc[:10,:]['img_file']

plt.figure(figsize=(12,15))



for num, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, f_name))

    plt.subplot(5, 2, num + 1)

    plt.title(comb[comb['img_file']==f_name].reset_index()['name'][0])

    plt.imshow(img)

    plt.axis('off')
import PIL

from PIL import ImageDraw



# merge를 통해 같은 종류의 차들끼리 묶여짐

tmp_imgs = comb[comb['name']=='Audi S4 Sedan 2012'].iloc[:10,:]['img_file']

plt.figure(figsize=(12,15))



for num, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH2, f_name))

    plt.subplot(5, 2, num + 1)

    plt.title(comb[comb['img_file']==f_name].reset_index()['name'][0])

    plt.imshow(img)

    plt.axis('off')
# 이미지를 그대로 읽어온다, 비교를 위해 시드고정

np.random.seed(seed=49)

tmp_imgs =  np.random.choice(comb['img_file'],3,)

plt.figure(figsize=(15,10))



# 이미지를 시각화한다

for i, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH2, f_name))

    plt.subplot(1, 3, i+1)

    plt.title(comb[comb['img_file']==f_name].reset_index()['name'][0])

    plt.imshow(img)

    plt.axis('off')
from scipy.ndimage import rotate



# 임의의 회전 각도(rotate_angle)을 구한 후, 이미지를 회전한다.

rotate_angle = np.random.randint(40) - 20

print('# 이미지 회전 : {}도'.format(rotate_angle))



# 이미지를 시각화한다.

np.random.seed(seed=49)

tmp_imgs =  np.random.choice(comb['img_file'],3,)

plt.figure(figsize=(15,10))

for i, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH2, f_name))

    img = rotate(img, rotate_angle)

    plt.subplot(1, 3, i+1)

    plt.title(comb[comb['img_file']==f_name].reset_index()['name'][0])

    plt.imshow(img)

    plt.axis('off')
import cv2

# 10x10 크기의 커널로 이미지를 흐린다

blur_degree = 10

print('{}x{} 커널 크기로 이미지 흐리기'.format(blur_degree, blur_degree))



# 이미지를 시각화한다.

np.random.seed(seed=49)

tmp_imgs =  np.random.choice(comb['img_file'],3,)

plt.figure(figsize=(15,10))



for i, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH2, f_name))

    #cv2사용시 array변경

    img = np.asarray(img)

    img = cv2.blur(img,(blur_degree,blur_degree))

    plt.subplot(1, 3, i+1)

    plt.title(comb[comb['img_file']==f_name].reset_index()['name'][0])

    plt.imshow(img)

    plt.axis('off')