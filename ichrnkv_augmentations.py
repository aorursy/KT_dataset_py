# импорт бибилиотек

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import os
import zipfile

# настройка размеров графиков
%pylab inline
pylab.rcParams['figure.figsize'] = (8, 8)

# смотрим на файлы
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# распакуем трейн

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('data')
# посмотрим на оригинальную фотку

img = cv2.imread('/kaggle/working/data/train/dog.3668.jpg',  cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img);
aug = A.HorizontalFlip(p=1)
image = aug(image=img)['image']

plt.imshow(image);
aug = A.GridDistortion(distort_limit = (0.8, 0.9), p=2)
image = aug(image=img)['image']

plt.imshow(image);
# сразу несколько

def compose_aug(p=0.5):
    return A.Compose([
    A.RandomBrightnessContrast(p=p),    
    A.RandomGamma(p=p),    
    A.CLAHE(p=p), 
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=p)
    ], p=p)


aug = compose_aug(p=10)
image = aug(image=img)['image']

plt.imshow(image);