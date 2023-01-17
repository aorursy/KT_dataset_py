import numpy as np

import pandas as pd

import cv2 as cv

import matplotlib.pyplot as plt

import os

from PIL import Image



%matplotlib inline
def _plot(img1, img2):

    plt.figure(figsize=(15,8))

    plt.subplot(1, 2, 1)

    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))

    plt.title("source_1")

    plt.subplot(1, 2, 2)

    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))

    plt.title("source_2");
def get_shape(i):

    # парсим форму картинки

    if i > data['img_shape_2'].shape[0]:

        return [100,100]

    shape = data['img_shape_2'][i][1:-1].split(", ")   # парсим форму картинки

    shape = np.array([int(d) for d in shape])

    return shape
data = pd.read_csv('../input/comparing-images/test.csv')  # картинки представляют из себя строки

data2 = data['img_2'][1][1:-1].split(", ") # выбираем картинку и переводим ее в массив

data3 = np.array([int(d) for d in data2])

shape = get_shape(1)

data4 = data3.reshape(shape)
data.shape
data4.shape
plt.imshow(data4);
def encode(img):

    '''

    Кодировка ответа для kaggle, на вход сюда подавать маску

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
mask_arr_sub = []    # маски отличий

for i in range(data.shape[0]):

    shape = get_shape(i)

    ans = np.zeros(shape)   # тут должны быть ваши ответы

    mask_arr_sub.append(encode(ans))

    print(i)
mask_arr_sub
def decode(mask, shape=(768, 768)):

    '''

    Восстановление маски отличий из кодировки

    mask: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
mask = decode(mask_arr_sub[0], shape=get_shape(1)[:-1])   # проверим правильность работы
plt.imshow(mask);
d = {"id":[i for i in range(12)],        # формируется словарь с индексами и масками

    'mask_arr': mask_arr_sub}
df = pd.DataFrame(data=d)          # Словарь в DataFrame
df.to_csv("sample_solution.csv",index=False)