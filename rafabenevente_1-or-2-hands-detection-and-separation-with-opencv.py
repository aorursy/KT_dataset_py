import cv2

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os
def AutoCanny(image, sigma=0.33):

    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))

    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(image, lower, upper)

    return edged
def GetImageForValidatation(image):

    # para retornar a imagem ao tamanho original fazer 1 / pelo fator de resize

    x = 0.25

    y = 0.3

    img = cv2.resize(image, (0,0), fx=x, fy=y)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    res_auto = AutoCanny(img_gray)



    count_px = np.zeros(res_auto.shape[1])

    # Ignora 20% da imagem para forcar somente no meio

    ini = int(count_px.shape[0] * 0.2)

    fim = int(count_px.shape[0] * 0.8)

    for i in range(res_auto.shape[1]):

        count_px[i] = np.count_nonzero(res_auto[:,i])





    min_px = np.argmin(count_px[ini:fim])+ini

    max_range = (count_px.shape[0] // 2) + (ini // 2)

    min_range = (count_px.shape[0] // 2) - (ini // 2)

    # print("Posicao do menor pixel:", np.argmin(count_px[ini:fim])+ini)

    image_crop = image

    if min_range <= min_px <= max_range :

        image_crop = image[:, :int(min_px * (1/x))]



    return cv2.cvtColor(cv2.resize(image_crop, (256,256)), cv2.COLOR_BGR2RGB)
test_df = pd.read_csv("../input/i2a2-bone-age-regression/test.csv")

images = np.array(

    [np.array(GetImageForValidatation(

                cv2.imread(os.path.join("../input/i2a2-bone-age-regression/images", fname))))

                for fname in test_df["fileName"].values])



for img in images:

    plt.imshow(img)

    plt.show()