import numpy as np

import pandas as pd

import cv2

import os

import matplotlib.pyplot as plt

import random

from PIL import Image

from tqdm import tqdm
!unzip ../input/kuzushiji/k49-train-imgs.npz && mv ../working/arr_0.npy k49-train-imgs.npy

!unzip ../input/kuzushiji/k49-train-labels.npz && mv ../working/arr_0.npy k49-train-labels.npy

!unzip ../input/kuzushiji/k49-train-imgs.npz && mv ../working/arr_0.npy k49-test-imgs.npy

!unzip ../input/kuzushiji/k49-train-labels.npz && mv ../working/arr_0.npy k49-test-labels.npy
!unzip ../input/kuzushiji/kmnist-train-imgs.npz && mv ../working/arr_0.npy kmnist-train-imgs.npy

!unzip ../input/kuzushiji/kmnist-train-labels.npz && mv ../working/arr_0.npy kmnist-train-labels.npy

!unzip ../input/kuzushiji/kmnist-train-imgs.npz && mv ../working/arr_0.npy kmnist-test-imgs.npy

!unzip ../input/kuzushiji/kmnist-train-labels.npz && mv ../working/arr_0.npy kmnist-test-labels.npy
os.listdir("../input/kuzushiji")
k49 = np.load('../working/k49-train-imgs.npy')

k49_labels = np.load('../working/k49-train-labels.npy')

k49_mapping = pd.read_csv("../input/kuzushiji/k49_classmap.csv")
kmnist = np.load('../working/kmnist-train-imgs.npy')

kmnist_labels = np.load('../working/kmnist-train-labels.npy')

kmnist_mapping = pd.read_csv("../input/kuzushiji/kmnist_classmap.csv")
def get_k49(show=False):

    idx = random.randint(0,len(k49)-1)

    img = k49[idx]

    if show:

        plt.imshow(img)

        plt.show()

    return img, k49_mapping.iloc[k49_labels[idx]].codepoint
sample = get_k49(show=True)
def get_kmnist(show=False):

    idx = random.randint(0,len(kmnist)-1)

    img = kmnist[idx]

    if show:

        plt.imshow(img)

        plt.show()

    return img, kmnist_mapping.iloc[kmnist_labels[idx]].codepoint
sample = get_kmnist(show=True)
def get_kuzushiji_kanji(show=False):

    kanji_list = os.listdir("../input/kuzushiji/kkanji/kkanji2/")

    selected_kanji = random.choice(kanji_list)

    image_list = os.listdir("../input/kuzushiji/kkanji/kkanji2/"+selected_kanji)

    selected_image = random.choice(image_list)

    image=cv2.imread("../input/kuzushiji/kkanji/kkanji2/{}/{}".format(selected_kanji,selected_image))

    if show:

        plt.imshow(image)

        plt.show()

    return image, selected_kanji
sample = get_kuzushiji_kanji(show=True)
def get_new_page(page_dimensions = (3900,2400), binary_mask=True):

    

    page = np.zeros(page_dimensions)

    labels = ""

    

    number_of_columns = random.randint(3,8)

    symbols_per_columns = random.randint(10,20)

    margin = random.randint(30,200)

    symbol_size = random.randint(100,250)

    

    for row in range(1,symbols_per_columns):

        for col in range(1,number_of_columns):

            x_location = int((page.shape[1]-margin*2)*col/number_of_columns)

            y_location = int((page.shape[0]-margin*2)*row/symbols_per_columns)

            symbol_size_variation = random.randint(0,10)

            #randomly pick a subtype from the KMNIST dataset.

            condition = random.randint(1,3)

            if condition==1:

                symbol, label = get_kmnist()

                symbol = cv2.resize(symbol, (symbol_size-symbol_size_variation, symbol_size-symbol_size_variation)) 

                if binary_mask:

                    ret,symbol = cv2.threshold(symbol.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                page[y_location:y_location+symbol_size-symbol_size_variation,x_location:x_location+symbol_size-symbol_size_variation] = symbol

            elif condition==2:

                symbol, label = get_k49()

                symbol = cv2.resize(symbol, (symbol_size-symbol_size_variation, symbol_size-symbol_size_variation)) 

                if binary_mask:

                    ret,symbol = cv2.threshold(symbol.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                page[y_location:y_location+symbol_size-symbol_size_variation,x_location:x_location+symbol_size-symbol_size_variation] = symbol

            elif condition==3:

                symbol, label = get_kuzushiji_kanji()

                symbol = cv2.resize(symbol, (symbol_size-symbol_size_variation, symbol_size-symbol_size_variation)) 

                symbol = symbol[:,:,0]

                if binary_mask:

                    ret,symbol = cv2.threshold(symbol.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                page[y_location:y_location+symbol_size-symbol_size_variation,x_location:x_location+symbol_size-symbol_size_variation] = symbol

            #Bug fixed in version 9. 

            labels += "{} {} {} {} {} ".format(label,str(x_location),str(y_location),str(symbol_size-symbol_size_variation),str(symbol_size-symbol_size_variation))



    return page, labels
def print_random_pages():

    sample_number = 10

    fig = plt.figure(figsize = (20,sample_number))

    for i in range(0,sample_number):

        ax = fig.add_subplot(2, 5, i+1)

        ax.imshow(get_new_page()[0])

    plt.tight_layout()

    plt.show()
print_random_pages()
dest_dir = "../working/synthetic-kmnist-pages"

os.mkdir(dest_dir)
kuzushiji_df = pd.DataFrame(columns=["image_id","labels"])

#The number of output files is limited to 500

number_of_pages = 400

with tqdm(total=number_of_pages) as pbar:

    for idx in range(0,number_of_pages):

        pbar.update(1)

        filename = "{}.png".format(idx)

        binary_image, labels = get_new_page()

        cv2.imwrite("{}/{}".format(dest_dir, filename), binary_image)

        kuzushiji_df = kuzushiji_df.append({"image_id":filename,"labels":labels}, ignore_index=True)

kuzushiji_df.to_csv("{}/synthetic_kmnist_pages.csv".format(dest_dir),index=False)
kuzushiji_df.head()