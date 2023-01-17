import numpy as np

from PIL import Image

import glob

import pickle



def read_image(image_path):

    image = Image.open(image_path)

    image = np.asarray(image)

    return image



def read_images():

    classes = glob.glob('./data/*')

    #print(classes) # ['./data/aloe_vera_gel', './data/baby_shoes'....]



    raw_images = []

    files_name = []

    for classes_path in classes:

        image_paths = glob.glob(f'{classes_path}/*.jpg')

        for image_path in image_paths:

            image = read_image(image_path) # image[0][0] -> array([42, 50, 37], dtype=uint8) RGB

            raw_images.append(image)

            files_name.append(image_path)

    return raw_images, files_name
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import matplotlib.image as mimage

plt.rcParams.update({'figure.max_open_warning': 0})



def draw_by_path(path, title):

    plt.figure()

    plt.suptitle(title, fontsize=20)

    _img = mimage.imread(path)

    plt.imshow(_img)

    

def draw_grid(ranking, name):

    #gs = gridspec.GridSpec(1, 10, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)

    gs = gridspec.GridSpec(1, 10)

    plt.figure(figsize=(15,1))

    plt.suptitle(name, fontsize=10)

    for g, idx in zip(gs, ranking[:10]):            

        ax = plt.subplot(g)

        _img = mimage.imread(files_name[idx])

        ax.imshow(_img)

        ax.set_xticks([])

        ax.set_yticks([])

        if ranking[0] == idx:

            ax.spines['right'].set_color('red')

            ax.spines['left'].set_color('red')

            ax.spines['bottom'].set_color('red')

            ax.spines['top'].set_color('red') 

            ax.set_title("input")

            ax.title.set_color('red')

        #ax.set_aspect('auto')     
def cosine_similarity(a, b):

    a = np.array(a)

    b = np.array(b)

    o = np.dot(a,b.T)

    return o





def L1(target, all_img):

    target = np.array(target)

    all_img = np.array(all_img)

    return np.sum(abs(all_img-target), axis = 1)





def to_color_hist(raw_images, quant_size = 8):

    hist_images = []

    rang = 256 / quant_size

    for count, image in enumerate(raw_images):

        v = [0] * (quant_size * quant_size * quant_size)

        image = np.asarray(image).reshape(-1,3)

        #image = np.asarray(image)

        

        col1 = image[:,0]/rang

        col2 = image[:,1]/rang

        col3 = image[:,2]/rang

        col1, col2, col3 = col1.astype(int), col2.astype(int), col3.astype(int)

        quant_li = col1 + col2*quant_size + col3*quant_size*quant_size

        

        # normalize

        for num in quant_li:

            v[int(num)]+= 1

        

        hist_images.append(v)



    return hist_images



def evaluate(images, files_name, cls = 0, matric = L1):

    if matric == cosine_similarity:

        ascending = -1

    else:

        ascending = 1

    

    from numpy import linalg as LA

    images = [img/LA.norm(img) for img in images]

    

    MAP = 0

    for leave_i in range(20):

        leave_i = 20*cls + leave_i

        score = matric(images[leave_i], images)

        ranking = score.argsort()[::ascending] # [100,120] 第一名的index 第二名的index

        AP = 0

        hit = 0

        count = 0

        for j in ranking:

            if j != leave_i:

                count += 1

                if files_name[j].split('/')[2] == files_name[leave_i].split('/')[2]:

                    hit += 1

                    AP += hit/count

                if hit == 19:

                    break

        MAP += AP/19



    #draw_by_path(files_name[leave_i], "input")

    if cls < 3:

        draw_grid(ranking, files_name[leave_i].split('/')[2])



    return MAP/20
raw_images, files_name = read_images()

hist_images = to_color_hist(raw_images, quant_size = 8)

import pickle

with open('hist_images.pkl', 'wb') as f:

    pickle.dump(hist_images, f, pickle.HIGHEST_PROTOCOL)

with open('files_name.pkl', 'wb') as f:

    pickle.dump(files_name, f, pickle.HIGHEST_PROTOCOL)



import pickle

with open('files_name.pkl', 'rb') as f:

    files_name = pickle.load(f)



with open('hist_images.pkl', 'rb') as f:

    hist_images = pickle.load(f)
import pandas as pd

df_col = [files_name[i*20].split('/')[2] for i in range(25)] 



df = pd.DataFrame(columns = ["mean"] + df_col)
M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(hist_images, files_name, i, L1)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'RGB histogram L1'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(hist_images, files_name, i, cosine_similarity)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'RGB histogram cosine similarity'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
import cv2

def get_HSV():

    classes = glob.glob('./data/*') #['./data/aloe_vera_gel', './data/baby_shoes'....]



    hsv_pic = []

    for classes_path in classes:

        image_paths = glob.glob(f'{classes_path}/*.jpg')

        for image_path in image_paths:

            img = cv2.imread(image_path)

            img = cv2.resize(img, (128, 128))

            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

            

            v = [0] * (18 * 4 * 4)

            img = np.asarray(img).reshape(-1,3)

            #image = np.asarray(image)



            col1 = img[:,0]/(180/18)

            col2 = img[:,1]/(256/4)

            col3 = img[:,2]/(256/4)

            col1, col2, col3 = col1.astype(int), col2.astype(int), col3.astype(int)

            quant_li = col1 + col2*18 + col3*18*4



            # normalize

            for num in quant_li:

                v[int(num)]+= 1

            hsv_pic.append(v)

    return hsv_pic



hsv_pic = get_HSV()

M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(hsv_pic, files_name, i, L1)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'HSV L1'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
MAP_list = []

for i in range(25):

    MAP = evaluate(hsv_pic, files_name, i, cosine_similarity)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'HSV cosine similarity'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
import cv2

def get_freq_hist():

    classes = glob.glob('./data/*') #['./data/aloe_vera_gel', './data/baby_shoes'....]



    freq_hist = []

    for classes_path in classes:

        image_paths = glob.glob(f'{classes_path}/*.jpg')

        for image_path in image_paths:

            img = cv2.imread(image_path,0) #GBR

            img = cv2.resize(img, (128, 128))

            f = np.fft.fft2(img)

            fshift = np.fft.fftshift(f)

            magnitude_spectrum = 20*np.log(np.abs(fshift))

            magnitude_spectrum = magnitude_spectrum.reshape((-1))

            freq_hist.append(magnitude_spectrum.tolist())

            

    return freq_hist

img = cv2.imread("./data/aloe_vera_gel/aloe_vera_gel_1.jpg", 0) #GBR

f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)

magnitude_spectrum = 20*np.log(np.abs(fshift))

#plt.figure()

plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
freq_hist = get_freq_hist()

M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(freq_hist, files_name, i, L1)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'frequency response L1'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
#!/usr/bin/env python

 

import numpy as np

import cv2

 

def draw_gabor(image1, image2):

    plt.subplot(121),plt.imshow(image1, cmap='gray')

    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122),plt.imshow(image2, cmap='gray')

    plt.title('Gabor'), plt.xticks([]), plt.yticks([])

    plt.show()

    

def draw_img(img, title):

    plt.figure(figsize=(1,1))

    plt.imshow(img, cmap='gray')#, cmap = 'gray')

    plt.title(title), plt.xticks([]), plt.yticks([])

    plt.show()

    

def build_filters():

    filters = []

    #ksize = 31

    ksize = 21

    #for theta in np.arange(0, np.pi, np.pi / 16):

    for theta in np.arange(0, np.pi, np.pi / 4):

        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) #Size ksize, double sigma, double theta

        kern /= 1.5*kern.sum()

        #draw_gabor(kern, cv2.getGaborKernel((ksize, ksize), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F))

        filters.append(kern)

    return filters

 

def process(img, filters):

    accum = np.zeros_like(img)

    for kern in filters:

        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)

        np.maximum(accum, fimg, accum) # (a, b, c) save to c'''

    return accum



def get_gabor_pic():

    classes = glob.glob('./data/*') #['./data/aloe_vera_gel', './data/baby_shoes'....]



    gabor_hist = []

    for classes_path in classes:

        image_paths = glob.glob(f'{classes_path}/*.jpg')

        for image_path in image_paths:

            img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (128, 128))

            #print(img.shape)

            filters = build_filters()

            res1 = process(img, filters)

            gabor_hist.append(res1.reshape(-1).tolist())

            

    return gabor_hist



def process_to_hist(img, filters):

    accum = np.zeros_like(img)

    hist = []

    for kern in filters:

        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)

        fimg = fimg.reshape(-1)

        hist.append(sum(fimg))

    return hist



def get_gabor_hist():

    classes = glob.glob('./data/*') #['./data/aloe_vera_gel', './data/baby_shoes'....]



    gabor_hist = []

    for classes_path in classes:

        image_paths = glob.glob(f'{classes_path}/*.jpg')

        print(classes_path)

        for image_path in image_paths:

            img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (128, 128))

            #print(img.shape)

            filters = build_filters()

            his = process_to_hist(img, filters)

            his = np.asarray(his)

            gabor_hist.append(his.reshape(-1))

    return gabor_hist

fil = build_filters()

    

plt.figure()

plt.subplot(141),plt.imshow(fil[0], cmap = 'gray')

plt.title('theta=0*pi'), plt.xticks([]), plt.yticks([])

plt.subplot(142),plt.imshow(fil[1], cmap = 'gray')

plt.title('theta=pi/4'), plt.xticks([]), plt.yticks([])

plt.subplot(143),plt.imshow(fil[2], cmap = 'gray')

plt.title('theta=2*pi/4'), plt.xticks([]), plt.yticks([])

plt.subplot(144),plt.imshow(fil[3], cmap = 'gray')

plt.title('theta=3*pi/4'), plt.xticks([]), plt.yticks([])



plt.show()
img_fn = './zebra.png'

img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)

filters = build_filters()

res1 = process(img, filters)

draw_gabor(img, res1)
#gabor_hist = get_gabor_hist()

gabor_hist = get_gabor_pic()



M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(gabor_hist, files_name, i, L1)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'Gabor L1'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
fusion_feature = [i+j for i,j in zip(hist_images, gabor_hist)]



M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(fusion_feature, files_name, i, L1)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'fusion feature L1'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
import numpy as np

from sklearn import random_projection



hist_images = np.asarray(hist_images) #(500,512) 500*512

transformer = random_projection.GaussianRandomProjection(int(hist_images.shape[1]*0.25))

X_new = transformer.fit_transform(hist_images)



M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(X_new, files_name, i, L1)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'RP(25%)'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
import numpy as np

from sklearn import random_projection



hist_images = np.asarray(hist_images) #(500,512) 500*512

transformer = random_projection.GaussianRandomProjection(int(hist_images.shape[1]*0.5))

X_new = transformer.fit_transform(hist_images)



M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(X_new, files_name, i, L1)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'RP(50%)'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
from keras.applications import VGG16

from keras.applications import InceptionV3

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

import numpy as np



model = InceptionV3(weights='imagenet', include_top=False)



def get_deep_feature():

    classes = glob.glob('./data/*') #['./data/aloe_vera_gel', './data/baby_shoes'....]



    deep_feature = []

    for classes_path in classes:

        image_paths = glob.glob(f'{classes_path}/*.jpg')

        print(classes_path)

        for image_path in image_paths:

            img = image.load_img(image_path, target_size=(224, 224))

            x = image.img_to_array(img)

            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)



            features = model.predict(x)

            features = features.flatten()

            deep_feature.append(features.tolist())

    return deep_feature

deep_feature = get_deep_feature()

M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(deep_feature, files_name, i, cosine_similarity)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'Deep feature L1'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]

print(f"Mean MAP: {M_MAP/25}")

df
import numpy as np

from sklearn import random_projection



deep_feature = np.asarray(deep_feature) #(500,512) 500*512

transformer = random_projection.GaussianRandomProjection(int(deep_feature.shape[1]*0.5))

X_new = transformer.fit_transform(deep_feature)



M_MAP = 0

MAP_list = []



for i in range(25):

    MAP = evaluate(X_new, files_name, i, L1)

    M_MAP += MAP

    MAP_list.append(MAP)

    

df = df.append(pd.DataFrame([MAP_list], columns = df_col))

df = df.rename(index={df.index[-1]: 'Random projection L1'})

df['mean'] = df[df_col].mean(axis=1)

df = df[["mean"] + df_col]
print(f"Mean MAP: {M_MAP/25}")

df