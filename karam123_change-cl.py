# importing libraries used in this notebook



from keras.preprocessing import image

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

import numpy as np

from sklearn.cluster import KMeans

import os, shutil, glob, os.path

from PIL import Image

image.LOAD_TRUNCATED_IMAGES = True

from tqdm.notebook import tqdm 

import glob

import math

import pickle

%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline
# command to download the dataset given in the problem statement

!wget --header="Host: doc-0g-2k-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9" --header="Referer: https://drive.google.com/" --header="Cookie: AUTH_t749vl4skj7cn551ncp80lljbg5da3jf_nonce=49nkonhl8kcc2" --header="Connection: keep-alive" "https://doc-0g-2k-docs.googleusercontent.com/docs/securesc/1s3jo9hqchijmcbr6j69dtugek0datcv/9e7mrnehkvsggebq8hvgcd27sm2mu26p/1596822900000/07496480791912752493/12633858806009087463/1VT-8w1rTT2GCE5IE5zFJPMzv7bqca-Ri?e=download&authuser=0&nonce=49nkonhl8kcc2&user=12633858806009087463&hash=m498o18dip5brnhau0oqhdkbcc1idmp7" -c -O 'dataset.zip'
# unziping the dataset folder

!unzip dataset.zip 
# location where the dataset is downloaded

out_dir = '/kaggle/working/dataset'
# storing the path of all the jpj file in filelist from our dataset

filelist = glob.glob(os.path.join(out_dir,'*.jpg'))
# sorting the path of our images

filelist.sort()

featurelist = []
#printing the path of first five images

filelist[0:5]
# using weights of vgg16 model trained on imagenet data for featurization

model = VGG16(weights='imagenet', include_top=False)
for i, imagepath in tqdm(enumerate(filelist)):

    img = image.load_img(imagepath)

    img_data = image.img_to_array(img)

    img_data = np.expand_dims(img_data, axis=0)

    img_data = preprocess_input(img_data)

    features = np.array(model.predict(img_data))

    featurelist.append(features.flatten())
# dumping the features of our images in a pickle file so that it can be used for further use

with open('featurelist', 'wb') as f: 

    pickle.dump(featurelist, f)
#loading the featurelist that consist of bottleneck features of all the images

with open('/kaggle/working/featurelist', 'rb') as f: 

    featurelist = pickle.load(f) 
# printing lenght of filelist

len(filelist)
# printing lenght of featurelist

len(featurelist)
#sorting the path of jpg images

filelist.sort()

filelist[0:10]
# loading the weights of vgg16 model that is train on imagenet data that consists of 1000 classes

model = VGG16(weights='imagenet', include_top=False)
# function that return the weight of a vector



def square_rooted(x):

    return math.sqrt(sum([a*a for a in x]))





# function to return the cosine similarity between two vectors



def cosine_similarity(x,y):

    numerator = sum(a*b for a,b in zip(x,y))

    denominator = square_rooted(x)*square_rooted(y)

    return numerator/float(denominator)

# function to find cosine similarity of a given image with all the images in the dataset.



def find_similar(input_image_path,input_features,filelist):

    for i in tqdm(range(len(filelist))):

        if filelist[i]== input_image_path:

            continue

        dic_store[filelist[i]] = cosine_similarity(input_features,featurelist[i])
# function to return the bottleneck feature of a given input image



def input_features(input_image_path):

    img = image.load_img(input_image_path)

    img_data = image.img_to_array(img)

    img_data = np.expand_dims(img_data, axis=0)

    img_data = preprocess_input(img_data)

    features = np.array(model.predict(img_data))  

    return features.flatten()
# function to represent the images as described in the problem statement



def print_images(N,input_image_path,sorted_x):

    columns = 3

    rows = ceil(N/3) +1

    fig=plt.figure(figsize=(8, 8))

    input_image = mpimg.imread(input_image_path)

    fig.add_subplot(rows, columns, 1)

    plt.imshow(input_image)



    i=3

    for key in sorted_x:

        i+=1

        if(i>N+3):

            break

        fig.add_subplot(rows, columns, i)

        img=mpimg.imread(key[0])

        imgplot = plt.imshow(img)

    plt.show()
# finding the cosine similarity between our image 3 and image 5



cosine_similarity(featurelist[2],featurelist[5]) # for demonstration
# dictionary to store the cosine similarity of input image with other images

dic_store = {}
# image for which you have to find similar images 



input_image_path = '/kaggle/working/dataset/1000.jpg' # write image path
# diaplaying input image

im = Image.open(input_image_path)

im
# will return input features of the input image

features = input_features(input_image_path)
# will find the cosine similarity of input image with other images



find_similar(input_image_path,features,filelist)
# sorting dictionary in descending order on values

sorted_x = sorted(dic_store.items(), key=lambda kv: kv[1],reverse = True)
# printing top ten cosine similaity values and the images path

sorted_x[0:10]
# define the number of simailar images you want

N = 10 #you can take any number
# will return the N images similar to input image

print_images(N,input_image_path,sorted_x)