from PIL import Image, ImageDraw, ImageFont

from os import listdir

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

import os

import gc

import sys

import seaborn as sns

import cv2

import shutil

from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm_notebook as tqdm



%matplotlib inline



print (os.listdir('../input/'))

print("Ready!")
fontsize = 50



# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip



font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
# This function takes in a filename of an image, and the labels in the string format given in train.csv, and returns an image containing the bounding boxes and characters annotated

def visualize_training_data(image_fn, labels):

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 5)

    

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')

    bbox_canvas = Image.new('RGBA', imsource.size)

    char_canvas = Image.new('RGBA', imsource.size)

    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character

    char_draw = ImageDraw.Draw(char_canvas)



    for codepoint, x, y, w, h in labels:

        x, y, w, h = int(x), int(y), int(w), int(h)

        char = unicode_map[codepoint] # Convert codepoint to actual unicode character



        # Draw bounding box around character, and unicode character next to it

        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))

        char_draw.text((x + w + fontsize/4, y + h/2 - fontsize), char, fill=(0, 0, 255, 255), font=font)



    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    return np.asarray(imsource)







def visualize_test_data(image_fn):

    

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    return np.asarray(imsource)
# This function takes in a filename of an image, and the labels in the string format given in a submission csv, and returns an image with the characters and predictions annotated.

def visualize_predictions(image_fn, labels):

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 3)

    

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')

    bbox_canvas = Image.new('RGBA', imsource.size)

    char_canvas = Image.new('RGBA', imsource.size)

    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character

    char_draw = ImageDraw.Draw(char_canvas)



    for codepoint, x, y in labels:

        x, y = int(x), int(y)

        char = unicode_map[codepoint] # Convert codepoint to actual unicode character



        # Draw bounding box around character, and unicode character next to it

        bbox_draw.rectangle((x-10, y-10, x+10, y+10), fill=(255, 0, 0, 255))

        char_draw.text((x+25, y-fontsize*(3/4)), char, fill=(255, 0, 0, 255), font=font)



    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    return np.asarray(imsource)
!ls ../input/
PATH = '../input/kuzushiji-recognition/'

df_train = pd.read_csv(PATH+'train.csv')

df_test = os.listdir(PATH+'test_images/')

unicode_map = {codepoint: char for codepoint, char in pd.read_csv(PATH+'unicode_translation.csv').values}

print ("TRAIN: ", df_train.shape)

print ("TEST: ", len(df_test))

df_train.head()
df_train.isnull().sum()
#df_train.dropna(inplace=True)

df_train.reset_index(inplace=True, drop=True)

print ("TRAIN: ", df_train.shape)
chars = {}



for i in range (df_train.shape[0]):

    try:

        a = [x for x in df_train.labels.values[i].split(' ') if x.startswith('U')]

        n_a = int(len(a))        

        for j in a:

            if j not in chars: chars[j]=1

            else:

                chars[j]+=1

                

        a = " ".join(a)

        

    except AttributeError:

        a = None

        n_a = 0

        

    df_train.loc[i,'chars'] = a

    df_train.loc[i,'n_chars'] = n_a

    

df_train.head()
print ("MAX chars in a picture= ", df_train.n_chars.max())

print ("MIN chars in a picture= ", df_train.n_chars.min())

print ("MEAN chars in a picture= ", df_train.n_chars.mean())
chars = pd.DataFrame(list(chars.items()), columns=['char', 'count'])

chars['jp_char'] = chars['char'].map(unicode_map)

print (" >> Chars dataframe <<")

print ("Number of chars: ",chars.shape[0])

chars.to_csv("chars_freq.csv",index=False)

chars.head()
chars.sort_values(by=['count'], ascending=False).head(10).reset_index()
sns.set(style="whitegrid")

plt.figure(figsize=(22,20))

ax = sns.barplot(y="char", x="count", data=chars.sort_values(by=['count'], ascending=False).head(100))

ax.set_title("Character frequency in images (top 100)")

plt.show()
print ('Total chars', chars.shape[0])

print ('< 10 freq', chars[chars['count'] <= 10].shape[0])
rare = chars[chars['count'] <= 10]

print (rare.shape)

rare.head()
rare.to_csv('rare_chars.csv', index=False)
lowchar = df_train[df_train.n_chars <= 10]

print ('lowchar images ',lowchar.shape[0], lowchar.shape[0]/ df_train.shape[0])

lowchar.head()
for image_fn in lowchar.image_id:

    image_fn = '../input/train_images/'+image_fn+'.jpg'

    imsource = Image.open(image_fn).convert('RGBA')

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    imsource = np.asarray(imsource)

    plt.figure(figsize=(10, 10))

    plt.title(image_fn)

    plt.axis("off")

    plt.imshow(imsource, interpolation='lanczos')

    plt.show() 
print (lowchar.shape)

lowchar.dropna(inplace=True)

print (lowchar.shape)

lowchar.to_csv('train_lowchar.csv',index=False)
df_train["title"]= df_train["image_id"].str.split("_", n = 1, expand = True)[0]

#df_train["chapter"]= df_train["image_id"].str.split("_", n = 2, expand = True)[1]

#df_train["page"]= df_train["image_id"].str.split("_", n = 3, expand = True)[2]

df_train.head()
print (df_train['title'].nunique())

df_train['title'].unique()[0:10]
book = df_train[df_train['title']== '200006663'].reset_index(drop=True)

book
def visualize_book(title, df_train):

    df_train[df_train['title']== title].reset_index(drop=True)

    print ('Book ', title)

    for i in book.index:

        img,labels,_,_,_ = book.values[i]

        viz = visualize_training_data(PATH+'train_images/{}.jpg'.format(img), labels)

        plt.figure(figsize=(15, 15))

        plt.title(img)

        plt.axis("off")

        plt.imshow(viz, interpolation='lanczos')

        plt.show()
visualize_book('200006663', df_train)
visualize_book('200014685-00002', df_train)
visualize_book('200014685-00003', df_train)
print ("TRAIN: ", df_train.shape)
np.random.seed(1337)



for i in range(2):

    img,labels,_,_,_ = df_train.values[np.random.randint(len(df_train))]

    viz = visualize_training_data(PATH+'train_images/{}.jpg'.format(img), labels)

    plt.figure(figsize=(15, 15))

    plt.title(img)

    plt.axis("off")

    plt.imshow(viz, interpolation='lanczos')

    plt.show()
for img in df_test[0:2]:

    viz = visualize_test_data(PATH+'test_images/{}'.format(img))

    plt.figure(figsize=(15, 15))

    plt.title(img)

    plt.axis("off")

    plt.imshow(viz, interpolation='lanczos')

    plt.show()
def get_char(img_id, labels):

    

    image_fn = '../input/train_images/{}.jpg'.format(img_id)

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 5)

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')

    img = np.asarray(imsource.convert("RGB"))



    info = []

    

    for idx, (codepoint, x, y, w, h) in enumerate(labels):

        x, y, w, h = int(x), int(y), int(w), int(h)

        try:

            char = unicode_map[codepoint] # Convert codepoint to actual unicode character

        except KeyError:

            char = "e" # https://www.kaggle.com/c/kuzushiji-recognition/discussion/100712#latest-580747

        

        # crop char

        #print (idx,x,y,w,h,char)

        crop_img = img[y:y+h, x:x+w]

        result = Image.fromarray(crop_img, mode='RGB')

        name = img_id+'_{}.jpg'.format(idx)

        result.save('kminst/'+name)

        

        info.append((name,codepoint))

        

    del imsource, img, result, name

    gc.collect()

    

    return info
!mkdir kminst

!ls
'''

generated = 0

info = []



for i in tqdm(df_train.index):

    img, labels,_,_ = df_train.values[i]

    info += get_char(img, labels)

    generated+= int(df_train[df_train['image_id']==img].n_chars)

    

    if (i+1)%500 == 0 or i==df_train.index[-1]:

        # save memory

        shutil.make_archive('kminst_'+str(i//500), 'zip', 'kminst')

        print (i+1,"\t>> generated ...", generated)

        shutil.rmtree('kminst', ignore_errors=True)

        os.mkdir('kminst')

'''
!rm -r kminst

!ls
#shutil.make_archive('kminst', 'zip', 'kminst')

#!rm -r kminst

#!ls
info[0:5]
infok = pd.DataFrame(columns=['char_id','unicode'])

infok['char_id'] = [i[0] for i in info]

infok['unicode'] = [i[1] for i in info]

print ("TOTAL KMNIST = ", infok.shape[0])

infok.to_csv('info.csv',index=False)

infok.head()
example = "200021660-00023_2"

"100249537_00013_2"

"hnsd007-039"

"100249537_00003_2"

"200014685-00003_1"

"200014685-00016_2"
def get_char_example(image_fn, labels):

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 5)

    

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')

    img = np.asarray(imsource.convert("RGB"))

    bbox_canvas = Image.new('RGBA', imsource.size)

    char_canvas = Image.new('RGBA', imsource.size)

    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character

    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y, w, h in labels:

        x, y, w, h = int(x), int(y), int(w), int(h)

        char = unicode_map[codepoint] # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it

        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))

        char_draw.text((x + w + fontsize/4, y + h/2 - fontsize), char, fill=(0, 0, 255, 255), font=font)

        

        # crop char

        print (x,y,w,h,char)

        crop_img = img[y:y+h, x:x+w]

        plt.axis("off")

        plt.imshow(np.asarray(crop_img), interpolation='lanczos')

        plt.show()



    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    return np.asarray(imsource)
img, labels,_,_ = df_train[df_train['image_id']=="umgy004-011"].values[0]

print ("IMAGE: ", img)

print (">> chars:", int(df_train[df_train['image_id']==img].n_chars),"\n")



viz = get_char_example(PATH+'train_images/{}.jpg'.format(img), labels)

plt.figure(figsize=(15, 15))

plt.title(img)

plt.axis("off")

plt.imshow(viz, interpolation='lanczos')

plt.show()
image_fn = '../input/test_images/test_030d9355.jpg'

pred_string = 'U+306F 1231 1465 U+304C 275 1652 U+3044 1495 1218 U+306F 436 1200 U+304C 800 2000 U+3044 1000 300' # Prediction string in submission file format

viz = visualize_predictions(image_fn, pred_string)



plt.figure(figsize=(15, 15))

plt.imshow(viz, interpolation='lanczos')