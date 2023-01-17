import numpy as np 

import pandas as pd 

import os

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

#set directory

MAIN_DIR = '../input/prostate-cancer-grade-assessment'

# load data

train = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv'))

# useful function for plotting counts

def plot_count(df, feature, title='', size=2):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='deep')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 9,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()

# useful function for plotting relative distributions 

def plot_relative_distribution(df, feature, hue, title='', size=2):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(x=feature, hue=hue, data=df, palette='deep')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
print(train.head())
plot_count(df=train, feature='data_provider', title = 'Data provider - count and percentage share')
plot_relative_distribution(df=train, feature='isup_grade', hue='data_provider', title = 'relative distribution of ISUP grade across data_provider', size=2)
import skimage.io

import PIL

path = os.path.join(MAIN_DIR, 'train_images')

biopsy = skimage.io.MultiImage(os.path.join(path, train.image_id.tolist()[0]+'.tiff'))

display(PIL.Image.fromarray(biopsy[-1]))
x = 1450

y = 1950

level = 1

width = 512

height = 512



patch = biopsy[level][y:y+height, x:x+width]





plt.figure()

plt.imshow(patch)

plt.show()
x = 1450*4

y = 1950*4

level = 0

width = 512

height = 512



patch = biopsy[level][y:y+height, x:x+width]





plt.figure()

plt.imshow(patch)

plt.show()
train_img_index = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv')).set_index('image_id')

import matplotlib

import matplotlib.pyplot as plt

mask_dir = '../input/prostate-cancer-grade-assessment/train_label_masks/'



#creating a function to take an image id and return an array of the image or mask as required

def id2array(id, type):

    if type == 'mask':

        if os.path.isfile(os.path.join(mask_dir + id + '_mask.tiff')) == True:

            array = skimage.io.MultiImage(os.path.join(mask_dir + id + '_mask.tiff'))[-1]

        else:

            print(no_mask_array)

            array = 0

    else:

        array = skimage.io.MultiImage(os.path.join(path + '/' + id + '.tiff'))[-1]

    return array



# we set up two colour maps as described above

cmap_rad = matplotlib.colors.ListedColormap(['black', 'gray', 'gray', 'yellow', 'orange', 'red'])

cmap_kar = matplotlib.colors.ListedColormap(['black', 'gray', 'purple'])





# this function will take 5 image ids and display an image and the related mask

def plot5(ids):

    img_arrays = [id2array(item, 'image') for item in ids]

    mask_arrays = [id2array(item, 'mask') for item in ids]

    fig, axs = plt.subplots(5, 2, figsize=(15,25))

    for i in range(0,5):

        image_id = ids[i]

        data_provider = train_img_index.loc[image_id, 'data_provider']

        gleason_score = train_img_index.loc[image_id, 'gleason_score']

        axs[i, 0].imshow(img_arrays[i])

        mask_array = mask_arrays[i]

        if data_provider == 'karolinska':

            axs[i, 1].imshow(mask_array[:,:,0], cmap=cmap_kar, interpolation='nearest', vmin=0, vmax=2)

        else:

            axs[i, 1].imshow(mask_array[:,:,0], cmap=cmap_rad, interpolation='nearest', vmin=0, vmax=5)

        for j in range(0,2):

            axs[i,j].set_title(f"ID: {image_id}\nSource: {data_provider} Gleason: {gleason_score}")

    plt.show()
plot5(train.image_id.tolist()[100:105])
from tensorflow.keras.models import load_model

model3 = load_model('../input/gl3-panda-training/gl3_model')

model3.load_weights('../input/gl3-panda-training/best.hdf5')



model4 = load_model('../input/gl4-panda-training/gl4_model')

model4.load_weights('../input/gl4-panda-training/best.hdf5')



model5 = load_model('../input/gl5-panda-training-hup/gl5_model')

model5.load_weights('../input/gl5-panda-training-hup/best.hdf5')



models = [model3, model4, model5]



DATA = '../input/prostate-cancer-grade-assessment/test_images'

TEST = '../input/prostate-cancer-grade-assessment/test.csv'

TRAIN = '../input/prostate-cancer-grade-assessment/train.csv'

SAMPLE = '../input/prostate-cancer-grade-assessment/sample_submission.csv'



testdf = pd.read_csv(TEST)

ids = testdf.image_id.tolist()

dp = testdf.data_provider.tolist()



N = 30

sz = 224



sub_df = pd.read_csv(SAMPLE)

import skimage.io

def id2tiles(id):

    results = []

    if os.path.exists(DATA):

        img = skimage.io.MultiImage(os.path.join(DATA,id+'.tiff'))[-2]

        shape = img.shape

        pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                    constant_values=255)

        img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

        img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

        if len(img) < N:

            img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

        idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

        img = img[idxs]

        for i in range(len(img)):

            rel_img = img[i]

            rel_img[:,:,0] = ((rel_img[:,:,0]/255) - 0.8094)/ 0.4055

            rel_img[:,:,1] = ((rel_img[:,:,1]/255) - 0.6067)/ 0.5094

            rel_img[:,:,2] = ((rel_img[:,:,2]/255) - 0.7383)/ 0.4158

            results.append(rel_img)

    return(results)

    

model_pred_k = load_model('../input/training-for-model-outputs-to-grade/pred_model_k')

model_pred_k.load_weights('../input/training-for-model-outputs-to-grade/model_kbest.hdf5')



model_pred_r = load_model('../input/fork-of-training-for-model-outputs-to-grade/pred_model_r')

model_pred_r.load_weights('../input/fork-of-training-for-model-outputs-to-grade/model_rbest.hdf5')



list_of_list_of_tiles = []

list_of_list_of_tiles = [id2tiles(item) for item in ids]



isups = []

preds_list = []

def tiles2pred(tiles):

    if os.path.exists(DATA):

        new_images = [np.reshape(item, [1,224,224,3]) for item in tiles]

        preds = np.zeros((N,3))

        for i in range(0,N):

            for j in range(0,3):

                preds[i,j] = models[j].predict(new_images[i])

        preds_list.append(preds)

        

for item in list_of_list_of_tiles:

    tiles2pred(item)

    

model_used = []

index = 0

for item in preds_list:

    features = item.reshape(1,90)

    if dp[index] == 'karolinska':

        pred_array = model_pred_k.predict(features)

        model_used.append('k')

        isup = pred_array.argmax()

        isups.append(isup)

    else:

        pred_array = model_pred_r.predict(features)

        model_used.append('r')

        isup = pred_array.argmax()

        isups.append(isup)

    index = index+1



if os.path.exists(DATA):

    sub_df = pd.DataFrame({'image_id': ids, 'isup_grade': isups})

    sub_df.to_csv("submission.csv", index=False)



sub_df.to_csv("submission.csv", index=False)