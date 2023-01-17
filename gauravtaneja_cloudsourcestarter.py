import os

import pandas as pd

import random

import numpy as np

from matplotlib import pyplot as plt

from glob import glob

from PIL import Image

import imageio

import cv2
data_path = '/kaggle/input/understanding_cloud_organization'

train_csv_path = os.path.join('/kaggle/input/understanding_cloud_organization','train.csv')

train_image_path = os.path.join('/kaggle/input/understanding_cloud_organization','train_images')



# set paths to train and test image datasets

TRAIN_PATH = '../input/understanding_cloud_organization/train_images/'

TEST_PATH = '../input/understanding_cloud_organization/test_images/'

def load_processdata(loc,**kwargs):

    nomaskvalue = kwargs.get('nomaskvalue',-1)

    

    train_df = pd.read_csv(loc).fillna(nomaskvalue)

    

    # split column

    split_df = train_df["Image_Label"].str.split("_", n = 1, expand = True)

    # add new columns to train_df

    train_df['img'] = split_df[0]

    train_df['lbl'] = split_df[1]

    

    del split_df

    

    # Create labeled cloud type dummies ( but why? idk)

    train_df['fish'] = np.where((train_df['lbl'].str.lower()=='fish') & (train_df['EncodedPixels']!=-1),1,0)

    train_df['sugar'] = np.where((train_df['lbl'].str.lower()=='sugar') & (train_df['EncodedPixels']!=-1),1,0)

    train_df['gravel'] = np.where((train_df['lbl'].str.lower()=='gravel') & (train_df['EncodedPixels']!=-1),1,0)

    train_df['flower'] = np.where((train_df['lbl'].str.lower()=='flower') & (train_df['EncodedPixels']!=-1),1,0)

    

    train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['lbl'], row['EncodedPixels']), axis = 1)



    return train_df



def get_image_sizes(train = True):

    '''

    Function to get sizes of images from test and train sets.

    INPUT:

        train - indicates whether we are getting sizes of images from train or test set

    '''

    if train:

        path = TRAIN_PATH

    else:

        path = TEST_PATH

        

    widths = []

    heights = []

    

    images = sorted(glob(path + '*.jpg'))

    

    max_im = Image.open(images[0])

    min_im = Image.open(images[0])

        

    for im in range(0, len(images)):

        image = Image.open(images[im])

        width, height = image.size

        

        if len(widths) > 0:

            if width > max(widths):

                max_im = image



            if width < min(widths):

                min_im = image



        widths.append(width)

        heights.append(height)

        

    return widths, heights, max_im, min_im



trdf = load_processdata(train_csv_path)

trdf.head()
# Lets look at some data on cloud type occurances

typecols = ['fish','sugar','gravel','flower']



co_occ = trdf.groupby('img')[typecols].sum().T.dot(trdf.groupby('img')[typecols].sum())

import seaborn as sns

sns.heatmap(co_occ, cmap = 'YlGnBu', annot=True, fmt="d")
trdf.head()
from sklearn import svm

from sklearn.model_selection import cross_val_score

for c in typecols:

    print(f'{c}: ')

    yvar=c

    xvars = [i for i in typecols if i!=yvar][0:2]

    X = trdf[xvars]

    y = trdf[yvar]

    clf = svm.SVC(kernel='linear', C=1.0)



    scores = cross_val_score(clf, X, y, cv=5)

    print("    Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf.fit(X, y)

    print(f'    {xvars},{list(clf.coef_)}')
result_list = []

import statsmodels.api as sm

trdf['const'] = 0

for c in typecols:

    print(f'-------{c}-------')

    for i in [1]:

        print(f'        Iteration: {i}')

        trdf_s = trdf.sample(frac=0.8, replace=False, random_state=i)

        yvar = c

        xvars = [i for i in typecols if i!=yvar][0:2]

        X = trdf_s[xvars]

        y = trdf_s[yvar]



        logit = sm.Logit(y, X)

        result = logit.fit()

        result_list.extend([result])

        print(result.summary())



del trdf['const']
# Label count freq

trdf.groupby('img')[typecols].sum().sum(axis=1).plot.hist(title='Freq of # labels per image')

plt.show()



trdf.groupby('img')[typecols].sum().sum(axis=0).plot(kind='bar',color='green',title='Occuarnce of the Cloud Types')

# Function to decode the run length mask

def rle_to_mask(rle_string, height, width):

    '''

    convert RLE(run length encoding) string to numpy array



    Parameters: 

    rle_string (str): string of rle encoded mask

    height (int): height of the mask

    width (int): width of the mask 



    Returns: 

    numpy.array: numpy array of the mask

    '''

    

    rows, cols = height, width

    

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]

        #print(rle_numbers)

        rle_pairs = np.array(rle_numbers).reshape(-1,2)

        #print(rle_pairs)

        img = np.zeros(rows*cols, dtype=np.uint8)

        #print(img)

        for index, length in rle_pairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img

# we will use the following function to decode our mask to binary and count the sum of the pixels for our mask.

def get_binary_mask_sum(encoded_mask):

    mask_decoded = rle_to_mask(encoded_mask, width=2100, height=1400)

    binary_mask = (mask_decoded > 0.0).astype(int)

    return binary_mask.sum()



# calculate sum of the pixels for the mask per cloud formation

trdf['mask_pixel_sum'] = trdf.apply(lambda x: get_binary_mask_sum(x['EncodedPixels']), axis=1)



# Hope I'm doing this right

trdf['mask_pixel_perc'] = trdf['mask_pixel_sum']/(2100*1400)

trdf.head()
trdf.groupby('lbl')['mask_pixel_perc'].describe()
trdf.loc[trdf.mask_pixel_perc>0,:].groupby('lbl')['mask_pixel_perc'].describe()
g = sns.FacetGrid(trdf, col="lbl")

g.map(plt.hist, "mask_pixel_perc")
g = sns.FacetGrid(trdf.loc[trdf.mask_pixel_perc>0,:], col="lbl")

g.map(plt.hist, "mask_pixel_perc")
for i in range(0,5):

    img = cv2.imread(os.path.join(train_image_path, trdf['img'][i]))

    mask_decoded = rle_to_mask(trdf['Label_EncodedPixels'][i][1], img.shape[0], img.shape[1])

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))

    ax[0].imshow(img)

    ax[1].imshow(mask_decoded)
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np



model = ResNet50(weights='imagenet')

top_preds=[]
def get_mask_cloud(img_path, img_id, label, mask):

    img = cv2.imread(os.path.join(img_path, img_id), 0)

    mask_decoded = rle_to_mask(mask, img.shape[0], img.shape[1])

    mask_decoded = (mask_decoded > 0.0).astype(int)

    img = np.multiply(img, mask_decoded)

    return img
# top_preds = []

# for i in range(0,1):#trdf.shape[0]):

#     img_path = os.path.join(train_image_path, trdf['img'][i])

#     img = image.load_img(img_path, target_size=(224, 224))

    

#     img_print = cv2.imread(os.path.join(train_image_path, trdf['img'][i]))

#     mask_decoded = get_mask_cloud(img_print, trdf['lbl'][i], trdf['EncodedPixels'][i])

#     #img = get_mask_cloud(train_image_path, sample['ImageId'], sample['Label'],sample['EncodedPixels'])

#     print(type(mask_decoded),type(image.img_to_array(img)))

#     print(mask_decoded)

#     print("+"*50)

#     print(image.img_to_array(img))

    

#     x = image.img_to_array(img)

#     x = np.expand_dims(x, axis=0)

#     x = preprocess_input(x)



#     preds = model.predict(x)

#     # decode the results into a list of tuples (class, description, probability)

#     # (one such list for each sample in the batch)

#     #print('Predicted:', decode_predictions(preds, top=2)[0])

#     top_preds.extend([decode_predictions(preds, top=2)[0]])



# #trdf['ResNet_toppreds'] = top_preds



# #trdf.head(50)