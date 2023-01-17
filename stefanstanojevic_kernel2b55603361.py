# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import tensorflow as tf

import tensorflow.keras as keras

import numpy as np

print('TensorFlow version: {}'.format(tf.__version__))



# Any results you write to the current directory are saved as output.
root_model = keras.models.load_model('../input/grapheme-root-model/grapheme_root_model.h5')

vowel_model = keras.models.load_model('../input/vowel-diacritic-model/vowel_model.h5')

consonant_model = keras.models.load_model('../input/consonant-diacritic-model/consonant_model.h5')
import pandas as pd

class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")

class_map_corrected = pd.read_csv("../input/bengaliai-cv19/class_map_corrected.csv")

sample_submission = pd.read_csv("../input/bengaliai-cv19/sample_submission.csv")

test = pd.read_csv("../input/bengaliai-cv19/test.csv")

train = pd.read_csv("../input/bengaliai-cv19/train.csv")

train_multi_diacritics = pd.read_csv("../input/bengaliai-cv19/train_multi_diacritics.csv")
import numpy as np

import matplotlib.pyplot as plt

from skimage.transform import resize

from tqdm import tqdm



heigth = 137;

width = 236;



def get_bounding_box(image):

    a,b = image.shape

    

    down = 0

    while (np.any(image[down,:] < 200) == False) and (down < a):

        down += 1



    up = a-1

    while (np.any(image[up,:] < 200) == False) and (up >= 0):

        up -= 1

  

    left = 0

    while (np.any(image[:,left] < 200) == False) and (left < b):

        left += 1



    right = b-1

    while (np.any(image[:,right] < 200) == False) and (right >= 0):

        right -= 1



    # in order to get the index of the last row/column that was actually zero

    down = max(down-1,0)

    up = min(up+1,a-1)

    left = max(left- 1,0)

    right = min(right+1,b-1)



    return (down,up,left,right)  



def resize_bounding_box(graph_image):

    a,b,c,d = get_bounding_box(graph_image)

    #plt.figure()

    #plt.imshow(graph_image)

    #plt.show()

    #plt.figure()

    #plt.imshow(graph_image[a:b+1,c:d+1])

    #plt.show()



    new_image = resize(graph_image[a:b+1,c:d+1],(50,100))

    #plt.figure()

    #plt.imshow(new_image)

    #plt.show()



    return new_image



def resize_everything(data_index):



    read = pd.read_parquet(f"../input/bengaliai-cv19/test_image_data_{data_index}.parquet")

    train_read = read.drop(['image_id'], axis=1, inplace=False)

    img_labels = read['image_id'].values

    full_array = train_read.values.reshape((-1,heigth,width))

    

    num_images,_,_ = full_array.shape

    X_full_resized = np.zeros((num_images,50,100))

    

    for i in tqdm(range(num_images)):

        X_full_resized[i,:,:] = resize_bounding_box(full_array[i,:,:])



    return (X_full_resized,img_labels)
def test_model(model, X_test, image_ids, task = '_grapheme_root'):

    y_test = np.argmax(model.predict(X_test),axis = 1)

    new_image_ids = []

    for i in range(image_ids.shape[0]):

        new_image_ids.append(image_ids[i] + task)

        

    return (new_image_ids, list(y_test))





def test_all_models(root_model, vowel_model, consonant_model, data_index = 0):

    

    X_test, image_ids = resize_everything(data_index)

    #X_full = train_df0.values.reshape((-1,heigth,width,1))



    num_images = X_test.shape[0]

    X_test = X_test.reshape((num_images,50,100,1))

    root_img_ids, y_root_test = test_model(root_model, X_test, image_ids, task = '_grapheme_root')

    vowel_img_ids, y_vowel_test = test_model(vowel_model, X_test, image_ids, task = '_vowel_diacritic')

    consonant_img_ids, y_consonant_test = test_model(consonant_model, X_test, image_ids, task = '_consonant_diacritic')

    return (root_img_ids + vowel_img_ids + consonant_img_ids, y_root_test + y_vowel_test + y_consonant_test)
sub_ids = []

sub_targets = []

for i in range(4):

    a,b = test_all_models(root_model, vowel_model, consonant_model, data_index = i)

    sub_ids = sub_ids + a

    sub_targets = sub_targets + b



path_save = 'submission.csv'

sub_data = pd.DataFrame(data={'row_id':sub_ids, 'target':sub_targets})

if not os.path.exists(path_save):

    os.mknod(path_save)

sub_data.to_csv(path_save, index = False)
our_submission = pd.read_csv("submission.csv")

our_submission.head(40)