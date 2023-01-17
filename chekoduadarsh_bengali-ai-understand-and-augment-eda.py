



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import seaborn as sns

import matplotlib.pyplot as plt

import random

# Any results you write to the current directory are saved as output.
Train_data = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

Test_data = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
Train_data.head()
class_map.head()
# count the number of classes for each target

n_classes = class_map.groupby(by=['component_type']).count()
n_classes.head()
sns.set(style="darkgrid")

k = ['vowel_diacritic','grapheme_root','consonant_diacritic']

sns.countplot(data = class_map ,x = 'component_type',order = k)

HEIGHT = 137

WIDTH = 236
b_train_data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')



imageid = b_train_data.iloc[:, 0]

image = b_train_data.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

f, ax = plt.subplots(5, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(25):

    ax[i].imshow(image[i], cmap='Greys')

def listToString(s):  

    

    # initialize an empty string 

    str1 = ""  

    

    # traverse in the string   

    for ele in s:  

        str1 += ele   

    

    # return string   

    return str1  

        
from wordcloud import WordCloud





s = listToString(Train_data['grapheme'] )



# Read the whole text.

text = s



# Generate a word cloud image

wordcloud = WordCloud().generate(text)



# Display the generated image:

# the matplotlib way:

import matplotlib.pyplot as plt





# take relative word frequencies into account, lower max_font_size

wordcloud = WordCloud(background_color="white",max_words=len(s),max_font_size=70, relative_scaling=.5, font_path = "/kaggle/input/bengalifont/Nikosh.ttf").generate(text)

plt.figure(figsize=(30, 30))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()


#sns.set(style="darkgrid")

plt.figure(figsize=(20, 10))

sns.countplot(data = Train_data ,x = 'grapheme_root',color="c")



type(image[1])
rand_num = random.randint(0,len(image))

rand_num
plt.imshow(image[rand_num])

plt.title("input_img")

plt.show()
flipped_img = np.fliplr(image[rand_num])

plt.imshow(flipped_img)

plt.title("flipped_img")

plt.show()
import skimage

img = image[rand_num]

img = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True)

plt.imshow(img)

plt.show()
from skimage.transform import rotate

img = image[rand_num]

img = rotate(img, 15)

plt.imshow(img)

plt.show()
from skimage.transform import rotate

img = image[rand_num]

img = rotate(img, -15)

plt.imshow(img)

plt.show()
import numpy as np

from skimage import exposure

img = image[rand_num]

v_min, v_max = np.percentile(img, (0.2, 99.8))

img = exposure.rescale_intensity(img, in_range=(v_min, v_max))



plt.imshow(img)

plt.show()
