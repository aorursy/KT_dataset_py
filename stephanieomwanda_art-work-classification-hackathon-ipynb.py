# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')      

# Used to ignore the warnings displayed by python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# uploading the dataset

# uploading the artist csv just to preview information on the artists



artsy=pd.read_csv("../input/best-artworks-of-all-time/artists.csv")
# a random preview of our dataset

artsy.take(np.random.permutation(len(artsy))[:15])
# shape of the dataset 

print('Our dataset has', artsy.shape[0], 'rows and', artsy.shape[1], 'columns')
# confirming the datatypes

artsy.dtypes
# statistical summary of the datasets

artsy.describe().transpose()
# checking for duplicates

artsy.duplicated().sum()
# check for null values



artsy.isnull().sum()
# getting infromation on the dataset

artsy.info
artsy.head()
# getting the unique features

artsy.nunique()
# dropping the irrelevant columns

artsy.drop(columns=['id','bio','wikipedia'],inplace =True)
# Image manipulation.

import PIL.Image

from IPython.display import display

from glob import glob

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

def plotImages(artist,directory):

    print(artist)

    multipleImages = glob(directory)

    plt.rcParams['figure.figsize'] = (15, 15)

    plt.subplots_adjust(wspace=0, hspace=0)

    i_ = 0

    for l in multipleImages[:25]:

        im = cv2.imread(l)

        im = cv2.resize(im, (128, 128)) 

        plt.subplot(5, 5, i_+1) #.set_title(l)

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

        i_ += 1
print(os.listdir("/kaggle/input/best-artworks-of-all-time/images/images"))
#Read Images

import os

from skimage import io

from PIL import Image

# import cv2

def upload_art_train_images(image_path,best_artwork,height, width):

    images = []

    labels = []

    # Loop across the three directories having wheat images.

    for category in best_artwork:

        # Append the wheat category directory into the main path

        full_image_path = image_path +  category + "/"

        # Retrieve the filenames from the all the three wheat directories.

        image_file_names = [os.path.join(full_image_path, f) for f in os.listdir(full_image_path)]

        # Read the image pixels

        for file in image_file_names:

#             image= cv2.imread(file)

            image=io.imread(file)

            # Append image into list

            image_from_array = Image.fromarray(image, 'RGB')

            #Resize image

            size_image = image_from_array.resize((height, width))

            #Append image into list

            images.append(np.array(size_image))

#             size_image = image_from_array.resize((height, width))

            #Append image into list

#             images.append(np.array(size_image))

            #images.append(image) # uncomment after check

            # Label for each image as per directory

            labels.append(category)

    return images, labels



## Invoke the function

#Image resize parameters

height = 30

width = 30

num_classes = 2

#Get number of classes

best_artwork = ['Claude_Monet', 'Alfred_Sisley']

train_images, train_labels = upload_art_train_images('/kaggle/input/best-artworks-of-all-time/images/images/',best_artwork,height,width)

from keras.utils.np_utils import to_categorical

y_train=np.array(labels)

y_train = to_categorical(y_train, num_classes)

plotImages("Jean-Michel Basquiat","/kaggle/input/new-images/basquiat/**")
plotImages("Keith Haring","/kaggle/input/new-images/haring/**")
plotImages("Vincent van Gogh","/kaggle/input/best-artworks-of-all-time/images/images/Vincent_van_Gogh/**")
artsy.columns
# We want to obtain the age of the artists so  I will split the death and birth year into two columns

# I will then drop the year column

artsy_year = pd.DataFrame(artsy.years.str.split(' ',2).tolist(),columns = ['birth','-','death'])

artsy_year.drop(["-"],axis=1,inplace=True)

artsy["birth"]=artsy_year.birth

artsy["death"]=artsy_year.death

artsy.drop(["years"],axis=1,inplace=True)
artsy["birth"]=artsy["birth"].apply(lambda x: int(x))

artsy["death"]=artsy["death"].apply(lambda x: int(x))
artsy2 = pd.DataFrame({'name': ['Jean-Michel Basquiat','Keith Haring'],

                            'birth': ['1960','1958'],

                            'death':['1988','1990'],

                            'genre': ['Neo-expressionism', 'Pop Art'],

                            'nationality': ['American', 'American'],

                            'paintings':[600, 79]})

frames= (artsy,artsy2)

art=pd.concat(frames,ignore_index=True)



art.birth=art.birth.astype('int')

art.death=art.death.astype('int')
art["age"]=art.death-art.birth
# specifying bins for when we visualize the distribustion

# creating a new column to show 



art['age']=art['age']

bins=[27,55,65,77,98]

labels=["young adult","early adult","adult","senior"]

art['age_group']=pd.cut(art['age'],bins,labels=labels)
# create function that obtains the century 

# creating a century column

art['century'] = (art['death'] // 100) + 1

art.take(np.random.permutation(len(art))[:52])
# Dropping more irrelevant columns 

art.drop(columns=['birth','death'], inplace= True)
# Calculating the mean of the numeric features

numeric = ['age', 'century', 'paintings']

for col in numeric:

  print(art[[col]].mean())
# Determining the mode of each of the numeric features



for col in numeric:

  print(art[[col]].mode())
# Identifying the median 



for col in numeric:

  print(art[[col]].median())
# The InterQuartile Range (IQR)

# IQR is also called the midspread or middle 50%



# Calculating IQR for the numeric features







for i in numeric:



  Q1 = art[i].quantile(0.25)

  Q3 = art[i].quantile(0.75)

  IQR = Q3 - Q1

  print(i, ':', IQR)
# findining outliers 

columns=['age','paintings','century']

fig, ax = plt.subplots(len(columns), figsize=(8,40))

for i, values in enumerate(columns):



    sns.boxplot(y=art[values], ax=ax[i])

    ax[i].set_title('Box plot - {}'.format(values), fontsize=8)

    ax[i].set_xlabel(values, fontsize=8)

plt.show()
# Distribution Plots

# plots to check for the distribution of the numeric features of our data



fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (20, 25))



for ax, name, data in zip(axes.flatten(), numeric, art):

  sns.distplot(art[name], hist = True, ax = ax, bins = 20, color = 'crimson')

  plt.suptitle('Boxplots for Numeric Features', fontsize = 16)

  plt.subplots_adjust()

  plt.tight_layout
# a plot showing the most popular genre

plt.style.use('fivethirtyeight')

art['genre'].value_counts().plot.bar()
# a plot showing the most popular genre grouped by the century

art['genre'].groupby('century').plot.bar()
# visualization showing the Age Group count per Art

art['age_group'].value_counts().plot.bar(rot =0)

plt.xlabel("age_group",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.title("Age Group count per Artist",fontsize=15)

plt.show()

plt.figure(figsize=(5,5))

art_genre = sns.countplot(y='genre',data=art)

art_genre





plt.figure(figsize=(5,5))

art_nationality = sns.countplot(y='nationality',data=art)

art_nationality



# visualization showing the Age Group count per Art

art['century'].value_counts().plot.bar(rot =0)

plt.xlabel("century",fontsize=15)

plt.ylabel("Count",fontsize=15)

plt.title("Century count per Artist",fontsize=15)

plt.show()



# matplotlib histogram

plt.hist(art['age'], color = 'blue', edgecolor = 'black',

         bins = int(180/5))







# seaborn histogram

sns.distplot(art['age'], hist=True, kde=False,

             bins=int(180/5), color = 'blue',

             hist_kws={'edgecolor':'black'})

# Add labels

plt.title('Histogram of Age of Artists')

plt.xlabel('Age (years)')

plt.ylabel('No. of Artists')
## TensorFlow and keras

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout