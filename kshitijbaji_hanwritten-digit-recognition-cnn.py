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



# Any results you write to the current directory are saved as output.
# the following three lines are suggested by the fast.ai course

%reload_ext autoreload

%autoreload 2

%matplotlib inline
# hide warnings

import warnings

warnings.simplefilter('ignore')
# the fast.ai library, used to easily build neural networks and train them

from fastai import *

from fastai.vision import *
# to get all files from a directory

import os



# to easier work with paths

from pathlib import Path



# to read and manipulate .csv-files

import pandas as pd
INPUT = Path("../input")

os.listdir(INPUT)
train_df = pd.read_csv(INPUT/"train.csv")

train_df.head(3)
test_df = pd.read_csv(INPUT/"test.csv")

test_df.head(3)
TRAIN = Path("../train")

TEST = Path("../test")
# Create training directory

for index in range(10):

    try:

        os.makedirs(TRAIN/str(index))

    except:

        pass
# Test whether creating the training directory was successful

sorted(os.listdir(TRAIN))
#Create test directory

try:

    os.makedirs(TEST)

except:

    pass
# import numpy to reshape array from flat (1x784) to square (28x28)

import numpy as np



# import PIL to display images and to create images from arrays

from PIL import Image



def saveDigit(digit, filepath):

    digit = digit.reshape(28,28)

    digit = digit.astype(np.uint8)



    img = Image.fromarray(digit)

    img.save(filepath)
# save training images

for index, row in train_df.iterrows():

    

    label,digit = row[0], row[1:]

    

    folder = TRAIN/str(label)

    filename = f"{index}.jpg"

    filepath = folder/filename

    

    digit = digit.values

    

    saveDigit(digit, filepath)
# save testing images

for index, digit in test_df.iterrows():



    folder = TEST

    filename = f"{index}.jpg"

    filepath = folder/filename

    

    digit = digit.values

    

    saveDigit(digit, filepath)
# import matplotlib to arrange the images properly

import matplotlib.pyplot as plt



def displayTrainingData():

    fig = plt.figure(figsize=(5,10))

    

    for rowIndex in range(1, 10):

        subdirectory = str(rowIndex)

        path = TRAIN/subdirectory

        images = os.listdir(path)

        for sampleIndex in range(1, 6):

            randomNumber = random.randint(0, len(images)-1)

            image = Image.open(path/images[randomNumber])

            ax = fig.add_subplot(10, 5, 5*rowIndex + sampleIndex)

            ax.axis("off")

            

            plt.imshow(image, cmap='gray')

        

    plt.show()

    

def displayTestingData():

    fig = plt.figure(figsize=(5, 10))

    

    paths = os.listdir(TEST)

    

        

    for i in range(1, 51):

        randomNumber = random.randint(0, len(paths)-1)

        image = Image.open(TEST/paths[randomNumber])

        

        ax = fig.add_subplot(10, 5, i)

        ax.axis("off")

        

        plt.imshow(image, cmap='gray')

    plt.show()
print('samples of training data')

displayTrainingData()
print('samples of testing data')

displayTestingData()
image_path = TEST/os.listdir(TEST)[9]

image = Image.open(image_path)

image_array = np.asarray(image)





fig, ax = plt.subplots(figsize=(15, 15))



img = ax.imshow(image_array, cmap='gray')



for x in range(28):

    for y in range(28):

        value = round(image_array[y][x]/255.0, 2)

        color = 'black' if value > 0.5 else 'white'

        ax.annotate(s=value, xy=(x, y), ha='center', va='center', color=color)



plt.axis('off')

plt.show()
# transforms

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(

    path = ("../train"),

    test = ("../test"),

    valid_pct = 0.2,

    bs = 16,

    size = 28,

    #num_workers = 0,

    ds_tfms = tfms

)
mnist_stats
data.normalize(mnist_stats)
# all the classes in data

print(data.classes)
learn = cnn_learner(data, base_arch=models.resnet18, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)
learn.fit_one_cycle(cyc_len=5)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(7, 7))
interp.plot_confusion_matrix()
class_score, y = learn.get_preds(DatasetType.Test)
probabilities = class_score[0].tolist()

[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]
class_score = np.argmax(class_score, axis=1)
class_score[0].item()
sample_submission =  pd.read_csv(INPUT/"sample_submission.csv")

display(sample_submission.head(2))

display(sample_submission.tail(2))