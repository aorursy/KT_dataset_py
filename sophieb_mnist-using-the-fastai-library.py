# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#We import all of the necessary libraries

import os

from pathlib import Path

from PIL import Image
PATH = Path('/kaggle/')

print(os.listdir(PATH))
#Load train and test datasets

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
#Check the contets of the train dataset

train.head()
test.info()
train.label.value_counts().plot(kind='bar');
#Create the images directory

images = Path(PATH/'images')

images.mkdir(parents=True, exist_ok=True)
#Create train and test subdirectories inside directory

images_train = Path(images/'train')

images_test = Path(images/'test')



images_train.mkdir(parents=True, exist_ok=True)

images_test.mkdir(parents=True, exist_ok=True)
#Create one directory per label inside the train subdirectory

for i in range(10):

    label_dir = Path(images_train/str(i))

    label_dir.mkdir(parents=True, exist_ok=True)
os.listdir(images)
def create_image_from_row(digit, dest_path):

    #convert digit to a 28x28 matrix

    mat = digit.reshape(28,28)

    mat = mat.astype(np.uint8)

    #convert the matrix to an image

    img = Image.fromarray(mat)

    #save the image to the train directory

    img.save(dest_path)
# save training images

for index, row in train.iterrows():

    #separte the label and the digit 

    label,digit = row[0], row[1:]

    #obtain the directory to save the image

    directory = images_train/str(label)

    #create a filename for the image

    filename = f"{index}.jpg"

    dest_path = directory/filename

    digit = digit.values

    

    create_image_from_row(digit, dest_path)
#save test images

for index, row in test.iterrows():

    #create a filename for the image

    filename = f"{index}.jpg"

    dest_path = images_test/filename

    digit = row.values

    

    create_image_from_row(digit, dest_path)
import matplotlib.pyplot as plt

import random



def displayTrainingData():

    fig = plt.figure(figsize=(5,10))

    

    for rowIndex in range(1, 10):

        subdirectory = str(rowIndex)

        path = images_train/subdirectory

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

    

    paths = os.listdir(images_test)

    for i in range(1, 51):

        randomNumber = random.randint(0, len(paths)-1)

        image = Image.open(images_test/paths[randomNumber])

        

        ax = fig.add_subplot(10, 5, i)

        ax.axis("off")

        

        plt.imshow(image, cmap='gray')

    plt.show()
print('A few samples of the training images.')

displayTrainingData()
print('A few samples of the testing images.')

displayTestingData()
from fastai.vision import *
data = ImageDataBunch.from_folder(images, train='train', test='test', valid_pct=0.2, bs=32,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet18, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-3))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
losses,idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(15,11))
class_score, y = learn.get_preds(DatasetType.Test)
probabilities = class_score[0].tolist()

[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]
class_score = np.argmax(class_score, axis=1)
INPUT = Path("../input/digit-recognizer")

TEST = Path("../test")
sample_submission =  pd.read_csv(INPUT/"sample_submission.csv")

display(sample_submission.head(2))

display(sample_submission.tail(2))
# remove file extension from filename

ImageId = [os.path.splitext(path)[0] for path in os.listdir(images_test)]

# typecast to int so that file can be sorted by ImageId

ImageId = [int(path) for path in ImageId]

# +1 because index starts at 1 in the submission file

ImageId = [ID+1 for ID in ImageId]
submission  = pd.DataFrame({

    "ImageId": ImageId,

    "Label": class_score

})

# submission.sort_values(by=["ImageId"], inplace = True)

submission.to_csv("submission.csv", index=False)

display(submission.head(3))

display(submission.tail(3))