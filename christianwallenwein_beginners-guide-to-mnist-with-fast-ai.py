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
INPUT = Path("../input/digit-recognizer")

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

    path = TRAIN,

    test = TEST,

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
# remove file extension from filename

ImageId = [os.path.splitext(path)[0] for path in os.listdir(TEST)]

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
import platform; platform.system()
flip_tfm = RandTransform(tfm=TfmPixel (flip_lr), kwargs={}, p=1, resolved={}, do_run=True, is_random=True, use_on_y=True)

folder = TRAIN/"3"

filename = os.listdir(folder)[0]

img = open_image(TRAIN/folder/filename)

display(img)

display(img.apply_tfms(flip_tfm))
tfms = get_transforms(do_flip=False)
learn = cnn_learner(data, base_arch=models.densenet169, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)
learn = cnn_learner(data, base_arch=models.densenet169, pretrained=False, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)
import torchvision.models
learn = Learner(data, torchvision.models.googlenet(), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)
learn = Learner(data, torchvision.models.googlenet(pretrained=True), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        # here you instantiate all the layers of the neural network and the activation function

        

    def forward(self, x):

        # here you define the forward propagation

        return x
# set the batch size

batch_size = 16



class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        # input is 28 pixels x 28 pixels x 3 channels

        # our original data was grayscale, so only one channel, but fast.ai automatically loads in the data as RGB

        self.conv1 = nn.Conv2d(3,16, 3, padding=1)

        self.conv2 = nn.Conv2d(16,32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7*7*32, 500)

        self.fc2 = nn.Linear(500, 10)

        self.relu = nn.ReLU()



    def forward(self, x):

        print (x.size())

        # x (28x28x3)

        x = self.conv1(x)

        # x (28x28x16)

        x = self.pool(x)

        # x (14x14x16)

        x = self.relu(x)

        

        x = self.conv2(x)

        # x (14x14x32)

        x = self.pool(x)

        # x (7x7x32)

        x = self.relu(x)



        # flatten images in batch

        print(x.size())

        x = x.view(-1,7*7*32)

        print(x.size())

        x = self.fc1(x)

        x = self.relu(x)

        

        x = self.fc2(x)

        x = self.relu(x)

        

        return x
learn = Learner(data, CNN(), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)