%reload_ext autoreload

%autoreload 2

%matplotlib inline

# FOR NON-FASTAI LIBRARIES

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# FOR ALL THE FASTAI LIBRARIES



from fastai.vision import *

from fastai.metrics import *





# make sure CUDA is available and enabled

print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
# mainDIR = "/kaggle/input/digit-recognizer/"

# os.listdir(mainDIR)

INPUT = Path("../input/digit-recognizer")

os.listdir(INPUT)
# train_df = pd.read_csv(mainDIR+ "train.csv")

train_df = pd.read_csv(INPUT/"train.csv")

train_df.head(3)
test_df = pd.read_csv(INPUT/"test.csv")

test_df.head(3)
# TRAIN = Path("/kaggle/train/")

# TEST = Path("/kaggle/test/")



TRAIN = Path("../train")

TEST = Path("../test")
#MAKE DIRECTORIES  FOR TRAINING FOLDER



for i in range(10):    

    try:         

        os.makedirs(TRAIN/str(i))       

    except:

        pass
#CHECK IF MAKING THE DIRECTORIES WORKED!!

sorted(os.listdir(TRAIN))
#LET'S MAKE THE TEST FOLDER 



try:

    os.makedirs(TEST)

except:

    pass
os.listdir(TEST)
# os.listdir(TEST)

if os.path.isdir(TRAIN):

    print('Train directory has been created')

else:

    print('Train directory creation failed.')



if os.path.isdir(TEST):

    print('Test directory has been created')

else:

    print('Test directory creation failed.')
from PIL import Image
def pix2img(pix_data, filepath):

    img_mat = pix_data.reshape(28,28)

    img_mat = img_mat.astype(np.uint8())

    

    img_dat = Image.fromarray(img_mat)

    img_dat.save(filepath)

    

    
# SAVE TRAINING IMAGES 



for idx, data in train_df.iterrows():

    

    label, data = data[0], data[1:]

    folder = TRAIN/str(label)

    

    fname = f"{idx}.jpg"

    filepath = folder/fname

    

    img_data = data.values

    

    pix2img(img_data,filepath)
# THE SAME PROCESS FOR TESTING DATA 

for idx, data in test_df.iterrows():

    

#     label, data = data[0], data[1:]

    folder = TEST

    

    fname = f"{idx}.jpg"

    filepath = folder/fname

    

    img_data = data.values

    

    pix2img(img_data,filepath)
def plotTrainImage():

    

    fig = plt.figure(figsize= (5,10))

    

    for rowIdx in range(1,10):

        

        foldNum = str(rowIdx)

        path = TRAIN/foldNum

        

        images = os.listdir(path)

        

        for sampleIdx in range(1,6):

            

            randNum = random.randint(0, len(images)-1)

            image = Image.open(path/images[randNum])

            ax = fig.add_subplot(10, 5, 5*rowIdx + sampleIdx)

            ax.axis("off")

            

            plt.imshow(image, cmap='gray')

            

    plt.show()      

    

print('plotting training images')

plotTrainImage()
# FUNCTION FOR PLOTTING TEST IMAGES 



def plotTestImage():

    

    fig = plt.figure(figsize=(5, 10))    

    paths = os.listdir(TEST)    

        

    for i in range(1, 51):

        randomNumber = random.randint(0, len(paths)-1)

        image = Image.open(TEST/paths[randomNumber])

        

        ax = fig.add_subplot(10, 5, i)

        ax.axis("off")

        

        plt.imshow(image, cmap='gray')

    plt.show()



print('plotting testing images')

plotTestImage()
# transforms which are a part of data augmentation

tfms = get_transforms(do_flip = False)

print('test : ',TEST)

print('train: ', TRAIN)

print(type(TEST))
data = ImageDataBunch.from_folder(



    path = ("../train"),

    test = ("../test"),

    valid_pct = 0.1,

#     bs = 16,

    bs = 256,    

    size = 28,

    num_workers = 0,

    ds_tfms = tfms

)
mnist_stats
# data.normalize(mnist_stats)

data.normalize(imagenet_stats)

print(data.classes)

print('There are', data.c, 'classes here')
#VERSION 1

# learn = cnn_learner(data, base_arch = models.resnet18, metrics = accuracy,model_dir="/tmp/models", callback_fns=ShowGraph )



#version 2

learn = cnn_learner(data, base_arch = models.resnet34, metrics = accuracy,model_dir="/tmp/models", callback_fns=ShowGraph )



# version 3

# learn = cnn_learner(data, base_arch = models.resnet50, metrics = accuracy,model_dir="/tmp/models", callback_fns=ShowGraph )
doc(fit_one_cycle)
learn.fit_one_cycle(3)
learn.save('model1')
learn.load('model1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(30 , slice(1e-3, 1e-2))
learn.show_results(3, figsize= (7,7))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(6, figsize=(7, 7))
interp.plot_confusion_matrix()
class_score , y = learn.get_preds(DatasetType.Test)
probabilities = class_score[0].tolist()

[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]
class_score = np.argmax(class_score, axis=1)
class_score[1].item()
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


