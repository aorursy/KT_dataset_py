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
import torchvision.models
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
# transforms
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(
    path = TRAIN,
    test = TEST,
    valid_pct = 0.2,
    bs = 256,
    size = 28,
    num_workers = 5,
    ds_tfms = tfms
).normalize(mnist_stats)
# all the classes in data
print(data.classes)
resnet34_learn = Learner(data, torchvision.models.resnet34(pretrained=True), metrics=[error_rate, accuracy, top_k_accuracy], model_dir="/tmp/models", callback_fns=ShowGraph)
resnet_learn = Learner(data, torchvision.models.resnet50(pretrained=True), metrics=[error_rate, accuracy, top_k_accuracy], model_dir="/tmp/models", callback_fns=ShowGraph)
googlenet_learn = Learner(data, torchvision.models.googlenet(pretrained=True), metrics=[error_rate, accuracy, top_k_accuracy], model_dir="/tmp/models", callback_fns=ShowGraph)
resnext_learn = Learner(data, torchvision.models.resnext50_32x4d(pretrained=True), metrics=[error_rate, accuracy, top_k_accuracy], model_dir="/tmp/models", callback_fns=ShowGraph)
wideres_learn = Learner(data, torchvision.models.wide_resnet50_2(pretrained=True), metrics=[error_rate, accuracy, top_k_accuracy], model_dir="/tmp/models", callback_fns=ShowGraph)
mobilenet_learn = Learner(data, torchvision.models.mobilenet_v2(pretrained=True), metrics=[error_rate, accuracy, top_k_accuracy], model_dir="/tmp/models", callback_fns=ShowGraph)
# for learn in model:
#     learn.lr_find()
#     learn.recorder.plot(suggestion=True)
%%time
resnet34_learn.fit_one_cycle(10)
%%time
resnet_learn.fit_one_cycle(10)
%%time
googlenet_learn.fit_one_cycle(10)
%%time
resnext_learn.fit_one_cycle(10)
%%time
wideres_learn.fit_one_cycle(10)
%%time
mobilenet_learn.fit_one_cycle(10)
model = [resnet_learn, googlenet_learn, resnext_learn, wideres_learn, mobilenet_learn]
for learn in model:
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(7, 7))
    interp.plot_confusion_matrix()
ImageId = [int(os.path.splitext(path)[0])+1 for path in os.listdir(TEST)]
model_name = ['resnet', 'googlenet', 'resnext', 'wideres', 'mobilenet']
i = 0
for learn in model:
    class_score, y = learn.get_preds(DatasetType.Test)
    class_score = np.argmax(class_score, axis=1)
    submission  = pd.DataFrame({"ImageId": ImageId,"Label": class_score})
    submission.to_csv("submission_"+str(model_name[i])+".csv", index=False)
    i += 1
