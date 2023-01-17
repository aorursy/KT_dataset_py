!pip3 install -q fastai==2.0.16
from fastai.vision.all import *
working_directory = "/kaggle/input/100-bird-species"
dls = ImageDataLoaders.from_folder(working_directory, 
                 item_tfms=Resize(340),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75), bs=16)
dls.show_batch(nrows=3, ncols=3)
learn = cnn_learner(dls, resnet18, pretrained=True, metrics=accuracy, model_dir="/kaggle/working")
learn.fine_tune(12)
learn.show_results()
learn.path = Path("/kaggle/working/")
learn.export('2020-10-26_1.6.0-birds-classifier-resnet18-224.pkl')
import torch
import os

test_dir = working_directory + "/test/"
list_dir = os.listdir(test_dir)
classes = learn.dls.categorize.vocab.items

print(list_dir)

test_predictions = []
target_predictions= []


for folder in list_dir:
  list_images = os.listdir(test_dir + folder + "/")
  for image_name in list_images:
    target_predictions.append(folder)
    prediction, _, _ = learn.predict(test_dir + folder + "/" + image_name)
    test_predictions.append(prediction)
correct = 0
total = 0

for element_index in range(len(test_predictions)):
  total += 1
  if target_predictions[element_index] == test_predictions[element_index]:
    correct += 1

print(f"Test set accuracy: {100*correct/total}")