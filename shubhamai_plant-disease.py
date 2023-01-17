import numpy as np # linear algebra
from fastai.vision import * # DL
# Data Argumentation
tfms = get_transforms(flip_vert=True)
# Loading & Preprocessing Data
data = ImageDataBunch.from_folder("../input/plantvillage-dataset/", train="color", valid_pct=0.1, size=256, bs=100)
data
data.batch_stats()
# Normalising Data 
data.normalize()
# Showing a smaple image
data.open(data.items[10])
# Showing a batch of images with corresponding labels
data.show_batch()
# Using ResNet18 model 
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.data.classes
# The Model Architecture
learn.model
# Training Model for 1 Epoch

#learn.fit(5, lr=1e-03)
learn.fit_one_cycle(1)
# Finding the best learning rate to increate overall accuracy

learn.lr_find(num_it=1000)
# Showing the results for training data

learn.show_results(ds_type=DatasetType.Train)
# # Showing the results for validation data

learn.show_results(ds_type=DatasetType.Valid)
learn.recorder.plot()
# Plotting loss 

learn.recorder.plot_losses()
# Plotting Accuracy 

learn.recorder.plot_metrics()
# Getting the confusing matrix

preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)
interp.confusion_matrix()
# Seeing in which class model get the most confused
interp.most_confused()
# Saving the trained model

learn.export(file = Path("/kaggle/working/export.pkl"))
