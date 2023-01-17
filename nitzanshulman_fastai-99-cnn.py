import os

import numpy as np

import pandas as pd



from pathlib import *

from fastai.vision import *
# Create dataset

path = '/kaggle/input/working-potholes/kaggle/working/pothole-detection-dataset'



fastai_data = ImageDataBunch.from_folder(path, train=".",

                                         valid_pct=0.2, ds_tfms=get_transforms(), 

                                         size=300, num_workers=4, 

                                         bs=32).normalize(imagenet_stats)
fastai_data.show_batch(figsize=(7,8))

fastai_data.classes, fastai_data.c, len(fastai_data.train_ds), len(fastai_data.valid_ds)
# Tricking torch not to download resnet

# Torch did not ran yet so cache directory does not exist

!mkdir -p /tmp/.cache/torch/checkpoints



# Copy resnet34.pth into torch cache

!cp /kaggle/input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
learn = cnn_learner(fastai_data, models.resnet34, metrics=error_rate, model_dir='/tmp/models')
learn.fit_one_cycle(4)
# Save model before unfreezing

learn.save('stage1')
def plot_confusion_matrix(learner):

    """

    Ploting a confusion matrix using a fastai cnn learner

    """

    interp = ClassificationInterpretation.from_learner(learner)

    interp.plot_confusion_matrix()
plot_confusion_matrix(learn)
unfreezed_learner = learn.load('stage1')
unfreezed_learner.unfreeze()
unfreezed_learner.fit_one_cycle(3, max_lr=slice(3e-5,3e-4))
plot_confusion_matrix(unfreezed_learner)
# Move data to a read/write dir

# input_path = "/kaggle/input/pothole-detection-dataset/"

# !rm -rf /kaggle/working/pothole-detection-dataset

# !cp -r  {input_path} /kaggle/working
# Create paths

# path = Path("/kaggle/working/pothole-detection-dataset")

# normal_path = path/"normal"

# pothole_path = path/"potholes"
# Verify and delete images it can not open

# verify_images(path=normal_path, delete=True)

# verify_images(path=pothole_path, delete=True)
# from zipfile import ZipFile



# Create zip file

# dirName = "/kaggle/working/pothole-detection-dataset"



# with ZipFile('/tmp/b.zip', 'w') as zipObj:

#    # Iterate over all the files in directory

#    for folderName, subfolders, filenames in os.walk(dirName):

#        for filename in filenames:

#            #create complete filepath of file in directory

#            filePath = os.path.join(folderName, filename)

#            # Add file to zip

#            zipObj.write(filePath)

# Move zip file to a place I can download from

# !cp  /tmp/b.zip /kaggle/working/b.zip