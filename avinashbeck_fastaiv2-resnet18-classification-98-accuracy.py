import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Installing the fastai v2
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
flower_types = "daisy", "dendelion", "rose", "sunflower", "tulip"
path = Path("/kaggle/input/flowers-recognition/flowers/")
fns = get_image_files(path)
fns
# Creating the DataBlock to quickly build Datasets and DataLoaders

flower = DataBlock(
        blocks = (ImageBlock, CategoryBlock),      # Specify the dependent and independent variables
        get_items = get_image_files,             # Takes a path and returns a list of all images
        splitter = RandomSplitter(valid_pct=0.2, seed = 42),# Random Splitting. If you already have a valid set use GrandParentSplitter
        get_y = parent_label,      # Function provided by Fastai that gets the name of the folder the file is in
        item_tfms = Resize(128))   # Resize all the images to 128 by 128
# Now lets create the DataLoader

dls = flower.dataloaders(path)
dls.valid.show_batch(max_n = 5, nrows = 1)
flower = flower.new(
            item_tfms = RandomResizedCrop(224, min_scale = 0.5),    # Image size = 224, min_scale: how much images to be selected at minimum
            batch_tfms = aug_transforms())    # Perform image augmentation on the entire batch on GPU

# Create new Dataloaders
dls = flower.dataloaders(path)
# Lets create a Learner and finetune it

learn = cnn_learner(dls, resnet18, metrics = accuracy)
learn.fine_tune(4)
# Plot Confusion Matrix

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows = 1)
# Manually remove any misclassified images from training to avoid model confusion
'''
cleaner = ImageClassifierCleaner(learn)
cleaner

'''

# To delete the incorrect images

'''
for idx in cleaner.delete():
    cleaner.fns[idx].unlink()         # unlink works as the same way as delete

'''

# It's a good idea to always save the model for future use

output_path = Path("/kaggle/working")
learn.export(output_path/"model")
