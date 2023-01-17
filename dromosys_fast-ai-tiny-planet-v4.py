import torch
print(torch.__version__)
import fastai
fastai.__version__
from fastai.vision import * 
from fastai import *
import fastai; 
fastai.show_install(1)
planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
data = ImageDataBunch.from_csv(planet, folder='train', size=128, suffix='.jpg', sep = ' ', ds_tfms=planet_tfms)
data = (ImageItemList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        #Where to find the data? -> in planet 'train' folder
        .random_split_by_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_df(sep=' ')
        #How to label? -> use the csv file
        .transform(planet_tfms, size=128)
        #Data augmentation? -> use tfms with a size of 128
        .databunch())                          
        #Finally -> use the defaults for conversion to databunch
data.show_batch(rows=2, figsize=(9,7))
learn = create_cnn(data, models.resnet18,  metrics=[accuracy_thresh])
learn.fit(5)
