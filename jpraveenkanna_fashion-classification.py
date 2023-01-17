import torch

#from fastai.vision import *

from fastai.vision import Path,get_image_files,open_image,get_transforms,ImageDataBunch,imagenet_stats,cnn_learner,models,ClassificationInterpretation

import pandas as pd

import numpy as np

import warnings

from fastai.metrics import error_rate





warnings.filterwarnings('ignore')

np.random.seed(2)
torch.cuda.get_device_name(0)
#Smaller version - dataset

base_path = Path('/kaggle/input/fashion-product-images-small/myntradataset')

images_path = base_path/'images'

base_path.ls()
df_fine_tuned = pd.read_csv('/kaggle/input/fashion-label-training/top_65_catagory.csv')

df_fine_tuned.info()
df_fine_tuned.head()
#Printing images that don't exist from the dataframe

invalid_img = []

for idx,value in enumerate(df_fine_tuned.name):

    path = images_path/str(value)

    if not path.is_file():

        invalid_img.append(idx)

        print(idx,path)

invalid_img
#Printing distinct catagories

len(df_fine_tuned.label.unique())
df_fine_tuned.groupby('label')['name'].count().nlargest(100)
print(df_fine_tuned['label'][4])

open_image(base_path/'images'/df_fine_tuned['name'][4])
tfms= get_transforms(max_rotate=25,do_flip=True,max_zoom=1.1,max_lighting=0.4)

data_fine_tuned = ImageDataBunch.from_df(images_path, df_fine_tuned, ds_tfms=tfms,size=60, bs=64).normalize(imagenet_stats)

learn_fine_tuned = cnn_learner(data_fine_tuned, models.resnet101, metrics=error_rate)

learn_fine_tuned.fit_one_cycle(4)

learn_fine_tuned.model_dir = "/kaggle/working"

learn_fine_tuned.save('Fashion_model-65-category')
learn_fine_tuned.lr_find()

learn_fine_tuned.recorder.plot()


learn_fine_tuned.unfreeze()

learn_fine_tuned.fit_one_cycle(6, max_lr=slice(1e-06,1e-03))
interp = ClassificationInterpretation.from_learner(learn_fine_tuned)



losses,idxs = interp.top_losses()



len(data_fine_tuned.valid_ds)==len(losses)==len(idxs)



interp.most_confused(min_val=2)
#Viewing top losses interpreted from the model

interp.plot_top_losses(9, figsize=(30,11))
#Trsting a image

print(df_fine_tuned['label'][4488])

test = open_image(base_path/'images'/df_fine_tuned['name'][4488])

test
#Running inference

learn_fine_tuned.predict(test)
#Predicted class 

data_fine_tuned.classes[56]
#list of labels 

data_fine_tuned.classes
#save model as pytorch

learn_fine_tuned.save('Fashion_model-fine_tuned')
#Saving model as binary format

learn_fine_tuned.path = Path("/kaggle/working")

learn_fine_tuned.export()
#Saving the label as csv file

labels = pd.DataFrame(data={"labels": data_fine_tuned.classes})

labels.to_csv("labels.csv", sep=',',index=False,header=False)