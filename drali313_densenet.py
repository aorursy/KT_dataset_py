import numpy as np 

import pandas as pd 

from pathlib import Path



from fastai.imports import *

from fastai import *

from fastai.vision import *



from tqdm import tqdm_notebook as tqdm



base_path = Path('/kaggle/input/plant-pathology-2020-fgvc7/')
def get_tag(row):

    if row.healthy:

        return "healthy"

    if row.multiple_diseases:

        return "multiple_diseases"

    if row.rust:

        return "rust"

    if row.scab:

        return "scab"
def transform_data(train_labels):

    train_labels.image_id = [image_id+'.jpg' for image_id in train_labels.image_id]

    train_labels['tag'] = [get_tag(train_labels.iloc[idx]) for idx in train_labels.index]

    train_labels.drop(columns=['healthy', 'multiple_diseases', 'rust', 'scab'], inplace=True)
train_labels = pd.read_csv(base_path/"train.csv")

path = base_path/"images"
transform_data(train_labels)

train_labels = train_labels.set_index("image_id")
train_labels['tag'].value_counts()
tfms = get_transforms(flip_vert=True,max_zoom=1.3,max_lighting=0.3,) 
src = (ImageList.from_folder(path)

      .filter_by_func(lambda fname: "Train" in fname.name)

      .split_by_rand_pct()

      .label_from_func(lambda o: train_labels.loc[o.name]['tag']))
data_224 = (src.transform(tfms, size=224)

       .databunch(bs=16)

       .normalize())
data_224.show_batch(4)


import torch 

import torchvision

model = torchvision.models.mnasnet1_0(pretrained=True)
model
model.classifier[1].out_features=4
# model.layer4[0].bn1.momentum=0.3
# model.layer4[0].bn1.momentum=0.1

# model.layer4[0].bn1.eps=1e-04

# model.layer4[0].bn2.momentum=0.2

# model.layer4[0].bn2.eps=1e-03

# model.layer4[0].bn3.momentum=0.15

# model.layer4[0].bn3.eps=1e-03

# # model.layer4[1].bn1.momentum=0.05

# model.layer4[1].bn1.eps=1e-04

# model.layer4[1].bn2.momentum=0.13

# model.layer4[1].bn2.eps=1e-03

# model.layer4[1].bn3.momentum=0.15

# model.layer4[1].bn3.eps=1e-03

# model.layer4[2].bn1.momentum=0.15

# model.layer4[2].bn1.eps=1e-04

# model.layer4[2].bn2.momentum=0.3

# model.layer4[2].bn2.eps=1e-03

# model.layer4[2].bn3.momentum=0.05

# model.layer4[2].bn3.eps=1e-06

model=model.cuda()
from fastai.callbacks import *



learn = Learner(data_224, model, metrics=[error_rate, accuracy,],model_dir='kaggle/working/model')
learn.model_dir = "/kaggle/working"
from fastai.callbacks import *

try:

    learn.fit(20,1e-4,callbacks=[SaveModelCallback(learn, every='imrpovement', monitor='accuracy')])

except :

    learn.fit(20,1e-4)

    learn.save('bestmodel')
learn
learn.unfreeze()
learn.load('bestmodel')

learn.fit_one_cycle(25,1e-6,callbacks=[SaveModelCallback(learn, every='imrpovement', monitor='accuracy')])
test_images = ImageList.from_folder(base_path/"images")

test_images.filter_by_func(lambda x: x.name.startswith("Test"))


test_df = pd.read_csv(base_path/"test.csv")

test_df['healthy'] = [0.0 for _ in test_df.index]

test_df['multiple_diseases'] = [0.0 for _ in test_df.index]

test_df['rust'] = [0.0 for _ in test_df.index]

test_df['scab'] = [0.0 for _ in test_df.index]

test_df = test_df.set_index('image_id')

        
for item in tqdm(test_images.items):

    name = item.name[:-4]

    img = open_image(item)

    preds = learn.predict(img)[2]



    test_df.loc[name]['healthy'] = preds[0]

    test_df.loc[name]['multiple_diseases'] = preds[1]

    test_df.loc[name]['rust'] = preds[2]

    test_df.loc[name]['scab'] = preds[3]

            
test_df
test_df.to_csv(f"/kaggle/working/resnet_result.csv")
test_df.to_csv(f"/kaggle/working/resnet_result1111.csv")