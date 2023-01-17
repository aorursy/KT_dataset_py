# Imports and read data

import numpy as np 

import pandas as pd

from fastai.vision import *

import os

        

data_dir = "/kaggle/input/digit-recognizer"

train_file = "train.csv"

test_file = "test.csv"
# Create fast.ai datasets from pandas

class CustomImageList(ImageList):

    def open(self, fn):

        img = fn.reshape(28,28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # Convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        return res



test = CustomImageList.from_csv_custom(path=data_dir, csv_name=test_file, imgIdx=0)



tfms = get_transforms(do_flip=False)

data = (CustomImageList.from_csv_custom(path=data_dir, csv_name=train_file, imgIdx=1)

                           .split_by_rand_pct(.2)

                           .label_from_df(cols='label')

                           .add_test(test, label=0)

                           .transform(tfms)

                           .databunch(bs=64, num_workers=0)

                           .normalize(imagenet_stats))

                          

data.show_batch(rows=3, figsize=(5,5))
# Create learner

learn = cnn_learner(data, base_arch=models.resnet50, metrics=[accuracy,error_rate], model_dir='/kaggle/working/models')

# Plot learning rate

learn.lr_find()

learn.recorder.plot()
# Train

learn.fit_one_cycle(5, 1e-2, wd=0.1)

# Train some more

learn.fit_one_cycle(5, wd=0.1)
learn.save('7')
# Predict answers 

learn.load('7')

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# Output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission_d2.csv', index=False)