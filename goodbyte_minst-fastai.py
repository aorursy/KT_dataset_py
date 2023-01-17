# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *

from fastai.vision import *



# Any results you write to the current directory are saved as output.
!ls ../input

!mkdir -p /tmp/.torch/models/

!cp ../input/pretrained-pytorch-models/* /tmp/.torch/models/

!cp ../input/resnet-from-fastai/* /tmp/.torch/models
path = Path('../input/digit-recognizer')
class CustomImageItemList(ImageItemList):

    def open(self, fn):

        img = fn.reshape(28, 28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 783.0, axis=1).values

        return res
test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)

data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')

                       .random_split_by_pct(.2)

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

                       .databunch(bs=64, num_workers=0)

                       .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet34
learn = create_cnn(data, arch, metrics=error_rate, model_dir='/tmp/models')
learn.lr_find()
learn.recorder.plot()
lr = 0.02
learn.fit_one_cycle(4, slice(lr))
learn.recorder.plot_losses()
learn.save('stage-1')
learn.load('stage-1')
interp = learn.interpret()
interp.plot_top_losses(9, figsize=(7,7))
preds, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)
# Bug in fastai? Why is this needed?

y = torch.argmax(preds, dim=1)
submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': y}, columns=['ImageId', 'Label'])

submission_df.head()
submission_df.to_csv('submission.csv', index=False)
!head submission.csv