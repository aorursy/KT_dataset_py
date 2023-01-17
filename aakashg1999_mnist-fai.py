# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from fastai import *

from fastai.vision import *



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path_train=Path("../input/train.csv")

path_test=Path("../input/test.csv")

path_sam_submit=Path("../input/sample_submission.csv")
traincsv=pd.read_csv(path_train)

testcsv=pd.read_csv(path_test)
traincsv.head()

testcsv.head()
class CustomImageItemList(ImageList):

    def open(self, fn):

        img = fn.reshape(28,28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        return res

    

test = CustomImageItemList.from_csv_custom(path=path_test,csv_name='' ,imgIdx=0)



data = (CustomImageItemList.from_csv_custom(path=path_train,csv_name='' )

                       .random_split_by_pct(.05)

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

                       .databunch(bs=64, num_workers=0))
data.c, len(data.classes)
learn = cnn_learner(data,models.resnet50, metrics=accuracy, model_dir='/kaggle/working/models')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,slice(1e-2))
learn.save('stage-1')
learn.fit_one_cycle(2,slice(1e-2))
learn.save('stage-2')
learn.fit_one_cycle(2,max_lr=1e-4)
learn.fit_one_cycle(2,slice(1e-3))
learn.save('stage-3')
learn.fit_one_cycle(2,slice(3e-2))
learn.save('stage-4')
learn.fit_one_cycle(2,slice(1e-2))
learn.save('stage-5')
# learn.fit_one_cycle(2,slice(1e-2))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,slice(2e-6,1e-2/10))
learn.save('stage-6')
learn.fit_one_cycle(2,slice(2e-6,1e-2/10))
learn.save('stage-7')

learn.unfreeze()
learn.fit_one_cycle(2,slice(2e-6,1e-2/10))
learn.load('stage-8')
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission.csv', index=False)