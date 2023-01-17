%reload_ext autoreload

%autoreload 2

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.vision import *

from fastai.metrics import *



import os

path = "../input/"

print(os.listdir(path))
class CustomImageList(ImageList):

    def open(self, fn):

        img = fn.reshape(28,28)

        img = np.stack((img,)*3, axis=-1)

        return Image(pil2tensor(img, dtype=np.float32))

    

    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res

    

    @classmethod

    def from_df_custom(cls, path:PathOrStr, df:DataFrame, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res
test = CustomImageList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)

data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)

                .split_by_rand_pct(.2)

                .label_from_df(cols='label')

                .add_test(test, label=0)

                .transform(get_transforms(do_flip=False))

                .databunch(bs=128, num_workers=0)

                .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(5,5))
learner = cnn_learner(data, models.resnet50, metrics=[ accuracy])
learner.data.batch_size = 1020
learner.lr_find()

learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(5, max_lr=slice(1.10E-02, 4.37E-02))
learner.save("stage1")
learner.unfreeze()

learner.lr_find()

learner.recorder.plot(suggestion=True)
learner.load("stage1")
learner.fit_one_cycle(10, max_lr=slice(1.0E-06,1.0E-04))
learner.recorder.plot_losses()
learner.save("stage2")
interp = ClassificationInterpretation.from_learner(learner)

interp.plot_confusion_matrix()
interp.top_losses()
interp.most_confused()
# get the predictions

predictions, *_ = learner.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission_imbalanced.csv', index=False)