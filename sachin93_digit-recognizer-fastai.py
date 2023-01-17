import numpy as np 

import pandas as pd 

from fastai import *

from fastai.vision import *
path = Path('../input/digit-recognizer')
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
data.show_batch(3, figsize=(6,6))
learn = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate], model_dir='/kaggle/working/')
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = 1e-2
learn.fit_one_cycle(4,slice(lr))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10,max_lr = slice(4e-6,4e-5))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission.csv', index=False)