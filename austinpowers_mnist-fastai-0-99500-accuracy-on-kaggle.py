import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings # remove Warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import *
import numpy as np

import pandas as pd

%matplotlib inline
path = '/kaggle/input/digit-recognizer/'
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_train.head()
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

df_test.head()
class CustomImageList(ImageList):

    def open(self, fn):

        if(fn.size == 785):

            fn = fn[1:]

        img = fn.reshape(28,28)

        img = np.stack((img,)*1, axis=-1)

        return Image(pil2tensor(img, dtype=np.float32))

    

    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res

    

    @classmethod

    def from_csv_custom_test(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        print(res)

        return res

    

    

    

    @classmethod

    def from_df_custom(cls, path:PathOrStr, df:DataFrame, imgIdx:int=1, header:str='', **kwargs)->'ItemList': 

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res
test = CustomImageList.from_csv_custom_test(path=path, csv_name='test.csv', imgIdx=0)
data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)

                .split_by_rand_pct(.02)

                .label_from_df(cols='label') #cols='label'

                .add_test(test, label=0)

                .transform(get_transforms(do_flip=False,max_rotate=15,max_warp=0.4))

                .databunch(bs=128, num_workers=0)

                .normalize(imagenet_stats))
data.train_ds
data.valid_ds
data.show_batch(rows=5, figsize=(7,7))

model_selected = models.resnet152
learn = cnn_learner(data, model_selected, 

                    metrics=error_rate,callback_fns=ShowGraph,model_dir='/kaggle/working').mixup()
learn.lr_find()
learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
learn.fit_one_cycle(30,min_grad_lr)

learn.save('r152-fit1')
data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)

                #.split_by_rand_pct(.02) --- this split has been removed

                .split_none()# this line replaces the previous split

                .label_from_df(cols='label') #cols='label'

                .add_test(test, label=0)

                .transform(get_transforms(do_flip=False,max_rotate=15,max_warp=0.4))

                .databunch(bs=128, num_workers=0)

                .normalize(imagenet_stats))
data.train_ds
learn.data = data
learn.fit_one_cycle(10,min_grad_lr)
learn.save('r152-fit2')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
learn.fit_one_cycle(10,min_grad_lr)

data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)

                .split_by_rand_pct(.02)

                .label_from_df(cols='label') #cols='label'

                .add_test(test, label=0)

                .transform(get_transforms(do_flip=False,max_rotate=15,max_warp=0.4))

                .databunch(bs=128, num_workers=0)

                .normalize(imagenet_stats))

learn.data = data
# get the predictions

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'label': labels})

submission_df.to_csv(f'submission.csv', index=False)
submission_df
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()

interp.plot_top_losses(9,cmap='gray')
