# !conda install -c fastai fastai --yes #using latest 1.0.48 as 1.0.46 learner will have read-only issue
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import numpy as np

import pandas as pd

import re
df_train = pd.read_csv('../input/train.csv')

df_train['fn'] = df_train.index

df_train.head()
class PixelImageItemList(ImageList):

    def open(self,fn):

        regex = re.compile(r'\d+')

        fn = re.findall(regex,fn)

        df = self.inner_df[self.inner_df.fn.values == int(fn[0])]

        df_fn = df[df.fn.values == int(fn[0])]

        img_pixel = df_fn.drop(labels=['label','fn'],axis=1).values

        img_pixel = img_pixel.reshape(28,28)

        img_pixel = np.stack((img_pixel,)*3,axis=-1)

        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))
src = (PixelImageItemList.from_df(df_train,'.',cols='fn')

      .split_by_rand_pct()

      .label_from_df(cols='label'))
data = (src.transform(tfms=(rand_pad(padding=5,size=28,mode='zeros'),[]))

       .databunch(num_workers=2,bs=128)

       .normalize(imagenet_stats))
data.show_batch(rows=3,figsize=(10,7))
print(data.train_ds[0][1]) #label

data.train_ds[0][0] #img
data.train_ds[0][0].data,data.train_ds[0][0].data.shape,data.train_ds[0][0].data.max()
learn = cnn_learner(data,models.resnet50,metrics=accuracy,model_dir='/kaggle/model')
learn.lr_find(end_lr=100)
learn.recorder.plot()
lr = 1e-2
learn.fit_one_cycle(5,slice(lr))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8,slice(2e-5,lr/5))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9,figsize=(6,6))
learn.show_results()
df_test = pd.read_csv('../input/test.csv')

df_test['label'] = 0

df_test['fn'] = df_test.index

df_test.head()
learn.data.add_test(PixelImageItemList.from_df(df_test,path='.',cols='fn'))
pred_test = learn.get_preds(ds_type=DatasetType.Test)
test_result = torch.argmax(pred_test[0],dim=1)

result = test_result.numpy()
# preds = learn.TTA(ds_type=DatasetType.Test)

# pred = torch.argmax(preds[0],dim=1)
final = pd.Series(result,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)

submission.to_csv('fastai-res34-0.992.csv',index=False)
submission.head()