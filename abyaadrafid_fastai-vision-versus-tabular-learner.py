import numpy as np

import pandas as pd

from fastai.vision import *

from fastai.tabular import *

import matplotlib.pyplot as plt

import re
path=Path('/kaggle/working/')
train=pd.read_csv('/kaggle/input/train.csv')

train['index'] = train.index

train.head()
class PixelList(ImageList):

    def open(self,index):

        regex = re.compile(r'\d+')

        fn = re.findall(regex,index)

        df_fn = self.inner_df[self.inner_df.index.values == int(fn[0])]

        img_pixel = df_fn.drop(labels=['label','index'],axis=1).values

        img_pixel = img_pixel.reshape(28,28)

        img_pixel = np.stack((img_pixel,)*3,axis=-1)

        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))
src = (PixelList.from_df(train,'.',cols='index')

      .split_by_rand_pct(0.1)

      .label_from_df(cols='label'))
tfms=get_transforms(rand_pad(padding=5,size=28,mode='zeros'))
data = (src.transform(tfms,size=128)

       .databunch(num_workers=5,bs=48)

       .normalize())

data.show_batch(rows=3,figsize=(10,7))
learner=cnn_learner(data,models.resnet101,metrics=[FBeta(),accuracy,error_rate])
learner.lr_find()

learner.recorder.plot(suggestion=True)
lr=9e-3
learner.fit_one_cycle(5,slice(lr))
learner.save('Resnet1')
learner.lr_find()

learner.recorder.plot(suggestion=True)
learner.freeze_to(2)
learner.fit_one_cycle(4,slice(5e-6,lr/50))
learner.save('Resnetfinal')
train=pd.read_csv('/kaggle/input/train.csv')

data=TabularDataBunch.from_df(path,train,dep_var='label',valid_idx=range(4000,6000))
tablearner=tabular_learner(data,layers=[200,100],ps=[0.001,0.01],emb_drop=0.004,metrics=accuracy)
tablearner.lr_find()

tablearner.recorder.plot(suggestion=True)
lr=2.2e-2
tablearner.fit_one_cycle(10,slice(lr),wd=0.1)
tablearner.lr_find()

tablearner.recorder.plot(suggestion=True)
tablearner.fit_one_cycle(4,4e-6,wd=0.1)
df_test = pd.read_csv('../input/test.csv')

df_test['label'] = 0

df_test['index'] = df_test.index

df_test.head()
learner.load('Resnetfinal')
learner.data.add_test(PixelList.from_df(df_test,path='.',cols='index'))
pred_test = learner.get_preds(ds_type=DatasetType.Test)

test_result = torch.argmax(pred_test[0],dim=1)

result = test_result.numpy()
final = pd.Series(result,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)

submission.to_csv('submit.csv',index=False)