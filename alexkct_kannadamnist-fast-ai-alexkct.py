!nvidia-smi
#accelerate pandas

# !pip install modin[ray]

# !pip install pandas
from fastai import *

from fastai.vision import *

import pandas as pd
path = '../input/Kannada-MNIST//'

path

! ls {path} -l
# df1 = pd.read_csv(path+'train.csv')

# df2 = pd.read_csv(path+'Dig-MNIST.csv')

# df = pd.concat([df1,df2],axis=0)

# df.to_csv('train.csv',index = False)
# df.iloc[:,1:] = df.iloc[:,1:].replace(range(128,254),'255')

# df.iloc[:,1:] = df.iloc[:,1:].replace(range(1,127),'0')
# df.to_csv('train.csv',index = False)
class CustomImageItemList(ImageList):

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
#best 0.986

# test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=1)

# data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')

#                        .split_by_rand_pct(0.2)

#                        .label_from_df(cols='label')

#                        .add_test(test, label=0)

#                        .transform(get_transforms(do_flip = False, max_rotate = 0.), size=49)

#                        .databunch(bs=256, num_workers=16)

#                        .normalize(mnist_stats))

# data
test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=1)

# DigMNIST = CustomImageItemList.from_csv_custom(path=path, csv_name='Dig-MNIST.csv')

data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')

                       .split_by_rand_pct(0.2)

#                       .split_by_idx(list(range(60000,70240)))

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

#                        .transform(get_transforms(do_flip = False))

                       .transform(get_transforms(do_flip = False, max_rotate = 0.,max_zoom = 1.), size=49)#, p_affine = 0.), size=49)

                       .databunch(bs=256, num_workers=16)

                       .normalize(mnist_stats))

data
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet50

# arch = models.resnet152

# arch = models.resnet18

arch
!mkdir -p /tmp/.cache/torch/checkpoints

!cp ../input/resnet50/resnet50-19c8e357.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth

# !cp ../input/resnet152/resnet152-b121ed2d.pth /tmp/.cache/torch/checkpoints/resnet152-b121ed2d.pth
# !cd ../input/radam-pytorch/RAdam/

# !ls
# !cp ../input/radam-pytorch/RAdam .

# import radam

# optar = partial(radam)
#learn = cnn_learner(data, arch,pretrained = False, loss_func = nn.CrossEntropyLoss(), metrics=[error_rate,accuracy], model_dir='../kaggle/working').to_fp16()



# learn = cnn_learner(data, arch,pretrained = False, opt_func = optar, loss_func = nn.CrossEntropyLoss(), metrics=[error_rate,accuracy], model_dir='../kaggle/working').to_fp16()

#learn = cnn_learner(data, models.densenet161, metrics=[error_rate, accuracy], model_dir="/tmp/model/", pretrained=False)
#FOR TESTING

learn = cnn_learner(data, arch,pretrained = False, loss_func = nn.MultiMarginLoss(), metrics=[error_rate,accuracy], model_dir='../kaggle/working')
# lr = 1e-02
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
# 0.9986

learn.fit_one_cycle(20)

learn.save('stage-1-50')
# #TEST

# learn.fit_one_cycle(3,1e-02)
# learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()
# lr = slice(1e-04, 1e-03)
# learn.fit_one_cycle(6,lr)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(20, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
tmp_df = pd.read_csv(path+'sample_submission.csv')

tmp_df.head()
for i in range(0,5000):

    img = learn.data.test_ds[i][0]

    tmp_df.loc[i]=[i,int(learn.predict(img)[1])]

tmp_df
tmp_df.to_csv('submission.csv',index=False)