from fastai import *

from fastai.vision import *

import pandas as pd
path = '../input/digit-recognizer/'

path

! ls {path} -l
df = pd.read_csv('../input/digit-recognizer/train.csv')

df2 = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')



df2.columns = df.columns

df_temp = pd.concat([df,df2], join='outer',axis=0 ,ignore_index=False ,sort=False).reset_index(drop=True)

df_temp.to_csv('train.csv',index=False)
df_temp.to_csv('train.csv',index=False)
# df = pd.read_csv(path+'train.csv')

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
test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)

# DigMNIST = CustomImageItemList.from_csv_custom(path=path, csv_name='Dig-MNIST.csv')

data = (CustomImageItemList.from_csv_custom(path='.', csv_name='train.csv')

                       .split_by_rand_pct(.2)

#                       .split_by_idx(list(range(60000,70240)))

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

#                        .transform(get_transforms(do_flip = False))

                       .transform(get_transforms(do_flip = False, max_rotate = 0.), size=49)

                       .databunch(bs=1024, num_workers=16)

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
learn = cnn_learner(data, arch,pretrained = False, metrics=[error_rate, accuracy], model_dir='../kaggle/working')

#learn = cnn_learner(data, models.densenet161, metrics=[error_rate, accuracy], model_dir="/tmp/model/", pretrained=False)
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-02
# learn.fit(20,lr=lr)

learn.fit_one_cycle(15, lr)

learn.save('stage-1-50')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(20, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
# test_df = pd.read_csv(path+'test.csv')

# test_df.head()
tmp_df = pd.read_csv(path+'sample_submission.csv')

tmp_df.head()
# test_df.drop('id',axis = 'columns',inplace = True)
tmp_array = tmp_df.values[:, :]
for i in range(28000):

    img = learn.data.test_ds[i][0]

    tmp_array[i,1] = int(learn.predict(img)[1])
tmp_array
tmp_df = pd.DataFrame(tmp_array,columns = ['ImageId','Label'])

tmp_df
tmp_df.to_csv('submission.csv',index=False)
import pandas as pd

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")