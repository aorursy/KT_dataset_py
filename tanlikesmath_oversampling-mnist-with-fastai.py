%reload_ext autoreload

%autoreload 2

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.vision import *

from fastai.metrics import *



import os

path = '../input'

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
learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir='/kaggle/working/models')
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4,max_lr=1e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10,max_lr = slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
# get the predictions

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission_orig.csv', index=False)
train_df = pd.read_csv(path+'/train.csv')

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df,test_size=0.2) # Here we will perform an 80%/20% split of the dataset, with stratification to keep similar distribution in validation set
train_df['label'].hist(figsize = (10, 5))
proportions = pd.DataFrame({0: [0.5],

                            1: [0.05],

                            2: [0.1],

                            3: [0.03],

                            4: [0.03],

                            5: [0.03],

                            6: [0.03],

                            7: [0.5],

                            8: [0.5],

                            9: [0.5],

                           })
imbalanced_train_df = train_df.groupby('label').apply(lambda x: x.sample(frac=proportions[x.name]))
imbalanced_train_df['label'].hist(figsize = (10, 5))
df = pd.concat([imbalanced_train_df,val_df])
data = (CustomImageList.from_df_custom(df=df,path=path, imgIdx=1)

                .split_by_idx(range(len(imbalanced_train_df)-1,len(df)))

                .label_from_df(cols='label')

                .add_test(test, label=0)

                .transform(get_transforms(do_flip=False))

                .databunch(bs=128, num_workers=0)

                .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir='/kaggle/working/models')
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4,max_lr=1e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10,max_lr = slice(1e-6,5e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
# get the predictions

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission_imbalanced.csv', index=False)
data.train_dl.dl.sampler
labels = []

for img,target in data.train_dl.dl:

    labels.append(target)

labels = torch.cat(labels)

plt.hist(labels)
np.bincount([data.train_dl.dataset.y[i].data for i in range(len(data.train_dl.dataset))])
type(np.max(np.bincount([data.train_dl.dataset.y[i].data for i in range(len(data.train_dl.dataset))])))
from torch.utils.data.sampler import WeightedRandomSampler



train_labels = data.train_dl.dataset.y.items

_, counts = np.unique(train_labels,return_counts=True)

class_weights = 1./counts

weights = class_weights[train_labels]

label_counts = np.bincount([learn.data.train_dl.dataset.y[i].data for i in range(len(learn.data.train_dl.dataset))])

total_len_oversample = int(learn.data.c*np.max(label_counts))

data.train_dl.dl.batch_sampler = BatchSampler(WeightedRandomSampler(weights,total_len_oversample), data.train_dl.batch_size,False)
labels = []

for img,target in data.train_dl:

    labels.append(target)

labels = torch.cat(labels)

plt.hist(labels)
class OverSamplingCallback(LearnerCallback):

    def __init__(self,learn:Learner):

        super().__init__(learn)

        self.labels = self.learn.data.train_dl.dataset.y.items

        _, counts = np.unique(self.labels,return_counts=True)

        self.weights = torch.DoubleTensor((1/counts)[self.labels])

        self.label_counts = np.bincount([self.learn.data.train_dl.dataset.y[i].data for i in range(len(self.learn.data.train_dl.dataset))])

        self.total_len_oversample = int(self.learn.data.c*np.max(self.label_counts))

        

    def on_train_begin(self, **kwargs):

        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(WeightedRandomSampler(weights,self.total_len_oversample), self.learn.data.train_dl.batch_size,False)
learn = cnn_learner(data, models.resnet18, metrics=[accuracy], callback_fns = [partial(OverSamplingCallback)], model_dir='/kaggle/working/models')
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4,1e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10,5e-4)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
# get the predictions

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission_oversampled.csv', index=False)