#install fastai v2



!pip install git+https://github.com/fastai/fastai_dev
%matplotlib inline



import PIL

import pydicom

import numpy as np

from pathlib import Path

from matplotlib import pyplot as plt

import os

import torch

import seaborn as sns

plt.style.use("seaborn")



#checking the input files

print(os.listdir("../input/rsna-intracranial-hemorrhage-detection/"))
#set input data path



dir_csv = '../input/rsna-intracranial-hemorrhage-detection'

dir_train_img = '../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'

dir_test_img = '../input/rsna-test-stage-1-images-png-224x/stage_1_test_png_224x'

dir_train_csv = "../input/rsnsa-intracranial-hemorrahage-detection/stage_1_train.csv"

dir_sample_submission_csv = "../input/rsnsa-intracranial-hemorrahage-detection/stage_1_sample_submission.csv"
#fastai libs



from fastai2.torch_basics import *

from fastai2.test import *

from fastai2.layers import *

from fastai2.data.all import *

from fastai2.optimizer import *

from fastai2.learner import *

from fastai2.metrics import *

from fastai2.vision.all import *

from fastai2.vision.learner import *

from fastai2.vision.models import *

from fastai2.callback.all import *
#get items



items = get_image_files(dir_train_img)

items = [i for i in items if '(copy)' not in i.name]
!mkdir -p data



#read the train csv

df_train = pd.read_csv(f'{dir_csv}/stage_1_train.csv')

df_train.head()
#check for any missing values in label

df_train.Label.isna().sum()
#refactor the csv file



df_train['fn'] = df_train.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.png')

df_train.columns = ['ID', 'probability', 'fn']

df_train['label'] = df_train.ID.apply(lambda x: x.split('_')[-1])

df_train.drop_duplicates('ID', inplace=True)
pivot = df_train.pivot(index='fn', columns='label', values='probability')

pivot.reset_index(inplace=True)

pivot.to_csv('data/train_pivot.csv', index=False)

pivot.head()
from collections import defaultdict



d = defaultdict(list)



for fn in df_train.fn.unique(): 

    d[fn]

    

for tup in df_train.itertuples():

    if tup.probability:

        d[tup.fn].append(tup.label)
#iterate through the items in 'd'



ks, vs = [], []



for k, v in d.items():

    ks.append(k), vs.append(' '.join(v))

    

#save the dataframe

pd.DataFrame(data = {'fn': ks, "labels": vs}).to_csv('data/train_labels_as_string.csv', index = False)
#define a class labeller



class Labeller():

    """path to label"""

    def __init__(self):

        self.df = pd.read_csv("data/train_labels_as_string.csv")

        self.df.set_index("fn", inplace = True)

        

    def __call__(self, path):

        fn = path.name

        labels_txt = self.df.loc[fn].labels

        if isinstance(labels_txt, float) or labels_txt == ' ': return []

        return labels_txt.split(' ')
#create an object of the class



labeler = Labeller()
classes = L(pd.read_csv("data/train_pivot.csv").columns.tolist()[1:])

classes
mcat = MultiCategorize(vocab = classes)

mcat.o2i = classes.val2idx()
mcat.o2i #mapping dictionary
tfms = [PILImage.create, [Labeller(), mcat, OneHotEncode()]]

ds_img_tfms = [ToTensor()]

dsrc = DataSource(items, tfms, splits = RandomSplitter()(items))
test_paths = get_image_files(dir_test_img)

test_tfms = [PILImage.create, [lambda x: np.array([0,0,0,0,0,0])]]

dsrc_test = DataSource([test_paths[0]] + test_paths, test_tfms, splits=[[0], L(range(len(test_paths))).map(lambda x: x + 1)])
dsrc_test[0]
means = [0.1627, 0.1348, 0.1373]

st_devs = [0.2961, 0.2605, 0.1889]



dataset_stats = (means, st_devs)

dataset_stats = broadcast_vec(1, 4, *dataset_stats)
#create train and test data bunch



ds_img_tfms = [ToTensor()]

dl_tfms = [Cuda(), IntToFloatTensor(), Normalize(*dataset_stats)]



#train data bunch

dbch = dsrc.databunch(after_item = ds_img_tfms, after_batch = dl_tfms, bs = 128, num_workers = 4)



#test data bunch

dbch_test = dsrc_test.databunch(after_item = ds_img_tfms, after_batch = dl_tfms, bs = 128, num_workers = 4)
model = create_cnn_model(resnet18, 6, -2, True)
model_segments = model[0][:6], model[0][6:], model[1]
def trainable_params_mod(model): return L(trainable_params(segment) for segment in model_segments)



#define optimization function.

opt_func = partial(Adam, wd=0.01, eps=1e-3)
#create a learner



learn = Learner(

    dbch,

    model,

    loss_func=BCEWithLogitsLossFlat(),

    metrics=[accuracy_multi],

    opt_func=opt_func,

    splitter=trainable_params_mod

)
#freeze



learn.freeze_to(-1)
#find learning rate



learn.lr_find()
#fit one cycle



learn.fit_one_cycle(2, 2e-2)
#save the model

learn.save('phase-1')
#load the model again

learn.load('phase-1');
#freeze

learn.freeze_to(-2)
learn.lr_find(start_lr=1e-8, end_lr=1e-1)
learn.fit_one_cycle(2, [1e-3, 1e-4, 1e-5]) #fit another cycle
learn.recorder.plot_loss() #plot the loss
learn.save('phase-2')



learn.load('phase-2');
learn.unfreeze()
learn.fit_one_cycle(5, [1e-4, 5e-4, 1e-3])
learn.save('phase-3')
learn.load('phase-3')
learn.metrics = [PrecisionMulti(), RecallMulti()]
learn.validate()
learn.dbunch = dbch_test
preds, targs = learn.get_preds()
#create a pred labels and probability

ids = []

labels = []



for path, pred in zip(test_paths, preds):

    for i, label in enumerate(classes):

        ids.append(f"{path.name.split('.')[0]}_{label}")

        predicted_probability = '{0:1.10f}'.format(pred[i].item())

        labels.append(predicted_probability)
#make submission file

pd.DataFrame({'ID': ids, 'Label': labels}).to_csv(f'submission.csv', index=False)