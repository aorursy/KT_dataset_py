# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
path=Path('/kaggle/input/imet-2020-fgvc7')
df = pd.read_csv(path/'train.csv')

df.head()
tfms = get_transforms(max_lighting=0.1, max_zoom=1.05, max_warp=0.)

#removed the vert flip
np.random.seed(42)

src = (ImageList.from_csv(path, 'train.csv', folder='train', suffix='.png')

       .split_by_rand_pct(0.2)

       .label_from_df(label_delim=' '))
data = (src.transform(tfms, size=224)

        .databunch(bs=200).normalize(imagenet_stats))

#swappng to good default for resnet34
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet34
# creating directories and copying the models to those directories

!mkdir -p /root/.cache/torch/checkpoints

!cp ../input/resnet34fastai/resnet34.pth /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = cnn_learner(data, arch, metrics=[acc_02, f_score])
! ls ../input
!cp ../input/resnet34stage1imet2020/stage-1-rn34.pth /kaggle/working

!cp ../input/resnet34stage1imet2020/stage-2-rn34-2.pth /kaggle/working

!cp ../input/resnet34imet2020stage2/stage-2-rn34\ \(1st\ try\).pth /kaggle/working/stage-2-rn34-1.pth

learn.model_dir = Path('/kaggle/working')
learn.load('stage-1-rn34')

learn.freeze()


#learn.lr_find()

#skip lrfind to save us 10 mins execution
#learn.recorder.plot()

#nothing to plot if didnt run lr_find

lr = 0.01
#learn.fit_one_cycle(1, slice(lr))
learn.save('stage-1-rn34') #save our stage one

#learn.load('stage-2-rn34-2') #load our pretrained finetuned model

learn.load('stage-2-rn34-1') #load our 1st pretrained finetuned model
learn.unfreeze()
#learn.lr_find()
#learn.recorder.plot()
#learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-5)) #did 10 on GCP
learn.path = Path('/kaggle/working')

learn.export()
test = ImageList.from_folder(path/'test')

len(test)
learn = load_learner(Path('/kaggle/working'), test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
thresh = 0.2

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
labelled_preds[:5]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'id':fnames, 'attribute_ids':labelled_preds})
outputpath = Path('/kaggle/working')

df.to_csv(outputpath/'submission.csv', index=False)