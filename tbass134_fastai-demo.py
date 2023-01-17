# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from os.path import join, exists, expanduser

from os import listdir, makedirs



cache_dir = expanduser(join('~', '.torch'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)
!cp ../input/fastai-model-48/stage3-sm.pth /tmp/.torch/models/stage3-sm.pth
# os.mkdir('./working')

print(os.listdir("../working"))
from fastai.vision import *

from fastai.metrics import error_rate
from pathlib import Path

path = Path('../input/histopathologic-cancer-detection');

path = Path('.');



path.ls()

df = pd.read_csv(path/'../input/histopathologic-cancer-detection/train_labels.csv'); df.head()
img_path = path/'../input/histopathologic-cancer-detection/train'/df.iloc[0].id

img = img_path.with_suffix('.tif')

open_image(img)
from PIL import Image

df_sample = df.copy()

df_sample = df.sample(n=10)

for index, row in df_sample.iterrows():

    file = '../input/histopathologic-cancer-detection/train/'+row.id+'.tif'

    image = Image.open(file)

    print(image.size)
src = (ImageItemList.from_csv(path/'../input/histopathologic-cancer-detection',csv_name='train_labels.csv', folder='train',suffix='.tif')

       .random_split_by_pct(0.2, seed=42)

       .label_from_df()

       .add_test_folder()

      )
tfms = get_transforms()

data = (src.transform(tfms, size=48)

        .databunch()

       .normalize(imagenet_stats))
data.show_batch(rows=3)
from sklearn.metrics import roc_auc_score



def auc_score(y_pred,y_true,tens=True):

    score = roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score = tensor(score)

    return score



learn = create_cnn(arch=models.resnet34, data=data, metrics=[error_rate,auc_score], model_dir='/tmp/.torch/models')
lr=0.001
learn.fit_one_cycle(1, slice(lr))
learn.save('stage1')
learn.load("stage1")

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, slice(1e-6, lr/5))
learn.save("stage2")

learn.load("stage2")

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, slice(1e-5, lr/5))
learn.save("stage3")
learn.load("stage3")

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, slice(1e-6, lr/5))
learn.save("stage3-sm")
learn.load("stage3-sm")
tfms = get_transforms()

data = (src.transform(tfms, size = 96)

       .databunch().normalize(imagenet_stats))

learn.data = data 
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, slice(1e-3))
# learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, slice(1e-6))
learn.save('stage3-lg')
# learn.fit_one_cycle(4, slice(1e-5))
# learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(4, slice(1e-6, lr/5))
# learn.save("stage4-lg")
# pred_score = auc_score(preds,y).item()

# print('The validation AUC is {}.'.format(pred_score))
# submissions = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

# id_list = list(submissions.id)
# preds,y = learn.TTA(ds_type=DatasetType.Test)

# pred_list = list(preds[:,1])
# pred_dict = dict((key, value.item()) for (key, value) in zip(learn.data.test_ds.items,pred_list))

# pred_ordered = [pred_dict[Path('../input/histopathologic-cancer-detection/test/' + id + '.tif')] for id in id_list]
# submissions = pd.DataFrame({'id':id_list,'label':pred_ordered})

# submissions.to_csv("../working/my_submission_{}.csv".format(pred_score),index = False)
# submissions.info()
# from IPython.display import HTML

# import pandas as pd

# import numpy as np

# import base64



# # function that takes in a dataframe and creates a text link to  

# # download it (will only work for files < 2MB or so)

# def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

#     csv = df.to_csv()

#     b64 = base64.b64encode(csv.encode())

#     payload = b64.decode()

#     html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

#     html = html.format(payload=payload,title=title,filename=filename)

#     return HTML(html)



# # create a random sample dataframe

# df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# # create a link to download the dataframe

# # create_download_link(df[])



# # ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 
# df_split = np.array_split(submissions, 4)
# create_download_link(submissions)
# len(df_split)