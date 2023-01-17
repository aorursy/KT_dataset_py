# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from fastai.vision import *





# Any results you write to the current directory are saved as output.
import os



os.environ["FASTAI_HOME"] = '/kaggle/.fastai/'



print(os.getenv('FASTAI_HOME'))
# path = Config.data_path()/'planet'

# path.mkdir(parents=True, exist_ok=True)

# path

import os



# os.path = '/kaggle/input/'

pd.read_csv('/kaggle/input/sample_submission_v2.csv').head(5)
!7za -bd -y -so x /kaggle/input/train-jpg.tar.7z | tar xf - -C /kaggle/input/
an_image_path = os.listdir( '/kaggle/input/train-tif-v2')[1]

an_image_path
an_image_path
from PIL import Image

Image.open('/kaggle/input/train-tif-v2/train_0.tif')
from fastai.vision import *
df_tags = pd.read_csv('/kaggle/input/train_v2.csv')

df_tags.head()
df_tags['tags'].value_counts() / len(df_tags)
df_tags['tags'].str.split()
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
np.random.seed(42)

src = (ImageList.from_csv('/kaggle/input', 'train_v2.csv', folder='train-jpg', suffix='.jpg')

       .split_by_rand_pct(0.2)

       .label_from_df(label_delim=' '))


data = (src.transform(tfms, size=128)

        .databunch().normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))

arch = models.resnet50

acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = cnn_learner(data, arch, metrics=[acc_02, f_score])